from fastapi import APIRouter, Request, Response, HTTPException, BackgroundTasks, Query
from app.models.whatsapp import WhatsAppWebhookPayload
from app.core.config import settings
import logging
import httpx
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/whatsapp", tags=["WhatsApp"])

@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    """Meta webhook verification handshake."""
    import os
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
    
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        logger.info("WhatsApp webhook verified successfully!")
        return Response(content=hub_challenge, media_type="text/plain")
    
    raise HTTPException(status_code=403, detail="Invalid verification token")

@router.post("/webhook")
async def receive_webhook(payload: dict, request: Request):
    """
    Receive incoming messages from WhatsApp.
    Returns 200 OK immediately.
    """
    print("\n\n" + "="*50)
    print("🔥 WEBHOOK RECEIVED FROM META! 🔥")
    print(payload)
    print("="*50 + "\n\n")
    try:
        # Extract messages safely
        entries = payload.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])
                
                if not messages:
                    continue
                    
                for message in messages:
                    phone = message.get("from")
                    message_id = message.get("id")
                    
                    # Create a dummy object to pass to process_whatsapp_message
                    class DummyMsg:
                        pass
                    msg_obj = DummyMsg()
                    msg_obj.type = message.get("type")
                    if msg_obj.type == "text":
                        class DummyText: pass
                        msg_obj.text = DummyText()
                        msg_obj.text.body = message.get("text", {}).get("body", "")
                    elif msg_obj.type == "interactive":
                        class DummyInteractive: pass
                        msg_obj.interactive = DummyInteractive()
                        inter = message.get("interactive", {})
                        msg_obj.interactive.type = inter.get("type")
                        if msg_obj.interactive.type == "button_reply":
                            class DummyReply: pass
                            msg_obj.interactive.button_reply = DummyReply()
                            msg_obj.interactive.button_reply.id = inter.get("button_reply", {}).get("id")
                        elif msg_obj.interactive.type == "list_reply":
                            class DummyReply: pass
                            msg_obj.interactive.list_reply = DummyReply()
                            msg_obj.interactive.list_reply.id = inter.get("list_reply", {}).get("id")

                    # 1. State machine parsing & fast replies
                    from app.services.whatsapp_flow_service import process_whatsapp_message
                    
                    host = str(request.base_url).rstrip("/")
                    
                    # We process the message asynchronously but wait for the FAST reply (Menu/Buttons)
                    # The generation is offloaded inside the flow service
                    await process_whatsapp_message(phone, msg_obj, message_id, host)
                        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        
    # ALWAYS return 200 immediately for Meta
    return Response(content="EVENT_RECEIVED", status_code=200)


from pydantic import BaseModel
class WorkerPayload(BaseModel):
    phone: str
    message_id: str
    context: dict = {}

@router.post("/generate_worker")
async def generate_worker(payload: WorkerPayload, background_tasks: BackgroundTasks):
    """
    This route runs in the background. Vercel will allow it to run up to maxDuration.
    Make sure to configure vercel.json to allow maxDuration (e.g. 300s) for this route.
    """
    logger.info(f"Worker started for {payload.phone}")
    
    from app.services.whatsapp_service import send_text_message, send_document_by_url
    
    # 2. DO HEAVY LIFTING
    async def heavy_work():
        import asyncio
        import uuid
        from app.models.test_generator import TestGenerationRequest, ChapterSection, DifficultyLevel, QuestionFormat
        from app.services.rag_service import retrieve_context
        from app.services.test_generator_service import generate_questions
        from app.services.export_service import generate_pdf
        from app.services.whatsapp_service import send_document_by_url, send_text_message
        from app.core.database import get_supabase_admin
        
        ctx = payload.context
        subject = ctx.get("subject", "Science")
        raw_class = str(ctx.get("class_grade") or ctx.get("class") or "10")
        class_val = raw_class.lower().replace("class", "").strip()
        raw_chapter = ctx.get("chapter", "Light")
        marks = int(ctx.get("marks", 20))
        
        try:
            # 1. Retrieve Context first (blocking)
            chunks = await asyncio.to_thread(retrieve_context, [raw_chapter], None, subject, class_val, 20)
            resolved_chapter = chunks[0]["chapter"] if chunks else raw_chapter
            
            # 2. Prepare request with resolved chapter name and correct Pydantic fields
            mcq_qty = max(1, marks // 5)
            short_qty = max(1, (marks - mcq_qty) // 3)
            
            req = TestGenerationRequest(
                exam_title=f"Class {class_val} {subject} - {resolved_chapter} Test",
                board="CBSE",
                class_grade=class_val,
                subject=subject,
                teacher_id=payload.phone,
                chapters=[
                    ChapterSection(
                        chapter=resolved_chapter,
                        quantity=mcq_qty,
                        difficulty=DifficultyLevel.MEDIUM,
                        format=QuestionFormat.MCQ,
                        marks_per_question=1
                    ),
                    ChapterSection(
                        chapter=resolved_chapter,
                        quantity=short_qty,
                        difficulty=DifficultyLevel.MEDIUM,
                        format=QuestionFormat.SHORT_ANSWER,
                        marks_per_question=3
                    )
                ]
            )
            
            # 3. Generate Questions (blocking)
            result = await asyncio.to_thread(generate_questions, req, chunks)
            raw_questions = result if isinstance(result, list) else result.get("questions", [])
            questions_dicts = [
                q.model_dump() if hasattr(q, "model_dump") else (q.dict() if hasattr(q, "dict") else q)
                for q in raw_questions
            ]
            
            # 4. Export to PDF (blocking)
            title = f"Class {class_val} {subject} - {resolved_chapter} ({marks} Marks)"
            pdf_bytes = await asyncio.to_thread(generate_pdf, questions_dicts, exam_title=title)
            
            # 5. Upload to Supabase Storage (public assignments bucket)
            from app.core.database import get_supabase
            supabase = get_supabase()
            file_name = f"whatsapp/{payload.phone}_{uuid.uuid4().hex}.pdf"
            await asyncio.to_thread(
                supabase.storage.from_("assignments").upload, 
                file_name, 
                pdf_bytes, 
                {"content-type": "application/pdf"}
            )
            public_url = supabase.storage.from_("assignments").get_public_url(file_name)
            
            # 6. Send to WhatsApp
            await send_document_by_url(
                payload.phone,
                public_url,
                f"{resolved_chapter}_Test.pdf",
                f"✅ Here is your AI generated test paper for *{resolved_chapter}*! 🎉"
            )
        except Exception as e:
            logger.error(f"Generation error in worker: {e}", exc_info=True)
            err_msg = str(e)
            if "No NCERT content found" in err_msg:
                friendly_msg = f"Sorry, could not find NCERT content for chapter '{raw_chapter}' in Class {class_val} {subject}. Please check the chapter name or try 'Light', 'Electricity', etc."
            elif "402" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                friendly_msg = "Gemini API credits exhausted. Please recharge credits in Google AI Studio or update your API key in .env."
            else:
                friendly_msg = f"Sorry, an error occurred while generating the test: {err_msg}"
            await send_text_message(payload.phone, friendly_msg)
            
    background_tasks.add_task(heavy_work)
    
    logger.info(f"Worker finished for {payload.phone}")
    return {"status": "success"}
