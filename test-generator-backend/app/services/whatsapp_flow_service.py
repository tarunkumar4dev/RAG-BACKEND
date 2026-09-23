import logging
from typing import Dict, Any, Optional
from app.services.whatsapp_service import send_text_message, send_interactive_buttons, send_interactive_list

logger = logging.getLogger(__name__)

# Mock database for session state (In a real app, use the Supabase whatsapp_sessions table)
# For the demo, we use an in-memory dictionary to hold state to avoid Supabase setup friction right now.
_sessions: Dict[str, Dict[str, Any]] = {}

async def process_whatsapp_message(phone: str, message: Any, message_id: str, host_url: str):
    """
    Main State Machine for WhatsApp Flow.
    """
    # Initialize session if not exists
    if phone not in _sessions:
        _sessions[phone] = {"current_step": "START", "context": {}, "processed_msg_ids": set()}
        
    session = _sessions[phone]
    
    # Idempotency check: ignore duplicate webhooks from Meta
    if "processed_msg_ids" not in session:
        session["processed_msg_ids"] = set()
        
    if message_id in session["processed_msg_ids"]:
        logger.info(f"Ignoring duplicate webhook for message_id: {message_id}")
        return
        
    session["processed_msg_ids"].add(message_id)
    
    current_step = session["current_step"]
    
    # Extract text from incoming message
    incoming_text = ""
    if message.type == "text":
        incoming_text = message.text.body.strip()
    elif message.type == "interactive":
        if message.interactive.type == "button_reply":
            incoming_text = message.interactive.button_reply.id
        elif message.interactive.type == "list_reply":
            incoming_text = message.interactive.list_reply.id

    logger.info(f"Processing message from {phone}. Input: {incoming_text}, Context: {session.get('context')}")

    # ── SMART INTENT / PREFIX-BASED ROUTER ──
    try:
        # Case 1: Greetings or Reset
        if incoming_text.lower() in ["hi", "hello", "menu", "reset", "start", "restart"]:
            session["context"] = {}
            await send_interactive_buttons(
                phone,
                "👋 *Welcome to A4AI Test Generator!*\n\nLet's create a test paper. Which class are you teaching?",
                [
                    {"id": "class_10", "title": "Class 10"},
                    {"id": "class_11", "title": "Class 11"},
                    {"id": "class_12", "title": "Class 12"}
                ]
            )
            return

        # Case 2: Class Selection (ID starts with 'class_')
        if incoming_text.startswith("class_"):
            class_val = incoming_text.replace("class_", "").strip()
            session["context"]["class_grade"] = class_val
            session["context"]["class"] = class_val
            
            # Show subject buttons based on class
            if class_val in ["9", "10"]:
                subj_buttons = [
                    {"id": "subj_science", "title": "Science"},
                    {"id": "subj_mathematics", "title": "Mathematics"},
                    {"id": "subj_english", "title": "English"}
                ]
            else:
                subj_buttons = [
                    {"id": "subj_physics", "title": "Physics"},
                    {"id": "subj_chemistry", "title": "Chemistry"},
                    {"id": "subj_mathematics", "title": "Mathematics"}
                ]
                
            await send_interactive_buttons(
                phone,
                f"Great! Class {class_val}. Which subject?",
                subj_buttons
            )
            return

        # Case 3: Subject Selection (ID starts with 'subj_')
        if incoming_text.startswith("subj_"):
            raw_subj = incoming_text.replace("subj_", "").strip().capitalize()
            class_val = str(session["context"].get("class_grade", "10"))
            
            # Normalization
            if class_val in ["9", "10"] and raw_subj.lower() in ["physics", "chemistry", "biology", "bio"]:
                subj_val = "Science"
            elif raw_subj.lower() in ["math", "maths"]:
                subj_val = "Mathematics"
            else:
                subj_val = raw_subj
                
            session["context"]["subject"] = subj_val
            
            await send_text_message(
                phone,
                f"Selected *{subj_val}* for Class {class_val}.\n\n📖 Please type the name of the chapter (e.g., Light, Electricity, or Triangles):"
            )
            return

        # Case 4: Marks Selection (ID starts with 'marks_')
        if incoming_text.startswith("marks_"):
            marks_val = int(incoming_text.replace("marks_", "").strip())
            session["context"]["marks"] = marks_val
            
            # Fallback if chapter wasn't set
            if not session["context"].get("chapter"):
                session["context"]["chapter"] = "Light"
                
            ch_name = session["context"]["chapter"]
            subj_name = session["context"].get("subject", "Science")
            class_name = session["context"].get("class_grade", "10")
            
            # Confirmation message with clear details
            await send_text_message(
                phone,
                f"🚀 *Generating your AI test paper...*\n\n📋 *Details:*\n• Class: {class_name}\n• Subject: {subj_name}\n• Chapter: {ch_name}\n• Total Marks: {marks_val}\n\n⏱️ This usually takes about 30-40 seconds. Please wait!"
            )
            
            # Trigger background worker
            import httpx
            import asyncio
            worker_url = f"{host_url}/api/v1/whatsapp/generate_worker"
            
            worker_payload = {
                "phone": phone,
                "message_id": message_id,
                "context": session["context"]
            }
            
            async def fire_and_forget():
                try:
                    async with httpx.AsyncClient() as client:
                        await client.post(worker_url, json=worker_payload, timeout=0.5)
                except httpx.ReadTimeout:
                    pass
                except Exception as ex:
                    logger.error(f"Failed to call generate_worker: {ex}")
                    
            asyncio.create_task(fire_and_forget())
            
            # Clear context for next run
            session["context"] = {}
            return

        # Case 5: Plain text (Chapter Name)
        if message.type == "text":
            chapter_name = incoming_text.strip()
            session["context"]["chapter"] = chapter_name
            subj_name = session["context"].get("subject", "Science")
            class_name = session["context"].get("class_grade", "10")
            
            await send_interactive_buttons(
                phone,
                f"Chapter saved: *{chapter_name}* ({subj_name}, Class {class_name}).\n\nHow many marks should this paper be?",
                [
                    {"id": "marks_20", "title": "20 Marks (Unit)"},
                    {"id": "marks_50", "title": "50 Marks (Half)"},
                    {"id": "marks_80", "title": "80 Marks (Full)"}
                ]
            )
            return

        # Fallback for unexpected input
        await send_text_message(phone, "Sorry, I didn't understand that. Type *Hi* to restart.")

    except Exception as e:
        logger.error(f"Error in flow: {e}", exc_info=True)
        await send_text_message(phone, "Oops! Something went wrong. Type *Hi* to restart.")

