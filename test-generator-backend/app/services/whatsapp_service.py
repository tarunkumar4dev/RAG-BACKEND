import httpx
import logging
import json
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

META_API_VERSION = "v20.0"

async def _send_meta_request(payload: dict) -> bool:
    """Send an outbound request to Meta Graph API."""
    phone_id = os.environ.get("WHATSAPP_PHONE_ID", "")
    token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
    
    if not phone_id or not token:
        logger.error("WhatsApp credentials missing in environment.")
        return False
        
    url = f"https://graph.facebook.com/{META_API_VERSION}/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=10.0)
            if resp.status_code not in (200, 201):
                logger.error(f"Meta API Error: {resp.status_code} - {resp.text}")
                return False
            return True
    except Exception as e:
        logger.error(f"Failed to send WhatsApp message: {e}")
        return False

async def send_text_message(phone: str, text: str) -> bool:
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "text",
        "text": {"body": text}
    }
    return await _send_meta_request(payload)

async def send_interactive_buttons(phone: str, body_text: str, buttons: List[Dict[str, str]]) -> bool:
    """
    buttons should be a list of dicts: [{"id": "btn1", "title": "Option 1"}] (max 3)
    """
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {
                        "type": "reply",
                        "reply": {"id": b["id"], "title": b["title"]}
                    } for b in buttons[:3]
                ]
            }
        }
    }
    return await _send_meta_request(payload)

async def send_interactive_list(phone: str, body_text: str, button_text: str, sections: List[dict]) -> bool:
    """
    sections format: [{"title": "Section 1", "rows": [{"id": "r1", "title": "Row 1", "description": "desc"}]}]
    """
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "header": {"type": "text", "text": "Select an option"},
            "body": {"text": body_text},
            "action": {
                "button": button_text,
                "sections": sections
            }
        }
    }
    return await _send_meta_request(payload)

async def send_document_by_url(phone: str, document_url: str, filename: str, caption: str = "") -> bool:
    payload = {
        "messaging_product": "whatsapp",
        "to": phone,
        "type": "document",
        "document": {
            "link": document_url,
            "filename": filename,
            "caption": caption
        }
    }
    return await _send_meta_request(payload)

