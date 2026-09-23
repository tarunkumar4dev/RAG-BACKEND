from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


# ═══════════════════════════════════════════════════════════════════════
# Incoming Meta Webhook Models
# ═══════════════════════════════════════════════════════════════════════

class TextContent(BaseModel):
    body: str

class InteractiveReply(BaseModel):
    id: str
    title: str

class InteractiveContent(BaseModel):
    type: str
    button_reply: Optional[InteractiveReply] = None
    list_reply: Optional[InteractiveReply] = None

class Message(BaseModel):
    from_: str = Field(alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[TextContent] = None
    interactive: Optional[InteractiveContent] = None

class ContactProfile(BaseModel):
    name: str

class Contact(BaseModel):
    profile: ContactProfile
    wa_id: str

class Value(BaseModel):
    messaging_product: str
    metadata: Dict[str, Any]
    contacts: Optional[List[Contact]] = None
    messages: Optional[List[Message]] = None

class Change(BaseModel):
    value: Value
    field: str

class Entry(BaseModel):
    id: str
    changes: List[Change]

class WhatsAppWebhookPayload(BaseModel):
    object: str
    entry: List[Entry]

