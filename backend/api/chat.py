from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    user_id: str

@router.post("/chat/completions")
async def chat_completion(request: ChatRequest):
    # In Phase 4, this integrates with ChromaDB (RAG) and an LLM API.
    # Currently simulates context-aware memory.
    
    user_message = request.messages[-1].content.lower()
    
    response = "I'm here to help you manage your wellness journey. Would you like to review your active Action Plan?"
    
    if "stress" in user_message:
        response = "I see you're mentioning stress. Last month, your assessment indicated a Moderate Risk. Are you experiencing any sleep disturbances along with this stress?"
    elif "diabetes" in user_message:
        response = "Diabetes is a chronic condition that affects how your body turns food into energy. Have you recently taken our General Wellness assessment?"
        
    return {
        "status": "success",
        "message": {
            "role": "assistant",
            "content": response
        }
    }
