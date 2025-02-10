from app.services.chroma_service import ChromaService
from app.services.openai_service import openai_service
from app.services.avalai_service import AvalaiService
from app.services.ollama_service import OllamaService
import uuid
import json
from typing import AsyncGenerator
from app.config.settings import settings

class RAGService:
    def __init__(self):
        self.collection_name = "RAG_COLLECTION"
        self.chroma_service = ChromaService()

        if settings.PROVIDER == "openai":
            self.provider_service = openai_service
        elif settings.PROVIDER == "avalai":
            self.provider_service = AvalaiService()
        elif settings.PROVIDER == "ollama":
            self.provider_service = OllamaService()
        else:
            raise ValueError("Invalid provider selected")

    def generate_response(self, messages: list, context: str, model: str = settings.CHAT_MODEL):
        messages_with_context = messages.copy()
        system_msg = next((msg for msg in messages_with_context if msg.role == "system"), None)
        
        if system_msg:
            system_msg.content = f"{system_msg.content}\n\nUse this context to answer the question:\n{context}"
        else:
            messages_with_context.insert(0, {"role": "system", "content": f"Use this context to enhance your responses:\n{context}"})

        messages_dict = [msg.model_dump() for msg in messages_with_context]
        return self.provider_service.create_chat_completion(messages_dict, model)

    async def generate_stream_response(self, messages: list, context: str, model: str = settings.CHAT_MODEL) -> AsyncGenerator[str, None]:
        messages_with_context = messages.copy()
        system_msg = next((msg for msg in messages_with_context if msg.role == "system"), None)
        
        if system_msg:
            system_msg.content = f"{system_msg.content}\n\nUse this context to answer the question:\n{context}"
        else:
            messages_with_context.insert(0, {"role": "system", "content": f"Use this context to enhance your responses:\n{context}"})

        messages_dict = [msg.model_dump() for msg in messages_with_context]
        async for chunk in self.provider_service.create_chat_completion_stream(messages_dict, model):
            yield chunk

rag_service = RAGService()
