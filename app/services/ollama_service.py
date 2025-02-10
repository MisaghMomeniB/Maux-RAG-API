from typing import List, Dict, AsyncGenerator
import json
import httpx
from app.config.settings import settings

class OllamaService:
    def __init__(self):
        self.base_url = settings.OLLAMA_API_URL  # آدرس API سرور Ollama
        self.model = settings.OLLAMA_MODEL  # مدلی که برای چت استفاده می‌شود

    async def create_chat_completion(self, messages: List[Dict], model: str = None):
        url = f"{self.base_url}/api/generate"
        model = model or self.model
        payload = {"model": model, "messages": messages}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def create_chat_completion_stream(self, messages: List[Dict], model: str = None) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/generate"
        model = model or self.model
        payload = {"model": model, "messages": messages, "stream": True}

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield json.dumps({"choices": [{"delta": {"content": line}}]})