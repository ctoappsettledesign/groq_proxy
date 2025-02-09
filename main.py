from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
import httpx
import json
from typing import Dict, Any
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from httpx import AsyncClient, Limits, Timeout
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    docs_url=None,    # Disable docs (Swagger UI)
    redoc_url=None    # Disable redoc
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add API key security scheme
api_key_header = APIKeyHeader(name="Authorization", auto_error=True)

# API key validation function
async def get_api_key(api_key: str = Depends(api_key_header)):
    # Remove 'Bearer ' prefix if present
    if api_key.startswith('Bearer '):
        api_key = api_key[7:]
    
    if api_key != "az-intital-key":
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key

# Replace the hardcoded GROQ_API_KEY with environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")
    
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Create a global AsyncClient with optimized settings
http_client = AsyncClient(
    limits=Limits(max_keepalive_connections=100, max_connections=100),
    timeout=Timeout(timeout=30.0, connect=5.0),
)

async def stream_groq_response(data: Dict[Any, Any]):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    async with http_client.stream(
        "POST",
        GROQ_API_URL,
        headers=headers,
        json=data,
        timeout=None
    ) as response:
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Groq API error")
        
        first_chunk = True
        async for line in response.aiter_lines():
            if line:
                if line.startswith("data: "):
                    line = line[6:]
                if line.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                
                try:
                    chunk = json.loads(line)
                    if not chunk.get('choices'):
                        continue
                        
                    delta = chunk['choices'][0].get('delta', {})
                    if delta and 'content' in delta and delta['content']:
                        content = delta['content']
                        if content.strip():
                            response_chunk = f"data: {line}\n\n"
                            if first_chunk:
                                first_chunk = False
                            yield response_chunk
                except json.JSONDecodeError:
                    continue

@app.get("/")
async def root():
    return {"message": "Server is running"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Dict[Any, Any], api_key: str = Depends(get_api_key)):
    # Ensure stream is set to True as we're creating a streaming endpoint
    request["stream"] = True
    
    return StreamingResponse(
        stream_groq_response(request),
        media_type="text/event-stream"
    )

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8006))
    uvicorn.run(app, host="0.0.0.0", port=port)
