from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import process_message


app = FastAPI(
    title="RAG E-commerce Customer Support Chatbot",
    description=(
        "Local API for the end-to-end customer support chatbot. "
        "The pipeline performs language detection, sentiment analysis, "
        "intent classification, routing, retrieval, and grounded response generation."
    ),
    version="1.0.0"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {
        "message": "RAG E-commerce Customer Support Chatbot API",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:
        result = process_message(
            message
        )

        if "error" in result:
            raise HTTPException(
                status_code=400,
                detail=result["error"]
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chatbot processing failed: {str(exc)}"
        )
