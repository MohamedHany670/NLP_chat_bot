from pathlib import Path
import json
import re
import os
import joblib

import torch
import torch.nn as nn

import chromadb
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from groq import Groq


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"
SENTIMENT_DIR = MODELS_DIR / "sentiment_model"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"


# =========================================================
# Environment variables
# =========================================================

load_dotenv(PROJECT_ROOT / ".env")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY was not found in the .env file."
    )

groq_client = Groq(
    api_key=groq_api_key
)


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# Load Language Detection Model
# =========================================================

language_model = joblib.load(
    MODELS_DIR / "language_model.pkl"
)

language_vectorizer = joblib.load(
    MODELS_DIR / "language_vectorizer.pkl"
)


language_names = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese"
}


def detect_language(text):
    normalized = text.strip().lower()

    # Remove simple punctuation so phrases like "hello!" still match.
    normalized_short = re.sub(
        r"[^\w\s]",
        "",
        normalized
    ).strip()

    english_short_phrases = {
        "hello",
        "hi",
        "hey",
        "hello there",
        "hi there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "thank you",
        "thanks",
        "thanks a lot",
        "thank you very much",
        "goodbye",
        "bye",
        "see you",
        "see you later"
    }

    if normalized_short in english_short_phrases:
        return "en", "English"

    vector = language_vectorizer.transform(
        [text]
    )

    language_code = language_model.predict(
        vector
    )[0]

    language_name = language_names.get(
        language_code,
        language_code
    )

    return language_code, language_name


# =========================================================
# Sentiment BiLSTM
# =========================================================

class BiLSTMSentimentClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_classes
        )

    def forward(self, input_ids):

        embedded = self.embedding(
            input_ids
        )

        _, (hidden, _) = self.lstm(
            embedded
        )

        hidden_forward = hidden[-2]
        hidden_backward = hidden[-1]

        hidden_combined = torch.cat(
            (
                hidden_forward,
                hidden_backward
            ),
            dim=1
        )

        output = self.dropout(
            hidden_combined
        )

        return self.fc(output)


# =========================================================
# Load Sentiment Files
# =========================================================

with open(
    SENTIMENT_DIR / "vocab.json",
    "r",
    encoding="utf-8"
) as file:
    sentiment_vocab = json.load(file)


with open(
    SENTIMENT_DIR / "config.json",
    "r",
    encoding="utf-8"
) as file:
    sentiment_config = json.load(file)


sentiment_model = BiLSTMSentimentClassifier(
    vocab_size=sentiment_config["vocab_size"],
    embedding_dim=sentiment_config["embedding_dim"],
    hidden_dim=sentiment_config["hidden_dim"],
    num_classes=sentiment_config["num_classes"]
)


sentiment_model.load_state_dict(
    torch.load(
        SENTIMENT_DIR / "sentiment_model.pt",
        map_location=device
    )
)

sentiment_model = sentiment_model.to(
    device
)

sentiment_model.eval()


id_to_sentiment = {
    int(key): value
    for key, value
    in sentiment_config[
        "id_to_sentiment"
    ].items()
}


MAX_LENGTH = sentiment_config[
    "max_length"
]


# =========================================================
# Sentiment preprocessing
# =========================================================

def clean_sentiment_text(text):
    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize_sentiment(text):
    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def encode_sentiment_text(text):

    tokens = tokenize_sentiment(
        text
    )

    ids = [
        sentiment_vocab.get(
            token,
            sentiment_vocab["<UNK>"]
        )
        for token in tokens
    ]

    ids = ids[:MAX_LENGTH]

    if len(ids) < MAX_LENGTH:

        ids += [
            sentiment_vocab["<PAD>"]
        ] * (
            MAX_LENGTH - len(ids)
        )

    return ids


def predict_sentiment(text):

    cleaned = clean_sentiment_text(
        text
    )

    encoded = encode_sentiment_text(
        cleaned
    )

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        output = sentiment_model(
            input_ids
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_id
        ].item()

    return (
        id_to_sentiment[predicted_id],
        confidence
    )


# =========================================================
# Intent Classification
# =========================================================

intent_model = joblib.load(
    MODELS_DIR / "intent_model.pkl"
)

intent_vectorizer = joblib.load(
    MODELS_DIR / "intent_vectorizer.pkl"
)


def clean_intent_text(text):

    text = str(text).lower()

    text = re.sub(
        r"\{\{.*?\}\}",
        " entity ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def predict_intent(text):

    cleaned = clean_intent_text(
        text
    )

    vector = intent_vectorizer.transform(
        [cleaned]
    )

    prediction = intent_model.predict(
        vector
    )[0]

    return prediction


# =========================================================
# Translation
# =========================================================

def translate_to_english(
    text,
    language_name
):

    if language_name == "English":
        return text

    completion = (
        groq_client
        .chat
        .completions
        .create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the customer "
                        "message into clear English. "
                        "Preserve the original "
                        "e-commerce support intent. "
                        "Return only the English "
                        "translation."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],

            temperature=0.0,
            max_completion_tokens=100
        )
    )

    return (
        completion
        .choices[0]
        .message
        .content
        .strip()
    )


# =========================================================
# RAG - Embedding Model
# =========================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================================================
# RAG - Chroma
# =========================================================

chroma_client = chromadb.PersistentClient(
    path=str(VECTOR_DB_DIR)
)

collection = (
    chroma_client
    .get_collection(
        name="customer_support"
    )
)


MAX_DISTANCE = 0.55


def retrieve_support(
    query,
    top_k=5
):

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    retrieved = []

    for i in range(
        len(results["documents"][0])
    ):

        retrieved.append({
            "instruction":
                results[
                    "documents"
                ][0][i],

            "response":
                results[
                    "metadatas"
                ][0][i]["response"],

            "intent":
                results[
                    "metadatas"
                ][0][i]["intent"],

            "category":
                results[
                    "metadatas"
                ][0][i]["category"],

            "distance":
                results[
                    "distances"
                ][0][i]
        })

    return retrieved


def filter_relevant_results(
    results
):

    return [
        item
        for item in results
        if item["distance"]
        <= MAX_DISTANCE
    ]


def build_context(
    retrieved_results
):

    context_parts = []

    for i, item in enumerate(
        retrieved_results,
        start=1
    ):

        context_parts.append(
            f"""
Support Example {i}

Customer Question:
{item["instruction"]}

Support Response:
{item["response"]}

Intent:
{item["intent"]}

Category:
{item["category"]}
"""
        )

    return "\n".join(
        context_parts
    )


# =========================================================
# Groq Grounded Response
# =========================================================

def generate_rag_response(
    original_message,
    retrieval_message,
    language,
    sentiment,
    intent,
    priority=False,
    top_k=5
):

    results = retrieve_support(
        retrieval_message,
        top_k=top_k
    )

    relevant_results = (
        filter_relevant_results(
            results
        )
    )

    if relevant_results:

        context = build_context(
            relevant_results
        )

    else:

        context = (
            "No sufficiently relevant "
            "customer-support context "
            "was found."
        )

    empathy_instruction = ""

    if sentiment == "negative":

        empathy_instruction = (
            "The customer appears upset "
            "or frustrated. Briefly "
            "acknowledge their frustration "
            "and respond empathetically."
        )

    priority_instruction = ""

    if priority:

        priority_instruction = (
            "This message should receive "
            "priority handling. If the "
            "available information does "
            "not fully resolve the issue, "
            "offer escalation to a human "
            "support agent."
        )

    system_prompt = f"""
You are an e-commerce customer support assistant.

Detected language: {language}
Sentiment: {sentiment}
Intent: {intent}

Rules:

1. Answer using ONLY the provided customer-support context.
2. Never invent policies, order details, dates, prices, tracking data, refund amounts, or customer information.
3. If the context is insufficient, clearly say you do not have enough information and offer human support.
4. Respond in {language}.
5. Keep the response concise, polite, and helpful.
6. Do not mention RAG, datasets, embeddings, or vector databases.
7. {empathy_instruction}
8. {priority_instruction}
"""

    user_prompt = f"""
Customer message:

{original_message}

Retrieved support context:

{context}

Provide the final customer-support response.
"""

    completion = (
        groq_client
        .chat
        .completions
        .create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.2,
            max_completion_tokens=500
        )
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
    )

    return answer, relevant_results


# =========================================================
# Direct Conversation Response
# =========================================================

def generate_conversation_response(
    message,
    language
):

    completion = (
        groq_client
        .chat
        .completions
        .create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly "
                        "e-commerce support assistant. "
                        f"Respond in {language}. "
                        "The message is only a greeting, "
                        "goodbye, or expression of thanks. "
                        "Reply briefly and naturally. "
                        "Do not invent any customer or "
                        "order information."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],

            temperature=0.2,
            max_completion_tokens=250
        )
    )

    return (
        completion
        .choices[0]
        .message
        .content
    )


# =========================================================
# Out-of-Scope Response
# =========================================================

def generate_out_of_scope_response(
    message,
    language
):

    completion = (
        groq_client
        .chat
        .completions
        .create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an e-commerce "
                        "customer support assistant. "
                        f"Respond in {language}. "
                        "Tell the customer in one or two complete sentences "
                        "that their request is outside the available "
                        "customer-support scope. "
                        "Offer human support if appropriate. "
                        "Do not answer the unrelated "
                        "question using general knowledge."
                    )
                },
                {
                    "role": "user",
                    "content": message
                }
            ],

            temperature=0.2,
            max_completion_tokens=500
        )
    )

    return (
        completion
        .choices[0]
        .message
        .content
    )


# =========================================================
# Complete Chatbot Pipeline
# =========================================================

def process_message(message):

    if not message or not message.strip():

        return {
            "error": "Message cannot be empty."
        }

    message = message.strip()

    # -----------------------------------------
    # 1. Language Detection
    # -----------------------------------------

    language_code, language_name = (
        detect_language(
            message
        )
    )

    # -----------------------------------------
    # 2. Translate for English-only classifiers
    # -----------------------------------------

    english_message = translate_to_english(
        message,
        language_name
    )

    # -----------------------------------------
    # 3. Sentiment Classification
    # -----------------------------------------

    sentiment, sentiment_confidence = (
        predict_sentiment(
            english_message
        )
    )

    # -----------------------------------------
    # 4. Intent Classification
    # -----------------------------------------

    intent = predict_intent(
        english_message
    )

    # -----------------------------------------
    # 5. Routing signals
    # -----------------------------------------

    negative_for_routing = (
        sentiment == "negative"
        and sentiment_confidence >= 0.90
    )

    # Complaint intent is the main priority signal.
    # Sentiment is used mainly to adjust response tone.
    priority = (
        intent == "complaint"
    )

    sentiment_for_response = sentiment

    if (
        sentiment == "negative"
        and not negative_for_routing
        and intent != "complaint"
    ):
        sentiment_for_response = "neutral"

    if intent == "complaint":
        sentiment_for_response = "negative"

    # -----------------------------------------
    # 6. Routing
    # -----------------------------------------

    retrieved_results = []

    if intent == "conversation":

        route = "direct_conversation"

        response = (
            generate_conversation_response(
                message,
                language_name
            )
        )

    elif intent == "out_of_scope":

        route = "human_escalation"

        response = (
            generate_out_of_scope_response(
                message,
                language_name
            )
        )

    else:

        if intent == "complaint":
            route = "priority_rag"

        elif negative_for_routing:
            route = "empathetic_rag"

        else:
            route = "rag"

        (
            response,
            retrieved_results
        ) = generate_rag_response(
            original_message=message,
            retrieval_message=english_message,
            language=language_name,
            sentiment=sentiment_for_response,
            intent=intent,
            priority=priority
        )

    # -----------------------------------------
    # Final Result
    # -----------------------------------------

    return {
        "message": message,

        "language_code":
            language_code,

        "language":
            language_name,

        "english_message":
            english_message,

        "sentiment":
            sentiment,

        "sentiment_confidence":
            round(
                sentiment_confidence,
                4
            ),

        "intent":
            intent,

        "priority":
            priority,

        "route":
            route,

        "response":
            response,

        "retrieved_count":
            len(retrieved_results)
    }


# =========================================================
# Local Test
# =========================================================

if __name__ == "__main__":

    print(
        "E-commerce Customer Support Chatbot"
    )

    print(
        "Type 'exit' to stop.\n"
    )

    while True:

        user_message = input(
            "Customer: "
        )

        if user_message.lower() == "exit":
            break

        result = process_message(
            user_message
        )

        print(
            "\nLanguage:",
            result.get("language")
        )

        print(
            "Sentiment:",
            result.get("sentiment")
        )

        print(
            "Sentiment Confidence:",
            result.get("sentiment_confidence")
        )

        print(
            "Intent:",
            result.get("intent")
        )

        print(
            "Priority:",
            result.get("priority")
        )

        print(
            "Route:",
            result.get("route")
        )

        print(
            "\nAssistant:",
            result.get("response")
        )

        print(
            "\n" + "-" * 70 + "\n"
        )
