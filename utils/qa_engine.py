import requests
import os
import streamlit as st

# Optional: if not using Streamlit secrets
OPENROUTER_API_KEY = st.secrets["openrouter"]["api_key"]

def answer_with_roberta(question, context_chunks):
    """
    Generates an MBA-style answer using DeepSeek R1T2 Chimera via OpenRouter API.
    """
    context = " ".join(context_chunks[:2])  # limit for better performance

    # MBA-style prompt
    prompt = (
        "You are an expert MBA case study solver. "
        "Based on the following case study, answer the question in a clear, structured, business-style format.\n\n"
        f"Case Study:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "tngtech/deepseek-r1t2-chimera:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {res.status_code} – {res.text}"
