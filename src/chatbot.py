import streamlit as st
from symptom_data import get_advice_for_symptom
from deep_translator import GoogleTranslator

# Example usage
translated_text = GoogleTranslator(source='auto', target='hi').translate("Hello")
print(translated_text)  # Prints "नमस्ते"


# Optional: OpenAI GPT integration
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -----------------------------
# Hardcoded symptom dictionary
# -----------------------------
SYMPTOM_ADVICE = {
    "fever": "Rest, drink plenty of fluids, and monitor your temperature. Seek medical help if fever persists more than 3 days.",
    "cough": "Stay hydrated, avoid irritants, and see a doctor if cough is severe or persistent.",
    "headache": "Rest in a dark room, stay hydrated. Consult a doctor if pain is severe or frequent.",
    "fatigue": "Take breaks, get enough sleep, and maintain good nutrition.",
    "sore throat": "Gargle warm salt water, drink fluids, and seek medical attention if it worsens.",
}

# -----------------------------
# GPT Client setup
# -----------------------------
def init_gpt_client():
    """Initialize OpenAI client if API key is available in Streamlit secrets"""
    if "openai_api_key" in st.secrets and OpenAI is not None:
        return OpenAI(api_key=st.secrets["openai_api_key"])
    return None

client = init_gpt_client()

# -----------------------------
# Chatbot response function
# -----------------------------
def get_chat_response(user_input, chat_history=None, target_lang="en"):
    """
    Returns chatbot response for a user input.
    Supports multi-turn chat and multilingual responses.
    
    Params:
    - user_input: str
    - chat_history: list of tuples [(sender, message)]
    - target_lang: language code (e.g., "en", "hi", "es")
    """
    responses = []

    # Normalize input
    user_input_clean = user_input.strip().lower()

    # 1️⃣ Check hardcoded dictionary
    for symptom, advice in SYMPTOM_ADVICE.items():
        if symptom in user_input_clean:
            responses.append(advice)

    # 2️⃣ Check CSV lookup
    if not responses:
        csv_advice = get_advice_for_symptom(user_input)
        if csv_advice:
            responses.append(csv_advice)

    # 3️⃣ GPT fallback
    if not responses and client is not None:
        messages = [
            {"role": "system",
             "content": "You are a helpful AI healthcare assistant. Provide concise, safe, non-diagnostic advice. Always recommend seeing a doctor for serious issues."}
        ]

        # Limit chat history to last 10 messages to avoid token overflow
        if chat_history:
            for sender, message in chat_history[-10:]:
                role = "user" if sender == "You" else "assistant"
                messages.append({"role": role, "content": message})

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            ai_reply = response.choices[0].message.content
            responses.append(ai_reply)
        except Exception as e:
            responses.append(f"⚠️ GPT API error: {e}")

    # 4️⃣ Default fallback
    if not responses:
        responses.append("I'm sorry, I couldn't find relevant advice. Please consult a doctor or provide more details.")

    # Combine responses
    final_reply = "\n\n".join(responses)

    # Translate to target language if needed
    if target_lang != "en":
        try:
            final_reply = GoogleTranslator(source='auto', target=target_lang).translate(final_reply)
        except Exception as e:
            final_reply += f"\n\n⚠️ Translation error: {e}"

    return final_reply
