import streamlit as st
import os
import re
import time
import random
from groq import Groq

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Buddy - English for Kids", page_icon="🧸", layout="centered")

# API Key সিকিউরলি লোড করা হচ্ছে
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("⚠️ DEBUG: API Key is completely missing from Secrets!")

LEVELS = {
    "L1": {
        "vocab": {"red", "blue", "ball", "cat", "dog", "mama", "dada", "one", "two", "three"},
        "max_len": 3,
        "prompt": "Use only very simple 1-3 word sentences. Teach colors, animals, family, numbers 1-5."
    },
    "L2": {
        "vocab": {"apple", "water", "happy", "sad", "run", "jump", "four", "five", "like", "want"},
        "max_len": 5,
        "prompt": "Use simple 3-5 word sentences. Teach food, actions, feelings, numbers 1-10."
    },
    "L3": {
        "vocab": {"under", "behind", "morning", "night", "because", "fast", "slow", "yesterday", "beautiful", "friend"},
        "max_len": 7,
        "prompt": "Use 4-7 word sentences. Teach prepositions, time, descriptive words, simple past."
    },
}

BLOCKLIST = [r"\b(kill|blood|gun|knife|sex|naked|drug)\b", r"\b(hate you|stupid|ugly)\b"]
SAFE_FALLBACKS = [
    "Let's try something fun instead! Do you like songs or stories?",
    "I'm here to help you learn and play. What makes you happy today?",
]

SEL_DIRECTIVES = {
    "self_awareness": "Name the emotion you notice and invite the child to name it too.",
    "self_management": "Offer a simple calming action like a big breath before continuing.",
    "social_awareness": "Ask a gentle perspective-taking question about a character.",
    "relationship_skills": "Praise turn-taking or sharing; model a polite phrase.",
    "responsible_decision": "Give two safe choices and let the child pick.",
}

# ---------------- SAFETY ----------------
def is_safe(text: str) -> bool:
    return not any(re.search(p, text, re.IGNORECASE) for p in BLOCKLIST)

# ---------------- MODE DETECTOR ----------------
SUPPORT_PATTERNS = [r"\b(sad|scared|angry|cry|hurt|hard|can't|cannot|don't want)\b", r"\b(tired|bored)\b"]
ENGAGEMENT_PATTERNS = [r"^\s*(ok|yes|no|fine|maybe)\s*$", r"\b(bored|nothing|dunno)\b"]
LEARNING_PATTERNS = [r"\b(what is|how say|what's this|is this|color|animal)\b"]

def detect_mode(text, history):
    text_lower = text.lower().strip()
    if any(re.search(p, text_lower) for p in SUPPORT_PATTERNS):
        return "SUPPORT"
    if any(re.search(p, text_lower) for p in ENGAGEMENT_PATTERNS) and len(text_lower.split()) <= 2:
        return "ENGAGEMENT"
    if any(re.search(p, text_lower) for p in LEARNING_PATTERNS):
        return "LEARNING"
    if history and history[-1].get("mode") == "LEARNING" and len(text_lower.split()) <= 3:
        return "LEARNING"
    return "CONVERSATION"

# ---------------- LEVEL MANAGER ----------------
def record_progress(text, profile):
    level = profile["level"]
    vocab = LEVELS[level]["vocab"]
    words = set(re.findall(r"[a-z]+", text.lower()))
    for w in words & vocab:
        profile["mastery"][w] = min(100, profile["mastery"].get(w, 0) + 10)

def maybe_promote(profile):
    level = profile["level"]
    vocab = LEVELS[level]["vocab"]
    mastered = sum(1 for w in vocab if profile["mastery"].get(w, 0) >= 80)
    if mastered / len(vocab) >= 0.7:
        order = ["L1", "L2", "L3"]
        idx = order.index(level)
        if idx < len(order) - 1:
            profile["level"] = order[idx + 1]

# ---------------- CONTEXT BUILDER ----------------
def build_prompt(profile, mode, sel):
    level = profile["level"]
    history = profile["history"][-5:]
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    return f"""You are "Buddy", a warm English teacher for a child under 7.
CURRENT LEVEL: {level}. {LEVELS[level]['prompt']}
CURRENT MODE: {mode}.
SEL DIRECTIVE: {sel}
CHILD PROFILE: mastered words = {[w for w,v in profile['mastery'].items() if v >= 80]}
RECENT CONVERSATION:
{history_text}

RULES:
- Use ONLY vocabulary appropriate for {level}.
- Keep sentences at or below {LEVELS[level]['max_len']} words.
- Never discuss violence, romance, politics, or anything unsafe for a child.
- If mode is SUPPORT: validate feelings first, do NOT teach.
- If mode is ENGAGEMENT: suggest a game or a song.
- If the child makes a language error, recast it gently instead of saying "wrong".
- End with a simple question or an invitation to speak.
"""

# ---------------- LLM (Streaming with gemma2-9b-it) ----------------
def get_llm_stream(prompt):
    if not GROQ_API_KEY or "gsk_" not in GROQ_API_KEY:
        yield "⚠️ DEBUG: API Key is missing or invalid format."
        return
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="gemma2-9b-it",  # Stable, fast non-reasoning model
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=80,
            stream=True,
        )
        got_content = False
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                got_content = True
                yield content
        if not got_content:
            yield "I am thinking... Can you say that again?"
    except Exception as e:
        yield f"❌ API Error: {str(e)}"

# ---------------- FALLBACK ----------------
def fallback_response(mode, level):
    if mode == "SUPPORT":
        return "It's okay to feel that way. Let's take a big breath together. In... and out. You are doing great."
    if mode == "ENGAGEMENT":
        return "Do you want to play a game or sing a song? Your choice!"
    
    l1_responses = [
        "Good job! Can you say 'red ball'?",
        "Wow, nice! What color is the cat?",
        "Let's learn a new word. Can you say 'dog'?",
        "Yay! Can you say 'I see a dog'?"
    ]
    
    if level == "L1":
        return random.choice(l1_responses)
    if level == "L2":
        return "Nice! Can you tell me what you like?"
    return "Well done! Can you say a longer sentence?"

# ---------------- SEL ----------------
def get_sel(mode, emotion_history):
    if mode == "SUPPORT":
        return SEL_DIRECTIVES["self_management"]
    if mode == "ENGAGEMENT":
        return SEL_DIRECTIVES["responsible_decision"]
    keys = list(SEL_DIRECTIVES.keys())
    return SEL_DIRECTIVES[keys[len(emotion_history) % len(keys)]]

# ---------------- SESSION ----------------
if "profile" not in st.session_state:
    st.session_state.profile = {
        "level": "L1",
        "mastery": {},
        "history": [],
        "mode": "LEARNING",
        "emotion_history": [],
    }

profile = st.session_state.profile

# ---------------- UI ----------------
st.title("🧸 Buddy - English for Kids")
st.caption(f"Level: **{profile['level']}**  |  Mode: **{profile['mode']}**")

for msg in profile["history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Say something to Buddy..."):
    # user message
    profile["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    start = time.time()

    # detect mode
    mode = detect_mode(prompt, profile["history"])
    profile["mode"] = mode
    if mode == "SUPPORT":
        profile["emotion_history"].append("support")

    # SEL
    sel = get_sel(mode, profile["emotion_history"])

    # context
    sys_prompt = build_prompt(profile, mode, sel)

    # LLM Streaming Output
    with st.chat_message("assistant"):
        response = st.write_stream(get_llm_stream(sys_prompt))

    # Output safety check
    if not is_safe(str(response)):
        response = SAFE_FALLBACKS[0]
        st.warning(response)

    # progress tracking (only in learning mode)
    if mode == "LEARNING":
        record_progress(prompt, profile)
        maybe_promote(profile)

    # add assistant message to history
    profile["history"].append({"role": "assistant", "content": str(response), "mode": mode})

    latency = time.time() - start
    st.caption(f"⚡ Total Response time: {latency:.2f}s")