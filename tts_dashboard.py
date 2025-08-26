# 📚 Imports
import subprocess
import re
import docx
import streamlit as st
import requests
import json
import base64
import zipfile
import os
from io import BytesIO
from gtts import gTTS 

api_key = st.secrets["OPENROUTER_API_KEY"]  # ✅ Cloud‑native secret loading

# 🧠 Load pronunciation overrides
with open("pronunciation.json", "r") as f:
    pron_dict = json.load(f)
# ✨ Add custom entries
custom_dict = {
    "NaCl": "sodium chloride",
    "DNA": "dee en ay",
    "RNA": "are en ay",
    "Euler": "oiler",
    "Mohr's circle": "more's circle"
}
pron_dict.update(custom_dict)
# 🔀 Split pronunciation dictionaries
pron_dict_local = {
    "NaCl": "sodium chloride"
}

pron_dict_elevenlabs = pron_dict.copy()
pron_dict_elevenlabs.update({
    "DNA": "dee en ay",
    "RNA": "are en ay",
    "Euler": "oiler",
    "Mohr's circle": "more's circle",
    "ligase": "lie gaze",
    "helicase": "hee lee case",
    "primase": "pry mace",
    "SSBs": "ess ess bees",
    "Okazaki": "oh kah zah kee"
})

# 🔍 Pronunciation fixer
def apply_pronunciation(text, engine="pyttsx3"):
    text = re.sub(r'([^\w\s])', '', text)
    replaced = {}
    active_dict = pron_dict_local if engine == "pyttsx3" else pron_dict_elevenlabs
    for word in sorted(active_dict, key=len, reverse=True):
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            text = re.sub(pattern, active_dict[word], text, flags=re.IGNORECASE)
            replaced[word] = active_dict[word]
    return text, replaced

# 🗂️ File loader
def read_uploaded_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    return None

# 🧩 Split MCQs
def split_into_mcqs(text):
    if not text or not isinstance(text, str):
        return []  # or return a default list of MCQs
    text = re.sub(r"\n{3,}", "\n\n", text)
    # ... rest of your MCQ splitting logic ...
    return mcqs

# 🛠 Save MP3
def get_output_filename(index):
    return f"mcq_{index:03}.mp3"

# 🗜️ Zipper
def create_zip_of_outputs():
    """
    Creates a ZIP file in memory from audio files stored in st.session_state["session_files"].
    Each entry in session_files should be a tuple: (filename, file_bytes).
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for filename, file_bytes in st.session_state.get("session_files", []):
            zipf.writestr(filename, file_bytes)
    zip_buffer.seek(0)
    return zip_buffer

from gtts import gTTS

def narrate_with_gtts(text, output_path, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(output_path)
        return True
    except Exception as e:
        st.warning(f"gTTS error: {e}")
        return False

import requests

def narrate_with_elevenlabs(text, output_path, voice_id, api_key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"❌ ElevenLabs API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❗ Exception during ElevenLabs narration: {e}")
        return False

def narrate_with_kittentts(text, output_path):
    try:
        from kittentts import KittenTTS
        import soundfile as sf
        hf_token = os.getenv("HF_API_KEY")
        model = KittenTTS("KittenML/kitten-tts-nano-0.1")
        audio = model.generate(text)
        sf.write(output_path, audio, 24000)
        return True
    except Exception as e:
        print(f"❌ Kitten TTS error: {e}")
        return False

# 📦 Export function using FFmpeg
def export_with_ffmpeg(path, bitrate, export_format):
    try:
        output_path = path.replace(".mp3", f"_converted.{export_format.lower()}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-b:a", f"{bitrate}k", output_path],
            check=True, capture_output=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Export failed: {e.stderr.decode()}")
        return None

# 🖥️ App UI
st.set_page_config(page_title="Narration Studio", layout="centered")
st.title("🎙️ Narration Studio • ElevenLabs + Local + GPT")

# 📥 Inputs
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📤 Upload .txt or .docx", type=["txt", "docx"])
with col2:
    raw_text = st.text_area("✍️ Or paste MCQs directly")

# 🎛 Sidebar controls
engine_choice = st.sidebar.radio("🔁 Choose Engine", ["Local (pyttsx3)", "ElevenLabs", "Kitten TTS"])
rate = st.sidebar.slider("🎛️ Speech Rate (WPM)", min_value=100, max_value=220, value=160, step=10)
pause_interval = st.sidebar.slider("⏸️ Pause Length (s)", min_value=0.0, max_value=3.0, value=0.5, step=0.1)

# 🔐 ElevenLabs (keep in old familiar place under pause slider)

# --- ElevenLabs Settings ---
if engine_choice == "ElevenLabs":
    st.sidebar.header("🔐 ElevenLabs Settings")
    api_key = st.secrets.get("ELEVENLABS_API_KEY")

    # Load voice list from secrets
    voice_options_str = st.secrets.get("VOICE_OPTIONS", "[]")
    voice_options = json.loads(voice_options_str)

    # Dropdown for voice names
    voice_names = [v["name"] for v in voice_options]
    selected_voice_name = st.sidebar.selectbox("🎙️ Choose Voice", voice_names)

    # Get matching ID for the selected voice
    voice_id = next((v["id"] for v in voice_options if v["name"] == selected_voice_name), None)

    # Optional: show which ID is in use (for debugging)
    st.sidebar.caption(f"Voice ID in use: {voice_id}")

    stability = st.sidebar.slider("🧘 Stability", 0.0, 1.0, 0.5)
    similarity_boost = st.sidebar.slider("🎨 Similarity Boost", 0.0, 1.0, 0.5)

# 🤖 GPT Generator (moved here so ElevenLabs stays above it)
st.sidebar.header("🤖 Generate with GPT")
asset_type = st.sidebar.selectbox("🎨 Asset Type", ["Script", "MCQs", "Voiceover Prompt"])
gpt_prompt = st.sidebar.text_area("🧠 GPT Prompt", "Explain Newton's laws in 3 MCQs")
generate_button = st.sidebar.button("✨ Generate")

# 📦 Export Settings
st.sidebar.header("📦 Export Settings")
export_format = st.sidebar.selectbox("🎞️ Format", ["MP3", "WAV"])
bitrate = st.sidebar.slider("📡 Bitrate (kbps)", 64, 320, 192)

# 🧠 GPT API Integration — OpenAI v1.x style
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")  # now reading from your .env file

if generate_button and gpt_prompt.strip():
    try:
        from openai import OpenAI
        if not api_key:
            st.error("❌ No OpenRouter API key found. Add OPENROUTER_API_KEY to your .env file")
        else:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            response = client.chat.completions.create(
                model="z-ai/glm-4.5-air:free",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for generating educational assets."},
                    {"role": "user", "content": gpt_prompt}
                ]
            )
            generated_text = response.choices[0].message.content
            raw_text = generated_text
            st.success("✅ MCQs generated with GLM‑4.5 Air")
            st.text_area("📄 Generated Text", generated_text, height=300)
            st.session_state["gpt_text"] = generated_text

    except Exception as e:
        st.error(f"❌ GPT error: {e}")
# 🧹 Clear GPT text button
if st.sidebar.button("🧹 Clear GPT Text", key="clear_gpt_button"):
    st.session_state["gpt_text"] = ""
    st.success("✅ GPT text cleared")

# 🧾 Override loader (file takes priority if uploaded)
if uploaded_file:
    raw_text = read_uploaded_file(uploaded_file)

engine_map = {
    "Local (pyttsx3)": "pyttsx3",
    "ElevenLabs": "elevenlabs",
    "Kitten TTS": "kittentts",
    "OpenTTS": "opentts"
}

# 📚 Batch Narration
st.subheader("📚 Narrate MCQ Batch")

if st.button("🎤 Narrate All"):
    combined_text = "\n\n".join(filter(None, [
        raw_text,
        read_uploaded_file(uploaded_file) if uploaded_file else "",
        st.session_state.get("gpt_text", "").strip()
    ]))

    if not combined_text.strip():
        st.warning("⚠️ Provide MCQs or generate with GPT")
    else:
        mcqs = split_into_mcqs(combined_text)
        mcqs = list(dict.fromkeys(mcqs))  # Now safe to deduplicate

        # 🔒 Create isolated session folder
        import time
        session_id = f"session_{int(time.time())}"
        st.session_state["session_id"] = session_id
        session_folder = os.path.join(os.getcwd(), session_id)
        os.makedirs(session_folder, exist_ok=True)
        st.session_state["session_folder"] = session_folder

        # Initialise in‑memory storage for this session
        st.session_state["session_files"] = []

        # Proceed with narration...
        st.info(f"🔄 Processing {len(mcqs)} files...")
        progress = st.progress(0)
        
        success = False

        for i, chunk in enumerate(mcqs, start=1):
            engine_key = engine_map.get(engine_choice, "unknown")
            cleaned, _ = apply_pronunciation(chunk, engine=engine_key)
            cleaned += "." * int(pause_interval * 2)
            path = os.path.join(session_folder, get_output_filename(i))

            if engine_choice == "gTTS":
                success = narrate_with_gtts(cleaned, path)
                if not success:
                    st.warning(f"⚠️ gTTS failed for MCQ #{i}")

            elif engine_choice == "ElevenLabs":
                if not voice_id or not api_key:
                    st.warning("❗ Missing ElevenLabs credentials")
                else:
                    success = narrate_with_elevenlabs(cleaned, path, voice_id, api_key)
                    if not success:
                        st.warning(f"⚠️ ElevenLabs failed for MCQ #{i}")

            elif engine_choice == "Kitten TTS":
                success = narrate_with_kittentts(cleaned, path)
                if not success:
                    st.warning(f"⚠️ Failed to narrate MCQ #{i}")

            # 🔁 Fallback logic
            if not success:
                st.info(f"🔁 Trying fallback engine for MCQ #{i}")

            # Cloud-safe fallback order
            fallback_order = ["Kitten TTS", "gTTS", "ElevenLabs"]

            for fallback in fallback_order:
                if fallback == engine_choice:
                    continue  # Skip the engine we already tried

                try:
                    if fallback == "Kitten TTS":
                        success = narrate_with_kittentts(cleaned, path)
                    elif fallback == "gTTS":
                        success = narrate_with_gtts(cleaned, path)
                    elif fallback == "ElevenLabs":
                        success = narrate_with_elevenlabs(cleaned, path, voice_id, api_key)

                    if success:
                        st.success(f"✅ Fallback succeeded with {fallback} for MCQ #{i}")
                        break

                except Exception as e:
                    st.warning(f"❌ Fallback {fallback} failed: {e}")

            progress.progress(i / len(mcqs))
        st.success("✅ Narration complete")

# List new audio files
session_folder = st.session_state.get("session_folder")
if session_folder and os.path.exists(session_folder):
    audio_files = sorted([
        f for f in os.listdir(session_folder)
        if f.lower().endswith((".mp3", ".wav"))
    ])

    if audio_files:
        for idx, f in enumerate(audio_files):
            file_path = os.path.join(session_folder, f)
            with open(file_path, "rb") as audio_file:
                file_data = audio_file.read()

            st.audio(file_data, format="audio/mp3")  # 👈 Add this line

            st.download_button(
                f"⬇️ Download {f}",
                file_data,
                file_name=f,
                key=f"download_{idx}_{f}"
            )

            if st.button(f"🗑️ Delete {f}", key=f"delete_{idx}"):
                os.remove(file_path)
                st.success(f"✅ Deleted {f}")
                st.rerun()

        zip_path = create_zip_of_outputs()
        with open(zip_path, "rb") as zf:
            st.download_button(
                label="📦 Download All as ZIP",
                data=zf,
                file_name="mcq_outputs.zip",
                mime="application/zip",
                key="download_zip"
            )
    else:
        st.info("📭 No audio files found in session folder.")
else:
    st.info("🎤 No session folder found. Click 'Narrate All' to generate audio files.")
