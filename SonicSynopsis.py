import streamlit as st
import requests
import yt_dlp
import tempfile
import shutil
import os
import whisper
import textwrap
from transformers import BartTokenizer
from deep_translator import GoogleTranslator

# Sidebar: Enter your Hugging Face API key
st.sidebar.title("Settings")
API_KEY = st.sidebar.text_input("Enter Hugging Face API Key:", type="password")
if not API_KEY:
    st.sidebar.warning("Please enter your Hugging Face API key.")

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
TOKENIZER = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
MAX_LENGTH = 1020

def download_youtube_audio(youtube_url):
    """Download audio from a YouTube URL and return the audio file path and temporary directory."""
    temp_dir = tempfile.mkdtemp()
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(youtube_url, download=True)
            audio_file = os.path.join(temp_dir, "audio.mp3")
            return audio_file, temp_dir
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None, None

def transcribe_audio(audio_path):
    """Transcribe the audio using Whisper."""
    model = whisper.load_model("tiny.en")
    result = model.transcribe(audio_path)
    return result["text"]

def truncate_text(text, max_length=MAX_LENGTH):
    """Truncate text to a maximum token length using the Bart tokenizer."""
    tokens = TOKENIZER.encode(text, truncation=True, max_length=max_length)
    return TOKENIZER.decode(tokens)

def summarize_text(text):
    """Summarize text using the Hugging Face Inference API."""
    payload = {
        "inputs": text,
        "parameters": {"max_length": 512, "min_length": 50, "do_sample": False}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        summary_output = response.json()
        # Expecting a list of dicts with key "summary_text"
        if isinstance(summary_output, list) and summary_output and "summary_text" in summary_output[0]:
            return summary_output[0]["summary_text"]
        else:
            st.error("Unexpected summary output format.")
            return None
    else:
        st.error("API request failed. Check your token or try again later.")
        return None

def translate_text(text, target_lang):
    """Translate text into the target language in chunks."""
    translator = GoogleTranslator(source='en', target=target_lang)
    chunks = textwrap.wrap(text, 1000)
    translated_chunks = [translator.translate(chunk) for chunk in chunks]
    return " ".join(translated_chunks)

# Initialize session state for transcript and summary if not already set
if "transcript" not in st.session_state:
    st.session_state["transcript"] = None
if "summary_text" not in st.session_state:
    st.session_state["summary_text"] = None

st.title("SonicSynopsis – Listen Less, Understand More!")
youtube_url = st.text_input("Enter YouTube URL:")

# Process the video: download, transcribe, and summarize.
if st.button("Process Video", key="process_video"):
    if youtube_url:
        st.info("Downloading audio...")
        audio_path, temp_dir = download_youtube_audio(youtube_url)
        
        if audio_path:
            st.success("Audio downloaded successfully!")
            st.info("Transcribing audio...")
            transcript = transcribe_audio(audio_path)
            st.session_state["transcript"] = transcript  # Save transcript in session state
            
            st.info("Summarizing transcript...")
            truncated_transcript = truncate_text(transcript)
            summary_text = summarize_text(truncated_transcript)
            
            if summary_text:
                st.session_state["summary_text"] = summary_text  # Save summary in session state
            else:
                st.error("Failed to generate summary.")
            
            # Clean up the temporary directory (removes audio file as well)
            if temp_dir:
                shutil.rmtree(temp_dir)
    else:
        st.error("Please enter a valid YouTube URL.")

# Display transcript if available
if st.session_state.get("transcript"):
    st.subheader("Transcript")
    st.text_area("Transcript", st.session_state["transcript"], height=200)

# Display summary and translation options if available
if st.session_state.get("summary_text"):
    st.subheader("Summary")
    st.write(st.session_state["summary_text"])
    
    st.subheader("Translate Summary")
    target_lang = st.selectbox("Select Language", ["ur", "fr", "es", "de", "zh", "ar"], key="target_lang")
    if st.button("Translate Summary", key="translate_summary"):
        translated_summary = translate_text(st.session_state["summary_text"], target_lang)
        st.write(translated_summary)
