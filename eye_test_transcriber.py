import os
import time
import math
import tempfile
import streamlit as st
from openai import OpenAI, RateLimitError
from moviepy.editor import VideoFileClip

# Initialize OpenAI client
client = OpenAI()

# -----------------------------
# Utility Functions
# -----------------------------
def split_video(file_path, chunk_duration=60):
    """Split video into chunks (in seconds). Returns list of chunk file paths."""
    video = VideoFileClip(file_path)
    duration = video.duration
    chunks = []

    st.info(f"Total video length: {math.ceil(duration)} seconds. Splitting into chunks...")

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(0, math.ceil(duration), chunk_duration):
            start = i
            end = min(i + chunk_duration, duration)
            chunk_file = os.path.join(tmpdir, f"chunk_{i//chunk_duration}.mp4")
            video.subclip(start, end).write_videofile(chunk_file, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            chunks.append(chunk_file)
        return chunks


def transcribe_with_retry(file_path, max_retries=5):
    """Transcribe a single audio/video file with retry on rate limit."""
    for attempt in range(max_retries):
        try:
            with open(file_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="vtt"
                )
            return transcript
        except RateLimitError:
            wait_time = (2 ** attempt) * 5  # exponential backoff
            st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            return None
    st.error("❌ Max retries exceeded. Please try again later.")
    return None


def combine_vtt(transcripts):
    """Combine multiple VTT transcripts into one."""
    combined = "WEBVTT\n\n"
    for t in transcripts:
        if not t:
            continue
        # Remove duplicate WEBVTT headers
        cleaned = t.replace("WEBVTT", "").strip()
        combined += cleaned + "\n\n"
    return combined


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Eye Test Video Transcriber", page_icon="👁️", layout="centered")

st.title("👁️ Eye Test Video Transcriber")
st.markdown("Upload an **MP4 eye test video** to get the transcribed text (as `.vtt` subtitle file).")

uploaded_file = st.file_uploader("🎥 Upload your Eye Test Video", type=["mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    st.info("📽 Processing video... please wait ⏳")
    chunks = split_video(video_path)

    all_transcripts = []
    progress = st.progress(0)
    total = len(chunks)

    for i, chunk_path in enumerate(chunks):
        st.write(f"🎧 Transcribing chunk {i+1}/{total}...")
        transcript = transcribe_with_retry(chunk_path)
        all_transcripts.append(transcript)
        progress.progress((i + 1) / total)

    if all_transcripts:
        final_vtt = combine_vtt(all_transcripts)
        output_file = os.path.splitext(uploaded_file.name)[0] + "_transcript.vtt"

        st.success("✅ Transcription completed successfully!")
        st.download_button(
            label="⬇️ Download VTT File",
            data=final_vtt,
            file_name=output_file,
            mime="text/vtt"
        )
        st.balloons()
    else:
        st.error("❌ No transcription output generated.")
