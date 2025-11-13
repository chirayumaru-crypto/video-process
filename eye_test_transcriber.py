import streamlit as st
from moviepy.editor import VideoFileClip
from openai import OpenAI
import tempfile
import re
import os

# Initialize client safely
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("🚨 Missing OpenAI API key in Streamlit secrets.toml!")
    st.stop()

# ✅ Fix: set env variable instead of passing in constructor
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()

st.set_page_config(page_title="AI Eye-Test Video Transcriber", layout="centered")
st.title("👁️ AI Eye-Test Video Transcriber")
st.caption("Upload an MP4 eye-test video and get an auto-corrected `.vtt` transcript.")

uploaded_video = st.file_uploader("🎞️ Upload Eye-Test Video (MP4)", type=["mp4"])

if uploaded_video:
    with st.spinner("⏳ Extracting audio and transcribing..."):
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        video = VideoFileClip(temp_video.name)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        video.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)

        with open(temp_audio.name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="vtt"
            )

        def clean_spoken_text(line):
            if not line or "-->" in line or line.strip() == "WEBVTT":
                return line

            replacements = {
                r"\buh+\b": "",
                r"\bum+\b": "",
                r"\bhai\b": "yes",
                r"\bokey+\b": "okay",
                r"\bma\b": "now",
                r"\bll\b": "will",
                r"\byaa+\b": "yeah",
                r"\btru+\b": "true",
                r"\bclrear\b": "clear",
                r"\bplaese\b": "please",
                r"\btets\b": "test",
                r"\bey\b": "eye"
            }
            for pattern, repl in replacements.items():
                line = re.sub(pattern, repl, line, flags=re.IGNORECASE)
            return re.sub(r"\s+", " ", line).strip()

        cleaned_lines = [clean_spoken_text(l) for l in transcript.splitlines()]
        cleaned_vtt = "\n".join(cleaned_lines)

        def label_speakers(vtt_text):
            blocks = vtt_text.split("\n\n")
            labeled_blocks = []
            speaker_toggle = True
            for block in blocks:
                if "-->" in block:
                    speaker = "Optometrist" if speaker_toggle else "Patient"
                    speaker_toggle = not speaker_toggle
                    parts = block.split("\n")
                    if len(parts) > 1:
                        parts[-1] = f"{speaker}: {parts[-1]}"
                    block = "\n".join(parts)
                labeled_blocks.append(block)
            return "\n\n".join(labeled_blocks)

        final_vtt = label_speakers(cleaned_vtt)

        st.success("✅ Transcription complete!")
        st.download_button(
            "💾 Download Cleaned Transcript (.vtt)",
            data=final_vtt,
            file_name="eye_test_transcript.vtt",
            mime="text/vtt"
        )

        st.text_area("🧾 Transcript Preview", final_vtt[:3000], height=300)

else:
    st.info("⬆️ Upload an MP4 file to start processing.")
