import streamlit as st
import torch
import pyaudio
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper model and processor
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Create the Streamlit interface
st.title("Real-time Voice Recognition")

# Function to transcribe speech
@st.cache(allow_output_mutation=True)
def transcribe_audio(audio):
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe(audio)

# Function to capture live audio and perform recognition
def recognize_voice():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    st.write("Speak now...")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = b"".join(frames)

    # Save the recorded audio to a temporary file
    temp_audio_file = "temp_audio.wav"
    sf.write(temp_audio_file, audio_data, RATE)

    # Perform speech recognition on the recorded audio
    result = transcribe_audio(temp_audio_file)
    return result

# UI components
if st.button("Start Recording"):
    result = recognize_voice()
    if result and "text" in result[0]:
        st.write("Recognized Text:")
        st.write(result[0]["text"])
