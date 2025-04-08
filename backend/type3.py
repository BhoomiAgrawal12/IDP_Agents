# full_app.py - Complete FastAPI Application

from fastapi import FastAPI
from pathlib import Path
import os
import yt_dlp
from moviepy.editor import VideoFileClip
import torchaudio
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from google.cloud import speech

app = FastAPI()

@app.post("/process-video")
async def process_video(video_url: str):
    video_dir = Path("video_data"); video_dir.mkdir(exist_ok=True)
    frame_dir = Path("mixed_data"); frame_dir.mkdir(exist_ok=True)
    audio_path = frame_dir / "output_audio.wav"
    audio_flac_path = frame_dir / "output_audio.flac"
    video_path = video_dir / "input_vid.webm"
    text_output_path = frame_dir / "output_text.txt"

    # 1. Download video
    ydl_opts = {
        'outtmpl': str(video_path).replace(".webm", ".%(ext)s"),
        'format': 'bestvideo+bestaudio/best',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        metadata = {
            "Author": info.get('uploader'),
            "Title": info.get('title'),
            "Views": info.get('view_count')
        }

    # 2. Extract frames
    clip = VideoFileClip(str(video_path))
    clip.write_images_sequence(os.path.join(frame_dir, "frame%04d.png"), fps=0.2)

    # 3. Extract and convert audio
    clip.audio.write_audiofile(str(audio_path))
    os.system(f"ffmpeg -i {audio_path} -c:a flac {audio_flac_path}")

    # 4. Transcribe using Whisper
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3").to("cuda" if torch.cuda.is_available() else "cpu")
    speech_array, sampling_rate = torchaudio.load(str(audio_flac_path))
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
        sampling_rate = 16000
    speech = speech_array.squeeze(0).numpy()
    inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 5. Transcribe using Google Cloud Speech-to-Text
    client = speech.SpeechClient()
    with open(audio_flac_path, 'rb') as f:
        audio = speech.RecognitionAudio(content=f.read())
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio)
    google_transcript = "\n".join([result.alternatives[0].transcript for result in response.results])

    # 6. Save Google STT output to file
    with open(text_output_path, "w") as file:
        file.write(google_transcript)

    os.remove(audio_path)  # Cleanup

    return {
        "metadata": metadata,
        "whisper_transcription": transcription,
        "google_transcription": google_transcript,
        "frames_folder": str(frame_dir),
        "text_saved_at": str(text_output_path)
    }
