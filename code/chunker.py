# chunked_transcriber.py
import whisper
import numpy as np

SAMPLE_RATE = 16000      # Whisper’s fixed sample rate
CHUNK_SECONDS = 30       # length of each chunk in seconds
CHUNK_SIZE = SAMPLE_RATE * CHUNK_SECONDS

def transcribe_full_audio(model: whisper.Whisper, audio_path: str) -> str:
    """
    Transcribe an entire audio file by splitting it into 30-second chunks.

    Args:
        model: A loaded whisper model (e.g., whisper.load_model("base")).
        audio_path: Path to the audio/video file.

    Returns:
        The concatenated transcription string.
    """
    # 1) Load the full waveform (resampled to 16 kHz mono)
    audio = whisper.load_audio(audio_path)
    # 2) Pad at the end so final chunk isn’t too short
    if len(audio) % CHUNK_SIZE != 0:
        pad_amount = CHUNK_SIZE - (len(audio) % CHUNK_SIZE)
        audio = np.concatenate([audio, np.zeros(pad_amount, dtype=audio.dtype)])

    transcripts = []

    # 3) Process chunk by chunk
    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i : i + CHUNK_SIZE]
        # 4) Build spectrogram & move to correct device
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)
        # 5) Optionally detect language once on first chunk
        if i == 0:
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)
            print(f"[#] Detected language: {lang}")

        # 6) Decode with defaults (greedy)
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        transcripts.append(result.text.strip())

    # 7) Join all chunk outputs
    full_text = " ".join(transcripts)
    return full_text


if __name__ == "__main__":
   main() 
