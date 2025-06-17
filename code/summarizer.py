import whisper
import os
import subprocess
import tempfile
from chunker import transcribe_full_audio
import time

def summarize_with_ollama(path,language, model="llama3"):
    # 1️⃣ Check file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")

    # 2️⃣ Load the transcript
    with open(path, "r", encoding="utf-8") as f:
        transcript = f.read()
        if (language=='en'):
            # english prompt
                prompt = (
                    "You are a helpful assistant that summarizes transcripts into a brief synopsis "
                        "and three key bullet-point takeaways.\n\n"
                        "TRANSCRIPT:\n"
                        f"{transcript}"
    )
        elif (language=='es'):
            # prompt en español
                prompt = (
                    "Eres un asistente que resume clases señalando los puntos más importantes de estas "
                        "listarás los puntos más importantes y crearás una breve descripción de ellos, todo en español.\n\n"
                        "TRANSCRIPT:\n"
                        f"{transcript}"
            )


    # 4️⃣ Call ollama run with that as the prompt argument
    cmd = ["ollama", "run", model, prompt]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Ollama failed: {result.stderr.strip()}")
    return result.stdout.strip()



def main():
    start_time=time.time()
    # —— Whisper transcription —— #
    print("Choose a Whisper model (e.g. tiny, base, small, medium, large):")
    model_choice = input().strip()
    model = whisper.load_model(model_choice)

    VALID_EXTS = ('.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', 'mp4')
    while True:
        audio_path = input("Insert audio file path: ").strip()
        if not os.path.isfile(audio_path):
            print(f"⚠️ File not found: {audio_path!r}")
            continue
        if not audio_path.lower().endswith(VALID_EXTS):
            print("⚠️ That doesn't look like an audio file. Supported: " +
                  ", ".join(VALID_EXTS))
            continue
        try: 
            transcript = transcribe_full_audio(model, audio_path)
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            break
        except Exception as e:
           print(f"⚠️ Couldn’t load audio (maybe corrupted?): {e}")
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device) #transforms 1d numpy array to logscaled mel spectrogram
    _, probs = model.detect_language(mel)#returns languages and their probs of being the langauge spoken 
    language = max(probs, key=probs.get)#gets the highest porobability language, prob.get gets the numerical value
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options) 
    print("\n=== Transcript ===\n", transcript)
    end_time_transcription = time.time()
    print("\ntranscription lasted ",(end_time_transcription-start_time)," seconds")
    # —— Write transcript to a temp file —— #
    # You can also hardcode a filename instead of tempfile
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tmp:
        tmp.write(transcript)
        tmp_path = tmp.name

    # —— Summarize with Ollama —— #
    print("\nSummarizing with llama3…")
    try:
        summary = summarize_with_ollama(tmp_path,language, model="llama3")
        print("\n=== Ollama Summary ===\n", summary)
        end_time_summary = time.time()
        print("\nsummary lasted ",(end_time_summary-start_time)," seconds")
    except Exception as e:
        print(f"❌ {e}")
    finally:
        # cleanup
        os.remove(tmp_path)


if __name__ == "__main__":
    main()

