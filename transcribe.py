from faster_whisper import WhisperModel
import os

# Add this to resolve the OpenMP library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the Whisper model with float32 to avoid the float16 warning
model = WhisperModel("base", compute_type="float32")

def transcribe_audio(file_path):
    try:
        segments, _ = model.transcribe(file_path, beam_size=5, language="en")
        transcription = " ".join(segment.text for segment in segments)
        print(f'{{"transcription": "{transcription.strip()}"}}\n', flush=True)
    except Exception as e:
        print(f'{{"error": "Error processing {file_path}: {str(e)}"}}\n', flush=True)

if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1]
    transcribe_audio(audio_file)
