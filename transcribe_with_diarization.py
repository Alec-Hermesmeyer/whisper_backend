import os
import json
import sys
import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from silero_vad import get_speech_timestamps, read_audio
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LocalDiarizer:
    def __init__(self):
        self.vad_sample_rate = 16000
        self.embedding_sample_rate = 16000
        self.load_models()

    def load_models(self):
        """Load models without external auth"""
        try:
            # SpeechBrain speaker embedding model
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="models/ecapa_embedding",
                run_opts={"device": "cpu"}  # Change to "cuda" if using GPU
            )
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Model loading failed: {str(e)}")
            sys.exit(1)

    def process_audio(self, file_path):
        """Full processing pipeline"""
        try:
            # 1. Read and validate audio
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != self.vad_sample_rate:
                waveform = self.resample_audio(waveform, sample_rate)

            # 2. Voice Activity Detection
            vad_segments = get_speech_timestamps(
                waveform.numpy()[0],
                model=None,  # Auto-loads default Silero VAD
                return_seconds=True,
                sampling_rate=self.vad_sample_rate
            )

            if not vad_segments:
                return {"error": "No speech detected"}

            # 3. Extract embeddings for each speech segment
            embeddings = []
            for seg in vad_segments:
                start_sample = int(seg["start"] * self.embedding_sample_rate)
                end_sample = int(seg["end"] * self.embedding_sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]
                embeddings.append(self.get_embedding(segment_waveform))

            # 4. Cluster embeddings to identify speakers
            speaker_labels = self.cluster_embeddings(np.array(embeddings))

            # 5. Format results
            return self.format_output(vad_segments, speaker_labels)

        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            return {"error": str(e)}

    def resample_audio(self, waveform, original_rate):
        """Resample audio to target rate"""
        resampler = torchaudio.transforms.Resample(
            orig_freq=original_rate,
            new_freq=self.embedding_sample_rate
        )
        return resampler(waveform)

    def get_embedding(self, waveform):
        """Extract speaker embedding from audio segment"""
        with torch.no_grad():
            return self.embedding_model.encode_batch(waveform).squeeze().cpu().numpy()

    def cluster_embeddings(self, embeddings, threshold=0.7):
        """Cluster embeddings into speakers"""
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average"
        )
        return clustering.fit_predict(embeddings)

    def format_output(self, segments, labels):
        """Structure the diarization results"""
        return {
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "speaker": f"SPEAKER_{label}"
                } for seg, label in zip(segments, labels)
            ],
            "speakers": list(set(f"SPEAKER_{label}" for label in labels))
        }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Usage: python diarize_local.py <audio_file.wav>"}))
        sys.exit(1)

    diarizer = LocalDiarizer()
    result = diarizer.process_audio(sys.argv[1])
    print(json.dumps(result))