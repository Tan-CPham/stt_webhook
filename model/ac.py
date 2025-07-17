import numpy as np
from numpy.typing import NDArray
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from model.download_model import download_hf_model
from util import audio as audio_util

class AstACModel:
    def __init__(
            self,
            model_repo_id: str,
            model_revision: str,
            model_local_dir: str,
            device: str,
    ):
        download_hf_model(
            model_repo_id=model_repo_id,
            model_revision=model_revision,
            model_local_dir=model_local_dir,
        )

        self.model = AutoModelForAudioClassification.from_pretrained(
            model_local_dir,
            local_files_only=True,
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_local_dir,
            local_files_only=True,
        )
        self.desired_sr = 16000

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self._warm_up_model()

    def predict(
            self, 
            audio: NDArray[np.float32],
            sample_rate: int,
            ) -> bool:
        
        if sample_rate != self.desired_sr:
            audio = audio_util.resample_audio(audio, sample_rate, self.desired_sr)

        features = self.feature_extractor(audio, sampling_rate=self.desired_sr, return_tensors="pt")
        features = {key: value.to(self.model.device) for key, value in features.items()}

        with torch.no_grad():
            outputs = self.model(**features)

        logits = outputs.logits
        speech_class_index = 0
        # speech_class_score = logits[0][speech_class_index].item()
        confidence_score = torch.softmax(logits, dim=1)[0][speech_class_index].item()
        
        return confidence_score > 0.4
        
    def _warm_up_model(self):
        dummy_audio = np.zeros(16000, dtype=np.float32)
        _ = self.predict(dummy_audio, 16000)