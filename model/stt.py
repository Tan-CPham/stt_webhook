import numpy as np
import torch
from numpy.typing import NDArray
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer, GenerationConfig
from util import audio as audio_util


class STTModel:
    def __init__(
            self,
            model_path: str,
            feature_extractor_path: str,
            tokenizer_path: str,
            hf_token: str,
            device: str,
            cache_dir: str = "weight",
    ):

        self.device = device
        self.model = SpeechEncoderDecoderModel.from_pretrained(model_path, use_auth_token=hf_token,
                                                               cache_dir=cache_dir).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path, token=hf_token,
                                                                      cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token, cache_dir=cache_dir)

        if torch.cuda.is_available() and self.device == "cuda":
            self.model = self.model.cuda()

        self._warm_up_model()

    def _decode_tokens(self, token_ids, skip_special_tokens=True, time_precision=0.02):
        timestamp_begin = self.tokenizer.vocab_size
        outputs = [[]]

        for token in token_ids:
            if token >= timestamp_begin:
                timestamp = f" |{(token - timestamp_begin) * time_precision:.2f}| "
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)

        outputs = [
            s if isinstance(s, str) else self.tokenizer.decode(s, skip_special_tokens=skip_special_tokens) for s in
            outputs
        ]
        return "".join(outputs).replace("< |", "<|").replace("| >", "|>")

    def _compute_confidence(self, scores: list[torch.Tensor]) -> float:
        """
        Compute confidence score as percentage from model output scores.
        Uses the average probability of the highest scoring tokens.
        """
        confidence_scores = []
        for score_tensor in scores:
            # Convert to probabilities using softmax
            score_probs = torch.nn.functional.softmax(score_tensor, dim=-1)
            # Get maximum probability for each position
            max_probs = torch.max(score_probs, dim=-1)[0]
            # Average the probabilities
            avg_prob = torch.mean(max_probs).item()
            confidence_scores.append(avg_prob)

        # Return average confidence as percentage
        return float(np.mean(confidence_scores) * 100)

    def _decode_wav(self, audio_wavs: list[torch.Tensor], prefix="") -> tuple[list[str], list[float]]:
        device = next(self.model.parameters()).device

        input_values = self.feature_extractor.pad(
            [{"input_values": feature} for feature in audio_wavs],
            padding=True,
            return_tensors="pt",
        )

        output_beam_ids = self.model.generate(
            input_values['input_values'].to(device),
            attention_mask=input_values['attention_mask'].to(device),
            decoder_input_ids=self.tokenizer.batch_encode_plus([prefix] * len(audio_wavs), return_tensors="pt")[
                                  'input_ids'][..., :-1].to(device),
            generation_config=GenerationConfig(decoder_start_token_id=self.tokenizer.bos_token_id),
            max_length=300,
            num_beams=3,
            no_repeat_ngram_size=4,
            num_return_sequences=1,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        confidence_scores = [self._compute_confidence(beam_scores) for beam_scores in zip(*output_beam_ids.scores)]
        output_texts = [self._decode_tokens(sequence) for sequence in output_beam_ids.sequences]

        return output_texts, confidence_scores

    def predict(self, audio: NDArray[np.float32], audio_sr: int) -> tuple[str, float]:
        """
        Predict text from audio and return confidence score.

        Returns:
            Tuple containing:
            - text (str): The predicted text
            - confidence (float): Confidence score as percentage (0-100)
        """
        if audio_sr != 16000:
            audio = audio_util.resample_audio(audio, audio_sr, 16000)

        audio_wavs = []
        audio_tensor = torch.from_numpy(audio)
        audio_wavs.append(audio_tensor)

        output_texts, confidence_scores = self._decode_wav(audio_wavs)
        return output_texts[0], confidence_scores[0]

    def _warm_up_model(self):
        dummy_audio = np.zeros(16000, dtype=np.float32)
        _ = self.predict(dummy_audio, 16000)
