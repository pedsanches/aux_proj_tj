import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from transcription_models.model_inference_class import ModelInference

class pierreguillou_whisper(ModelInference):
    def __init__(self):
        processor = AutoProcessor.from_pretrained("pierreguillou/whisper-medium-portuguese")
        model = AutoModelForSpeechSeq2Seq.from_pretrained("pierreguillou/whisper-medium-portuguese")

        super().__init__(model, torch.device("cpu"))
        self.processor = processor

    def transcript(self, audio) -> str:
        with torch.no_grad():
            speech_array, sampling_rate = librosa.load(audio, sr=16_000)
            inputs = self.processor(speech_array, return_tensors="pt")
            generated_ids = self.model.generate(inputs=inputs.input_features)
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcription_reviewed = self.review_transcript(transcription)
            transcription_class = self.classify_transcript(transcription_reviewed)
            output = {  'transcription': transcription_reviewed,
                        #'sucesso_exec': transcription_class['sucesso_exec'], 
                        #'sucesso_recon': transcription_class['sucesso_recon'], 
                        #'acao': transcription_class['acao'], 
                        #'conf_acao': transcription_class['conf_acao'], 
                        'token': transcription_class['token'], 
                        #'conf_token': transcription_class['conf_token'], 
                        #'aux': transcription_class['aux'],
                        'model used': 'wav2vec2'
            }

        return output
    
    def raw_transcript(self, audio) -> str:
        with torch.no_grad():
            speech_array, sampling_rate = librosa.load(audio, sr=16_000)
            inputs = self.processor(speech_array, return_tensors="pt")
            generated_ids = self.model.generate(inputs=inputs.input_features)
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            transcription_reviewed = self.review_transcript(transcription)
            output = { 'transcription': transcription_reviewed,
                        'model used': 'whisper'
            }

        return output