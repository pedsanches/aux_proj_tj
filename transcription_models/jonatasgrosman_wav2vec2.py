import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from transcription_models.model_inference_class import ModelInference

class jonatasgrosman_wav2vec2(ModelInference):
   def __init__(self):
      processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-portuguese")
      model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-portuguese")

      super().__init__( model, torch.device("cpu"))
      self.processor = processor

   def transcript(self, audio) -> str:
      with torch.no_grad():
         speech_array, sampling_rate = librosa.load(audio, sr=16_000)
         inputs = self.processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
         logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
         predicted_ids = torch.argmax(logits, dim=-1)
         transcription = self.processor.batch_decode(predicted_ids)

         transcription_reviewed = self.review_transcript(transcription[0])

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
         inputs = self.processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")
         logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
         predicted_ids = torch.argmax(logits, dim=-1)
         transcription = self.processor.batch_decode(predicted_ids)

         transcription_reviewed = self.review_transcript(transcription[0])
         
         output = { 'transcription': transcription_reviewed,
                     'model used': "wav2vec2"
         }
         
      return output