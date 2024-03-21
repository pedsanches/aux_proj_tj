import torch
import librosa
import torchaudio
import uuid
import os
from seamless_communication.models.inference import Translator

from transcription_models.model_inference_class import ModelInference

class facebook_seamless(ModelInference):
   def __init__(self):
      #model = torch.jit.load("./transcription_models/unity_on_device_s2t.ptl")
      self.translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=torch.device("cpu"))

      #super().__init__(model, torch.device("cpu"))

   def transcript(self, audio) -> str:
      audio_path = f"./transcription_models/tmp/audio-{uuid.uuid4()}.ogg"
      with open(audio_path, "wb") as f:
         f.write(audio.read())

      with torch.no_grad():
         transcribed_text, _, _ = self.translator.predict(audio_path, "asr", 'por')
      
         print(transcribed_text)

         #transcription_reviewed = self.review_transcript(transcribed_text)

         #transcription_class = transcribed_text
         #transcription_class = self.classify_transcript(transcription_reviewed)
         
         output = { 'transcription': transcribed_text,
                     #'token': transcribed_text,
                     'model used': "seamless"
         }

      try:
         os.remove(audio_path)

      except:
         print("Algo deu Errado na deleção do arquivo")
         
      return output