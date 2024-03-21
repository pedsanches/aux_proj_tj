import torch
from text_to_num import alpha2digit
from sentence_classification.recon_intencao import Recon_Itencao
from sentence_classification.faiss import TextClassifier
faiss=TextClassifier()
recon_iten = Recon_Itencao()


class ModelInference:
    def __init__(self, model, device):
        # Setar o número de threads em 1 para melhorar a paralelização
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        
        self.device = device
        self.model = model.to(self.device)
        

    def review_transcript(self, text):
        reviewed_text = alpha2digit(
            text, "pt", ordinal_threshold=0
        )
        return reviewed_text

    def classify_transcript(self, text):
        retorno = faiss.classifier(text)
        print(retorno)
        return retorno
