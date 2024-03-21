import torch
import numpy as np
import pandas as pd
from sentence_transformers  import SentenceTransformer
import faiss


df = pd.read_csv("./sentence_classification/utils/labels.csv")

# Encode categories into numeric labels
category_mapping = {category: idx for idx, category in enumerate(df['category'].unique())}
df['category_id'] = df['category'].map(category_mapping)


class TextClassifier:
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('ricardo-filho/bert-base-portuguese-cased-nli-assin-2')        
        self.device = torch.device("cpu")
        self.threshold = 1
        self.k = 3
        self.labeled_embeddings = self.sentence_transformer.encode(df['text'].tolist())
        faiss.normalize_L2(self.labeled_embeddings)
        self.embedding_dim = self.labeled_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(np.array(self.labeled_embeddings))
        self.train_labels = train_labels = df['category_id']

    def classifier(self, text):
        text_embedding = self.sentence_transformer.encode(text)
        text_embedding = np.array([text_embedding])

        faiss.normalize_L2(text_embedding)

        distances, ann = self.index.search(text_embedding, self.k)

        if distances[0].mean() > self.threshold:
            predicted_category = "CHOOSING"

        else:
            # Vote for the most frequent class among the k nearest neighbors
            predicted_label = np.bincount(self.train_labels[ann[0]]).argmax()
            # Map the numeric label back to the original category
            predicted_category = [category for category, idx in category_mapping.items() if idx == predicted_label][0]

        return {'token': predicted_category}