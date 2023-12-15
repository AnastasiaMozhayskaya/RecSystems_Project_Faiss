import pandas as pd
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
model = BertModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def filter_by_ganre(df: pd.DataFrame, ganre_list: list):
    filtered_df = df[df['ganres'].apply(lambda x: any(g in ganre_list for g in(x)))]
    filt_ind = filtered_df.index.to_list()
    return filt_ind


def embed_user(filt_ind: list, embeddings:np.array, user_text: str, n=100):
    tokens = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**tokens)
        user_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().reshape(1, -1)
    return user_embedding

def sort_embeddings(user_embedding: np.array, embeddings: np.array):
    distances = np.linalg.norm(embeddings - user_embedding, axis=1)  # Расчет эвклидова расстояния
    sorted_indices = np.argsort(distances)[::-1]  # Сортировка индексов в порядке убывания
    sorted_embeddings = embeddings[sorted_indices]  # Сортировка эмбеддингов
    return sorted_embeddings