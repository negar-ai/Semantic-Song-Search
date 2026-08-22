import numpy as np
DEFUALT_ALPHA = 0.65 # ALPHA is the weight we choose for lyrics embeddings

def combine_title_lyrics(title_embedding, lyrics_embedding, title_text, alpha = DEFUALT_ALPHA):
    a = 1 if title_text.strip() == '' else alpha
    combined = (1 - a) * title_embedding + a * lyrics_embedding
    return combined / np.linalg.norm(combined)

def combine_title_lyrics_batch(title_embeddings, lyrics_embeddings, title_texts, alpha = DEFUALT_ALPHA):
    is_empty_title = np.array([t.strip() == "" for t in title_texts])
    a = np.where(is_empty_title, 1, alpha)
    combined = (1 - a)[:,None] * title_embeddings + a[:, None] * lyrics_embeddings
    return combined / np.linalg.norm(combined, axis=1, keepdims=True)
