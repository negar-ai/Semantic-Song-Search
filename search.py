import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_query

# load the embeddings and metadata
title_embeddings = np.load('embeddings/title_embeddings.npy')
lyrics_embeddings = np.load('embeddings/lyrics_embeddings.npy')
song_embeddings = np.load('embeddings/song_embeddings.npy')
data = pd.read_csv('embeddings/metadata.csv')
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "missing someone after breakup"
query = preprocess_query(query)

query_embedding = model.encode([query], normalize_embeddings=True)

similarity_score = cosine_similarity(query_embedding, song_embeddings).flatten()

top_indices = similarity_score.argsort()[-3:][::-1]
results = []
for index in top_indices:
    results.append({
        'Title': data.iloc[index]['Title'],
        'Artist': data.iloc[index]['Artist'],
        'Similarity': similarity_score[index],
        'Lyrics': data.iloc[index]['Lyrics']
    })
print(results)