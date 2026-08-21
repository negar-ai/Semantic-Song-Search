import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# nltk.download('stopwords')

# url = "Lyrics_dataset/Songs.csv"
data = pd.read_csv('dataset/Songs.csv')
data = data.dropna(subset=["Lyrics"]).copy()

print(data.head())

stop_words = set(stopwords.words('english'))


def preprocess_lyrics(lyrics):
    lyrics = lyrics.lower()
    # Remove dataset artifacts
    lyrics = re.sub(  r'\d*embed(?:share)?\s*urlcopyembedcopy', ' ',  lyrics  )
    # Remove punctuation
    lyrics = ''.join(char for char in lyrics if char.isalnum() or char.isspace())
    # Normalize whitespace
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    return lyrics

def preprocess_title(title):
    title = title.lower()
    # Remove punctuation
    title = ''.join(char for char in title if char.isalnum() or char.isspace())
    # Normalize whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def preprocess_query(query):
    query = query.lower()
    # Remove punctuation
    query = ''.join(char for char in query if char.isalnum() or char.isspace())
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    return query

query = "missing someone after breakup"

data['Lyrics_cleaned'] = data['Lyrics'].apply(preprocess_lyrics)
data['title_cleaned'] = data['Title'].apply(preprocess_title)

query = preprocess_query(query)

# print(data[['Lyrics_cleaned', 'Lyrics']].head())

def search_songs(query, vectorizer, tfidf_matrix, data, k=5):
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[-k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            'Title': data.iloc[index]['Title'],
            'Artist': data.iloc[index]['Artist'],
            'Similarity': similarity_scores[index],
            'Lyrics': data.iloc[index]['Lyrics']
        })

    return results

def chunk_text(text, chunk_size=100, overlap=20):

    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if len(chunk) < 20:
            break
        chunks.append(" ".join(chunk))
    return chunks

vectorizer = TfidfVectorizer(ngram_range=(2,2), max_df=0.8)

tfidf_matrix = vectorizer.fit_transform(data['Lyrics_cleaned'])
query_vector = vectorizer.transform([query])

# results = search_songs(
#     query,
#     vectorizer,
#     tfidf_matrix,
#     data,
#     k=5
# )

# for i, result in enumerate(results, start=1):

#     print(f"\nRank {i}")
#     print(f"Title: {result['Title']}")
#     print(f"Artist: {result['Artist']}")
#     print(f"Similarity: {result['Similarity']:.4f}")
#     print(f"Lyrics: {result['Lyrics']}")

# Sentence Transformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
lyrics_list = data['Lyrics_cleaned'].tolist()
title_list = data['Title_cleaned'].fillna('').tolist()
all_chunks = []
chunk_song_indices = []

for song_index, lyrics in enumerate(data["Lyrics_cleaned"]):

    chunks = chunk_text(
        lyrics,
        chunk_size=100,
        overlap=20
    )

    for chunk in chunks:

        all_chunks.append(chunk)
        chunk_song_indices.append(song_index)

print("Total chunks:", len(all_chunks))
title_embeddings = model.encode(title_list, batch_size= 32, normalize_embeddings=True, show_progress_bar= True)
# lyrics_embeddings = model.encode(lyrics_list, batch_size= 32, normalize_embeddings=True, show_progress_bar= True)
chunk_embeddings = model.encode(all_chunks, batch_size= 32, normalize_embeddings=True, show_progress_bar= True)
print("Chunk embeddings shape:", chunk_embeddings.shape)


song_embeddings = np.zeros((len(data), chunk_embeddings.shape[1]))

for song_index in range(len(data)):
    song_chunk_indices = [
        i for i, idx in enumerate(chunk_song_indices) if idx == song_index
    ]

    song_embeddings[song_index] = np.mean(
        chunk_embeddings[song_chunk_indices],axis = 0)




# print("Title embeddings:", title_embeddings.shape)
# print("Lyrics embeddings:", lyrics_embeddings.shape)

# weights = [
#     (0.0, 1.0),
#     (0.1, 0.9),
#     (0.2, 0.8),
#     (0.3, 0.7),
#     (0.5, 0.5),
#     (0.7, 0.3),
#     (1.0, 0.0)
# ]

# song_embeddings = ( TITLE_WEIGHT * title_embeddings +  LYRICS_WEIGHT * lyrics_embeddings )
song_embeddings = song_embeddings / np.linalg.norm( song_embeddings, axis=1, keepdims=True )

query_embedding = model.encode([query], normalize_embeddings=True)

# for title_weight, lyrics_weight in weights:

#     song_embeddings = (
#         title_weight * title_embeddings +
#         lyrics_weight * lyrics_embeddings
#     )

#     song_embeddings = song_embeddings / np.linalg.norm(
#         song_embeddings,
#         axis=1,
#         keepdims=True
#     )

#     similarities = cosine_similarity(
#         query_embedding,
#         song_embeddings
#     ).flatten()

#     top_indices = similarities.argsort()[-5:][::-1]

#     print(
#         f"\n{'='*50}"
#         f"\nTitle: {title_weight:.1f} | "
#         f"Lyrics: {lyrics_weight:.1f}"
#         f"\n{'='*50}"
#     )

#     for rank, index in enumerate(top_indices, start=1):

#         print(
#             f"{rank}. "
#             f"{data.iloc[index]['Title']} - "
#             f"{data.iloc[index]['Artist']} "
#             f"({similarities[index]:.4f})"
        # )
TITLE_WEIGHT = 0.3
LYRICS_WEIGHT = 0.7

# song_embeddings = ( TITLE_WEIGHT * title_embeddings +  LYRICS_WEIGHT * lyrics_embeddings )
# song_embeddings = ( TITLE_WEIGHT * title_embeddings +  LYRICS_WEIGHT * chunk_embeddings )
song_embeddings = song_embeddings / np.linalg.norm( song_embeddings, axis=1, keepdims=True )
query_embedding = model.encode([query], normalize_embeddings=True)
embeddings = model.encode(lyrics_list)
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
