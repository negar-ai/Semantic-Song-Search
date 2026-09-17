import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_query
from song_lookup import find_song
from embedding_utils import combine_title_lyrics, embed_lyrics_single, combine_title_lyrics_batch

# load the embeddings and metadata
title_embeddings = np.load('embeddings/title_embeddings.npy')
lyrics_embeddings = np.load('embeddings/lyrics_embeddings.npy')
data = pd.read_csv('embeddings/metadata.csv')

song_embeddings = combine_title_lyrics_batch(
    title_embeddings, lyrics_embeddings, data['Title_cleaned'].fillna('').tolist(), alpha=1.0
)
song_embeddings = song_embeddings / np.linalg.norm(song_embeddings, axis=1, keepdims=True)
np.save('embeddings/song_embeddings.npy', song_embeddings)

# song_embeddings = np.load('embeddings/song_embeddings.npy')
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def _rank_by_embedding(query_embedding, k=5, exclude_index=None):
    similarity_score = cosine_similarity(query_embedding, song_embeddings).flatten()
    if exclude_index is not None:
        similarity_score[exclude_index] = -1  # keep the queried song out of its own results
    top_indices = np.argsort(similarity_score)[-k:][::-1]
    return [{
        "song": data.iloc[i]["song"],
        "artist": data.iloc[i]["artist"],
        'Similarity': similarity_score[i],
        "text": data.iloc[i]["text"]
        }
        for i in top_indices]

def search_by_text(query_text, k=5):
    query = preprocess_query(query_text)
    query_embedding = model.encode([query], normalize_embeddings=True)
    return { "status": "ok", "results": _rank_by_embedding(query_embedding, k=k)}

def search_by_song(title, artist=None, k=5):
    lookup_result = find_song(data, title, artist = artist)
    if lookup_result["status"] in ("not_found", "ambiguous", "fuzzy"):
        return lookup_result # caller/UI should ask user to disambiguous

    matched_song = lookup_result["match"]
    matched_index = matched_song.name

    title_embedding = model.encode([matched_song["Title_cleaned"]], normalize_embeddings=True)[0]
    lyrics_embedding = embed_lyrics_single(model, matched_song["Lyrics_cleaned"])
    # lyrics_embedding = model.encode([matched_song["Lyrics_cleaned"]], normalize_embeddings=True)[0]

    query_embedding = combine_title_lyrics(title_embedding, lyrics_embedding, matched_song["Title_cleaned"]).reshape(1,-1)

    return {
        "status": "ok",
        "matched_song": {"song": matched_song["song"], "artist": matched_song["artist"]},
        "results": _rank_by_embedding(query_embedding, k=k, exclude_index=matched_index)
    }

if __name__ == "__main__":
    print(search_by_text("a holly song about jesus and how he sacrificed himself for me"))
    # print(search_by_text("being naughty in finding love"))
    # print(search_by_song("blanck spcace", artist="taylor swift"))
