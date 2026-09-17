import numpy as np
import os

DEFAULT_ALPHA = 0.9 # ALPHA is the weight we choose for lyrics embeddings
DEFAULT_CHUNK_SIZE = 150
DEFAULT_CHUNK_OVERLAP = 30

def combine_title_lyrics(title_embedding, lyrics_embedding, title_text, alpha = DEFAULT_ALPHA):
    a = 1 if title_text.strip() == '' else alpha
    combined = (1 - a) * title_embedding + a * lyrics_embedding
    return combined / np.linalg.norm(combined)

def combine_title_lyrics_batch(title_embeddings, lyrics_embeddings, title_texts, alpha = DEFAULT_ALPHA):
    is_empty_title = np.array([t.strip() == "" for t in title_texts])
    a = np.where(is_empty_title, 1, alpha)
    combined = (1 - a)[:,None] * title_embeddings + a[:, None] * lyrics_embeddings
    return combined / np.linalg.norm(combined, axis=1, keepdims=True)

def checkpointed_encode(model, texts, checkpoint_path, batch_size = 32):
    emb_path = f"{checkpoint_path}_embeddings.npy"
    progress_path = f"{checkpoint_path}_progress.txt"

    total = len(texts)
    dim = model.get_embedding_dimension()

    if os.path.exists(emb_path) and os.path.exists(progress_path):
        embeddings = np.load(emb_path)
        with open(progress_path) as f:
            done = int(f.read().strip())
        print(f"Resuming from checkpoint: {done}/{total} already encoded...")
    else:
        embeddings = np.zeros((total, dim), dtype=np.float32)
        done = 0

    while done < total:
        batch_end = min(done + batch_size, total)
        batch = texts[done:batch_end]

        batch_embeddings = model.encode(batch, normalize_embeddings = True)
        embeddings[done:batch_end] = batch_embeddings

        done = batch_end

        np.save(emb_path, embeddings)
        with open(progress_path, 'w') as f :
            f.write(str(done))
        if done % (batch_size * 10) == 0 or done == total:
            print(f"Encoded {done}/{total}")

    os.remove(progress_path)
    return embeddings

def chunk_text(text, chunk_size = DEFAULT_CHUNK_SIZE, overlap = DEFAULT_CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if len(chunk) < 20:
            break
        chunks.append(" ".join(chunk))
    return chunks

def build_chunk_mapping(texts, chunk_size= DEFAULT_CHUNK_SIZE, overlap = DEFAULT_CHUNK_OVERLAP):
    """
    Chunk every song's lyrics and keep track of which song each chunk came from.
    Returns:
      all_chunks: flat list of chunk strings, across all songs
      chunk_song_indices: same length list, chunk_song_indices[i] = which song all_chunks[i] belongs to
    """
    all_chunks = []
    chunk_song_indices = []

    for song_idx, text in enumerate(texts):
        for chunk in chunk_text(text, chunk_size = chunk_size, overlap = overlap ):
            all_chunks.append(chunk)
            chunk_song_indices.append(song_idx)
    return all_chunks, chunk_song_indices


def pool_chunk_embeddings(chunk_embeddings, chunk_song_indices, num_songs):
    """
    Mean-pool chunk-level embeddings back into one embedding per song.
    Vectorized with np.add.at instead of looping per song (matters once
    num_songs is large — a per-song boolean mask loop is O(num_songs * num_chunks)).
    """
    dim = chunk_embeddings.shape[1]
    chunk_song_indices = np.asarray(chunk_song_indices)

    sums = np.zeros((num_songs, dim), dtype=np.float64)
    counts = np.zeros(num_songs, dtype=np.int64)
    np.add.at(sums, chunk_song_indices, chunk_embeddings)
    np.add.at(counts, chunk_song_indices, 1)

    pooled = sums / counts[:, None]
    pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims = True)
    return pooled.astype(np.float32)

def embed_lyrics_single(model, lyrics_cleaned, chunk_size = DEFAULT_CHUNK_SIZE, overlap = DEFAULT_CHUNK_OVERLAP):
    """
    Same chunk + mean-pool logic, for a single song at query time
    (used by search_by_song), so it matches how the database was built.
    """
    chunks = chunk_text(lyrics_cleaned, chunk_size=chunk_size, overlap=overlap)
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
    pooled = chunk_embeddings.mean(axis=0)
    return pooled / np.linalg.norm(pooled)
