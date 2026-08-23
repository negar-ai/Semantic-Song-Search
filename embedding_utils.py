import numpy as np
import os

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