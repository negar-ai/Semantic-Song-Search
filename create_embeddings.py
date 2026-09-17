import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocessing import preprocess_lyrics, preprocess_title, preprocess_artist
from embedding_utils import combine_title_lyrics_batch, checkpointed_encode, build_chunk_mapping, pool_chunk_embeddings


data = pd.read_csv('dataset/song60k.csv')
data = data.dropna(subset=["text"]).reset_index(drop=True)
# stop_words = set(stopwords.words('english'))
data['Lyrics_cleaned'] = data["text"].apply(preprocess_lyrics)
data['Title_cleaned'] = data["song"].apply(preprocess_title)
data['Artist_cleaned'] = data["artist"].apply(preprocess_artist)

# Sentence Transformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
lyrics_list = data['Lyrics_cleaned'].tolist()
title_list = data['Title_cleaned'].fillna('').tolist()

lyrics_chunks, chunk_song_indices = build_chunk_mapping(lyrics_list)


title_embeddings = checkpointed_encode(model, title_list,
checkpoint_path='embeddings/checkpoints/title', batch_size= 32)
# lyrics_embeddings = checkpointed_encode(model, lyrics_list,
# checkpoint_path='embeddings/checkpoints/lyrics', batch_size= 32)

# ++
chunk_embeddings = checkpointed_encode(model, lyrics_chunks,
checkpoint_path='embeddings/checkpoints/lyrics', batch_size=32)
lyrics_embeddings = pool_chunk_embeddings(chunk_embeddings, chunk_song_indices, num_songs=len(data))
# ++




# DEFAULT_ALPHA = 0.8 # ALPHA is the weight we choose for lyrics embeddings
# is_empty_title = data['Title_cleaned'].str.strip().eq('')
# alpha = np.where(is_empty_title, 1, DEFAULT_ALPHA)
song_embeddings = combine_title_lyrics_batch(title_embeddings, lyrics_embeddings, data['Title_cleaned'].tolist())
# save the embeddings for later use
np.save('embeddings/song_embeddings.npy', song_embeddings)
np.save('embeddings/title_embeddings.npy' ,title_embeddings)
np.save('embeddings/lyrics_embeddings.npy' ,lyrics_embeddings)


# save metadata
data[["song", "artist", "text", 'Title_cleaned','Artist_cleaned', 'Lyrics_cleaned']].to_csv('embeddings/metadata.csv', index=False)

# print(data[["song", 'Title_cleaned', 'Lyrics_cleaned']].head())
