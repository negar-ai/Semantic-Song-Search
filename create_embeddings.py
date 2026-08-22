import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from preprocessing import preprocess_lyrics, preprocess_title, preprocess_artist
from embedding_utils import combine_title_lyrics_batch

# nltk.download('stopwords')

data = pd.read_csv('dataset/Songs.csv')
data = data.dropna(subset=["Lyrics"]).reset_index(drop=True)
# stop_words = set(stopwords.words('english'))
data['Lyrics_cleaned'] = data['Lyrics'].apply(preprocess_lyrics)
data['Title_cleaned'] = data['Title'].apply(preprocess_title)
data['Artist_cleaned'] = data['Artist'].apply(preprocess_artist)

# Sentence Transformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
lyrics_list = data['Lyrics_cleaned'].tolist()
title_list = data['Title_cleaned'].fillna('').tolist()

title_embeddings = model.encode(title_list, batch_size= 32, normalize_embeddings=True, show_progress_bar= True)
lyrics_embeddings = model.encode(lyrics_list, batch_size= 32, normalize_embeddings=True, show_progress_bar= True)

defaut_ALPHA = 0.65 # ALPHA is the weight we choose for lyrics embeddings
is_empty_title = data['Title_cleaned'].str.strip().eq('')
alpha = np.where(is_empty_title, 1, defaut_ALPHA)
song_embeddings = combine_title_lyrics_batch(title_embeddings, lyrics_embeddings, data['Title_cleaned'].tolist())

# save the embeddings for later use
np.save('embeddings/song_embeddings.npy', song_embeddings)
np.save('embeddings/title_embeddings.npy' ,title_embeddings)
np.save('embeddings/lyrics_embeddings.npy' ,lyrics_embeddings)


# save metadata
data[['Title', 'Artist', 'Lyrics', 'Title_cleaned','Artist_cleaned', 'Lyrics_cleaned']].to_csv('embeddings/metadata.csv', index=False)

# print(data[['Title', 'Title_cleaned', 'Lyrics_cleaned']].head())
