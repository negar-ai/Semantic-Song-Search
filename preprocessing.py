import re

from matplotlib.pyplot import title


def preprocess_lyrics(lyrics):
    lyrics = lyrics.lower()
    # Remove dataset artifacts
    lyrics = re.sub(  r'\d*embed(?:share)?\s*urlcopyembedcopy', ' ',  lyrics  )
    # Remove punctuation
    lyrics = ''.join(char for char in lyrics if char.isalnum() or char.isspace())
    return re.sub(r'\s+', ' ', lyrics).strip()  # Normalize whitespace

def preprocess_title(title):
    title = title.lower()
    # Remove punctuation
    title = ''.join(char for char in title if char.isalnum() or char.isspace())
    return re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace

def preprocess_query(query):
    query = query.lower()
    # Remove punctuation
    query = ''.join(char for char in query if char.isalnum() or char.isspace())
    return re.sub(r'\s+', ' ', query).strip()  # Normalize whitespace
