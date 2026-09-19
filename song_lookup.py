from rapidfuzz import fuzz, process
from preprocessing import preprocess_title, preprocess_lyrics, preprocess_artist
import pandas as pd

def find_song(data, title, artist=None, threshold=80, top_k = 5):
    title_clean = preprocess_artist(title)
    candidates = data

    # filter by artist if given
    if artist is not None:
        artist_clean = preprocess_artist(artist)
        artist_scores = process.extract(artist_clean, candidates['Artist_cleaned'],
        scorer = fuzz.WRatio, limit = None)
        good_artist_indices = [idx for _, score, idx in artist_scores if score >= threshold]
        if good_artist_indices:
            candidates = candidates.loc[good_artist_indices]

    # 1. exact match
    exact_matches = candidates[candidates['Title_cleaned'] == title_clean]
    if len(exact_matches) == 1:
        return {"status": "exact", "match": exact_matches.iloc[0]}
    if len(exact_matches) > 1:
        return {"status": "ambiguous", "candidates" : exact_matches.to_dict('records')}

    # 2. no exact match
    results = process.extract(title_clean, candidates['Title_cleaned'],
    scorer=fuzz.WRatio, limit = top_k)

    # results = list of (matched_string, score, index_in_candidates)
    strong_matches = [(candidates.loc[idx], score) for _, score, idx
    in results if score >= threshold]
    if not strong_matches:
        return {"status": "not_found"}
    if len(strong_matches) == 1:
        return {"status":"exact", "match": strong_matches[0][0]}
    return {"status": "fuzzy", "candidates": strong_matches}


# data = pd.read_csv('dataset/Songs.csv')
# data = data.dropna(subset=["Lyrics"]).reset_index(drop=True)
# # stop_words = set(stopwords.words('english'))
# data['Lyrics_cleaned'] = data['Lyrics'].apply(preprocess_lyrics)
# data['Title_cleaned'] = data['Title'].apply(preprocess_title)

# print(find_song(data, 'shape', 'ed 4 sheeran'))