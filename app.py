from huggingface_hub import hf_hub_download
import os

REPO_ID = "negar-ai/semantic-song-embeddings"  # same as before, replace with yours

def ensure_embeddings_downloaded():
    os.makedirs('embeddings', exist_ok=True)
    files_needed = [
        "song_embeddings.npy",
        "title_embeddings.npy",
        "lyrics_embeddings.npy",
        "metadata.csv",
    ]
    for filename in files_needed:
        local_path = os.path.join('embeddings', filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename} from Hugging Face Hub...")
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
            )
            # hf_hub_download saves to its own cache location; copy it to where search.py expects it
            import shutil
            shutil.copy(downloaded_path, local_path)
        else:
            print(f"{filename} already present, skipping download.")

ensure_embeddings_downloaded()

from flask import Flask, render_template, request, jsonify
from search import search_by_text, search_by_song

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/search_text', methods=['POST'])
def api_search_text():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400

    result = search_by_text(query, k=5)
    return jsonify(result)

@app.route('/api/search_song', methods=['POST'])
def api_search_song():
    data = request.get_json()
    title = data.get('title', '').strip()
    artist = data.get('artist', '').strip() or None
    
    if not title:
        return jsonify({"status": "error", "message": "Title cannot be empty"}), 400

    result = search_by_song(title, artist=artist, k=5)
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)