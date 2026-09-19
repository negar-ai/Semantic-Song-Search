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
    app.run(debug=True, port=5000)