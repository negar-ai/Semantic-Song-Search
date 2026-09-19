// Search on Enter key
document.getElementById('text-query').addEventListener('keydown', e => {
  if (e.key === 'Enter') searchByText();
});

document.getElementById('song-title').addEventListener('keydown', e => {
  if (e.key === 'Enter') searchBySong();
});

document.getElementById('song-artist').addEventListener('keydown', e => {
  if (e.key === 'Enter') searchBySong();
});

// Tab switching
document.querySelectorAll('.tab-button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active')
    document.getElementById('results').innerHTML = '';
  });
  
});

async function searchByText() {
  const query = document.getElementById('text-query').value;
  if (!query.trim()) return;

  const button = document.getElementById('text-search-btn');
  showLoading(button);
  try {
    const response = await fetch('api/search_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await response.json();
    renderResults(data);
  } catch (err) {
    showError();
  } finally {
    hideLoading(button);
  }
}

async function searchBySong(){
  const title = document.getElementById('song-title').value
  const artist = document.getElementById('song-artist').value
  if (!title.trim()) return;

  const button = document.getElementById('song-search-btn');
  showLoading(button);
  try {
    const response = await fetch('api/search_song', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, artist  })
    });
    const data = await response.json();
    renderResults(data);
  } catch (err) {
    showError();
  } finally {
    hideLoading(button);
  }
}

function showLoading(button) {
  button.disabled = true;
  const container = document.getElementById('results');
  container.innerHTML = `
    <div class="loading">
      <div class="spinner"></div>
      <span>Searching for songs…</span>
    </div>
  `;
}

function hideLoading(button) {
  button.disabled = false;
}

function showError() {
  const container = document.getElementById('results');
  container.innerHTML = '<p>Something went wrong while searching. Please try again.</p>';
}

function renderResults(data) {
  const container = document.getElementById('results')
  container.innerHTML = '';

  if (data.status === 'not_found') {
    container.innerHTML = '<p>No matching song found 😰</p>';
    return;
  }

  if (data.status === 'ambiguous') {
    container.innerHTML += `<p>Multiple possible matches 🙄 try adding an artist name.</p>`;
    return;
  }

  if (data.matched_song) {
    container.innerHTML += `<p> 🥳 Songs similar to ${data.matched_song.song} by ${data.matched_song.artist}:</p>`;
  }

  data.results.forEach(r => {
    const card = document.createElement('div')
    card.className = 'result-card'
    card.innerHTML = `
      <h3>${r.song}</h3>
      <div class="artist">${r.artist}</div>
      <div class="similarity">Similarity: ${r.Similarity.toFixed(3)}</div>
    `;
    container.appendChild(card);
  })
}