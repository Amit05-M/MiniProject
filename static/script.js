const startButton       = document.getElementById('startButton');
const videoPreview      = document.getElementById('videoPreview');
const videoOverlay      = document.getElementById('videoOverlay');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const inputStatus       = document.getElementById('inputStatus');
const fallbackCard      = document.getElementById('fallbackCard');
const loadingLayer      = document.getElementById('loadingLayer');
const resultSummary     = document.getElementById('resultSummary');
const songResults       = document.getElementById('songResults');

let currentStream = null;

function setStatus(msg, color = '') {
  inputStatus.textContent = msg;
  inputStatus.style.color = color || 'var(--muted)';
}

function showLoading(on) {
  loadingLayer.classList.toggle('hidden', !on);
  startButton.disabled = on;
}

function clearResults() {
  resultSummary.innerHTML = 'Analyzing your mood…';
  songResults.innerHTML = '';
}

/* ---- WAV encoder ---- */
function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const ws = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  const f2pcm = (v2, off, inp) => {
    for (let i = 0; i < inp.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, inp[i]));
      v2.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  };
  ws(0,'RIFF'); v.setUint32(4,36+samples.length*2,true); ws(8,'WAVE'); ws(12,'fmt ');
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,sampleRate,true); v.setUint32(28,sampleRate*2,true);
  v.setUint16(32,2,true); v.setUint16(34,16,true); ws(36,'data');
  v.setUint32(40,samples.length*2,true); f2pcm(v,44,samples);
  return new Blob([buf], { type: 'audio/wav' });
}

function flattenBuffers(buffers) {
  const total = buffers.reduce((a, b) => a + b.length, 0);
  const out = new Float32Array(total);
  let offset = 0;
  buffers.forEach(b => { out.set(b, offset); offset += b.length; });
  return out;
}

async function recordAudio(stream, seconds = 4) {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) return null;
  const ctx = new AC();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const bufs = [];
  proc.onaudioprocess = e => bufs.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ctx.destination);
  await new Promise(r => setTimeout(r, seconds * 1000));
  src.disconnect(); proc.disconnect(); await ctx.close();
  return encodeWav(flattenBuffers(bufs), ctx.sampleRate || 22050);
}

function snapshot() {
  const c = document.createElement('canvas');
  c.width = videoPreview.videoWidth || 640;
  c.height = videoPreview.videoHeight || 480;
  c.getContext('2d').drawImage(videoPreview, 0, 0);
  return c.toDataURL('image/png');
}

function toDataURL(blob) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onloadend = () => res(r.result);
    r.onerror = rej;
    r.readAsDataURL(blob);
  });
}

async function callAnalyze(photo, audio) {
  const res = await fetch('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ photo, audio })
  });
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return res.json();
}

function buildSongCard(song) {
  const embed = song.youtubeEmbedId
    ? `<div class="song-embed"><iframe src="https://www.youtube.com/embed/${song.youtubeEmbedId}?rel=0" allowfullscreen loading="lazy"></iframe></div>`
    : '<p class="no-embed">No YouTube preview available</p>';

  const spotifyBtn = song.spotify
    ? `<a href="${song.spotify}" target="_blank" rel="noreferrer" class="btn-link btn-spotify">▷ Spotify</a>` : '';
  const ytBtn = song.youtubeUrl
    ? `<a href="${song.youtubeUrl}" target="_blank" rel="noreferrer" class="btn-link btn-youtube">▷ YouTube</a>` : '';

  return `
    <div class="song-card">
      <div class="song-meta">
        <div class="song-title">${song.title}</div>
        <div class="song-artist">${song.artist}</div>
        <span class="song-emotion">${song.emotion}</span>
      </div>
      <div class="song-links">${spotifyBtn}${ytBtn}</div>
      ${embed}
    </div>`;
}

async function runDetection() {
  clearResults();
  showLoading(true);
  fallbackCard.classList.add('hidden');
  videoOverlay.classList.add('hidden');
  cameraPlaceholder.classList.remove('hidden');

  try {
    setStatus('Requesting camera & microphone…');
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: true
    });

    videoPreview.srcObject = currentStream;
    cameraPlaceholder.classList.add('hidden');
    await new Promise(r => {
      const t = setTimeout(r, 3000);
      videoPreview.onloadeddata = () => { clearTimeout(t); r(); };
    });
    await videoPreview.play();
    videoOverlay.classList.remove('hidden');
    setStatus('Live feed active — capturing 4 seconds…', 'var(--green)');

    const audioBlob = await recordAudio(currentStream, 4);
    const photoData = snapshot();
    videoOverlay.classList.add('hidden');

    setStatus('Sending to server for analysis…', 'var(--accent)');
    const audioData = audioBlob ? await toDataURL(audioBlob) : null;
    const result = await callAnalyze(photoData, audioData);

    /* Show emotion summary */
    const details = [];
    if (result.facePrediction) details.push(`Face: <b>${result.facePrediction.label}</b>`);
    if (result.voicePrediction) details.push(`Voice: <b>${result.voicePrediction.label}</b>`);
    resultSummary.innerHTML = `
      Detected: <strong style="color:var(--text);font-size:1.1rem">${result.emotion}</strong>
      <span style="color:var(--muted);font-size:0.8rem"> · ${Math.round(result.confidence * 100)}% confidence · source: ${result.source}</span>
      ${details.length ? `<div style="margin-top:4px;font-size:0.8rem;color:var(--muted)">${details.join(' &nbsp;·&nbsp; ')}</div>` : ''}
    `;

    if (!result.songs || result.songs.length === 0) {
      songResults.innerHTML = '<div class="fallback" style="display:block">No songs found for this emotion. Try again!</div>';
    } else {
      songResults.innerHTML = result.songs.map(buildSongCard).join('');
    }

    setStatus('Done! Enjoy your playlist 🎵', 'var(--green)');

  } catch (err) {
    console.error(err);
    fallbackCard.classList.remove('hidden');
    resultSummary.innerHTML = `<span style="color:#fca5a5">Error: ${err.message}</span>`;
    setStatus('Something went wrong. Try again.', 'var(--red)');
  } finally {
    showLoading(false);
    videoOverlay.classList.add('hidden');
    if (currentStream) {
      currentStream.getTracks().forEach(t => t.stop());
      videoPreview.srcObject = null;
      currentStream = null;
    }
    cameraPlaceholder.classList.remove('hidden');
  }
}

startButton.addEventListener('click', runDetection);
