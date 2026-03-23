'use strict';

// ─────────────────────────────────────────────
// Session state
// ─────────────────────────────────────────────

const SESSION = {
  active:       false,
  queue:        [],       // [{passage_ref, level, added_at, last_attempt_date, …}]
  queueIdx:     0,
  timeLimit:    1,        // minutes; 1 = just one verse; 0 = all
  startedAt:    null,
  priority:     'struggling',

  // Current verse
  passageRef:   null,
  segments:     [],
  segmentIdx:   0,

  // Phase: 'idle'|'preparing'|'playing'|'listening'|'scoring'|'feedback'
  phase:        'idle',

  // Audio playback
  currentAudioEl:      null,
  lastSegmentAudioUrl: null,  // URL for replay (server path, not object URL)

  // Recording
  mediaStream:     null,
  recorder:        null,
  recordingChunks: [],
  discardRecording: false,

  // VAD
  audioCtx:    null,
  analyser:    null,
  silenceTimer: null,
  voiceActive: false,
  _vadFrame:   null,

  // Double-tap
  lastTapTime: 0,
  tapTimer:    null,
};

// Session-start overlay selections
let _startTime     = 1;
let _startPriority = 'struggling';

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

function $id(id) { return document.getElementById(id); }

// ─────────────────────────────────────────────
// Session start overlay
// ─────────────────────────────────────────────

function openSessionStart() {
  if (!state || !state.verses || state.verses.length === 0) {
    showToast('Add some verses first.');
    return;
  }
  openModal('modal-session-start');
}

function _setupSegControl(containerId, onChange) {
  $id(containerId).addEventListener('click', e => {
    const btn = e.target.closest('.seg-btn');
    if (!btn) return;
    $id(containerId).querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    onChange(btn.dataset.value);
  });
}

// ─────────────────────────────────────────────
// Queue building
// ─────────────────────────────────────────────

function buildQueue(verses, priority, timeLimit) {
  let pool = [...verses];

  switch (priority) {
    case 'struggling':
      pool.sort((a, b) => {
        const ld = Math.max(0, a.level) - Math.max(0, b.level);
        if (ld !== 0) return ld;
        const ad = a.last_attempt_date ? new Date(a.last_attempt_date) : new Date(0);
        const bd = b.last_attempt_date ? new Date(b.last_attempt_date) : new Date(0);
        return ad - bd;
      });
      break;
    case 'neglected':
      pool.sort((a, b) => {
        const ad = a.last_attempt_date ? new Date(a.last_attempt_date) : new Date(0);
        const bd = b.last_attempt_date ? new Date(b.last_attempt_date) : new Date(0);
        return ad - bd;
      });
      break;
    case 'recent':
      pool.sort((a, b) => new Date(b.added_at) - new Date(a.added_at));
      break;
    case 'random':
      for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [pool[i], pool[j]] = [pool[j], pool[i]];
      }
      break;
  }

  return timeLimit === 1 ? pool.slice(0, 1) : pool;
}

// ─────────────────────────────────────────────
// Session lifecycle
// ─────────────────────────────────────────────

async function beginSession() {
  closeModal('modal-session-start');

  const verses = (state && state.verses) || [];
  if (verses.length === 0) return;

  // Request mic permission upfront
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: _audioConstraints() });
    s.getTracks().forEach(t => t.stop());
  } catch (_) {
    showToast('Microphone access is required for practice sessions.');
    return;
  }

  SESSION.queue      = buildQueue(verses, _startPriority, _startTime);
  SESSION.queueIdx   = 0;
  SESSION.timeLimit  = _startTime;
  SESSION.startedAt  = Date.now();
  SESSION.active     = true;

  _showSessionScreen();
  _practiceNextVerse();
}

function _showSessionScreen() {
  $id('session-screen').classList.add('active');
  document.body.style.overflow = 'hidden';
}

function _hideSessionScreen() {
  $id('session-screen').classList.remove('active');
  document.body.style.overflow = '';
  SESSION.active = false;
  SESSION.phase  = 'idle';
}

async function _practiceNextVerse() {
  if (SESSION.queueIdx >= SESSION.queue.length) {
    _endSession(); return;
  }

  // Time-limited sessions: stop after the limit expires (checked between verses)
  if (SESSION.timeLimit > 1) {
    const elapsed = (Date.now() - SESSION.startedAt) / 60_000;
    if (elapsed >= SESSION.timeLimit) {
      _endSession(); return;
    }
  }

  const verse = SESSION.queue[SESSION.queueIdx];
  SESSION.passageRef  = verse.passage_ref;
  SESSION.segments    = [];
  SESSION.segmentIdx  = 0;

  _setPhase('preparing');

  try {
    const data = await apiGet(`/sessions/${encodeURIComponent(SESSION.passageRef)}/segments`);
    SESSION.segments = data.segments;
  } catch (_) {
    showToast(`Could not load ${SESSION.passageRef} — skipping`);
    SESSION.queueIdx++;
    _practiceNextVerse();
    return;
  }

  _practiceSegment();
}

async function _practiceSegment() {
  if (!SESSION.active) return;

  if (SESSION.segmentIdx >= SESSION.segments.length) {
    await _finishVerse(); return;
  }

  _setPhase('playing');

  const ref = SESSION.passageRef;
  const idx = SESSION.segmentIdx;
  const audioUrl = `/sessions/${encodeURIComponent(ref)}/segments/${idx}/audio`;

  // Ensure recording is fully stopped before any playback begins
  _stopRecording(true);

  // Tear down previous playback
  if (SESSION.currentAudioEl) {
    SESSION.currentAudioEl.pause();
    SESSION.currentAudioEl = null;
  }

  SESSION.lastSegmentAudioUrl = audioUrl;

  const audio = new Audio(audioUrl);
  SESSION.currentAudioEl = audio;

  audio.onended = () => {
    if (SESSION.phase === 'playing') setTimeout(_startListening, 500);
  };
  audio.onerror = () => {
    // No audio available — go straight to listening after the same delay
    if (SESSION.phase === 'playing') setTimeout(_startListening, 500);
  };

  audio.play().catch(() => {
    if (SESSION.phase === 'playing') setTimeout(_startListening, 500);
  });
}

function _startListening() {
  if (!SESSION.active || SESSION.phase === 'feedback') return;
  _setPhase('listening');
  _startRecording();
}

// ─────────────────────────────────────────────
// Recording + VAD
// ─────────────────────────────────────────────

const VAD_THRESHOLD = 0.015;
const SILENCE_MS    = 1500;

function _audioConstraints() {
  const deviceId = localStorage.getItem('micDeviceId');
  return deviceId ? { deviceId: { exact: deviceId } } : true;
}

async function _startRecording() {
  SESSION.recordingChunks  = [];
  SESSION.discardRecording = false;

  try {
    SESSION.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: _audioConstraints() });
  } catch (_) {
    showToast('Microphone unavailable');
    _skipSegment(); return;
  }

  // Audio analysis for VAD + pulse
  SESSION.audioCtx = new AudioContext();
  const source = SESSION.audioCtx.createMediaStreamSource(SESSION.mediaStream);
  SESSION.analyser = SESSION.audioCtx.createAnalyser();
  SESSION.analyser.fftSize = 512;
  SESSION.analyser.smoothingTimeConstant = 0.4;
  source.connect(SESSION.analyser);

  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : 'audio/webm';

  SESSION.recorder = new MediaRecorder(SESSION.mediaStream, { mimeType });
  SESSION.recorder.ondataavailable = e => {
    if (e.data.size > 0) SESSION.recordingChunks.push(e.data);
  };
  SESSION.recorder.onstop = _onRecordingStop;
  SESSION.recorder.start(100);

  SESSION.voiceActive  = false;
  SESSION.silenceTimer = null;
  _vadLoop();
}

function _vadLoop() {
  if (!SESSION.analyser || !SESSION.recorder || SESSION.recorder.state !== 'recording') return;

  const buf = new Float32Array(SESSION.analyser.fftSize);
  SESSION.analyser.getFloatTimeDomainData(buf);

  let rms = 0;
  for (const s of buf) rms += s * s;
  rms = Math.sqrt(rms / buf.length);

  const hasVoice = rms > VAD_THRESHOLD;

  if (hasVoice !== SESSION.voiceActive) {
    SESSION.voiceActive = hasVoice;
    $id('session-pulse-wrap').classList.toggle('pulse-active', hasVoice);
  }

  if (!hasVoice) {
    if (!SESSION.silenceTimer) {
      SESSION.silenceTimer = setTimeout(_stopRecording, SILENCE_MS);
    }
  } else {
    clearTimeout(SESSION.silenceTimer);
    SESSION.silenceTimer = null;
  }

  SESSION._vadFrame = requestAnimationFrame(_vadLoop);
}

function _stopRecording(discard = false) {
  SESSION.discardRecording = discard;

  clearTimeout(SESSION.silenceTimer);
  SESSION.silenceTimer = null;

  if (SESSION._vadFrame) {
    cancelAnimationFrame(SESSION._vadFrame);
    SESSION._vadFrame = null;
  }

  if (SESSION.recorder && SESSION.recorder.state === 'recording') {
    SESSION.recorder.stop();
  }

  if (SESSION.mediaStream) {
    SESSION.mediaStream.getTracks().forEach(t => t.stop());
    SESSION.mediaStream = null;
  }

  if (SESSION.audioCtx) {
    SESSION.audioCtx.close().catch(() => {});
    SESSION.audioCtx = null;
  }

  $id('session-pulse-wrap').classList.remove('pulse-active');
}

async function _onRecordingStop() {
  if (SESSION.discardRecording) return;
  if (!SESSION.active) return;

  _setPhase('scoring');

  const blob = new Blob(SESSION.recordingChunks, { type: 'audio/webm' });

  const fd = new FormData();
  fd.append('passage_ref', SESSION.passageRef);
  fd.append('segment_idx', SESSION.segmentIdx);
  fd.append('audio', blob, 'recording.webm');

  try {
    const res = await fetch('/sessions/score', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(res.status);
  } catch (_) {
    showToast('Could not save segment — skipping');
    _skipSegment(); return;
  }

  SESSION.segmentIdx++;
  _practiceSegment();
}

function _skipSegment() {
  // No audio file will be written for this segment; the finish endpoint
  // handles missing files gracefully by excluding them from the stitched audio.
  SESSION.segmentIdx++;
  _practiceSegment();
}

// ─────────────────────────────────────────────
// Finish verse + feedback
// ─────────────────────────────────────────────

async function _finishVerse() {
  let result;
  try {
    const res = await fetch('/sessions/finish', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        passage_ref:   SESSION.passageRef,
        segment_count: SESSION.segments.length,
      }),
    });
    if (res.status === 422) {
      // Server found no audio (all segments skipped) — advance silently
      _advanceQueue(); return;
    }
    if (!res.ok) throw new Error(res.status);
    result = await res.json();
  } catch (_) {
    showToast('Could not save results');
    _advanceQueue(); return;
  }

  _showFeedback(result);
}

function _showFeedback(result) {
  _setPhase('feedback');

  $id('feedback-ref').textContent = result.passage_ref;

  const pct  = Math.round(result.overall_score * 100);
  const pill = $id('feedback-score-pill');
  pill.textContent = `${pct}%`;
  pill.className   = pct >= 80 ? 'score-good' : pct >= 50 ? 'score-ok' : 'score-poor';

  // Full-passage word-level diff
  const diffEl = $id('feedback-diff');
  diffEl.innerHTML = '';

  (result.diff || []).forEach(token => {
    const span = document.createElement('span');
    span.className = `diff-${token.status}`;
    span.textContent = token.word;
    diffEl.appendChild(span);
    if (token.status !== 'inserted') {
      diffEl.appendChild(document.createTextNode(' '));
    }
  });

  // Raw transcript (debug)
  const transcriptEl = $id('feedback-transcript');
  if (result.transcript && result.transcript.trim()) {
    transcriptEl.textContent = result.transcript;
    transcriptEl.hidden = false;
  } else {
    transcriptEl.textContent = '';
    transcriptEl.hidden = true;
  }

  // Level change line
  const lvlEl = $id('feedback-level');
  if (result.level_after !== result.level_before) {
    const up = result.level_after > result.level_before;
    lvlEl.textContent = `${up ? '↑' : '↓'} Level ${result.level_before + 1} → ${result.level_after + 1}`;
    lvlEl.style.color = up ? 'var(--lvl-adv-fg)' : 'var(--destructive)';
  } else {
    lvlEl.textContent = `Level ${result.level_after + 1}`;
    lvlEl.style.color = 'var(--text-muted)';
  }
}

function _advanceQueue() {
  SESSION.queueIdx++;
  _practiceNextVerse();
}

function _endSession() {
  _cleanup();
  _hideSessionScreen();
  showToast('Session complete');
  if (window.loadVerses) loadVerses();
}

// ─────────────────────────────────────────────
// Controls: restart / end / continue
// ─────────────────────────────────────────────

function _restartVerse() {
  _stopRecording(true);
  if (SESSION.currentAudioEl) {
    SESSION.currentAudioEl.pause();
    SESSION.currentAudioEl = null;
  }
  SESSION.segmentIdx = 0;
  _setPhase('preparing');
  // Segments already loaded — jump straight into playback
  _practiceSegment();
}

function _endSessionEarly() {
  _stopRecording(true);
  if (SESSION.currentAudioEl) {
    SESSION.currentAudioEl.pause();
    SESSION.currentAudioEl = null;
  }
  _hideSessionScreen();
  if (window.loadVerses) loadVerses();
}

function _cleanup() {
  _stopRecording(true);
  if (SESSION._vadFrame) {
    cancelAnimationFrame(SESSION._vadFrame);
    SESSION._vadFrame = null;
  }
  if (SESSION.currentAudioEl) {
    SESSION.currentAudioEl.pause();
    SESSION.currentAudioEl = null;
  }
}

// ─────────────────────────────────────────────
// Replay (double-tap)
// ─────────────────────────────────────────────

function _replayLastSegment() {
  if (!SESSION.lastSegmentAudioUrl) return;
  _stopRecording(true);

  if (SESSION.currentAudioEl) {
    SESSION.currentAudioEl.pause();
    SESSION.currentAudioEl = null;
  }

  _setPhase('playing');

  const audio = new Audio(SESSION.lastSegmentAudioUrl);
  SESSION.currentAudioEl = audio;
  audio.onended = () => {
    if (SESSION.phase === 'playing') setTimeout(_startListening, 500);
  };
  audio.play().catch(() => {});
}

// ─────────────────────────────────────────────
// Tap / double-tap detection
// ─────────────────────────────────────────────

function _onSessionTap(e) {
  // Ignore taps on control buttons or feedback panel buttons
  if (e.target.closest('#session-controls') || e.target.closest('#session-feedback-view')) return;

  const now      = Date.now();
  const isDouble = (now - SESSION.lastTapTime) < 280;
  SESSION.lastTapTime = now;

  if (isDouble) {
    clearTimeout(SESSION.tapTimer);
    SESSION.tapTimer = null;
    _replayLastSegment();
    return;
  }

  if (SESSION.phase === 'listening') {
    SESSION.tapTimer = setTimeout(() => {
      SESSION.tapTimer = null;
      if (SESSION.phase === 'listening') _stopRecording();
    }, 280);
  }
}

// ─────────────────────────────────────────────
// Phase management → drives all CSS state
// ─────────────────────────────────────────────

const _PHASES = ['preparing', 'playing', 'listening', 'scoring', 'feedback'];

function _setPhase(phase) {
  SESSION.phase = phase;

  const screen = $id('session-screen');
  _PHASES.forEach(p => screen.classList.toggle(`phase-${p}`, p === phase));

  // Reference
  $id('session-ref-label').textContent = SESSION.passageRef || '';

  // Segment indicator
  if (SESSION.segments.length > 0 && phase !== 'feedback') {
    const disp = Math.min(SESSION.segmentIdx + 1, SESSION.segments.length);
    $id('session-segment-label').textContent = `${disp} of ${SESSION.segments.length}`;
  } else {
    $id('session-segment-label').textContent = '';
  }

  // Status label
  const labels = {
    preparing: 'Loading…',
    playing:   'Listen…',
    listening: 'Say it back',
    scoring:   'Checking…',
    feedback:  '',
  };
  $id('session-status-label').textContent = labels[phase] || '';
}

// ─────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {

  // Practice button
  document.getElementById('btn-practice').addEventListener('click', openSessionStart);

  // Session start overlay controls
  _setupSegControl('time-seg',     v => { _startTime     = parseInt(v, 10); });
  _setupSegControl('priority-seg', v => { _startPriority = v; });

  $id('btn-cancel-session').addEventListener('click', () => closeModal('modal-session-start'));
  $id('modal-session-start').querySelector('.modal-backdrop')
    .addEventListener('click', () => closeModal('modal-session-start'));

  $id('btn-start-session').addEventListener('click', beginSession);

  // Session screen controls
  $id('btn-restart-verse').addEventListener('click', _restartVerse);
  $id('btn-end-session').addEventListener('click', _endSessionEarly);
  $id('btn-continue-session').addEventListener('click', () => _advanceQueue());

  // Tap / double-tap on session screen
  $id('session-screen').addEventListener('click', _onSessionTap);

  // Escape key: close session screen only if in feedback or idle
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if (SESSION.phase === 'feedback') { _advanceQueue(); }
  });

});
