'use strict';

// ─────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────

const LEVEL_LABELS = [
  '3 words', '5 words', '8 words', '12 words',
  'Full verse', 'Full passage', 'Mastered',
];

// Canonical Bible book order for biblical sorting
const BIBLE_BOOKS = [
  'Genesis','Exodus','Leviticus','Numbers','Deuteronomy',
  'Joshua','Judges','Ruth','1 Samuel','2 Samuel',
  '1 Kings','2 Kings','1 Chronicles','2 Chronicles',
  'Ezra','Nehemiah','Esther','Job','Psalms','Proverbs',
  'Ecclesiastes','Song of Solomon','Song of Songs','Isaiah','Jeremiah',
  'Lamentations','Ezekiel','Daniel','Hosea','Joel','Amos',
  'Obadiah','Jonah','Micah','Nahum','Habakkuk','Zephaniah',
  'Haggai','Zechariah','Malachi',
  'Matthew','Mark','Luke','John','Acts',
  'Romans','1 Corinthians','2 Corinthians','Galatians','Ephesians',
  'Philippians','Colossians','1 Thessalonians','2 Thessalonians',
  '1 Timothy','2 Timothy','Titus','Philemon','Hebrews',
  'James','1 Peter','2 Peter','1 John','2 John','3 John','Jude',
  'Revelation',
];

// ─────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────

const state = {
  /** @type {Array<{passage_ref:string, added_at:string, level:number, level_name:string, last_attempt_date:string|null}>} */
  verses: [],
  sortKey: 'added',    // 'added' | 'biblical' | 'proficiency'
  sortDir: 'asc',      // 'asc' | 'desc'
  deleteMode: false,
  /** passage_ref pending delete confirmation */
  pendingDelete: null,
  loading: true,
};

// ─────────────────────────────────────────────
// API helpers
// ─────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const res = await fetch(path, options);
  return res;
}

async function apiGet(path) {
  const res = await apiFetch(path);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

async function apiDelete(path) {
  const res = await apiFetch(path, { method: 'DELETE' });
  if (!res.ok) throw new Error(`DELETE ${path} → ${res.status}`);
}

// ─────────────────────────────────────────────
// Data loading
// ─────────────────────────────────────────────

async function loadVerses() {
  setLoading(true);

  let list;
  try {
    list = await apiGet('/verses');
  } catch (e) {
    setLoading(false);
    showToast('Failed to load verses. Is the server running?');
    return;
  }

  // Render immediately with what we have — level shows as 'new' until progress arrives
  state.verses = list.map(v => ({
    passage_ref:       v.passage_ref,
    added_at:          v.added_at,
    level:             -1,           // -1 = not yet loaded
    level_name:        'WORDS_3',
    last_attempt_date: null,
  }));

  setLoading(false);
  renderList();

  // Fetch progress for all verses in parallel, update cards as they arrive
  const fetches = list.map(async (v, i) => {
    try {
      const p = await apiGet(`/progress/${encodeURIComponent(v.passage_ref)}`);
      state.verses[i] = {
        ...state.verses[i],
        level:             p.level,
        level_name:        p.level_name,
        last_attempt_date: p.last_attempt_date,
      };
      // Patch just this card in the DOM rather than re-rendering everything
      patchCard(state.verses[i]);
    } catch (_) {
      // If progress fetch fails, default to level 0
      state.verses[i] = { ...state.verses[i], level: 0 };
      patchCard(state.verses[i]);
    }
  });

  await Promise.allSettled(fetches);

  // Re-render if sorting by proficiency so order reflects fetched data
  if (state.sortKey === 'proficiency') renderList();
}

// ─────────────────────────────────────────────
// Sorting
// ─────────────────────────────────────────────

function parseRef(ref) {
  const m = ref.match(/^(.+?)\s+(\d+):(\d+)/);
  if (!m) return { bookIdx: 999, chapter: 0, verse: 0 };
  const bookIdx = BIBLE_BOOKS.findIndex(
    b => b.toLowerCase() === m[1].toLowerCase()
  );
  return {
    bookIdx: bookIdx === -1 ? 999 : bookIdx,
    chapter: parseInt(m[2], 10),
    verse:   parseInt(m[3], 10),
  };
}

function getSorted() {
  return [...state.verses].sort((a, b) => {
    let cmp = 0;

    if (state.sortKey === 'biblical') {
      const pa = parseRef(a.passage_ref), pb = parseRef(b.passage_ref);
      cmp = (pa.bookIdx - pb.bookIdx) || (pa.chapter - pb.chapter) || (pa.verse - pb.verse);
    } else if (state.sortKey === 'added') {
      cmp = new Date(a.added_at) - new Date(b.added_at);
    } else { // proficiency
      // Treat unloaded (-1) as level 0 for sorting
      cmp = Math.max(0, a.level) - Math.max(0, b.level);
    }

    return state.sortDir === 'asc' ? cmp : -cmp;
  });
}

// ─────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────

function relativeTime(dateStr, prefix = '') {
  if (!dateStr) return null;
  const days = Math.floor((Date.now() - new Date(dateStr)) / 86_400_000);
  if (days === 0) return prefix ? `${prefix} today` : 'Today';
  if (days === 1) return prefix ? `${prefix} yesterday` : 'Yesterday';
  if (days < 7)  return `${prefix ? prefix + ' ' : ''}${days}d ago`;
  if (days < 30) return `${prefix ? prefix + ' ' : ''}${Math.floor(days / 7)}w ago`;
  return `${prefix ? prefix + ' ' : ''}${Math.floor(days / 30)}mo ago`;
}

function subtitleFor(v) {
  if (v.last_attempt_date) {
    return `Last practiced ${relativeTime(v.last_attempt_date)}`;
  }
  return relativeTime(v.added_at, 'Added') || 'Added recently';
}

function levelClass(level) {
  if (level < 0)  return 'level-early';   // loading state
  if (level >= 6) return 'level-mastered';
  if (level >= 4) return 'level-advanced';
  if (level >= 2) return 'level-developing';
  return 'level-early';
}

function levelLabel(level) {
  if (level < 0) return '·';   // loading placeholder
  return `Level ${level + 1}`;
}

// ─────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────

function setLoading(on) {
  state.loading = on;
  document.getElementById('loading-state').classList.toggle('hidden', !on);
}

function renderList() {
  const list  = document.getElementById('verse-list');
  const empty = document.getElementById('empty-state');
  const sorted = getSorted();

  empty.hidden = state.loading || sorted.length > 0;

  list.innerHTML = sorted.map(v => cardHtml(v)).join('');
  list.classList.toggle('delete-mode', state.deleteMode);

  bindCardEvents(list);
}

function cardHtml(v) {
  const lbl  = levelLabel(v.level);
  const lcls = levelClass(v.level);
  const sub  = escapeHtml(subtitleFor(v));
  const ref  = escapeHtml(v.passage_ref);
  const rAttr = escapeAttr(v.passage_ref);

  return `<li class="verse-item" data-ref="${rAttr}">
  <div class="verse-content">
    <div class="verse-top">
      <span class="verse-ref">${ref}</span>
      <span class="verse-level ${lcls}">${lbl}</span>
    </div>
    <div class="verse-sub">${sub}</div>
  </div>
  <button class="delete-btn" aria-label="Remove ${rAttr}">−</button>
</li>`;
}

/** Update a single card's level badge and subtitle without re-rendering the whole list. */
function patchCard(v) {
  const item = document.querySelector(`.verse-item[data-ref="${CSS.escape(v.passage_ref)}"]`);
  if (!item) return;

  const badge = item.querySelector('.verse-level');
  if (badge) {
    badge.textContent = levelLabel(v.level);
    badge.className   = `verse-level ${levelClass(v.level)}`;
  }

  const sub = item.querySelector('.verse-sub');
  if (sub) sub.textContent = subtitleFor(v);
}

function openSettings() {
  document.getElementById('sort-key-select').value = state.sortKey;
  document.getElementById('sort-dir-select').value = state.sortDir;
  populateMicSelect();
  openModal('modal-settings');
}

// ─────────────────────────────────────────────
// Card event binding
// ─────────────────────────────────────────────

function bindCardEvents(list) {
  list.querySelectorAll('.verse-item').forEach(item => {
    bindLongPress(item, () => enterDeleteMode());

    const delBtn = item.querySelector('.delete-btn');
    delBtn.addEventListener('click', e => {
      e.stopPropagation();
      confirmDelete(item.dataset.ref);
    });
  });

  // Click outside a delete-btn while in delete mode → exit
  list.addEventListener('click', e => {
    if (state.deleteMode && !e.target.closest('.delete-btn')) {
      exitDeleteMode();
    }
  });
}

// ─────────────────────────────────────────────
// Long-press detection
// ─────────────────────────────────────────────

function bindLongPress(el, callback) {
  let timer   = null;
  let startX  = 0;
  let startY  = 0;
  const HOLD  = 550;   // ms
  const DRIFT = 10;    // px

  function start(x, y) {
    startX = x; startY = y;
    el.classList.add('pressing');
    timer = setTimeout(() => {
      el.classList.remove('pressing');
      callback();
    }, HOLD);
  }

  function cancel() {
    clearTimeout(timer);
    el.classList.remove('pressing');
  }

  function move(x, y) {
    if (Math.abs(x - startX) > DRIFT || Math.abs(y - startY) > DRIFT) cancel();
  }

  // Mouse
  el.addEventListener('mousedown',  e => start(e.clientX, e.clientY));
  el.addEventListener('mousemove',  e => move(e.clientX, e.clientY));
  el.addEventListener('mouseup',    cancel);
  el.addEventListener('mouseleave', cancel);

  // Touch
  el.addEventListener('touchstart', e => {
    const t = e.touches[0];
    start(t.clientX, t.clientY);
  }, { passive: true });
  el.addEventListener('touchmove', e => {
    const t = e.touches[0];
    move(t.clientX, t.clientY);
  }, { passive: true });
  el.addEventListener('touchend',    cancel);
  el.addEventListener('touchcancel', cancel);
}

// ─────────────────────────────────────────────
// Delete mode
// ─────────────────────────────────────────────

function enterDeleteMode() {
  if (state.deleteMode) return;
  state.deleteMode = true;
  document.getElementById('verse-list').classList.add('delete-mode');
  if (navigator.vibrate) navigator.vibrate(40);
}

function exitDeleteMode() {
  state.deleteMode = false;
  document.getElementById('verse-list').classList.remove('delete-mode');
}

function confirmDelete(ref) {
  state.pendingDelete = ref;
  document.getElementById('delete-ref').textContent = ref;
  openModal('modal-delete');
}

async function executeDelete() {
  const ref = state.pendingDelete;
  if (!ref) return;

  try {
    await apiDelete(`/verses/${encodeURIComponent(ref)}`);
  } catch (e) {
    closeModal('modal-delete');
    showToast('Could not remove verse. Please try again.');
    return;
  }

  state.verses = state.verses.filter(v => v.passage_ref !== ref);
  state.pendingDelete = null;
  closeModal('modal-delete');
  exitDeleteMode();
  renderList();
  showToast(`Removed ${ref}`);
}

// ─────────────────────────────────────────────
// Add verse flow
// ─────────────────────────────────────────────

let _addAbort = null;

function openAddModal() {
  document.getElementById('ref-input').value     = '';
  document.getElementById('add-error').hidden    = true;
  document.getElementById('add-status').hidden   = true;
  document.getElementById('btn-confirm-add').disabled    = false;
  document.getElementById('btn-confirm-add').textContent = 'Add';
  openModal('modal-add');
  // Delay focus so the modal animation has started
  setTimeout(() => document.getElementById('ref-input').focus(), 120);
}

async function submitAddVerse() {
  const ref = document.getElementById('ref-input').value.trim();
  if (!ref) return;

  const errorEl   = document.getElementById('add-error');
  const statusEl  = document.getElementById('add-status');
  const confirmEl = document.getElementById('btn-confirm-add');

  errorEl.hidden  = true;
  statusEl.hidden = false;
  statusEl.textContent = 'Saving verse…';
  confirmEl.disabled   = true;
  confirmEl.textContent = 'Adding…';

  // Cycle status messages while the server is working (TTS + alignment is slow)
  const statusMessages = ['Saving verse…', 'Generating audio…', 'Running alignment…'];
  let msgIdx = 0;
  const msgTimer = setInterval(() => {
    msgIdx = (msgIdx + 1) % statusMessages.length;
    if (!statusEl.hidden) statusEl.textContent = statusMessages[msgIdx];
  }, 4000);

  _addAbort = new AbortController();

  let res;
  try {
    res = await fetch('/verses', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ passage_ref: ref }),
      signal:  _addAbort.signal,
    });
  } catch (e) {
    clearInterval(msgTimer);
    if (e.name === 'AbortError') return;
    _showAddError('Could not reach the server. Please try again.');
    return;
  } finally {
    _addAbort = null;
  }

  clearInterval(msgTimer);

  if (res.status === 409) {
    _showAddError('This verse is already in your list.');
    return;
  }

  if (!res.ok) {
    let detail = 'Something went wrong. Please try again.';
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    _showAddError(detail);
    return;
  }

  const newVerse = await res.json();

  // Optimistically add with level 0; then fetch real progress
  const entry = {
    passage_ref:       newVerse.passage_ref,
    added_at:          newVerse.added_at,
    level:             0,
    level_name:        'WORDS_3',
    last_attempt_date: null,
  };
  state.verses.push(entry);

  closeModal('modal-add');
  renderList();

  // Fetch progress in background (non-blocking)
  apiGet(`/progress/${encodeURIComponent(ref)}`).then(p => {
    const i = state.verses.findIndex(v => v.passage_ref === ref);
    if (i !== -1) {
      state.verses[i] = { ...state.verses[i], level: p.level, level_name: p.level_name, last_attempt_date: p.last_attempt_date };
      patchCard(state.verses[i]);
    }
  }).catch(() => {});
}

function _showAddError(msg) {
  const errorEl   = document.getElementById('add-error');
  const statusEl  = document.getElementById('add-status');
  const confirmEl = document.getElementById('btn-confirm-add');

  errorEl.textContent  = msg;
  errorEl.hidden       = false;
  statusEl.hidden      = true;
  confirmEl.disabled   = false;
  confirmEl.textContent = 'Add';
}

// ─────────────────────────────────────────────
// Modal management
// ─────────────────────────────────────────────

function openModal(id) {
  const el = document.getElementById(id);
  el.setAttribute('aria-hidden', 'false');
  // Trigger animation on next frame
  requestAnimationFrame(() => el.setAttribute('aria-hidden', 'false'));
}

function closeModal(id) {
  const el = document.getElementById(id);
  el.setAttribute('aria-hidden', 'true');
}

function isModalOpen(id) {
  return document.getElementById(id).getAttribute('aria-hidden') === 'false';
}

// ─────────────────────────────────────────────
// Toast
// ─────────────────────────────────────────────

let _toastTimer = null;

function showToast(msg) {
  let toast = document.getElementById('toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.remove('show'), 2800);
}

// ─────────────────────────────────────────────
// Sort controls
// ─────────────────────────────────────────────

function setSortKey(key) {
  state.sortKey = key;
  renderList();
}

function setSortDir(dir) {
  state.sortDir = dir;
  renderList();
}

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeAttr(str) {
  return String(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// ─────────────────────────────────────────────
// Microphone device enumeration
// ─────────────────────────────────────────────

async function populateMicSelect() {
  const select = document.getElementById('mic-select');
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;

  // enumerateDevices may return empty labels until permission is granted;
  // we populate lazily when the settings modal opens so labels are available
  // after the session (which requests mic permission) has run at least once.
  let devices;
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (_) {
    return;
  }

  const inputs = devices.filter(d => d.kind === 'audioinput');

  // Rebuild options (keep the "Default" option at index 0)
  select.innerHTML = '<option value="">Default</option>';
  inputs.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Microphone (${d.deviceId.slice(0, 8)}…)`;
    select.appendChild(opt);
  });

  // Restore saved selection
  const saved = localStorage.getItem('micDeviceId');
  if (saved) select.value = saved;
}

// ─────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {

  // ── FAB ──────────────────────────────────────
  document.getElementById('fab').addEventListener('click', openAddModal);

  // ── Practice button wired in session.js ──────

  // ── Settings ──────────────────────────────────
  document.getElementById('btn-settings').addEventListener('click', openSettings);
  document.getElementById('btn-close-settings').addEventListener('click', () => closeModal('modal-settings'));
  document.getElementById('sort-key-select').addEventListener('change', e => setSortKey(e.target.value));
  document.getElementById('sort-dir-select').addEventListener('change', e => setSortDir(e.target.value));

  // ── Add modal ─────────────────────────────────
  document.getElementById('btn-cancel-add').addEventListener('click', () => {
    if (_addAbort) _addAbort.abort();
    closeModal('modal-add');
  });

  document.getElementById('btn-confirm-add').addEventListener('click', submitAddVerse);

  document.getElementById('ref-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') submitAddVerse();
  });

  // ── Delete modal ──────────────────────────────
  document.getElementById('btn-cancel-delete').addEventListener('click', () => {
    state.pendingDelete = null;
    closeModal('modal-delete');
  });

  document.getElementById('btn-confirm-delete').addEventListener('click', executeDelete);

  // ── Backdrop clicks ───────────────────────────
  document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
    backdrop.addEventListener('click', () => {
      const modal = backdrop.closest('.modal');
      if (modal.id === 'modal-add') {
        if (_addAbort) _addAbort.abort();
      }
      if (modal.id === 'modal-delete') state.pendingDelete = null;
      closeModal(modal.id);

    });
  });

  // ── Keyboard ──────────────────────────────────
  document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;

    if (state.deleteMode) {
      exitDeleteMode();
      return;
    }

    ['modal-add', 'modal-delete', 'modal-settings', 'modal-session-start'].forEach(id => {
      if (isModalOpen(id)) {
        if (id === 'modal-add' && _addAbort) _addAbort.abort();
        if (id === 'modal-delete') state.pendingDelete = null;
        closeModal(id);
      }
    });
  });

  // ── Microphone selector ───────────────────────
  populateMicSelect();
  document.getElementById('mic-select').addEventListener('change', e => {
    const val = e.target.value;
    if (val) localStorage.setItem('micDeviceId', val);
    else     localStorage.removeItem('micDeviceId');
  });

  // ── Initial load ──────────────────────────────
  loadVerses();
});
