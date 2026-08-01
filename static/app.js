// ─── FeedPulse Client-Side Logic ───

// ---------------------------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------------------------
function getTheme() {
  return localStorage.getItem('fp-theme') || 'dark';
}

function applyTheme(theme) {
  const html = document.documentElement;
  if (theme === 'dark') {
    html.classList.add('dark');
  } else {
    html.classList.remove('dark');
  }
  // Toggle icon visibility
  const iconDark = document.getElementById('theme-icon-dark');
  const iconLight = document.getElementById('theme-icon-light');
  if (iconDark && iconLight) {
    iconDark.classList.toggle('hidden', theme === 'light');
    iconLight.classList.toggle('hidden', theme === 'dark');
  }
}

function toggleTheme() {
  const next = getTheme() === 'dark' ? 'light' : 'dark';
  localStorage.setItem('fp-theme', next);
  applyTheme(next);
}

// Apply on load
applyTheme(getTheme());

// ---------------------------------------------------------------------------
// Search / filter
// ---------------------------------------------------------------------------
let _activeCategory = 'all';

function filterArticles() {
  const query = (document.getElementById('search-input')?.value || '').toLowerCase().trim();
  const cards = document.querySelectorAll('.article-card');
  let visible = 0;

  cards.forEach(card => {
    const title = card.dataset.title || '';
    const desc = card.dataset.desc || '';
    const source = (card.dataset.source || '').toLowerCase();
    const cat = card.dataset.category || 'all';

    const matchesSearch = !query || title.includes(query) || desc.includes(query) || source.includes(query);
    const matchesCat = _activeCategory === 'all' || cat === _activeCategory;

    if (matchesSearch && matchesCat) {
      card.classList.remove('hidden');
      visible++;
    } else {
      card.classList.add('hidden');
    }
  });

  // Toggle no-results message
  const noResults = document.getElementById('no-results');
  if (noResults) {
    noResults.classList.toggle('hidden', visible > 0);
  }
}

// Debounced search input
let _searchTimer;
function onSearchInput() {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(filterArticles, 150);
}

document.addEventListener('DOMContentLoaded', () => {
  const mainSearch = document.getElementById('search-input');
  if (mainSearch) mainSearch.addEventListener('input', onSearchInput);

  // Sync mobile search
  document.querySelectorAll('.search-mobile').forEach(el => {
    el.addEventListener('input', () => {
      if (mainSearch) mainSearch.value = el.value;
      onSearchInput();
    });
  });
});

// ---------------------------------------------------------------------------
// Category switching
// ---------------------------------------------------------------------------
function switchCategory(cat, btn) {
  _activeCategory = cat;

  // Update active tab styling
  document.querySelectorAll('.category-tab').forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');

  // If we have server-rendered data, do client-side filtering
  filterArticles();

  // Also fetch from API for the freshest data
  fetchCategory(cat);
}

function fetchCategory(cat) {
  const url = cat === 'all' ? '/api/feeds' : `/api/feeds?category=${encodeURIComponent(cat)}`;
  fetch(url)
    .then(r => r.json())
    .then(data => {
      renderArticles(data.articles || []);
      updateStats(data);
    })
    .catch(err => console.error('Fetch error:', err));
}

// ---------------------------------------------------------------------------
// Render articles dynamically
// ---------------------------------------------------------------------------
function renderArticles(articles) {
  const container = document.getElementById('articles-container');
  if (!container) return;

  // Hide initial loading
  const initialLoading = document.getElementById('initial-loading');
  if (initialLoading) initialLoading.classList.add('hidden');

  if (articles.length === 0) {
    container.innerHTML = '';
    const noResults = document.getElementById('no-results');
    if (noResults) noResults.classList.remove('hidden');
    return;
  }

  const noResults = document.getElementById('no-results');
  if (noResults) noResults.classList.add('hidden');

  const query = (document.getElementById('search-input')?.value || '').toLowerCase().trim();

  container.innerHTML = articles
    .filter(a => {
      if (!query) return true;
      return (a.title || '').toLowerCase().includes(query)
          || (a.description || '').toLowerCase().includes(query)
          || (a.source || '').toLowerCase().includes(query);
    })
    .map(a => articleHTML(a))
    .join('');
}

function articleHTML(a) {
  const scoreClass = a.score >= 10
    ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
    : a.score >= 5
      ? 'bg-blue-500/15 text-blue-400 border border-blue-500/20'
      : 'bg-gray-500/15 text-gray-400 border border-gray-500/20';

  const desc = a.description
    ? `<p class="mt-2 text-sm text-gray-400 leading-relaxed line-clamp-2">${escapeHtml(a.description)}</p>`
    : '';

  const date = a.date ? `<span>${escapeHtml(a.date)}</span>` : '';

  return `
    <article class="article-card group relative p-4 sm:p-5 rounded-xl bg-navy-800/50 border border-white/5 hover:border-blue-500/30 hover:bg-navy-800/80 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/5 hover:-translate-y-0.5"
             data-source="${escapeAttr(a.source)}"
             data-title="${escapeAttr((a.title||'').toLowerCase())}"
             data-desc="${escapeAttr((a.description||'').toLowerCase())}">
      <div class="flex flex-col sm:flex-row sm:items-start gap-3">
        <div class="flex-1 min-w-0">
          <a href="${escapeAttr(a.link)}" target="_blank" rel="noopener noreferrer"
             class="text-[15px] font-semibold leading-snug text-gray-100 hover:text-blue-400 transition-colors line-clamp-2 block">
            ${escapeHtml(a.title)}
          </a>
          ${desc}
        </div>
        <div class="flex sm:flex-col items-center sm:items-end gap-2 flex-shrink-0">
          <span class="score-pill inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-bold ${scoreClass}">
            <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>
            ${a.score}
          </span>
        </div>
      </div>
      <div class="mt-3 flex items-center gap-3 text-xs text-gray-500">
        <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-white/5 font-medium text-gray-400">${escapeHtml(a.source)}</span>
        ${date}
      </div>
    </article>`;
}

function escapeHtml(s) {
  const el = document.createElement('span');
  el.textContent = s || '';
  return el.innerHTML;
}

function escapeAttr(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
function updateStats(data) {
  const el = (id) => document.getElementById(id);
  if (data.total != null && el('stat-total')) el('stat-total').textContent = data.total;
  if (data.source_count != null && el('stat-sources')) el('stat-sources').textContent = data.source_count;
  if (data.last_refresh && el('stat-refresh')) el('stat-refresh').textContent = data.last_refresh;
}

// ---------------------------------------------------------------------------
// Refresh
// ---------------------------------------------------------------------------
function triggerRefresh() {
  const icon = document.getElementById('refresh-icon');
  const overlay = document.getElementById('loading-overlay');

  if (icon) icon.classList.add('animate-spin-slow');
  if (overlay) overlay.classList.remove('hidden');

  fetch('/api/refresh', { method: 'POST' })
    .then(() => {
      // Poll until done
      pollRefresh();
    })
    .catch(err => {
      console.error('Refresh error:', err);
      if (icon) icon.classList.remove('animate-spin-slow');
      if (overlay) overlay.classList.add('hidden');
    });
}

function pollRefresh() {
  setTimeout(() => {
    fetch('/api/feeds')
      .then(r => r.json())
      .then(data => {
        if (data.refreshing) {
          pollRefresh();
        } else {
          const icon = document.getElementById('refresh-icon');
          const overlay = document.getElementById('loading-overlay');
          if (icon) icon.classList.remove('animate-spin-slow');
          if (overlay) overlay.classList.add('hidden');

          renderArticles(data.articles || []);
          updateStats(data);

          // Update category counts
          if (data.categories) {
            document.querySelectorAll('.category-tab').forEach(tab => {
              const cat = tab.dataset.category;
              const countEl = tab.querySelector('.cat-count');
              if (countEl && cat === 'all') {
                countEl.textContent = data.total || 0;
              } else if (countEl && data.categories[cat] != null) {
                countEl.textContent = data.categories[cat];
              }
            });
          }

          if (data.error) {
            console.warn('Refresh error:', data.error);
          }
        }
      })
      .catch(() => pollRefresh());
  }, 2000);
}

// ---------------------------------------------------------------------------
// Auto-poll for initial data (page loads before first fetch completes)
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  const initialLoading = document.getElementById('initial-loading');
  if (initialLoading && !initialLoading.classList.contains('hidden')) {
    // Data not ready yet — poll
    const poll = () => {
      fetch('/api/feeds')
        .then(r => r.json())
        .then(data => {
          if (data.total > 0) {
            renderArticles(data.articles || []);
            updateStats(data);
            initialLoading.classList.add('hidden');

            // Update sidebar counts
            if (data.categories) {
              document.querySelectorAll('.category-tab').forEach(tab => {
                const cat = tab.dataset.category;
                const countEl = tab.querySelector('.cat-count');
                if (countEl && cat === 'all') {
                  countEl.textContent = data.total || 0;
                } else if (countEl && data.categories[cat] != null) {
                  countEl.textContent = data.categories[cat];
                }
              });
            }
          } else if (data.error) {
            initialLoading.innerHTML = `
              <svg class="w-12 h-12 text-red-400 mb-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/></svg>
              <h2 class="text-lg font-semibold text-red-300 mb-2">Something went wrong</h2>
              <p class="text-sm text-gray-500 max-w-sm">${escapeHtml(data.error)}</p>
              <button onclick="location.reload()" class="mt-4 px-4 py-2 rounded-lg bg-blue-500/20 text-blue-400 text-sm font-medium hover:bg-blue-500/30 transition">Retry</button>
            `;
          } else {
            setTimeout(poll, 3000);
          }
        })
        .catch(() => setTimeout(poll, 3000));
    };
    setTimeout(poll, 3000);
  }
});
