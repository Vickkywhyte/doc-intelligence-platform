# Streamlit UI Specification

**Related docs:** `07_API_SPEC.md`, `06_EVAL_SPEC.md`
**File:** `ui/app.py`

---

## 1. Layout overview

Single-page Streamlit app with three tabs. Page config:

```python
st.set_page_config(
    page_title="Document Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```

**Tabs:**
1. **Chat** — ask questions, see answers with citations
2. **Eval Dashboard** — the differentiator, on display
3. **Runs History** — past experiments

A persistent **sidebar** shows:
- Current config summary (chunking, embedder, generator)
- Cost accrued this session
- Link to `/docs` (FastAPI)
- "Provider: local / hosted" badge with opt-in toggle

---

## 2. Tab 1 — Chat

### 2.1 Layout

Two-column split (2:1):

**Left (chat):**
- Header: "Ask the corpus"
- Chat history (message bubbles; user right-aligned, assistant left-aligned)
- Assistant bubbles include:
  - Answer text with inline citation chips rendered clickable
  - Footer line: `model · latency · $cost`
  - Refusal responses rendered in muted style with an "I don't know" badge
- Input row: text box + "Ask" button (Enter submits)
- "Clear chat" icon button (top-right of the column)

**Right (retrieved chunks):**
- Header: "Sources for last answer"
- For each retrieved chunk (rank ordered):
  - Rank + score chip
  - Doc filename + page
  - Chunk text (collapsed to 3 lines by default; expandable)
  - "Copy chunk_id" icon button
- If the last answer is a refusal: header shows "No relevant sources found"
  and lists the top-1 chunk in muted style with its low score visible

### 2.2 Behavior

- On submit:
  - Show a spinner (`Thinking...`)
  - Call `POST /query`
  - Append assistant bubble + update right column
  - Increment session cost counter
- On error: red banner with the error envelope's `message`
- Clicking a citation chip:
  - Scrolls right column to the corresponding chunk
  - Brief highlight animation (Streamlit `st.markdown` with CSS class)

### 2.3 Empty state

- Greeting: "Ask a question about the documents in your corpus."
- Three example question chips (from `data/eval/golden.jsonl`, `type=factual`)
  that populate the input on click.

---

## 3. Tab 2 — Eval Dashboard

The demo. Everything here reads from `results/runs/*.json` — no hardcoded data.

### 3.1 Top row — run selector + summary cards

**Run selector:** dropdown listing runs by
`{run_id} · {n_configs} configs · {timestamp}`. Default = most recent.

**Summary cards** (6 cards, evenly spaced):
| Card | Value |
|---|---|
| Best answer accuracy | `0.87 (recursive · bge-small)` |
| Best P@5 | `0.83 (semantic · openai-small)` |
| Lowest hallucination | `0.05 (recursive · openai-small)` |
| Refusal correctness | `0.93` |
| Best cost/1k | `$0.00 (minilm)` |
| Total run cost | `$0.31` |

Each card shows the **best config** for that metric in parentheses.

### 3.2 Heatmap

- **Rows:** chunking strategy (fixed, recursive, semantic)
- **Cols:** embedding model (openai-small, bge-small, minilm)
- **Cell:** selected metric (default `answer_accuracy`)
- **Metric selector:** dropdown above the chart; every metric from
  `06_EVAL_SPEC.md §6` is selectable, plus a "direction" annotation
  (↑ better / ↓ better)
- **Color scale:** Plotly `RdYlGn` for ↑ metrics (green = best);
  inverted for ↓ metrics (green = lowest)
- **Hover:** shows all metrics for that cell + CI half-width

### 3.3 Hallucination bar chart

- One bar per config (9 bars)
- Y-axis: `hallucination_rate`
- Bars sorted ascending (best first)
- Error bars = 95% CI half-width
- Hover: config name, rate, CI, and count of hallucinated claims across the
  run

### 3.4 Cost-vs-quality scatter

- **X-axis:** `cost_per_1k_q_usd` (log scale if spread > 10×)
- **Y-axis:** `answer_accuracy`
- **Point per config:**
  - Color by chunking strategy
  - Marker shape by embedding family (circle = local, diamond = hosted)
  - Size by `latency_ms_p50` (bigger = slower)
- **Labels:** every point labeled with a short config code
  (e.g., `rec · bge`)
- **Quadrant annotation:** "best value" upper-left quadrant highlighted

### 3.5 Full comparison table

Streamlit `st.dataframe`, sortable, all 9 rows × all metrics:

| Chunking | Embedding | P@5 | R@5 | MRR | NDCG@5 | Acc | SemSim | Halluc | Faith | Refusal | Latency | $/1k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

- Column headers include direction arrows (↑/↓).
- Best cell in each column highlighted (bold + light background).
- CSV download button above the table.

### 3.6 Drill-down

Click a row in the comparison table → expand a per-question table:

| Q ID | Type | Question | Answer | Refused | P@5 | Acc | Halluc | Faith | Judge rationale |
|---|---|---|---|---|---|---|---|---|---|

- Click a question → expand to show:
  - Retrieved chunks (with scores)
  - Hallucinated claims (list, red)
  - Faithful statements (list, green)
  - Full judge verdict JSON
  - Cost + latency for that question

### 3.7 Empty state

If no runs exist:
- Message: "No evaluation runs yet. Run `make eval-matrix` and refresh."
- Button: "Trigger eval run" → calls `POST /eval/run` (async), shows
  progress via a polling banner.

---

## 4. Tab 3 — Runs History

### 4.1 Table

| Run ID | Config hash | Chunking × Embedding | Started | Duration | Best metric | Cost | Status |
|---|---|---|---|---|---|---|---|

- Sortable, filterable by status.
- Click a row → opens a detail panel (same as Tab 2's drill-down, but
  read-only view of a historic run).

### 4.2 Actions per run

- **"Open in Dashboard"** → switches to Tab 2 with that run selected
- **"Download report"** → GETs `/eval/runs/{id}/report` and offers as a file
- **"Download JSON"** → offers the raw `EvalResult` as a file
- **"Re-run"** → copies the config to `configs/replay_{run_id}.yaml` and
  triggers `POST /eval/run`

### 4.3 Empty state

- "No runs yet. The first matrix run will appear here."

---

## 5. Sidebar

### 5.1 Current config

Read-only display of the canonical config (`configs/experiment_matrix.yaml`):
```
Chunking:      recursive (512, 64)
Embeddings:    bge_small_en_v15
Retrieval:     k=5, mmr=false
Generator:     openai/gpt-4o-mini (temp=0)
Judge:         openai/gpt-4o-mini (v1)
```

### 5.2 Session cost

Running total: `Session cost: $0.0142`. Resets on page reload.

### 5.3 Provider badge

```
Provider: LOCAL (no documents leave this machine)
```
or, if hosted embeddings/LLM are configured:
```
Provider: HOSTED via OpenRouter (documents sent to third parties)
[ ] I understand and consent
```
The checkbox must be checked before hosted mode is active. See
`11_SECURITY_PRIVACY.md`.

### 5.4 Links

- `FastAPI docs ↗` → opens `http://localhost:8000/docs` in a new tab
- `GitHub ↗` (if `REPO_URL` is set in env)
- `About this project` → expands a short pitch (the 30-second version)

---

## 6. Design constraints

- **No custom CSS** beyond minimal utility classes for chat bubbles.
  Streamlit-native look is fine; shippability > aesthetics.
- **All charts via Plotly** (`plotly>=5.20`). Interactive, embeddable,
  works offline.
- **No secrets in the UI.** API keys live in `.env` only.
- **No file writes from the UI** except when the user clicks "Download"
  (browser-handled).
- **Works offline** for local-embedding + local-generator configs.

---

## 7. Accessibility

- All charts have a data table equivalent below them (Streamlit expander).
- Color scales on heatmaps are also annotated with numeric values in cells.
- No reliance on color alone to convey meaning in the comparison table
  (best cell uses bold + background, not just color).

---

## 8. Performance targets

| Interaction | Target |
|---|---|
| Tab switch | < 200 ms |
| Chat submit → answer rendered | < 5 s p95 (hosted) |
| Dashboard load (9 configs) | < 2 s |
| Drill-down expand | < 500 ms |
| Full page reload | < 3 s |

If a target is missed, cache the underlying JSON in `@st.cache_data`.

---

## 9. What the UI does NOT do (v1)

- Document management (upload UI exists, deletion doesn't)
- Multi-user sessions
- Auth
- Live updating of in-progress runs (poll manually or reload)
- Model selection at query time (config-only)
- Fine-grained per-chunk feedback / thumbs up/down

These are documented as v2 candidates.

---

## 10. File breakdown

```
ui/
├── app.py                  # st.set_page_config + tabs + sidebar
├── components/
│   ├── chat_tab.py         # Tab 1
│   ├── eval_tab.py         # Tab 2
│   └── runs_tab.py         # Tab 3
├── charts.py               # plotly helpers (heatmap, bar, scatter)
└── state.py                # st.session_state keys + helpers
```

Each component is a single `render_*()` function. `app.py` calls them.
Charts are pure functions `(data) -> plotly.graph_objects.Figure`.