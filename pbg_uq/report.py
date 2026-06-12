"""Self-contained HTML report for pbg-uq results.

Provides a single public function::

    generate_html_report(result, parameter_names, observable_names=None) -> str

All styling, SVG charts, and the interactive PCE explorer are inlined.
No external file or CDN dependency; the returned string is a complete HTML document.

Sections included
-----------------
- Header with key metrics (n_params, n_obs, n_train_samples, PCE order)
- Top-driver callout
- Sobol indices: grouped bar chart (S_Ti + S_i) + ranked table
- Per-output relative error table (train; test if available, else CV)
- Interactive PCE response explorer (client-side JS, Legendre evaluation)

Sections pruned vs. uqEcoli source
------------------------------------
- Cell-cycle / growth-stratified strategies (by-generation, by-lineage, per-θ)
- Morris prescreening
- Multi-condition GSA
- Manifest / provenance block (no manifest in pbg-uq)
- Dashboard-specific external assets
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone

import numpy as np

from pbg_uq.results import UQPCResult


# ── Color palette ──────────────────────────────────────────────────────────────

_COLORS = [
    "#6366f1",  # indigo
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#3b82f6",  # blue
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#f97316",  # orange
    "#84cc16",  # lime
]

_VIRIDIS = [
    "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
]


def _viridis(t: float) -> str:
    """Map *t* ∈ [0,1] to a Viridis hex colour."""
    t = max(0.0, min(1.0, t))
    idx = t * (len(_VIRIDIS) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(_VIRIDIS) - 1)
    frac = idx - lo
    c1, c2 = _VIRIDIS[lo], _VIRIDIS[hi]
    r = int(int(c1[1:3], 16) * (1 - frac) + int(c2[1:3], 16) * frac)
    g = int(int(c1[3:5], 16) * (1 - frac) + int(c2[3:5], 16) * frac)
    b = int(int(c1[5:7], 16) * (1 - frac) + int(c2[5:7], 16) * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


# ── SVG builders ───────────────────────────────────────────────────────────────


def _svg_bar_chart(
    total: dict[str, float],
    first: dict[str, float] | None = None,
    *,
    width: int = 700,
    height: int = 320,
    color_total: str = _COLORS[0],
    color_first: str = _COLORS[1],
) -> str:
    """Grouped bar chart of Sobol indices as inline SVG."""
    params = list(total.keys())
    n = len(params)
    if n == 0:
        return ""
    has_first = first is not None and bool(first)

    vals_t = [total.get(p, 0.0) for p in params]
    vals_f = [first.get(p, 0.0) for p in params] if has_first else []
    max_val = max(max(vals_t), max(vals_f) if vals_f else 0.0, 0.01)
    max_val = math.ceil(max_val * 10) / 10

    ml, mr, mt, mb = 55, 20, 30, 90
    pw = width - ml - mr
    ph = height - mt - mb
    gw = pw / n
    bw = gw * 0.35 if has_first else gw * 0.6
    gap = gw * 0.05

    L = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">'
    ]

    # Grid lines + Y-axis labels
    for i in range(6):
        yv = max_val * i / 5
        yp = mt + ph - (yv / max_val) * ph
        L.append(
            f'<line x1="{ml}" y1="{yp:.1f}" x2="{width - mr}" y2="{yp:.1f}" '
            f'stroke="#333355" stroke-width="1"/>'
        )
        L.append(
            f'<text x="{ml - 8}" y="{yp + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#94a3b8">{yv:.2f}</text>'
        )

    # Bars
    for i, p in enumerate(params):
        xb = ml + i * gw + gap
        ht = (vals_t[i] / max_val) * ph
        yt = mt + ph - ht
        xt = xb + (0 if has_first else (gw - bw) / 2 - gap)
        L.append(
            f'<rect x="{xt:.1f}" y="{yt:.1f}" width="{bw:.1f}" height="{ht:.1f}" '
            f'fill="{color_total}" rx="3">'
            f'<title>{html.escape(p)}\nS_Ti = {vals_t[i]:.4f}</title></rect>'
        )
        if has_first:
            hf = (vals_f[i] / max_val) * ph
            yf = mt + ph - hf
            xf = xb + bw + gap
            L.append(
                f'<rect x="{xf:.1f}" y="{yf:.1f}" width="{bw:.1f}" height="{hf:.1f}" '
                f'fill="{color_first}" rx="3">'
                f'<title>{html.escape(p)}\nS_i = {vals_f[i]:.4f}</title></rect>'
            )
        # X-axis label
        xc = ml + i * gw + gw / 2
        L.append(
            f'<text x="{xc:.1f}" y="{mt + ph + 16}" text-anchor="end" font-size="11" '
            f'fill="#e2e8f0" transform="rotate(-35 {xc:.1f} {mt + ph + 16})">'
            f'{html.escape(p)}</text>'
        )

    # Legend
    lx, ly = ml + 10, mt + 8
    L.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{color_total}" rx="2"/>')
    L.append(f'<text x="{lx + 16}" y="{ly + 10}" font-size="11" fill="#e2e8f0">Total (S_Ti)</text>')
    if has_first:
        L.append(
            f'<rect x="{lx + 100}" y="{ly}" width="12" height="12" fill="{color_first}" rx="2"/>'
        )
        L.append(
            f'<text x="{lx + 116}" y="{ly + 10}" font-size="11" fill="#e2e8f0">First (S_i)</text>'
        )

    L.append("</svg>")
    return "\n".join(L)


# ── Table builders ─────────────────────────────────────────────────────────────


def _ranking_table(
    total: dict[str, float],
    first: dict[str, float] | None = None,
) -> str:
    """Parameter ranking table with inline bar + interaction column."""
    has_first = first is not None and bool(first)
    rows = []
    for p, t in total.items():
        f = first.get(p, 0.0) if has_first else 0.0
        inter = t - f if has_first else float("nan")
        rows.append((p, t, f, inter))
    rows.sort(key=lambda r: -r[1])

    L = ["<table>", "<thead><tr>", "<th>Rank</th><th>Parameter</th><th>S_Ti (Total)</th>"]
    if has_first:
        L.append("<th>S_i (First)</th><th>Interaction</th>")
    L.extend(["</tr></thead>", "<tbody>"])

    for rank, (p, t, f, inter) in enumerate(rows, 1):
        pct = min(t * 100, 100)
        bar = f'<div class="bar" style="width:{pct:.0f}%"></div>'
        L.append(
            f'<tr><td class="rank">{rank}</td>'
            f"<td><code>{html.escape(p)}</code></td>"
            f'<td><div class="bar-cell">{bar}<span>{_fmt(t)}</span></div></td>'
        )
        if has_first:
            inter_s = _fmt(inter) if math.isfinite(inter) else "-"
            L.append(f"<td>{_fmt(f)}</td><td>{inter_s}</td>")
        L.append("</tr>")

    L.extend(["</tbody>", "</table>"])
    return "\n".join(L)


def _relative_error_table(
    relerr_train: "np.ndarray",
    observable_names: list[str],
    relerr_test: "np.ndarray | None" = None,
    relerr_cv: "np.ndarray | None" = None,
    cv_n_folds: int = 0,
) -> str:
    """Per-output relative error table (train; test or CV if available)."""
    has_test = relerr_test is not None
    has_cv = relerr_cv is not None and not has_test

    header_cols = ["<th>Output</th>", "<th>Rel. Err. (train)</th>"]
    if has_test:
        header_cols.append("<th>Rel. Err. (test)</th>")
    if has_cv:
        folds_label = str(cv_n_folds) if cv_n_folds else ""
        header_cols.append(f"<th>Rel. Err. (CV{folds_label})</th>")

    L = ["<table>", "<thead><tr>", "".join(header_cols), "</tr></thead>", "<tbody>"]

    n = len(observable_names)
    for i in range(n):
        tr_val = float(relerr_train[i]) if i < len(relerr_train) else float("nan")
        row = (
            f"<tr><td><code>{html.escape(observable_names[i])}</code></td>"
            f'<td style="text-align:right;font-variant-numeric:tabular-nums">{tr_val:.4f}</td>'
        )
        if has_test:
            te_val = float(relerr_test[i]) if i < len(relerr_test) else float("nan")
            row += (
                f'<td style="text-align:right;font-variant-numeric:tabular-nums">'
                f"{te_val:.4f}</td>"
            )
        if has_cv:
            cv_val = float(relerr_cv[i]) if i < len(relerr_cv) else float("nan")
            row += (
                f'<td style="text-align:right;font-variant-numeric:tabular-nums">'
                f"{cv_val:.4f}</td>"
            )
        row += "</tr>"
        L.append(row)

    L.extend(["</tbody>", "</table>"])
    return "\n".join(L)


# ── Interactive PCE explorer ───────────────────────────────────────────────────


def _build_surrogate_json(surrogate) -> dict | None:
    """Convert a PCESurrogate into the JSON shape the JS explorer expects.

    Shape expected by JS:
        pop_coeffs  – list[list[float]], shape (n_outputs, n_terms)
        multi_indices – list[list[int]], shape (n_terms, n_params)
        bounds      – list[[lo, hi]], shape (n_params, 2)

    Returns None when the surrogate lacks required arrays.
    """
    if surrogate is None:
        return None
    coeffs = surrogate.coefficients   # (n_terms,) or (n_terms, n_outputs)
    mi = surrogate.multi_indices       # (n_terms, n_params)

    if coeffs is None or mi is None:
        return None

    # Normalise to (n_outputs, n_terms)
    if coeffs.ndim == 1:
        pop_coeffs = [coeffs.tolist()]
    else:
        # (n_terms, n_outputs) → transpose to (n_outputs, n_terms)
        pop_coeffs = coeffs.T.tolist()

    if surrogate.input_bounds is not None:
        bounds_list = surrogate.input_bounds.tolist()
    else:
        n_p = mi.shape[1] if mi.ndim > 1 else surrogate.input_dim
        bounds_list = [[-1.0, 1.0]] * n_p

    return {
        "pop_coeffs": pop_coeffs,
        "multi_indices": mi.tolist(),
        "bounds": bounds_list,
    }


def _interactive_css() -> str:
    return """
/* Interactive PCE explorer */
.explorer-grid {
  display: grid;
  grid-template-columns: 280px 1fr;
  gap: 24px;
  margin-top: 16px;
}
@media (max-width: 700px) {
  .explorer-grid { grid-template-columns: 1fr; }
}
.slider-panel { display: flex; flex-direction: column; gap: 14px; }
.slider-group label {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 4px;
  color: var(--text-dim);
}
.slider-group label code { color: var(--text); }
.slider-group input[type=range] {
  width: 100%;
  accent-color: var(--accent);
  height: 6px;
  cursor: pointer;
}
.slider-group .bounds {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: var(--text-dim);
  margin-top: 2px;
}
.pred-panel { display: flex; flex-direction: column; gap: 16px; }
.pred-readout {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
}
.pred-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  text-align: center;
}
.pred-card .pred-label {
  font-size: 10px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.pred-card .pred-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--accent-light);
  font-variant-numeric: tabular-nums;
  margin-top: 2px;
}
.explorer-controls {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.explorer-controls button {
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 6px 14px;
  font-size: 12px;
  cursor: pointer;
  font-weight: 500;
}
.explorer-controls button:hover { opacity: 0.85; }
.obs-toggles { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
.obs-toggle {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 4px 12px;
  font-size: 11px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  color: var(--text-dim);
  cursor: pointer;
  user-select: none;
  transition: all 0.15s ease;
}
.obs-toggle:hover { border-color: var(--accent); }
.obs-toggle.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.obs-toggle .color-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 5px;
  vertical-align: middle;
}
"""


def _interactive_section_html() -> str:
    return """
<div class="explorer-controls">
  <button id="pce-reset">Reset to Midpoint</button>
  <button id="pce-select-all">Select All</button>
  <button id="pce-select-none" style="background:var(--surface2);color:var(--text-dim);border:1px solid var(--border);">Clear</button>
</div>
<div class="obs-toggles" id="pce-obs-toggles"></div>
<div class="explorer-grid">
  <div class="slider-panel" id="pce-sliders"></div>
  <div class="pred-panel">
    <div class="pred-readout" id="pce-readout"></div>
  </div>
</div>"""


def _interactive_js(surr_json: str, param_names_json: str, obs_names_json: str) -> str:
    """Client-side JS for real-time Legendre PCE evaluation (no server required)."""
    return f"""
<script>
(function() {{
  const S = {surr_json};
  const PARAMS = {param_names_json};
  const OBS = {obs_names_json};
  const nP = PARAMS.length;
  const nObs = OBS.length;
  const bounds = S.bounds;
  const mi = S.multi_indices;
  const popC = S.pop_coeffs;
  const nBasis = mi.length;

  // Evaluate a single-output PCE via Legendre recurrence
  function legendreEval(xNorm, coeffs) {{
    let maxOrd = 0;
    for (let t = 0; t < nBasis; t++)
      for (let p = 0; p < nP; p++)
        if (mi[t][p] > maxOrd) maxOrd = mi[t][p];
    const P = [];
    for (let p = 0; p < nP; p++) {{
      P[p] = new Float64Array(maxOrd + 1);
      P[p][0] = 1.0;
      if (maxOrd >= 1) P[p][1] = xNorm[p];
      for (let n = 2; n <= maxOrd; n++)
        P[p][n] = ((2*n - 1) * xNorm[p] * P[p][n-1] - (n-1) * P[p][n-2]) / n;
    }}
    let result = 0;
    for (let t = 0; t < coeffs.length; t++) {{
      let term = coeffs[t];
      for (let p = 0; p < nP; p++) term *= P[p][mi[t][p]];
      result += term;
    }}
    return result;
  }}

  function getXNorm() {{
    const xPhys = [];
    for (let i = 0; i < nP; i++) {{
      const sl = document.getElementById('pce-slider-' + i);
      xPhys.push(parseFloat(sl.value));
    }}
    const xNorm = [];
    for (let i = 0; i < nP; i++) {{
      const lo = bounds[i][0], hi = bounds[i][1];
      xNorm.push(2 * (xPhys[i] - lo) / (hi - lo + 1e-15) - 1);
    }}
    return {{ xPhys, xNorm }};
  }}

  const selected = new Set();

  function update() {{
    const {{ xPhys, xNorm }} = getXNorm();
    for (let i = 0; i < nP; i++) {{
      document.getElementById('pce-val-' + i).textContent = xPhys[i].toFixed(4);
    }}
    const popPreds = [];
    for (let o = 0; o < nObs; o++) popPreds.push(legendreEval(xNorm, popC[o]));
    for (let o = 0; o < nObs; o++) {{
      const el = document.getElementById('pce-pred-' + o);
      if (el) {{
        el.textContent = popPreds[o].toFixed(4);
        el.parentElement.style.borderColor = selected.has(o) ? 'var(--accent)' : 'var(--border)';
      }}
    }}
  }}

  const CHART_COLORS = ['#6366f1','#f59e0b','#10b981','#ef4444','#3b82f6',
                         '#8b5cf6','#ec4899','#14b8a6','#f97316','#84cc16'];

  function syncToggleStyles() {{
    document.querySelectorAll('.obs-toggle').forEach(function(pill) {{
      const idx = parseInt(pill.dataset.idx);
      pill.classList.toggle('active', selected.has(idx));
    }});
  }}

  function init() {{
    const panel = document.getElementById('pce-sliders');
    if (!panel) return;

    // Build parameter sliders
    for (let i = 0; i < nP; i++) {{
      const lo = bounds[i][0], hi = bounds[i][1];
      const mid = (lo + hi) / 2;
      const div = document.createElement('div');
      div.className = 'slider-group';
      div.innerHTML =
        '<label><code>' + PARAMS[i] + '</code>'
        + '<span id="pce-val-' + i + '">' + mid.toFixed(4) + '</span></label>'
        + '<input type="range" id="pce-slider-' + i + '" min="' + lo + '" max="' + hi
        + '" step="' + ((hi - lo) / 200) + '" value="' + mid + '">'
        + '<div class="bounds"><span>' + lo.toFixed(3) + '</span><span>' + hi.toFixed(3) + '</span></div>';
      panel.appendChild(div);
      document.getElementById('pce-slider-' + i).addEventListener('input', update);
    }}

    // Build observable toggle pills
    const toggles = document.getElementById('pce-obs-toggles');
    for (let o = 0; o < nObs; o++) {{
      const pill = document.createElement('span');
      pill.className = 'obs-toggle active';
      pill.dataset.idx = o;
      const col = CHART_COLORS[o % CHART_COLORS.length];
      pill.innerHTML = '<span class="color-dot" style="background:' + col + '"></span>' + OBS[o];
      selected.add(o);
      pill.addEventListener('click', function() {{
        const idx = parseInt(this.dataset.idx);
        if (selected.has(idx)) selected.delete(idx); else selected.add(idx);
        syncToggleStyles();
        update();
      }});
      toggles.appendChild(pill);
    }}

    // Build prediction readout cards
    const readout = document.getElementById('pce-readout');
    for (let o = 0; o < nObs; o++) {{
      const card = document.createElement('div');
      card.className = 'pred-card';
      card.innerHTML = '<div class="pred-label">' + OBS[o]
        + '</div><div class="pred-value" id="pce-pred-' + o + '">-</div>';
      card.style.cursor = 'pointer';
      card.dataset.idx = o;
      card.addEventListener('click', function() {{
        const idx = parseInt(this.dataset.idx);
        if (selected.has(idx)) selected.delete(idx); else selected.add(idx);
        syncToggleStyles();
        update();
      }});
      readout.appendChild(card);
    }}

    document.getElementById('pce-reset').addEventListener('click', function() {{
      for (let i = 0; i < nP; i++) {{
        const lo = bounds[i][0], hi = bounds[i][1];
        document.getElementById('pce-slider-' + i).value = (lo + hi) / 2;
      }}
      update();
    }});
    document.getElementById('pce-select-all').addEventListener('click', function() {{
      for (let o = 0; o < nObs; o++) selected.add(o);
      syncToggleStyles(); update();
    }});
    document.getElementById('pce-select-none').addEventListener('click', function() {{
      selected.clear(); syncToggleStyles(); update();
    }});

    update();
  }}

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', init);
  }} else {{
    init();
  }}
}})();
</script>"""


# ── Public entry point ─────────────────────────────────────────────────────────


def generate_html_report(
    result: UQPCResult,
    parameter_names: list[str],
    observable_names: list[str] | None = None,
) -> str:
    """Generate a self-contained HTML report string from a UQPCResult.

    Parameters
    ----------
    result:
        Fitted UQPCResult from ``fit_pce_and_sobol`` / ``ForwardUQ.quantify()``.
    parameter_names:
        Names of the input parameters (same order as ``result.sobol.first_order``).
    observable_names:
        Names of the output observables. Falls back to ``y0, y1, ...`` if ``None``.

    Returns
    -------
    str
        Complete self-contained HTML document (no external deps).
    """
    n_params = len(parameter_names)
    n_obs = len(result.relerr_train)

    if observable_names is None:
        observable_names = [f"y{i}" for i in range(n_obs)]

    # ── Sobol indices ──────────────────────────────────────────────────────────
    so = result.sobol
    total_arr = np.asarray(so.total_order, dtype=float)
    first_arr = np.asarray(so.first_order, dtype=float)
    # Multi-output: average across outputs axis
    if total_arr.ndim > 1:
        total_arr = np.mean(total_arr, axis=0)
    if first_arr.ndim > 1:
        first_arr = np.mean(first_arr, axis=0)

    total_dict = {p: float(total_arr[i]) for i, p in enumerate(parameter_names)}
    first_dict = {p: float(first_arr[i]) for i, p in enumerate(parameter_names)}

    top_param = max(total_dict, key=total_dict.get)
    top_val = total_dict[top_param]

    sobol_bar = _svg_bar_chart(total_dict, first_dict)
    sobol_rank = _ranking_table(total_dict, first_dict)

    # ── Relative errors ────────────────────────────────────────────────────────
    relerr_test = result.relerr_test
    relerr_cv = result.relerr_cv if relerr_test is None else None
    cv_n_folds = result.cv_n_folds if relerr_cv is not None else 0
    err_table = _relative_error_table(
        result.relerr_train,
        observable_names,
        relerr_test=relerr_test,
        relerr_cv=relerr_cv,
        cv_n_folds=cv_n_folds,
    )

    # Error section description
    if relerr_test is not None:
        err_desc = "Test error is shown (independent held-out set)."
    elif relerr_cv is not None:
        err_desc = f"Cross-validation ({cv_n_folds}-fold) error is shown (no independent test set)."
    else:
        err_desc = "Only training error is available."

    # ── Interactive PCE explorer ───────────────────────────────────────────────
    surr_data = _build_surrogate_json(result.surrogate)
    has_explorer = surr_data is not None
    surr_json_str = json.dumps(surr_data) if surr_data else "{}"
    params_json = json.dumps(parameter_names)
    obs_json = json.dumps(observable_names)

    # ── Summary stats ──────────────────────────────────────────────────────────
    n_train = result.X_train.shape[0] if result.X_train is not None else "-"
    pce_order = result.surrogate.polynomial_order if result.surrogate is not None else "-"
    now = datetime.now(tz=timezone.utc).strftime("%B %d, %Y")

    # ── Pre-compute template fragments ─────────────────────────────────────────
    css_explorer = _interactive_css() if has_explorer else ""
    explorer_body = (
        f"""
<div class="strategy-divider">Interactive PCE Explorer</div>

<div class="section">
  <h2>Surrogate Predictor <span class="badge">Interactive</span></h2>
  <p class="section-desc">
    Drag parameter sliders to evaluate the fitted PCE surrogate in real time.
    Predicted observable values update instantly &mdash; no simulation required.
    Use this to explore &ldquo;what if&rdquo; scenarios: tune parameters until the
    predicted outputs match a desired state.
  </p>
  {_interactive_section_html()}
</div>
"""
        if has_explorer
        else ""
    )
    js_block = (
        _interactive_js(surr_json_str, params_json, obs_json) if has_explorer else ""
    )

    obs_pills = "".join(
        f'<span class="obs-pill">{html.escape(o)}</span>' for o in observable_names
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UQ Report</title>
<style>
:root {{
  --bg: #0f0f1a;
  --surface: #1a1a2e;
  --surface2: #222240;
  --border: #2d2d50;
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --accent: #6366f1;
  --accent-light: #818cf8;
  --success: #10b981;
  --warning: #f59e0b;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}}
.container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}

/* Header */
.header {{
  text-align: center;
  padding: 48px 0 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 40px;
}}
.header h1 {{
  font-size: 28px; font-weight: 700;
  letter-spacing: -0.5px; margin-bottom: 8px;
}}
.header .subtitle {{ color: var(--text-dim); font-size: 14px; }}

/* Metrics */
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 14px; margin-bottom: 36px;
}}
.metric {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 18px; text-align: center;
}}
.metric .value {{ font-size: 26px; font-weight: 700; color: var(--accent-light); }}
.metric .label {{
  font-size: 11px; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px;
}}

/* Sections */
.section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 28px; margin-bottom: 24px;
}}
.section h2 {{
  font-size: 18px; font-weight: 600;
  margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
}}
.section h2 .badge {{
  font-size: 11px; background: var(--accent); color: #fff;
  padding: 2px 8px; border-radius: 10px; font-weight: 500;
}}
.section-desc {{
  color: var(--text-dim); font-size: 13px;
  margin-bottom: 16px; line-height: 1.5;
}}

/* Tables */
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{
  text-align: left; padding: 10px 12px;
  border-bottom: 2px solid var(--border);
  color: var(--text-dim); font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
}}
td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); }}
tr:last-child td {{ border-bottom: none; }}
tr:hover {{ background: var(--surface2); }}
code {{
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 12px; background: var(--surface2);
  padding: 2px 6px; border-radius: 4px;
}}
.rank {{ color: var(--text-dim); font-weight: 600; width: 40px; }}
.bar-cell {{ display: flex; align-items: center; gap: 8px; min-width: 140px; }}
.bar {{ height: 8px; background: var(--accent); border-radius: 4px; min-width: 2px; }}
.bar-cell span {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}

/* Charts */
.chart-container {{
  display: flex; justify-content: center;
  padding: 8px 0; overflow-x: auto;
}}
svg text {{ font-family: 'Inter', -apple-system, sans-serif; }}

/* Observable pills */
.obs-list {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
.obs-pill {{
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 20px; padding: 4px 14px;
  font-size: 12px; font-family: monospace; color: var(--text-dim);
}}

/* Callouts */
.callout {{
  border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;
  display: flex; align-items: center; gap: 16px;
}}
.callout .icon {{ font-size: 28px; line-height: 1; }}
.callout .detail {{ flex: 1; }}
.callout .detail .title {{ font-weight: 600; font-size: 15px; margin-bottom: 2px; }}
.callout .detail .desc {{ font-size: 13px; color: var(--text-dim); }}
.callout-info {{
  background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(99,102,241,0.05));
  border: 1px solid rgba(99,102,241,0.3);
}}

/* Dividers */
.strategy-divider {{
  text-align: center; color: var(--text-dim);
  font-size: 12px; text-transform: uppercase;
  letter-spacing: 1px; margin: 36px 0 24px;
  display: flex; align-items: center; gap: 16px;
}}
.strategy-divider::before, .strategy-divider::after {{
  content: ""; flex: 1; height: 1px; background: var(--border);
}}

/* Footer */
.footer {{
  text-align: center; padding: 32px 0 16px;
  color: var(--text-dim); font-size: 12px;
  border-top: 1px solid var(--border); margin-top: 24px;
}}

@media (max-width: 600px) {{
  .container {{ padding: 20px 12px; }}
  .metrics {{ grid-template-columns: repeat(2, 1fr); }}
  .header h1 {{ font-size: 22px; }}
  .section {{ padding: 16px; }}
}}
{css_explorer}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>Uncertainty Quantification Report</h1>
  <div class="subtitle">Generated {now}</div>
</div>

<!-- Key metrics -->
<div class="metrics">
  <div class="metric"><div class="value">{n_params}</div><div class="label">Parameters</div></div>
  <div class="metric"><div class="value">{n_obs}</div><div class="label">Observables</div></div>
  <div class="metric"><div class="value">{n_train}</div><div class="label">Train Samples</div></div>
  <div class="metric"><div class="value">{pce_order}</div><div class="label">PCE Order</div></div>
</div>

<!-- Top driver callout -->
<div class="callout callout-info">
  <div class="icon">&#x1F3AF;</div>
  <div class="detail">
    <div class="title">Top Driver: <code>{html.escape(top_param)}</code></div>
    <div class="desc">Explains {top_val:.1%} of total output variance
      (S<sub>Ti</sub>&nbsp;=&nbsp;{_fmt(top_val)}).</div>
  </div>
</div>

<!-- Observables list -->
<div class="section">
  <h2>Tracked Observables</h2>
  <p class="section-desc">Output variables used to compute sensitivity indices.</p>
  <div class="obs-list">{obs_pills}</div>
</div>

<div class="strategy-divider">Sobol Sensitivity Analysis</div>

<!-- Sobol bar chart + ranked table -->
<div class="section">
  <h2>Sobol Indices</h2>
  <p class="section-desc">
    Total-order (S<sub>Ti</sub>) and first-order (S<sub>i</sub>) Sobol indices
    computed from the fitted PCE surrogate via analytic variance decomposition.
    S<sub>Ti</sub> includes interaction effects; S<sub>i</sub> captures only the
    direct contribution. Parameters are ranked by S<sub>Ti</sub>.
  </p>
  <div class="chart-container">{sobol_bar}</div>
  <div style="margin-top:20px;">{sobol_rank}</div>
</div>

<div class="strategy-divider">Surrogate Accuracy</div>

<!-- Relative error table -->
<div class="section">
  <h2>Relative Error by Output</h2>
  <p class="section-desc">
    Per-output relative L2 error of the PCE surrogate predictions. {err_desc}
  </p>
  <div style="overflow-x:auto;">{err_table}</div>
</div>

{explorer_body}

<div class="footer">
  pbg-uq &mdash; Uncertainty Quantification Report &mdash; {now}
</div>

</div>
{js_block}
</body>
</html>"""
