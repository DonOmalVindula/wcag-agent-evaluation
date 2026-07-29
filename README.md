# WCAG–Agent Evaluation: A Dual-Evaluation Framework

**Evaluating Web Accessibility for AI Agents: Measuring the Impact of WCAG Compliance on Autonomous Web Agent Performance**

Omal Vindula Wijegunawardana | W2053390 (UoW) / 20231723 (IIT) | MSc Advanced Software Engineering
Informatics Institute of Technology / University of Westminster

## Purpose

This framework jointly evaluates real-world websites on two axes — WCAG 2.2 accessibility
compliance (via axe-core) and LLM-based agent task success (via Playwright browser automation) —
and statistically correlates the two. It was used to run **5,369 agent episodes across 199 live
websites** in three observation modes (Accessibility Tree, raw DOM, screenshot), producing the
first empirical quantification of the accessibility–agent-performance relationship:

| Headline finding | Value |
|---|---|
| WCAG compliance vs. agent success (Spearman) | ρ = +0.362, p = 1.8×10⁻⁷ (n = 196) |
| Verified success: low / medium / high compliance sites | 33.6% / 59.7% / 71.5% |
| Text-based vs. screenshot observation | ~+18 pp (p < 10⁻¹⁵) |
| Accessibility Tree vs. raw DOM | equal accuracy, **58% fewer input tokens** |

The complete experimental dataset is included (`data/`), so **all statistical results and figures
in the dissertation can be reproduced in minutes at zero cost** without re-running the agent sweep.

## Requirements

**Software**
- Python ≥ 3.11 (developed on 3.14)
- macOS or Linux (Windows should work but is untested)
- ~2 GB disk (Playwright's Chromium build is ~300 MB)

**Hardware** — any modern machine; the full study ran on a MacBook (Apple Silicon), 3 concurrent
browser sessions, no GPU required.

**Languages / libraries / frameworks**
- Python (asyncio) throughout
- [Playwright](https://playwright.dev/python/) — browser automation & CDP Accessibility Tree access
- [axe-core](https://github.com/dequelabs/axe-core) 4.10.2 (vendored) — WCAG auditing
- pandas / SciPy / scikit-learn / matplotlib — statistics and figures
- httpx — Anthropic Messages API client
- pytest — test suite

## Installation

```bash
git clone <this repository>
cd wcag-agent-evaluation

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # installs runtime + dev dependencies from pyproject.toml
python -m playwright install chromium
```

## Configuration & external services

The **agent execution phase only** calls the Anthropic Messages API (model: `claude-sonnet-5`)
and requires an API key. Create a file named `.env` in the repository root:

```text
ANTHROPIC_API_KEY=sk-ant-...
```

No other external service, account, or credential is used. **No API key is required to run the
audits, the analysis, or the tests** — only to re-run agent episodes.

> **Cost warning:** re-executing the full 5,373-episode sweep costs roughly **US$410** in API
> usage (measured, July 2026 pricing) and ~30 hours wall-clock at the default concurrency of 3.
> Use `--limit-sites` for small pilots (~$0.05 per episode).

## Dataset

All data is included in `data/`:

| Path | Contents |
|---|---|
| `data/processed/final_sample.{csv,json}` | Stratified 200-site sample (5 sectors × 40, Tranco-ranked) |
| `data/processed/audit_detailed.json` | Per-rule WCAG audit of 196 sites (July 2026) |
| `data/raw/candidates_scored.json` | The 216 scored candidates the sample was drawn from |
| `data/results/agent_runs.jsonl` | **Main dataset: 5,369 agent episodes** — one JSON object per line with site, task, observation mode, repetition, outcome, independent verification flag, full action trajectory, and exact token usage |
| `data/results/excluded_robots.json` | Sites excluded by the robots.txt gate (reuters.com) |
| `data/analysis/` | Derived tables (CSV), statistical test results (JSON), publication figures (PNG/PDF) |

To rebuild the sample from scratch instead (optional): collect candidates with the utilities in
`src/utils/`, then run `python -m src.accessibility.audit_sample`.

## Running the framework

**Quick demo** (no API key needed — audits two sites and prints scores):

```bash
python demo.py
```

**Module 1 — accessibility audit:**

```bash
python -m src.accessibility.run_audit https://www.gov.uk https://www.amazon.com
python -m src.accessibility.audit_sample        # full-sample audit -> data/processed/audit_detailed.json
```

**Module 2 — agent task execution** (requires `ANTHROPIC_API_KEY`):

```bash
# Single task, verbose:
python -m src.agent.run_agent --url https://www.gov.uk --task-desc "Find the page about renewing a passport"

# Small pilot (5 sites, 1 repetition, ~$2):
python -m src.agent.batch_runner --limit-sites 5 --reps 1 --out data/results/pilot.jsonl

# Full evaluation matrix (199 sites x 3 tasks x 3 modes x 3 reps; ~$410, ~30 h):
python -m src.agent.batch_runner --out data/results/agent_runs.jsonl
```

The batch runner checkpoints after every episode: re-running the same command resumes where it
stopped and automatically retries infrastructure errors. It also honours each site's
`robots.txt` and throttles to 3 concurrent sessions (configurable via `--concurrency`).

**Module 3 — statistical analysis** (free; reproduces the dissertation's results from the
included dataset):

```bash
python -m src.analysis.run_analysis     # correlations, descriptives, feature importance, figures
python -m src.analysis.inferential      # Kruskal-Wallis, Friedman/Wilcoxon, Steiger tests + final figures
```

Outputs land in `data/analysis/` and match Chapter 6 of the dissertation.

## Tests

```bash
python -m pytest tests/ -v
```

50 tests covering the LLM action parser (including malformed-response regression cases found
during piloting), action execution, task generation, independent success verification,
checkpoint/resume semantics, the robots.txt gate, statistical functions (validated on synthetic
data with known ground truth), audit loading, and the Accessibility Tree formatter. All tests are
offline — no network or API calls. Captured output: `tests/evidence/pytest_output.txt`.

## Known limitations

- Automated auditing detects at most ~50% of WCAG failures (Vigo et al., 2013); compliance
  scores are proxies and the reported correlation is attenuated, not inflated.
- The sampled (popular) sites cluster at the accessible end of the spectrum; the low-compliance
  tail of the web is under-represented.
- All episodes used a single agent architecture on one model (`claude-sonnet-5`); absolute
  success rates are configuration-specific, though all comparisons are internally controlled.
- A small number of heavily bot-defended sites (e.g. bestbuy.com) refuse automated sessions
  entirely; 4 of 5,373 episodes (0.07%) were unrecoverable for this reason.
- Live websites change: re-running the sweep will not reproduce episode-level results exactly
  (the included dataset preserves the exact study data; the analysis pipeline is deterministic
  given that data).

## Data statement

The dataset records the behaviour of an automated agent on **publicly accessible web pages**,
collected 26–28 July 2026 with per-site load comparable to a single human visitor, honouring
`robots.txt`. It contains no personal data, no user accounts, and no content from behind
authentication. Accessibility findings are machine-generated axe-core results of the kind
routinely published (e.g. HTTP Archive, WebAIM Million) and reflect the audited pages at
collection time only.

## Default credentials / test accounts

None — the framework uses no logins and creates no accounts.

## Licence

Code: MIT (see `LICENSE`). Dataset: released for research use with attribution.
