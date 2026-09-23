import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ.get("DREAMQAS_RESULTS_ROOT", REPO / "runs")).expanduser().resolve()
OUTPUTS = Path(os.environ.get("DREAMQAS_ANALYSIS_OUT", REPO / "outputs")).expanduser().resolve()


def campaign_root(name):
    return str(RESULTS / name)


def require_campaign(path):
    p = Path(path)
    if not p.is_dir() or not any(p.rglob("eval*.jsonl")):
        raise FileNotFoundError(f"No evaluation data in {p}. Set DREAMQAS_RESULTS_ROOT; see REPRODUCE.md.")
