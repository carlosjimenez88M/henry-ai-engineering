"""Evalúa el catálogo de las cuatro clases actuales; live consume API."""

import argparse
import json
from pathlib import Path

from henry_agents.config import ROOT
from henry_agents.cultural import evaluate_catalog

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["offline", "live"], default="offline")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
result = evaluate_catalog(args.mode)
text = json.dumps(result, indent=2, ensure_ascii=False)
output = args.output or ROOT / "reports" / f"catalog-evaluation-{args.mode}.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(text)
print(text)
if any(result[key] < 1 for key in ("retrieval_exact", "citations_valid", "abstention_ok")):
    raise SystemExit(1)
