"""Save complete baseline/adapter predictions for an honest qualitative comparison.

Run after NB5. This diagnostic leaves the frozen baseline and verdict untouched.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import evaluate as ev, generate
from labkit.config import get_tier
from peft import PeftModel


def main():
    target = [json.loads(line) for line in (ROOT / "data/eval_target.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    regression = [json.loads(line) for line in (ROOT / "data/eval_regression.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    model, tok = generate.load_base(get_tier("T4"))
    bp, _ = generate.generate_batch(model, tok, [r["input"] for r in target], system=generate.OPTIMIZED_PROMPT, label="qualitative/base-target")
    br, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression], max_new_tokens=96, label="qualitative/base-regression")
    del model
    generate.free_memory()
    model, tok = generate.load_base(get_tier("T4"))
    model = PeftModel.from_pretrained(model, str(ROOT / "adapters/correct"))
    model.eval()
    fp, _ = generate.generate_batch(model, tok, [r["input"] for r in target], system=generate.NAIVE_PROMPT, label="qualitative/ft-target")
    fr, _ = generate.generate_batch(model, tok, [r["instruction"] for r in regression], max_new_tokens=96, label="qualitative/ft-regression")
    rows = []
    for kind, data, baseline, fine_tune in [("target", target, bp, fp), ("regression", regression, br, fr)]:
        for i, (item, b, f) in enumerate(zip(data, baseline, fine_tune)):
            expected = item["label"] if kind == "target" else item["keywords"]
            score = ev.triage_field_accuracy if kind == "target" else ev.keyword_recall
            rows.append({"group": kind, "i": i, "input": item["input"] if kind == "target" else item["instruction"], "expected": expected, "baseline": b, "fine_tune": f, "baseline_score": score(b, expected), "ft_score": score(f, expected)})
    (ROOT / "results/qualitative_comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    wins = [r for r in rows if r["group"] == "target" and r["ft_score"] > r["baseline_score"]]
    losses = [r for r in rows if r["ft_score"] < r["baseline_score"]]
    ties = [r for r in rows if r["group"] == "target" and r["ft_score"] == r["baseline_score"]]
    selected = wins[:2] + losses[:2] + ties[:1]
    (ROOT / "results/qualitative_selected.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SELECTED_EXAMPLES", json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
