"""Build unified task files and stratified splits.

    python scripts/prepare_data.py            # uses data/raw/{bird,spider}
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skillevo.data import load_bird, load_spider, save_tasks, stratified_split  # noqa: E402
from skillevo.sqlexec import gold_result  # noqa: E402

FRACTIONS = {
    "bird": {"train": 0.5, "val": 0.2, "test": 0.3},      # evolve here
    "spider": {"train": 0.3, "val": 0.2, "test": 0.5},    # transfer test (+ optional evolution)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird", default="data/raw/bird/minidev/MINIDEV")
    ap.add_argument("--spider", default="data/raw/spider/spider_data")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    splits, summary = {}, {}
    for name, loader, root in (("bird", load_bird, a.bird), ("spider", load_spider, a.spider)):
        if not Path(root).exists():
            print(f"skip {name}: {root} not found")
            continue
        tasks = loader(root)
        ok = [t for t in tasks if Path(t.db_path).exists() and gold_result(t.db_path, t.gold_sql) is not None]
        print(f"{name}: {len(tasks)} tasks, {len(tasks) - len(ok)} dropped (missing db / gold SQL fails)")
        save_tasks(ok, out / f"{name}.jsonl")
        splits[name] = stratified_split(ok, FRACTIONS[name], a.seed)
        diff = {t.id: t.difficulty for t in ok}
        summary[name] = {
            "n_tasks": len(ok), "n_databases": len({t.db_id for t in ok}),
            "difficulty": dict(Counter(t.difficulty for t in ok)),
            "splits": {k: {"n": len(v), "difficulty": dict(Counter(diff[i] for i in v))} for k, v in splits[name].items()},
        }
    (out / "splits.json").write_text(json.dumps(splits))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
