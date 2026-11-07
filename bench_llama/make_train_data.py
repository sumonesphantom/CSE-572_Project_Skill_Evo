"""Build LoRA fine-tuning data (mlx-lm chat JSONL) in exactly the prompt format used at test time.

Sources (none overlap the evaluation sets):
  * Spider official train (train_spider + train_others): 8,659 questions on 146 databases that are
    disjoint from the Spider dev databases our Spider test sample comes from.
  * The shared repo's bird:train split (minus the 2 duplicated mini-dev questions that also sit in eval).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillevo.sqlexec import execute  # noqa: E402

from bench_llama.evaluate import VARIANTS, load_sets  # noqa: E402
from bench_llama.solver import build_messages  # noqa: E402

V = VARIANTS["sc5_nofs"]          # same schema/prompt as the chosen inference setup
SP = ROOT / "data/raw/spider/spider_data"


def example(task):
    msgs = build_messages(task, V, None)
    return {"messages": msgs + [{"role": "assistant", "content": f"```sql\n{task.gold_sql.strip()}\n```"}]}


def main(n_valid=200, seed=0):
    items, dropped = [], 0
    for f in ("train_spider.json", "train_others.json"):
        for i, x in enumerate(json.load(open(SP / f))):
            db = SP / "database" / x["db_id"] / f"{x['db_id']}.sqlite"
            t = SimpleNamespace(id=f"sptrain-{f[:-5]}-{i}", dataset="spider", db_id=x["db_id"], db_path=str(db),
                                question=x["question"], evidence="", gold_sql=x["query"])
            if not db.exists() or execute(str(db), t.gold_sql, 10)[1]:
                dropped += 1
                continue
            items.append(t)
    _, _, bird_train = load_sets()                         # already excludes anything in eval sets
    items += [t for t in bird_train if t.dataset == "bird"]
    rng = random.Random(seed)
    rng.shuffle(items)
    out = ROOT / "data/finetune"
    out.mkdir(parents=True, exist_ok=True)
    rows = [example(t) for t in items]
    with open(out / "valid.jsonl", "w") as f:
        for r in rows[:n_valid]:
            f.write(json.dumps(r) + "\n")
    with open(out / "train.jsonl", "w") as f:
        for r in rows[n_valid:]:
            f.write(json.dumps(r) + "\n")
    lens = sorted(sum(len(m["content"]) for m in r["messages"]) // 4 for r in rows)
    print(f"examples: {len(rows)} (train {len(rows) - n_valid}, valid {n_valid}); dropped {dropped} with failing gold SQL")
    print(f"approx tokens per example: median {lens[len(lens)//2]}, p95 {lens[int(.95*len(lens))]}, max {lens[-1]}")
    print(f"bird examples: {sum(t.dataset == 'bird' for t in items)}")


if __name__ == "__main__":
    main()
