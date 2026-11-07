"""Phase D: build chat-format SFT data for `mlx_lm.lora` (LoRA on Phi-4-mini, locally with MLX).

    python -m phibench.sft --mix d1 --out finetune/data/d1
    python -m phibench.sft --mix d2 --synsql data/raw/synsql/train-00000-of-00026.parquet --out finetune/data/d2

Mixes (training data only - never val/test):
  d1  Spider train_spider.json (3,000 sampled) + the repo's spider:train split + bird:train x3
      (the same sources Venkat planned for Qwen)
  d2  d1 + a seeded slice of SynSQL-2.5M (Apache-2.0; synthetic databases, so it never touches
      BIRD/Spider databases). Its schema is stored as CREATE TABLE text, so those prompts use DDL.

Every BIRD/Spider prompt is built by the same `build_messages` used at evaluation (a3_rules:
compact schema + guidelines, no examples), and the target is the gold SQL in a ```sql block.
mlx_lm's mask_prompt puts the loss on that answer only. Questions whose text matches an
evaluation question are dropped (BIRD mini-dev lists question 138 twice: one copy is in val).
Examples longer than --max-tokens are dropped, because mlx_lm would cut the answer off.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skillevo.data import Task  # noqa: E402
from skillevo.solver import SYSTEM as REPO_SYSTEM  # noqa: E402

from phibench.metrics import gold_rows  # noqa: E402
from phibench.run import EVAL_SETS, SPIDER_TRAIN, eval_ids, load_data  # noqa: E402
from phibench.strategies import GUIDELINES, STRATEGIES, build_messages  # noqa: E402

TRAIN_PROMPT = "a3_rules"
TOKENIZER = "microsoft/Phi-4-mini-instruct"


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def answer(sql: str) -> dict:
    return {"role": "assistant", "content": f"```sql\n{sql.strip()}\n```"}


def task_example(task) -> dict | None:
    if not Path(task.db_path).exists() or gold_rows(task.db_path, task.gold_sql) is None:
        return None
    return {"messages": build_messages(task, STRATEGIES[TRAIN_PROMPT]) + [answer(task.gold_sql)],
            "source": task.dataset, "id": task.id}


def synsql_example(row: dict) -> dict | None:
    q, schema, sql = (row.get("question") or "").strip(), (row.get("schema") or "").strip(), (row.get("sql") or "").strip()
    if not q or not schema or not sql or len(schema) > 12_000:
        return None
    user = "\n\n".join([f"Database schema:\n{schema}", GUIDELINES, f"Question: {q}"])
    return {"messages": [{"role": "system", "content": REPO_SYSTEM}, {"role": "user", "content": user}, answer(sql)],
            "source": "synsql", "id": f"synsql-{row.get('_row', 0)}"}


def spider_train_tasks(path: Path = SPIDER_TRAIN) -> list[Task]:
    out = []
    for i, x in enumerate(json.loads(Path(path).read_text())):
        db = Path(path).parent / "database" / x["db_id"] / f"{x['db_id']}.sqlite"
        out.append(Task(f"spider-train-{i}", "spider", x["db_id"], str(db), x["question"], "", x["query"]))
    return out


def load_synsql(paths: list[str], n: int, seed: int) -> list[dict]:
    import pandas as pd
    frames = [pd.read_parquet(p, columns=["question", "schema", "sql"]) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["_row"] = range(len(df))
    take = df.sample(n=min(n * 2, len(df)), random_state=seed).to_dict("records")  # 2x: some get filtered
    out = [e for e in (synsql_example(r) for r in take) if e]
    return out[:n]


def token_counter(name: str = TOKENIZER):
    """Exact count with the Phi-4-mini chat template when transformers is available; else chars/3.5."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)

        def count(ex):
            ids = tok.apply_chat_template(ex["messages"], tokenize=True)
            return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        return count, "exact"
    except Exception as e:  # noqa: BLE001 - offline or no transformers
        print(f"tokenizer unavailable ({type(e).__name__}); estimating tokens as chars/3.5")
        return (lambda ex: int(sum(len(m["content"]) for m in ex["messages"]) / 3.5)), "estimate"


def build(mix: str, tasks: dict, splits: dict, spider_train: list[Task], synsql: list[dict] | None = None,
          spider_max: int = 3000, bird_repeat: int = 3, valid_n: int = 100, seed: int = 0,
          max_tokens: int = 4000, counter=None) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(seed)
    eval_q = {norm(tasks[i].question) for s in EVAL_SETS for i in eval_ids(splits, s) if i in tasks}
    eval_id = {i for s in EVAL_SETS for i in eval_ids(splits, s)}
    stats = Counter()

    def keep(t):
        if t.id in eval_id or norm(t.question) in eval_q:
            stats["dropped_eval_overlap"] += 1
            return False
        return True

    sp = [t for t in spider_train if keep(t)]
    rng.shuffle(sp)
    sp = sp[:spider_max]
    sp_split = [tasks[i] for i in splits["spider"]["train"] if i in tasks and keep(tasks[i])]
    bird = [tasks[i] for i in splits["bird"]["train"] if i in tasks and keep(tasks[i])]
    train = [e for e in (task_example(t) for t in sp + sp_split) if e]
    bird_ex = [e for e in (task_example(t) for t in bird) if e]
    train += bird_ex * bird_repeat
    if mix == "d2":
        if not synsql:
            raise ValueError("mix d2 needs SynSQL rows (--synsql)")
        train += synsql
    elif mix != "d1":
        raise ValueError(f"unknown mix {mix!r}")
    val_tasks = [tasks[i] for i in splits["spider"]["val"] if i in tasks]
    rng.shuffle(val_tasks)
    valid = [e for e in (task_example(t) for t in val_tasks[:valid_n]) if e]
    if counter is not None:
        def fits(ex):
            n = counter(ex)
            stats["max_tokens_seen"] = max(stats["max_tokens_seen"], n)
            stats["tokens_total"] += n
            if n > max_tokens:
                stats["dropped_too_long"] += 1
                return False
            return True
        train = [e for e in train if fits(e)]
        valid = [e for e in valid if counter(e) <= max_tokens]
    rng.shuffle(train)
    stats.update({f"train_{k}": v for k, v in Counter(e["source"] for e in train).items()})
    stats["train"], stats["valid"] = len(train), len(valid)
    return train, valid, dict(stats)


def write(out: Path, train, valid, stats):
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w") as f:
            for e in rows:
                f.write(json.dumps({"messages": e["messages"]}) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", choices=["d1", "d2"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synsql", nargs="*", default=[], help="SynSQL-2.5M parquet shard(s) with question/schema/sql")
    ap.add_argument("--synsql-n", type=int, default=4000)
    ap.add_argument("--spider-max", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    tasks, splits = load_data()
    syn = load_synsql(a.synsql, a.synsql_n, a.seed) if a.mix == "d2" else None
    counter, how = token_counter()
    train, valid, stats = build(a.mix, tasks, splits, spider_train_tasks(), syn, a.spider_max,
                                max_tokens=a.max_tokens, seed=a.seed, counter=counter)
    stats["token_count"] = how
    write(Path(a.out), train, valid, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
