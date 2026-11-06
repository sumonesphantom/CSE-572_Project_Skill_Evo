"""Build chat-format SFT data for mlx_lm.lora from Spider train + our BIRD/Spider train splits.

    python finetune/build_sft.py --strategy zs_rules --spider-max 3000 --out finetune/data/zs_rules

Prompts are produced by the *same* bench.strategies.build_messages used at evaluation time, so the
fine-tuned model sees exactly the prompt format it is scored with. Only train splits are used:
val/test questions never enter training (valid.jsonl is a 60-question slice of spider:val for the
training-loss curve; it is reported, never used for model selection on test).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skillevo.data import Task, load_tasks  # noqa: E402
from bench.metrics import gold_rows  # noqa: E402
from bench.strategies import LADDER, build_messages  # noqa: E402

SPIDER = ROOT / "data/raw/spider/spider_data"


def spider_train_tasks() -> list[Task]:
    out = []
    for i, x in enumerate(json.loads((SPIDER / "train_spider.json").read_text())):
        db = SPIDER / "database" / x["db_id"] / f"{x['db_id']}.sqlite"
        out.append(Task(f"spider-train-{i}", "spider", x["db_id"], str(db), x["question"], "", x["query"]))
    return out


def example(task, strat) -> dict | None:
    if not Path(task.db_path).exists() or gold_rows(task.db_path, task.gold_sql) is None:
        return None
    msgs = build_messages(task, strat)  # zero-shot prompt (no demos/skills) at train time
    return {"messages": msgs + [{"role": "assistant", "content": f"```sql\n{task.gold_sql.strip()}\n```"}]}


def token_counter(name):
    """Exact Qwen chat-template token count (all Qwen2.5 sizes share one tokenizer); chars/3 fallback offline."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        def count(ex):
            ids = tok.apply_chat_template(ex["messages"], tokenize=True)
            return len(ids["input_ids"] if hasattr(ids, "keys") else ids)  # transformers 5 returns a dict
        return count
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer unavailable ({e}); estimating tokens as chars/3")
        return lambda ex: sum(len(m["content"]) for m in ex["messages"]) // 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="zs_rules", choices=[k for k, v in LADDER.items() if not v.k_shots and not v.cot])
    ap.add_argument("--spider-max", type=int, default=3000, help="subsample of the 7,000 Spider train questions")
    ap.add_argument("--bird-repeat", type=int, default=3, help="upsample BIRD train (247 q's) so it is not drowned by Spider")
    ap.add_argument("--out", default="finetune/data/zs_rules")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="drop examples longer than this (must stay under max_seq_length in the LoRA config, "
                         "otherwise mlx_lm truncates the end, i.e. the SQL answer)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    strat = LADDER[a.strategy]
    tasks = {}
    for f in ("bird", "spider"):
        tasks.update(load_tasks(ROOT / f"data/processed/{f}.jsonl"))
    splits = json.loads((ROOT / "data/processed/splits.json").read_text())

    sp = spider_train_tasks()
    rng.shuffle(sp)
    train = [example(t, strat) for t in sp[: a.spider_max]]
    train += [example(tasks[i], strat) for i in splits["spider"]["train"]]
    bird = [example(tasks[i], strat) for i in splits["bird"]["train"]]
    train += bird * a.bird_repeat
    n_tok = token_counter(a.tokenizer)
    before = sum(1 for x in train if x)
    train = [x for x in train if x and n_tok(x) <= a.max_tokens]
    dropped = before - len(train)
    rng.shuffle(train)
    val_ids = rng.sample(splits["spider"]["val"], 60)  # spider:val is not used by any eval set
    valid = [x for x in (example(tasks[i], strat) for i in val_ids) if x and n_tok(x) <= a.max_tokens]

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    chars = sum(len(json.dumps(r)) for r in train)
    print(json.dumps({"train": len(train), "valid": len(valid), "dropped_too_long": dropped,
                      "approx_train_tokens": chars // 4, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
