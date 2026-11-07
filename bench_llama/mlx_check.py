"""Run the fine-tuned model directly in MLX (no Ollama import) on N BIRD-val questions, greedy,
same prompt as the 'pruned' variant, and compare with the Ollama-served copy."""
import json, sys, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
sys.path.insert(0, ".")
import types; sys.modules.setdefault("requests", types.ModuleType("requests"))
from bench_llama.evaluate import load_sets, VARIANTS
from bench_llama.solver import build_messages
from bench_llama.extract import extract_sql
from skillevo.sqlexec import execute, is_correct

model_path, n = sys.argv[1], int(sys.argv[2])
adapter = sys.argv[3] if len(sys.argv) > 3 else None
tasks, sets, _ = load_sets()
ids = sets["val"][:n]
model, tok = load(model_path, adapter_path=adapter)
v = VARIANTS["pruned"]
ok = 0; t0 = time.time(); out = []
for tid in ids:
    t = tasks[tid]
    prompt = tok.apply_chat_template(build_messages(t, v, None), add_generation_prompt=True, tokenize=False)
    text = generate(model, tok, prompt=prompt, max_tokens=300, sampler=make_sampler(temp=0.0))
    sql = extract_sql(text)
    c = is_correct(t.db_path, sql, t.gold_sql)[0]
    ok += bool(c); out.append({"task_id": tid, "sql": sql, "correct": bool(c)})
print(f"{model_path} adapter={adapter}: {ok}/{len(ids)} = {100*ok/len(ids):.1f}%  ({time.time()-t0:.0f}s)")
for r in out[:5]: print("  ", r["sql"][:160])
