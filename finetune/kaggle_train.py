"""Phase D on a free Kaggle GPU: QLoRA on Phi-4-mini, merge, convert to Ollama's Q4_K_M GGUF.

Runs inside a Kaggle notebook (GPU T4 / P100, internet on). Free; no API keys.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python kaggle_train.py --runs base,d1,d2
    python kaggle_train.py --runs d1 --steps 400        # one adapter

For each run it writes /kaggle/working/<run>/:
    adapter/                   LoRA adapter (training runs only)
    train_log.json             loss curve, tokens/s, settings
    phi4mini-<run>-Q4_K_M.gguf the merged model in the same quantisation as Ollama's phi4-mini
    DONE                       marker: a rerun skips finished runs

Training matches the MLX config written for the Mac (except that the frozen base runs in fp16, not
4-bit, which fits a 16 GB T4 and is several times faster there): LoRA rank 16 (alpha 32 = scale 2.0) on the
last 16 of 32 layers, dropout 0.05, effective batch 8, lr 1e-4 cosine with 40 warmup steps, loss on
the SQL answer only. Memory: Phi-4-mini's vocabulary has 200k tokens, so full-sequence logits for an
8k-token prompt take several GB. The loss here runs the output layer only on answer positions.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = "microsoft/Phi-4-mini-instruct"
TARGETS = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
LLAMA_CPP = "https://github.com/ggml-org/llama.cpp"


# ----------------------------------------------------------------------------- data
def find_data(mix: str, root: str) -> tuple[Path, Path]:
    hits = sorted(glob.glob(f"{root}/**/{mix}/train.jsonl", recursive=True))
    if not hits:
        raise FileNotFoundError(f"no {mix}/train.jsonl under {root}: is the dataset attached to the notebook?")
    d = Path(hits[0]).parent
    return d / "train.jsonl", d / "valid.jsonl"


def load_jsonl(path: Path) -> list[list[dict]]:
    with open(path) as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def _ids(x) -> list[int]:
    return list(x["input_ids"] if hasattr(x, "keys") else x)


def encode(tok, messages: list[dict], max_len: int):
    """(input_ids, labels) with labels -100 on the prompt, or None if too long / template mismatch."""
    prompt = _ids(tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    full = _ids(tok.apply_chat_template(messages, tokenize=True))
    if len(full) > max_len or full[:len(prompt)] != prompt or len(full) == len(prompt):
        return None
    return full, [-100] * len(prompt) + full[len(prompt):]


def encode_all(tok, rows, max_len):
    out, skipped = [], 0
    for m in rows:
        e = encode(tok, m, max_len)
        if e is None:
            skipped += 1
        else:
            out.append(e)
    return out, skipped


# ----------------------------------------------------------------------------- model
def avoid_gqa_sdpa():
    """transformers passes enable_gqa=True to SDPA when there is no attention mask. PyTorch has no
    memory-efficient kernel for that on a T4, so it falls back to the math kernel, which builds the
    full heads x L x L score matrix (2.4 GB at 7k tokens). Repeating the KV heads instead keeps
    SDPA on the memory-efficient kernel."""
    try:
        from transformers.integrations import sdpa_attention
        if hasattr(sdpa_attention, "use_gqa_in_sdpa"):
            sdpa_attention.use_gqa_in_sdpa = lambda *a, **k: False
            return True
    except ImportError:
        pass
    return False


def load_model(base: str, four_bit: bool, n_lora_layers: int, rank: int):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8      # T4 / P100: no bf16 -> fp16
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    kw = {"dtype": dtype, "attn_implementation": "sdpa"}
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    if cuda:
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    for p in model.parameters():
        p.requires_grad_(False)
    # Non-reentrant checkpointing needs no gradient on the inputs, so backward stops at the
    # lowest LoRA layer instead of running through all 32 layers.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    n_layers = model.config.num_hidden_layers
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=TARGETS, bias="none",
                     layers_to_transform=list(range(max(0, n_layers - n_lora_layers), n_layers)), task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():          # trainable LoRA weights in fp32 (GradScaler needs fp32 params)
        if p.requires_grad:
            p.data = p.data.float()
    model.config.use_cache = False
    return model, dtype


def answer_loss(model, input_ids, labels):
    """Mean cross-entropy over answer tokens, running the output layer on those positions only."""
    import torch.nn.functional as F
    inner = model.get_base_model()                 # the causal-LM with LoRA layers injected
    hidden = inner.model(input_ids=input_ids, use_cache=False).last_hidden_state
    h, tgt = hidden[:, :-1, :], labels[:, 1:]
    mask = tgt != -100
    logits = inner.lm_head(h[mask]).float()
    return F.cross_entropy(logits, tgt[mask]), int(mask.sum())


# ----------------------------------------------------------------------------- training
def train(args, run: str, out: Path) -> dict:
    import torch
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    tr_path, va_path = find_data(run, args.data_root)
    train_rows, skipped = encode_all(tok, load_jsonl(tr_path), args.max_len)
    valid_rows, _ = encode_all(tok, load_jsonl(va_path)[: args.val_n], args.max_len)
    print(f"[{run}] {len(train_rows)} train examples ({skipped} skipped), {len(valid_rows)} valid", flush=True)
    patched = avoid_gqa_sdpa()
    model, dtype = load_model(args.base, args.four_bit, args.lora_layers, args.rank)
    model.print_trainable_parameters()
    print(f"[{run}] dtype {dtype}, 4-bit base: {args.four_bit}, GQA-SDPA patch: {patched}", flush=True)
    dev = next(p for p in model.parameters() if p.requires_grad).device
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)
    use_scaler = dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    amp = torch.autocast(device_type="cuda", dtype=dtype) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    if dev.type == "cuda":   # fail fast (in the step-0 validation pass) rather than fall back to the O(L^2) math kernel
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]).__enter__()

    order, pos = list(range(len(train_rows))), 0
    random.shuffle(order)

    def nxt():
        nonlocal pos
        if pos >= len(order):
            random.shuffle(order)
            pos = 0
        i = order[pos]
        pos += 1
        ids, lab = train_rows[i]
        return torch.tensor([ids], device=dev), torch.tensor([lab], device=dev)

    def val_loss():
        model.eval()
        tot = n = 0
        with torch.no_grad(), amp:
            for ids, lab in valid_rows:
                loss, k = answer_loss(model, torch.tensor([ids], device=dev), torch.tensor([lab], device=dev))
                tot, n = tot + loss.item() * k, n + k
        model.train()
        return tot / max(n, 1)

    log = {"run": run, "settings": {k: v for k, v in vars(args).items() if k not in ("runs",)},
           "n_train": len(train_rows), "skipped": skipped, "steps": [], "val": []}
    log["val"].append({"step": 0, "loss": val_loss()})
    print(f"[{run}] step 0 val loss {log['val'][-1]['loss']:.4f}", flush=True)
    model.train()
    t0, toks = time.time(), 0
    for step in range(1, args.steps + 1):
        losses = []
        for _ in range(args.grad_acc):
            ids, lab = nxt()
            with amp:
                loss, _k = answer_loss(model, ids, lab)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"[{run}] loss is {loss.item()} at step {step}: stop and lower --lr")
            scaler.scale(loss / args.grad_acc).backward()
            losses.append(loss.item())
            toks += ids.shape[1]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        rec = {"step": step, "loss": sum(losses) / len(losses), "lr": sched.get_last_lr()[0],
               "tok_per_s": round(toks / (time.time() - t0), 1), "elapsed_min": round((time.time() - t0) / 60, 1)}
        log["steps"].append(rec)
        if step == 5:
            proj = (time.time() - t0) / 5 * args.steps / 3600
            print(f"[{run}] projected training time for {args.steps} steps: {proj:.1f} h "
                  f"(--max-hours {args.max_hours})", flush=True)
        if step % args.log_every == 0 or step == 1:
            print(f"[{run}] step {step}/{args.steps} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                  f"{rec['tok_per_s']} tok/s {rec['elapsed_min']} min", flush=True)
        out_of_time = (time.time() - t0) / 3600 > args.max_hours
        if out_of_time:
            log["stopped_early_at_step"] = step
            print(f"[{run}] --max-hours {args.max_hours} reached at step {step}: saving and stopping", flush=True)
        if step % args.eval_every == 0 or step == args.steps or out_of_time:
            log["val"].append({"step": step, "loss": val_loss()})
            print(f"[{run}] step {step} val loss {log['val'][-1]['loss']:.4f}", flush=True)
            model.save_pretrained(out / "adapter")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
        if out_of_time:
            break
    model.save_pretrained(out / "adapter")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    model = opt = None  # free GPU memory before the merge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return log


# ----------------------------------------------------------------------------- merge + GGUF
def original_files(base: str) -> Path:
    """The base model's own non-weight files (config, tokenizer), as published."""
    p = Path(base)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(base, allow_patterns=["*.json", "*.txt", "*.model", "*.py", "*.tiktoken"]))


def merge(base: str, adapter: Path | None, dst: Path):
    """Merged fp16 weights + the original config/tokenizer files. Re-saving those through a newer
    transformers rewrites them (e.g. rope settings, tokenizer class), and llama.cpp's converter
    reads the published format."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map="cpu")
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(dst, safe_serialization=True)
    del model
    gc.collect()
    for f in original_files(base).iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, dst / f.name)


def sh(cmd: list[str], cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_llama_cpp(root: Path) -> Path:
    src = root / "llama.cpp"
    if not src.exists():
        sh(["git", "clone", "--depth", "1", LLAMA_CPP, str(src)])
    # The converter imports its bundled gguf-py itself. Installing its requirements file could
    # replace the notebook's CUDA build of torch, so only the two small extras are added.
    sh([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"])
    q = src / "build" / "bin" / "llama-quantize"
    if not q.exists():
        sh(["cmake", "-B", "build", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF"], cwd=src)
        sh(["cmake", "--build", "build", "--target", "llama-quantize", "-j", "4"], cwd=src)
    return src


def to_gguf(merged: Path, out_file: Path, tmp: Path):
    src = ensure_llama_cpp(tmp)
    f16 = tmp / (out_file.stem + "-f16.gguf")
    sh([sys.executable, str(src / "convert_hf_to_gguf.py"), str(merged), "--outtype", "f16", "--outfile", str(f16)])
    shutil.rmtree(merged, ignore_errors=True)
    sh([str(src / "build" / "bin" / "llama-quantize"), str(f16), str(out_file), "Q4_K_M"])
    f16.unlink(missing_ok=True)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="base,d1,d2", help="comma list: base (control, no training), d1, d2")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--data-root", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--tmp", default="/tmp/phi4mini")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-layers", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--val-n", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=9.5, help="stop training (and save) before Kaggle's 12 h limit")
    ap.add_argument("--4bit", dest="four_bit", action="store_true",
                    help="4-bit NF4 base (QLoRA). Default: fp16 base, which fits a 16 GB T4 and trains much faster")
    ap.add_argument("--no-gguf", dest="gguf", action="store_false", help="stop after training (+ merge)")
    a = ap.parse_args(argv)
    out_root, tmp = Path(a.out), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for run in a.runs.split(","):
        out = out_root / run
        if (out / "DONE").exists():
            print(f"[{run}] already done, skipping", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        adapter = None
        if run != "base":
            train(a, run, out)
            adapter = out / "adapter"
        merged = tmp / f"merged-{run}"
        merge(a.base, adapter, merged)
        if a.gguf:
            to_gguf(merged, out / f"phi4mini-{run}-Q4_K_M.gguf", tmp)
        (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"[{run}] finished -> {out}", flush=True)


if __name__ == "__main__":
    main()
