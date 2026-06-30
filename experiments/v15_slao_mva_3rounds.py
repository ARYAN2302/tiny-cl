"""
V15: SLAO + MVA — 3-Round Loop with Threshold Tracking
======================================================

Answers the question v14 couldn't: does pass^5 trend up across rounds 2 and 3,
or does it bounce around near +0.07 in a way that looks like noise?

v14 showed MVA composes with SLAO (+0.070 pass^5, 1 round). But chi-sq=1.8
vs 3.84 needed means 1 round is close to a coin flip. v15 runs 3 MVA rounds
to see if the signal compounds.

ARCHITECTURE
------------
Rounds 1-3: Standard v13 SLAO (train domains A->B->C, SLAO merge after each)
Rounds 4-6: MVA self-improvement rounds (certainty-validated SQuAD, SLAO merge)

Each MVA round:
  1. Generate answers on fresh SQuAD sample
  2. Compute certainty distribution (LOGGED — does it shift across rounds?)
  3. Adaptive threshold (50th percentile of THIS round's distribution)
  4. Validate, train, SLAO merge
  5. Eval: pass^5 + domain perplexity

WHAT WE LEARN
-------------
- Does pass^5 trend up monotonically (signal) or bounce (noise)?
- Does the certainty distribution shift across MVA rounds?
- Does validation precision hold or degrade?
- Does domain PPL drift accumulate or stabilize?

FRAMING (unchanged from v14)
----------------------------
Self-improvement via REDUCED-ERROR FILTERING, not self-improvement via correct
self-generated data. MVA's certainty gate filters out 60-88% of wrong answers
(precision 79-94%), reducing wrong-answer contamination from ~30% (naive) to
~6-21% (MVA). The pass^5 improvement reflects training on fewer mistakes.

USAGE: Copy-paste into one Kaggle cell with T4 GPU. ~4-5 hours.
"""

import subprocess, sys, os, json, time, random, math, gc, re
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIG
# ============================================================================

MODEL_ID = "LiquidAI/LFM2.5-350M"
OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("/home/z/my-project/download")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# v13's minimal LoRA config — proven on SLAO, used for consistency
LORA_RANK = 32
LORA_ALPHA = 32       # v13 uses alpha=rank (not 2*rank)
LORA_DROPOUT = 0.05
LORA_TARGETS = ["in_proj", "out_proj"]  # v13's minimal config

# Training config (from v13)
TRAIN_LR = 2e-4
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
DOMAIN_EPOCHS = 1
MVA_EPOCHS = 3          # MVA gets more epochs — less data (~15K tokens vs 1M)
BATCH_SIZE = 8
CONTEXT_LENGTH = 512

# MVA config (from validated v5 script)
# v14 fix: ADAPTIVE threshold. The fixed 17.0 was tuned on the BASE model in v5.
# After SLAO merging, the certainty distribution shifts and 0/200 clear the threshold.
# Adaptive = take the top 50% by certainty (median split), whatever the actual number is.
# This is NOT "tuning for better pass^5" — it's "make the filter operational on post-SLAO model."
# The reduced-error-filtering framing is preserved: we still filter by certainty, just adaptively.
ADAPTIVE_THRESHOLD = True
CERTAINTY_THRESHOLD = 17.0  # only used if ADAPTIVE_THRESHOLD = False
CERTAINTY_PERCENTILE = 50   # take top 50% (median split) when adaptive
N_MVA_QUESTIONS = 200   # training pool
N_MVA_HOLDOUT = 100     # pass^5 eval set
PASS_K = 5
GEN_MAX_NEW_TOKENS = 60
GEN_TEMPERATURE = 0.7

# Experiment
SEED = 42
DOMAIN_ORDER = ["A", "B", "C"]
N_MVA_ROUNDS = 3  # v15: 3 MVA rounds instead of 1 (v14) to test compounding

# LFM2.5 layer structure
CONV_LAYER_IDS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYER_IDS = [2, 5, 8, 10, 12, 14]

DOMAINS = {
    "A": {"name": "medical",  "display": "Medical",
           "dataset": "epfl-llm/guidelines", "field": "clean_text"},
    "B": {"name": "code",     "display": "Code",
           "dataset": "iamtarun/python_code_instructions_18k_alpaca", "field": "output"},
    "C": {"name": "creative", "display": "Creative",
           "dataset": "roneneldan/TinyStories", "field": "text"},
}

# ============================================================================
# DEPS
# ============================================================================

def _ensure_deps():
    missing = []
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                                   "transformers>=5.0.0", "packaging"])
    except ImportError:
        missing.extend(["transformers>=5.0.0", "packaging"])
    for pkg in ["peft", "datasets", "accelerate", "scipy"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

_ensure_deps()

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora.layer import LoraLayer

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATA
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
        self.n_chunks = max(1, len(token_ids) // context_length)
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        s = idx * self.context_length
        e = s + self.context_length
        chunk = self.token_ids[s:e]
        return {"input_ids": chunk, "labels": chunk.clone()}

def prepare_domain(domain_key, tokenizer, max_tokens=1_000_000):
    from datasets import load_dataset
    d = DOMAINS[domain_key]
    print(f"  Loading: {d['display']}")
    ds = load_dataset(path=d["dataset"], split="train")
    texts = [t for t in ds[d["field"]] if t and len(t.strip()) > 10]
    random.shuffle(texts)
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if len(all_tokens) >= max_tokens: break
    token_ids = torch.tensor(all_tokens[:int(max_tokens)], dtype=torch.long)
    print(f"    {len(token_ids):,} tokens")
    n_val = min(int(len(token_ids) * 0.1), 100_000)
    n_train = len(token_ids) - n_val
    return TextDataset(token_ids[:n_train], CONTEXT_LENGTH), \
           TextDataset(token_ids[n_train:n_train + n_val], CONTEXT_LENGTH)

def load_squad_pairs(n_questions):
    """Load SQuAD (passage, question, answer) triples."""
    from datasets import load_dataset
    print(f"  Loading SQuAD ({n_questions} questions)...")
    squad = load_dataset("rajpurkar/squad", split="validation")
    passages = {}
    for ex in squad:
        ctx = ex["context"]
        if ctx not in passages:
            passages[ctx] = []
        answers = ex["answers"]["text"]
        if answers:
            passages[ctx].append({"q": ex["question"], "a": answers[0], "paragraph": ctx})
    passage_list = list(passages.keys())
    random.shuffle(passage_list)
    pairs = []
    for ctx in passage_list:
        qs = passages[ctx]
        random.shuffle(qs)
        for q in qs[:2]:
            pairs.append((q["paragraph"], q["q"], q["a"]))
        if len(pairs) >= n_questions: break
    print(f"    {len(pairs)} questions from {len(set(p[0] for p in pairs))} passages")
    return pairs[:n_questions]

# ============================================================================
# MODEL
# ============================================================================

def load_base():
    print(f"  Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="eager",
    )
    return model, tokenizer

def create_model():
    model, tokenizer = load_base()
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    conv_c, attn_c = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if any(f"layers.{idx}." in name for idx in CONV_LAYER_IDS): conv_c += 1
        elif any(f"layers.{idx}." in name for idx in ATTN_LAYER_IDS): attn_c += 1
    print(f"  LoRA: {conv_c} conv + {attn_c} attn = {conv_c+attn_c} modules (rank={LORA_RANK})")
    return model, tokenizer

# ============================================================================
# SLAO UTILS (from v13, unchanged)
# ============================================================================

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

@torch.no_grad()
def compute_ppl(model, dataset, max_samp=1024):
    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    tot_loss, tot_tok, nb = 0.0, 0, 0
    for batch in loader:
        if nb * 8 >= max_samp: break
        out = model(input_ids=batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
        nt = batch["labels"].numel()
        tot_loss += out.loss.item() * nt; tot_tok += nt; nb += 1
    model.train()
    return math.exp(tot_loss / tot_tok) if tot_tok > 0 else float("inf")

def train_phase(model, dataset, epochs=DOMAIN_EPOCHS):
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=TRAIN_LR, weight_decay=TRAIN_WD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    gs, tl = 0, 0.0
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
            opt.step(); tl += out.loss.item(); gs += 1
            if gs % 50 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")
    return gs, tl

def eval_all_domains(model, val_ds):
    return {pk: compute_ppl(model, val_ds[pk]) for pk in DOMAIN_ORDER if val_ds.get(pk) is not None}

# ============================================================================
# SLAO CORE (from v13, unchanged)
# ============================================================================

def slao_extract_ortho_A(model):
    ortho_A = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        A = module.lora_A["default"].weight.data.float()
        Q, R = torch.linalg.qr(A.T.contiguous())
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        ortho_A[name] = Q.T
    return ortho_A

def slao_init(model, ortho_A, prev_ft_B):
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        if name in ortho_A:
            module.lora_A["default"].weight.data.copy_(
                ortho_A[name].to(DEVICE).to(module.lora_A["default"].weight.data.dtype))
        B_key = f"{name}.lora_B.default.weight"
        if B_key in prev_ft_B:
            module.lora_B["default"].weight.data.copy_(
                prev_ft_B[B_key].to(DEVICE).to(module.lora_B["default"].weight.data.dtype))

def slao_merge(merged_state, ft_state, task_num):
    lam = 1.0 / math.sqrt(task_num)
    new_merged = {}
    for key in ft_state:
        ft_val = ft_state[key]
        if key in merged_state:
            if "lora_A" in key:
                new_merged[key] = ft_val.cpu().clone()
            elif "lora_B" in key:
                old_val = merged_state[key]
                new_merged[key] = (old_val + lam * (ft_val - old_val)).cpu().clone()
            else:
                new_merged[key] = ft_val.cpu().clone()
        else:
            new_merged[key] = ft_val.cpu().clone()
    print(f"  [SLAO-MERGE] Task {task_num}: A=replace, B=interpolate(λ={lam:.4f})")
    return new_merged

# ============================================================================
# MVA UTILS (from validated v5 script, adapted to v13 infrastructure)
# ============================================================================

def build_prompt(paragraph, question):
    return f"Passage: {paragraph}\n\nQuestion: {question}\n\nAnswer (brief, factual):"

def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def ground_truth_check(answer, gold):
    norm_answer = normalize_answer(answer)
    norm_gold = normalize_answer(gold)
    if not norm_gold: return False
    gold_tokens = set(norm_gold.split())
    answer_tokens = set(norm_answer.split())
    if not gold_tokens: return False
    return len(gold_tokens & answer_tokens) / len(gold_tokens) >= 0.5

def generate(model, tokenizer, prompt, max_new_tokens=GEN_MAX_NEW_TOKENS,
             temperature=GEN_TEMPERATURE, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=do_sample, top_p=0.95, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def compute_certainty(model, tokenizer, paragraph, question, answer):
    """INTUITOR self-certainty: KL(U || p_theta) averaged over answer tokens."""
    try:
        prompt = build_prompt(paragraph, question)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024, add_special_tokens=True)
        answer_ids = tokenizer(" " + answer, return_tensors="pt",
                               add_special_tokens=False)["input_ids"][0]
        full_ids = torch.cat([prompt_ids["input_ids"][0], answer_ids], dim=0).unsqueeze(0)
        inputs = {"input_ids": full_ids.to(DEVICE),
                  "attention_mask": torch.ones_like(full_ids).to(DEVICE)}
        answer_start = prompt_ids["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model(**inputs)
        answer_logits = outputs.logits[0, answer_start - 1:-1, :]
        if answer_logits.shape[0] == 0: return 0.0
        log_probs = F.log_softmax(answer_logits, dim=-1)
        vocab_size = log_probs.shape[-1]
        return (-math.log(vocab_size) - log_probs.mean(dim=-1)).mean().item()
    except Exception as e:
        print(f"    [certainty error] {e}")
        return 0.0

def measure_pass_k(model, tokenizer, holdout_pairs, k=PASS_K):
    """pass^k = P(all k samples correct). Also pass@k = P(>=1 correct)."""
    pass_k_results, pass_at_k_results = [], []
    for paragraph, question, gold in holdout_pairs:
        samples = [generate(model, tokenizer, build_prompt(paragraph, question))
                   for _ in range(k)]
        correct_flags = [ground_truth_check(s, gold) for s in samples]
        n_correct = sum(correct_flags)
        pass_k_results.append(1.0 if n_correct == k else 0.0)
        pass_at_k_results.append(1.0 if n_correct > 0 else 0.0)
    return {
        "pass_k": float(np.mean(pass_k_results)),
        "pass_at_k": float(np.mean(pass_at_k_results)),
        "n_questions": len(holdout_pairs),
    }

def build_mva_training_dataset(tokenizer, validated_pairs):
    """Build a TextDataset from validated (paragraph, question, answer) triples.
    Each pair is formatted as prompt + answer + eos, then concatenated into a token stream.
    """
    all_tokens = []
    for paragraph, question, answer in validated_pairs:
        text = build_prompt(paragraph, question) + " " + answer + tokenizer.eos_token
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    print(f"    MVA training stream: {len(token_ids):,} tokens")
    return TextDataset(token_ids, CONTEXT_LENGTH)

# ============================================================================
# RUN MVA ROUND (the new round 4)
# ============================================================================

def run_mva_round(model, tokenizer, merged_state, val_ds, squad_holdout, task_num=4, mva_round_num=1):
    """
    MVA self-improvement round:
    1. Load fresh SQuAD sample (different each round via seed offset)
    2. Generate answers with current model
    3. Compute certainty distribution (LOGGED — track shifts across rounds)
    4. Adaptive threshold (50th percentile of THIS round's distribution)
    5. Validate, train, SLAO merge
    6. Eval: domain perplexity + pass^5 on SQuAD holdout
    """
    print(f"\n{'='*70}")
    print(f"ROUND {task_num}: MVA SELF-IMPROVEMENT (MVA round {mva_round_num})")
    print(f"{'='*70}")

    # --- Baseline metrics (before MVA) ---
    print("  Measuring pre-MVA metrics...")
    pre_ppls = eval_all_domains(model, val_ds)
    pre_pass5 = measure_pass_k(model, tokenizer, squad_holdout)
    print(f"  Pre-MVA: pass^5={pre_pass5['pass_k']:.3f}, pass@5={pre_pass5['pass_at_k']:.3f}")
    print(f"  Pre-MVA PPL: " + " | ".join(f"{k}: {v:.2f}" for k, v in pre_ppls.items()))

    # --- Load SQuAD questions for MVA training (fresh sample each round) ---
    # Use seed offset so each MVA round gets different questions
    # (otherwise the model trains on the same 200 questions every round)
    original_seed = random.getstate()
    random.seed(SEED + mva_round_num * 1000)  # different sample each round
    squad_train = load_squad_pairs(N_MVA_QUESTIONS)
    random.setstate(original_seed)  # restore RNG state

    # --- Generate answers + compute certainty (ALL questions first) ---
    print(f"\n  Generating {len(squad_train)} answers + computing certainty...")
    all_results = []
    t_start = time.time()
    for i, (paragraph, question, gold) in enumerate(squad_train):
        answer = generate(model, tokenizer, build_prompt(paragraph, question))
        certainty = compute_certainty(model, tokenizer, paragraph, question, answer)
        correct = ground_truth_check(answer, gold)
        all_results.append({
            "paragraph": paragraph, "question": question, "gold": gold,
            "answer": answer, "certainty": certainty, "correct": correct,
        })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"    [{i+1}/{len(squad_train)}] ~{elapsed*(len(squad_train)-i-1)/(i+1):.0f}s remaining")

    # --- Certainty distribution diagnostic ---
    certainties = [r["certainty"] for r in all_results]
    cert_arr = np.array(certainties)
    print(f"\n  Certainty distribution on post-SLAO model:")
    print(f"    min={cert_arr.min():.2f}  max={cert_arr.max():.2f}  mean={cert_arr.mean():.2f}")
    print(f"    median={np.median(cert_arr):.2f}  std={cert_arr.std():.2f}")
    print(f"    percentiles: 25th={np.percentile(cert_arr,25):.2f}  "
          f"50th={np.percentile(cert_arr,50):.2f}  75th={np.percentile(cert_arr,75):.2f}")
    print(f"    (v5 base model had mean=16.68, threshold=17.0 — SLAO shifts this distribution)")

    # --- Apply adaptive or fixed threshold ---
    if ADAPTIVE_THRESHOLD:
        threshold = float(np.percentile(cert_arr, CERTAINTY_PERCENTILE))
        print(f"\n  ADAPTIVE threshold: {threshold:.2f} ({CERTAINTY_PERCENTILE}th percentile)")
        print(f"    → Takes top {100-CERTAINTY_PERCENTILE}% by certainty (~{len(all_results) * (100-CERTAINTY_PERCENTILE) // 100} pairs)")
    else:
        threshold = CERTAINTY_THRESHOLD
        print(f"\n  FIXED threshold: {threshold}")

    validated_pairs = []
    n_correct_validated = 0
    n_wrong_validated = 0
    for r in all_results:
        if r["certainty"] >= threshold:
            validated_pairs.append((r["paragraph"], r["question"], r["answer"]))
            if r["correct"]: n_correct_validated += 1
            else: n_wrong_validated += 1

    n_validated = len(validated_pairs)
    precision = n_correct_validated / max(n_validated, 1)
    print(f"\n  Validation results:")
    print(f"    Validated: {n_validated}/{len(squad_train)} ({100*n_validated/len(squad_train):.1f}%)")
    print(f"    Precision: {n_correct_validated}/{n_validated} = {100*precision:.1f}%")
    print(f"    Wrong answers in training: {n_wrong_validated} ({100*n_wrong_validated/max(n_validated,1):.1f}%)")
    print(f"    (Known flaw — self-improvement via reduced-error filtering,")
    print(f"     not via correct self-generated data)")

    if n_validated == 0:
        print("  [SKIP] No validated pairs — MVA round aborted")
        return {"pre": {"ppl": pre_ppls, "pass5": pre_pass5},
                "mva": None, "reason": "no validated pairs"}

    # --- SLAO init: orthogonalize A, load merged B ---
    print(f"\n  SLAO init for MVA round...")
    ortho_A = slao_extract_ortho_A(model)
    prev_ft_B = {k: v for k, v in merged_state.items() if "lora_B" in k}
    slao_init(model, ortho_A, prev_ft_B)

    # --- Train on validated pairs ---
    print(f"\n  Training on {n_validated} validated pairs ({MVA_EPOCHS} epochs)...")
    mva_dataset = build_mva_training_dataset(tokenizer, validated_pairs)
    gs, tl = train_phase(model, mva_dataset, epochs=MVA_EPOCHS)

    # --- SLAO merge ---
    ft_state = get_lora_state(model)
    merged_state = slao_merge(merged_state, ft_state, task_num)
    set_lora_state(model, merged_state)

    # --- Post-MVA metrics ---
    print(f"\n  Measuring post-MVA metrics...")
    post_ppls = eval_all_domains(model, val_ds)
    post_pass5 = measure_pass_k(model, tokenizer, squad_holdout)
    print(f"  Post-MVA: pass^5={post_pass5['pass_k']:.3f}, pass@5={post_pass5['pass_at_k']:.3f}")
    print(f"  Post-MVA PPL: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_ppls.items()))

    # --- Compute deltas ---
    pass5_delta = post_pass5["pass_k"] - pre_pass5["pass_k"]
    ppl_deltas = {k: post_ppls[k] - pre_ppls[k] for k in DOMAIN_ORDER}

    print(f"\n  DELTAS (MVA round):")
    print(f"    pass^5: {pre_pass5['pass_k']:.3f} → {post_pass5['pass_k']:.3f} ({pass5_delta:+.3f})")
    for k in DOMAIN_ORDER:
        print(f"    PPL({k}): {pre_ppls[k]:.2f} → {post_ppls[k]:.2f} ({ppl_deltas[k]:+.2f})")

    return {
        "mva_round": mva_round_num,
        "task_num": task_num,
        "pre": {"ppl": pre_ppls, "pass5": pre_pass5},
        "post": {"ppl": post_ppls, "pass5": post_pass5},
        "validation": {
            "n_validated": n_validated, "n_correct": n_correct_validated,
            "n_wrong": n_wrong_validated, "precision": precision,
            "threshold": threshold,
            "adaptive": ADAPTIVE_THRESHOLD,
            "certainty_distribution": {
                "min": float(cert_arr.min()), "max": float(cert_arr.max()),
                "mean": float(cert_arr.mean()), "median": float(np.median(cert_arr)),
                "std": float(cert_arr.std()),
            },
        },
        "deltas": {"pass5": pass5_delta, "ppl": ppl_deltas},
        "merged_state": merged_state,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("V15: SLAO + MVA — 3-ROUND LOOP WITH THRESHOLD TRACKING")
    print(f"Seed: {SEED} | Order: {'→'.join(DOMAIN_ORDER)} → {N_MVA_ROUNDS} MVA rounds")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}")
    print(f"Domain epochs: {DOMAIN_EPOCHS}, MVA epochs: {MVA_EPOCHS}")
    print(f"Certainty threshold: {'ADAPTIVE (50th percentile)' if ADAPTIVE_THRESHOLD else CERTAINTY_THRESHOLD}")
    print(f"Device: {DEVICE}")
    print()

    # --- Load model + tokenizer ---
    model, tokenizer = create_model()

    # --- Prepare domain data ---
    print("\nPreparing domain data...")
    phases_data, val_ds = {}, {}
    for pk in DOMAIN_ORDER:
        t, v = prepare_domain(pk, tokenizer)
        phases_data[pk] = t
        val_ds[pk] = v

    # --- Load SQuAD holdout for pass^5 ---
    print("\nLoading SQuAD holdout for pass^5...")
    squad_holdout = load_squad_pairs(N_MVA_HOLDOUT)

    # --- Rounds 1-3: Standard SLAO ---
    merged_state = None
    prev_ft_state = None
    results = {"rounds": {}, "seed": SEED, "config": {
        "lora_rank": LORA_RANK, "lora_targets": LORA_TARGETS,
        "adaptive_threshold": ADAPTIVE_THRESHOLD,
        "certainty_percentile": CERTAINTY_PERCENTILE,
        "domain_epochs": DOMAIN_EPOCHS, "mva_epochs": MVA_EPOCHS,
    }}

    for task_num, pk in enumerate(DOMAIN_ORDER, 1):
        print(f"\n{'='*70}")
        print(f"ROUND {task_num}: {DOMAINS[pk]['display']} (SLAO)")
        print(f"{'='*70}")

        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        if task_num > 1:
            ortho_A = slao_extract_ortho_A(model)
            prev_ft_B = {k: v for k, v in prev_ft_state.items() if "lora_B" in k}
            slao_init(model, ortho_A, prev_ft_B)
            print(f"  SLAO init for task {task_num}")

        gs, tl = train_phase(model, phases_data[pk])
        prev_ft_state = get_lora_state(model)

        if merged_state is None:
            merged_state = prev_ft_state.copy()
        else:
            merged_state = slao_merge(merged_state, prev_ft_state, task_num)

        set_lora_state(model, merged_state)
        ppls = eval_all_domains(model, val_ds)
        pass5 = measure_pass_k(model, tokenizer, squad_holdout)

        results["rounds"][f"round_{task_num}"] = {
            "domain": pk, "ppl": ppls, "pass5": pass5,
            "avg_loss": tl / max(gs, 1),
        }
        print(f"  PPL: " + " | ".join(f"{k}: {v:.2f}" for k, v in ppls.items()))
        print(f"  pass^5: {pass5['pass_k']:.3f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # --- Rounds 4-6: MVA (3 rounds to test compounding) ---
    for mva_round in range(1, N_MVA_ROUNDS + 1):
        task_num = 3 + mva_round  # rounds 4, 5, 6
        print(f"\n{'#'*70}")
        print(f"# MVA ROUND {mva_round}/{N_MVA_ROUNDS} (SLAO task {task_num})")
        print(f"{'#'*70}")
        mva_results = run_mva_round(model, tokenizer, merged_state, val_ds, squad_holdout,
                                     task_num=task_num, mva_round_num=mva_round)
        results["rounds"][f"round_{task_num}_mva_{mva_round}"] = mva_results
        # Update merged_state for next round
        if "merged_state" in mva_results:
            merged_state = mva_results["merged_state"]
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # --- Final verdict ---
    print_verdict(results)

    # --- Save ---
    save_results = json.loads(json.dumps(results, default=str))
    with open(OUTPUT_DIR / "v15_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR}/v15_results.json")


def print_verdict(results):
    print(f"\n{'='*70}")
    print("V15 VERDICT — SLAO + MVA 3-ROUND LOOP")
    print(f"{'='*70}")

    rounds = results.get("rounds", {})

    # --- Full trajectory table ---
    print(f"\nFull trajectory (3 SLAO + 3 MVA rounds):")
    print(f"{'Round':<18} {'A (med)':<10} {'B (code)':<10} {'C (creat)':<10} {'pass^5':<10} {'cert mean':<10} {'thresh':<10} {'prec':<8}")
    print("-" * 92)

    # Domain rounds 1-3
    for i in range(1, 4):
        r = rounds.get(f"round_{i}", {})
        if not r: continue
        ppl = r.get("ppl", {})
        pass5 = r.get("pass5", {})
        pk_val = pass5.get("pass_k", 0) if isinstance(pass5, dict) else 0
        label = f"R{i} ({r.get('domain', '?')})"
        print(f"{label:<18} {ppl.get('A',0):<10.2f} {ppl.get('B',0):<10.2f} "
              f"{ppl.get('C',0):<10.2f} {pk_val:<10.3f} {'-':<10} {'-':<10} {'-':<8}")

    # MVA rounds 4-6
    mva_pass5_trajectory = []
    mva_deltas = []
    mva_thresholds = []
    mva_cert_means = []
    mva_precisions = []
    for mva_round in range(1, N_MVA_ROUNDS + 1):
        task_num = 3 + mva_round
        key = f"round_{task_num}_mva_{mva_round}"
        r = rounds.get(key, {})
        if not r or not r.get("post"): continue
        post_ppl = r["post"]["ppl"]
        post_pass5 = r["post"]["pass5"]["pass_k"]
        val = r.get("validation", {})
        cert_dist = val.get("certainty_distribution", {})
        thresh = val.get("threshold", 0)
        prec = val.get("precision", 0)
        label = f"R{task_num} (MVA{mva_round})"
        print(f"{label:<18} {post_ppl.get('A',0):<10.2f} {post_ppl.get('B',0):<10.2f} "
              f"{post_ppl.get('C',0):<10.2f} {post_pass5:<10.3f} "
              f"{cert_dist.get('mean',0):<10.2f} {thresh:<10.2f} {100*prec:<8.1f}")
        mva_pass5_trajectory.append(post_pass5)
        mva_deltas.append(r["deltas"]["pass5"])
        mva_thresholds.append(thresh)
        mva_cert_means.append(cert_dist.get("mean", 0))
        mva_precisions.append(prec)

    # --- MVA trajectory analysis ---
    if len(mva_pass5_trajectory) >= 2:
        print(f"\n{'='*70}")
        print("MVA TRAJECTORY ANALYSIS")
        print(f"{'='*70}")

        print(f"\npass^5 trajectory across MVA rounds:")
        for i, pk in enumerate(mva_pass5_trajectory, 1):
            print(f"  MVA round {i}: {pk:.3f}  (delta this round: {mva_deltas[i-1]:+.3f})")

        # Does it trend up?
        if len(mva_pass5_trajectory) >= 2:
            first = mva_pass5_trajectory[0]
            last = mva_pass5_trajectory[-1]
            total_gain = last - first
            print(f"\n  Total pass^5 gain across {len(mva_pass5_trajectory)} MVA rounds: {total_gain:+.3f}")
            print(f"  Average per-round: {total_gain/len(mva_pass5_trajectory):+.3f}")

            # Monotonic check
            monotonic_up = all(mva_pass5_trajectory[i] >= mva_pass5_trajectory[i-1]
                               for i in range(1, len(mva_pass5_trajectory)))
            bounced = any(mva_deltas[i] < 0 for i in range(1, len(mva_deltas)))
            if monotonic_up:
                print(f"  Trajectory: MONOTONIC UP (signal — each round improves or holds)")
            elif bounced:
                print(f"  Trajectory: BOUNCED (some rounds regressed — likely noise)")
            else:
                print(f"  Trajectory: MIXED (not monotonic but no regressions)")

        # Certainty distribution shift
        if len(mva_cert_means) >= 2:
            print(f"\nCertainty distribution across MVA rounds:")
            for i, (cm, th) in enumerate(zip(mva_cert_means, mva_thresholds), 1):
                print(f"  Round {i}: mean={cm:.2f}, threshold={th:.2f}")
            cert_shift = mva_cert_means[-1] - mva_cert_means[0]
            print(f"  Shift across rounds: {cert_shift:+.2f}")
            if abs(cert_shift) < 0.5:
                print(f"  → Certainty distribution STABLE across rounds (good — threshold doesn't need re-tuning)")
            else:
                print(f"  → Certainty distribution SHIFTED — adaptive threshold is necessary")

        # Precision trend
        if len(mva_precisions) >= 2:
            print(f"\nValidation precision across MVA rounds:")
            for i, p in enumerate(mva_precisions, 1):
                print(f"  Round {i}: {100*p:.1f}%")
            prec_trend = mva_precisions[-1] - mva_precisions[0]
            if prec_trend > 0.05:
                print(f"  → Precision IMPROVING (model getting better at self-assessment)")
            elif prec_trend < -0.05:
                print(f"  → Precision DEGRADING (filter becoming less reliable)")
            else:
                print(f"  → Precision STABLE (filter reliability holds across rounds)")

    # --- Final verdict ---
    print(f"\n{'='*70}")
    print("FRAMING: Self-improvement via reduced-error filtering")
    print("  MVA's certainty gate filters out wrong answers (precision shown above)")
    print("  Some wrong answers leak through — this is 'fewer mistakes in training,")
    print("  not 'verified-correct self-generated data.' That's the claim that survives.")
    print(f"{'='*70}")

    print(f"\nVERDICT:")
    if len(mva_pass5_trajectory) >= 2:
        total_gain = mva_pass5_trajectory[-1] - mva_pass5_trajectory[0]
        if total_gain > 0.10 and monotonic_up:
            print(f"  COMPOUNDS — pass^5 gained {total_gain:+.3f} over {len(mva_pass5_trajectory)} rounds, monotonic")
            print(f"  → Self-improvement via reduced-error filtering compounds on SLAO")
            print(f"  → Build full Architecture C (10 rounds) — the signal is real")
        elif total_gain > 0.05:
            print(f"  TRENDS UP — pass^5 gained {total_gain:+.3f} but not perfectly monotonic")
            print(f"  → Signal is real but noisy — Architecture C worth trying with more rounds")
        elif total_gain > 0:
            print(f"  MARGINAL — pass^5 gained only {total_gain:+.3f} over {len(mva_pass5_trajectory)} rounds")
            print(f"  → Gain doesn't compound — single-round +0.07 was mostly noise")
            print(f"  → Reconsider Architecture C; threshold-tuning or different signal needed")
        else:
            print(f"  DOES NOT COMPOUND — pass^5 declined or flat ({total_gain:+.3f})")
            print(f"  → MVA + SLAO doesn't compound; fundamental integration issue")
    else:
        print(f"  INSUFFICIENT DATA — less than 2 MVA rounds completed")


if __name__ == "__main__":
    main()
