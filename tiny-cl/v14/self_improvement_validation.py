"""
Self-Improvement Validation — Seed 123 confirmation run
=======================================================

One file, one cell, one run. Tests seed 123 to confirm v5's seed 42 result.

v5 seed 42 result (already done):
- MVA pass^5 = 0.710 vs baseline 0.530 (+18pp) and naive 0.590 (+12pp)

This run: seed 123. If MVA still beats naive by >5pp, the result is robust.

Usage on Kaggle (T4 GPU):
    !python self_improvement_validation.py

Runtime: ~90 min on T4
"""

# ============================================================================
# CONFIG
# ============================================================================

import os
from pathlib import Path

MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2.5-350M")
OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("/home/z/my-project/download")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment sizes
N_PHASE1_PAIRS = 100
N_PHASE2_TRAIN = 200
N_PHASE2_HOLDOUT = 100
N_CONSENSUS_SAMPLES = 4
PASS_K = 5

# Validation gate — certainty (strongest signal, r=0.341 in seed 42)
VALIDATION_SIGNAL = "certainty"
CONSENSUS_THRESHOLD = 0.80
CERTAINTY_THRESHOLD = 17.0

# LoRA config — corrected per official LFM2 docs
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj",
                       "in_proj", "w1", "w2", "w3"]
LORA_LR = 2e-4
LORA_EPOCHS = 3
LORA_BATCH_SIZE = 2
LORA_GRAD_ACCUM_STEPS = 8
LORA_WARMUP_RATIO = 0.1
LORA_LR_SCHEDULER = "cosine"
LORA_WEIGHT_DECAY = 0.1
LORA_ADAM_BETA2 = 0.95
LORA_MAX_GRAD_NORM = 1.0

GEN_MAX_NEW_TOKENS = 60
QUIZ_MAX_NEW_TOKENS = 15
GEN_TEMPERATURE = 0.7
CAUSAL_TOP_K_TOKENS = 20

# The one seed we're confirming
SEED = 123

# ============================================================================
# DEPENDENCY INSTALLATION
# ============================================================================

import subprocess, sys

def _ensure_deps():
    """Install required packages if missing."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            print(f"  Upgrading transformers {transformers.__version__} -> 5.0+ for LFM2...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                                   "transformers>=5.0.0", "packaging"])
            import importlib; importlib.reload(transformers)
    except ImportError:
        missing.extend(["transformers>=5.0.0", "packaging"])
    for pkg in ["peft", "trl", "scipy", "accelerate", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

_ensure_deps()

# ============================================================================
# IMPORTS
# ============================================================================

import json, math, random, re, time, warnings, string
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from trl import SFTTrainer, SFTConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

try:
    from scipy.stats import pointbiserialr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore")

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# UTILITIES
# ============================================================================

def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = " ".join(s.split())
    return s

def ground_truth_check(answer: str, gold: str) -> bool:
    """Check if answer contains gold answer tokens (>=50% overlap)."""
    norm_answer = normalize_answer(answer)
    norm_gold = normalize_answer(gold)
    if not norm_gold:
        return False
    gold_tokens = set(norm_gold.split())
    answer_tokens = set(norm_answer.split())
    if not gold_tokens:
        return False
    overlap = len(gold_tokens & answer_tokens) / len(gold_tokens)
    return overlap >= 0.5

def build_prompt(paragraph: str, question: str) -> str:
    return (f"Passage: {paragraph}\n\n"
            f"Question: {question}\n\n"
            f"Answer (brief, factual):")

def generate(model, tokenizer, prompt: str, max_new_tokens: int = GEN_MAX_NEW_TOKENS,
             temperature: float = GEN_TEMPERATURE, do_sample: bool = True) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=do_sample, top_p=0.95, pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_tokenizer():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  Architecture: {model.config.architectures}")
    print(f"  Vocab: {model.config.vocab_size}, Hidden: {model.config.hidden_size}")
    if hasattr(model.config, 'layer_types'):
        n_attn = sum(1 for t in model.config.layer_types if t == 'full_attention')
        n_conv = sum(1 for t in model.config.layer_types if t == 'conv')
        print(f"  Layers: {n_attn} attention + {n_conv} conv")
    # Verify LoRA targets exist
    target_names = {name.split('.')[-1] for name, _ in model.named_modules()}
    missing = [m for m in LORA_TARGET_MODULES if m not in target_names]
    if missing:
        print(f"  WARNING: LoRA targets not found: {missing}")
    else:
        print(f"  LoRA targets verified: {LORA_TARGET_MODULES}")
    return model, tokenizer

# ============================================================================
# SIGNAL COMPUTATIONS
# ============================================================================

def compute_certainty(model, tokenizer, paragraph, question, answer):
    """INTUITOR self-certainty: KL(U || p_theta) averaged over answer tokens."""
    try:
        prompt = build_prompt(paragraph, question)
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024, add_special_tokens=True)
        answer_text = " " + answer
        answer_ids = tokenizer(answer_text, return_tensors="pt",
                               add_special_tokens=False)["input_ids"][0]
        full_ids = torch.cat([prompt_ids["input_ids"][0], answer_ids], dim=0).unsqueeze(0)
        inputs = {"input_ids": full_ids.to(model.device),
                  "attention_mask": torch.ones_like(full_ids).to(model.device)}
        answer_start = prompt_ids["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model(**inputs)
        answer_logits = outputs.logits[0, answer_start - 1:-1, :]
        if answer_logits.shape[0] == 0:
            return 0.0
        log_probs = F.log_softmax(answer_logits, dim=-1)
        vocab_size = log_probs.shape[-1]
        kl_per_position = -math.log(vocab_size) - log_probs.mean(dim=-1)
        return kl_per_position.mean().item()
    except Exception as e:
        print(f"    [certainty error] {e}")
        return 0.0

def compute_consensus(model, tokenizer, paragraph, question, original_answer):
    """TTRL-style: generate N samples, return agreement fraction with most common."""
    all_answers = [normalize_answer(original_answer)]
    for _ in range(N_CONSENSUS_SAMPLES):
        ans = generate(model, tokenizer, build_prompt(paragraph, question),
                       max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=0.7, do_sample=True)
        all_answers.append(normalize_answer(ans))
    counter = Counter(all_answers)
    return counter.most_common(1)[0][1] / len(all_answers)

def compute_quiz_score(model, tokenizer, paragraph, question, answer):
    """Self-quiz: 3 YES/UNCLEAR questions. Returns fraction non-UNCLEAR."""
    quiz_prompts = [
        f"Passage: {paragraph}\nQuestion: {question}\nAnswer: {answer}\n\n"
        f"Does the answer contain a specific factual claim? Answer YES or UNCLEAR.\nAnswer:",
        f"Passage: {paragraph}\nQuestion: {question}\nAnswer: {answer}\n\n"
        f"Is the answer complete? Answer YES or UNCLEAR.\nAnswer:",
        f"Passage: {paragraph}\nQuestion: {question}\nAnswer: {answer}\n\n"
        f"Does the answer contain unsupported claims? Answer NO or UNCLEAR.\nAnswer:",
    ]
    non_unclear = 0
    for qp in quiz_prompts:
        response = generate(model, tokenizer, qp, max_new_tokens=QUIZ_MAX_NEW_TOKENS,
                            temperature=0.3, do_sample=True)
        upper = response.upper().strip()
        if not ("UNCLEAR" in upper and "YES" not in upper and "NO" not in upper):
            non_unclear += 1
    return non_unclear / 3

# ============================================================================
# SQUAD DATASET
# ============================================================================

def load_squad_pairs(n_questions):
    """Load SQuAD validation set, sample diverse passages."""
    from datasets import load_dataset
    print(f"  Loading SQuAD (target: {n_questions} questions)...")
    try:
        squad = load_dataset("rajpurkar/squad", split="validation")
    except Exception as e:
        print(f"    SQuAD load failed: {e}")
        return None

    passages = {}
    for ex in squad:
        ctx = ex["context"]
        if ctx not in passages:
            passages[ctx] = []
        answers = ex["answers"]["text"]
        if answers:
            passages[ctx].append({"q": ex["question"], "a": answers[0], "paragraph": ctx})

    print(f"    SQuAD: {len(passages)} passages, {sum(len(v) for v in passages.values())} questions")

    passage_list = list(passages.keys())
    random.shuffle(passage_list)
    pairs = []
    for ctx in passage_list:
        qs = passages[ctx]
        random.shuffle(qs)
        for q in qs[:2]:
            pairs.append((q["paragraph"], q["q"], q["a"]))
        if len(pairs) >= n_questions:
            break
    pairs = pairs[:n_questions]
    print(f"    Sampled {len(pairs)} questions from {len(set(p[0] for p in pairs))} passages")
    return pairs

# ============================================================================
# PHASE 1: SIGNAL VALIDATION
# ============================================================================

def run_phase1(model, tokenizer):
    print("\n" + "=" * 70)
    print("PHASE 1: SIGNAL VALIDATION")
    print("=" * 70)

    all_pairs = load_squad_pairs(N_PHASE1_PAIRS + 50)  # extra for safety
    if all_pairs is None:
        print("  FATAL: SQuAD unavailable. Cannot proceed.")
        return None
    random.shuffle(all_pairs)
    phase1_pairs = all_pairs[:N_PHASE1_PAIRS]

    results = []
    t_start = time.time()
    for i, (paragraph, question, gold) in enumerate(phase1_pairs):
        answer = generate(model, tokenizer, build_prompt(paragraph, question),
                          max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=0.7, do_sample=True)
        certainty = compute_certainty(model, tokenizer, paragraph, question, answer)
        consensus = compute_consensus(model, tokenizer, paragraph, question, answer)
        quiz = compute_quiz_score(model, tokenizer, paragraph, question, answer)
        correct = ground_truth_check(answer, gold)
        results.append({
            "question": question, "gold": gold, "answer": answer,
            "certainty": certainty, "consensus": consensus, "quiz_score": quiz,
            "correct": correct,
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{N_PHASE1_PAIRS}] {elapsed:.0f}s elapsed, "
                  f"~{elapsed*(N_PHASE1_PAIRS-i-1)/(i+1):.0f}s remaining")

    correct_arr = np.array([1 if r["correct"] else 0 for r in results])
    certainty_arr = np.array([r["certainty"] for r in results])
    consensus_arr = np.array([r["consensus"] for r in results])
    quiz_arr = np.array([r["quiz_score"] for r in results])

    certainty_z = (certainty_arr - certainty_arr.mean()) / certainty_arr.std() if certainty_arr.std() > 0 else certainty_arr

    correlations = {}
    for name, arr in [("certainty", certainty_z), ("consensus", consensus_arr), ("quiz_score", quiz_arr)]:
        if arr.std() < 1e-6 or correct_arr.std() < 1e-6:
            correlations[name] = {"correlation": 0.0, "p_value": None}
        elif SCIPY_AVAILABLE:
            r, p = pointbiserialr(correct_arr, arr)
            correlations[name] = {"correlation": float(r), "p_value": float(p)}
        else:
            r = np.corrcoef(correct_arr, arr)[0, 1]
            correlations[name] = {"correlation": float(r) if not np.isnan(r) else 0.0, "p_value": None}

    accuracy = float(correct_arr.mean())
    best = max(correlations.items(), key=lambda x: abs(x[1]["correlation"]))
    best_corr = abs(best[1]["correlation"])

    if best_corr > 0.40:
        decision = "PROCEED"
    elif best_corr < 0.20:
        decision = "FALLBACK"
    else:
        decision = "PROCEED_WITH_CAUTION"

    print(f"\n  Pairs: {len(results)}, Accuracy: {accuracy*100:.1f}%")
    print(f"  Correlations with correctness:")
    for name, stats in correlations.items():
        p_str = f"p={stats['p_value']:.4f}" if stats['p_value'] is not None else "p=N/A"
        print(f"    {name:<12} r={stats['correlation']:+.4f}  ({p_str})")
    print(f"  Decision: {decision} (best={best[0]}, r={best_corr:.3f})")

    q1_output = {
        "n_pairs": len(results), "accuracy": accuracy, "seed": SEED,
        "correlations": correlations, "decision": decision,
        "per_pair_results": results,
    }
    with open(OUTPUT_DIR / "results_q1.json", "w") as f:
        json.dump(q1_output, f, indent=2)

    return q1_output

# ============================================================================
# PHASE 2: MVA SELF-IMPROVEMENT TEST
# ============================================================================

def run_validation_on_train_set(model, tokenizer, train_pairs):
    """Generate answers + validate using selected signal (certainty or consensus)."""
    print(f"\n  Validation gate: {VALIDATION_SIGNAL} >= "
          f"{CERTAINTY_THRESHOLD if VALIDATION_SIGNAL == 'certainty' else CONSENSUS_THRESHOLD}")
    results = []
    t_start = time.time()
    for i, (paragraph, question, gold) in enumerate(train_pairs):
        primary = generate(model, tokenizer, build_prompt(paragraph, question),
                           max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=0.7, do_sample=True)
        consensus = compute_consensus(model, tokenizer, paragraph, question, primary)
        certainty = compute_certainty(model, tokenizer, paragraph, question, primary)
        quiz = compute_quiz_score(model, tokenizer, paragraph, question, primary)
        if VALIDATION_SIGNAL == "certainty":
            validated = certainty >= CERTAINTY_THRESHOLD
        else:
            validated = consensus >= CONSENSUS_THRESHOLD
        correct = ground_truth_check(primary, gold)
        results.append({
            "paragraph": paragraph, "question": question, "gold": gold,
            "answer": primary, "consensus_score": consensus, "certainty": certainty,
            "quiz_score": quiz, "validated": validated, "correct": correct,
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            sig_val = f"{certainty:.2f}" if VALIDATION_SIGNAL == "certainty" else f"{consensus:.2f}"
            print(f"    [{i+1}/{len(train_pairs)}] {VALIDATION_SIGNAL}={sig_val} "
                  f"validated={validated} correct={correct}  "
                  f"(~{elapsed*(len(train_pairs)-i-1)/(i+1):.0f}s remaining)")
    return results

def prepare_training_data(quiz_results, condition):
    """Build training pairs: naive=all, mva=validated, oracle=full-sentence gold."""
    pairs = []
    if condition == "naive":
        for r in quiz_results:
            pairs.append((r["question"], r["answer"], r["paragraph"]))
    elif condition == "mva":
        for r in quiz_results:
            if r["validated"]:
                pairs.append((r["question"], r["answer"], r["paragraph"]))
    elif condition == "oracle":
        for r in quiz_results:
            # Full sentence to preserve fluency (v3 fix — bare gold broke oracle)
            pairs.append((r["question"], f"The answer is {r['gold']}.", r["paragraph"]))
    return pairs

def lora_finetune(base_model, tokenizer, training_pairs, condition_name):
    """LoRA fine-tune with corrected LFM2 config. Tries TRL, falls back to manual."""
    if not PEFT_AVAILABLE or len(training_pairs) == 0:
        return base_model

    print(f"  Fine-tuning ({condition_name}) on {len(training_pairs)} pairs, {LORA_EPOCHS} epochs...")
    print(f"    r={LORA_R}, alpha={LORA_ALPHA}, lr={LORA_LR}, targets={LORA_TARGET_MODULES}")
    print(f"    AdamW beta2={LORA_ADAM_BETA2}, wd={LORA_WEIGHT_DECAY}, warmup={LORA_WARMUP_RATIO}, {LORA_LR_SCHEDULER}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, bias="none", target_modules=LORA_TARGET_MODULES,
    )

    # Try TRL SFTTrainer first
    if TRL_AVAILABLE:
        try:
            return _finetune_trl(base_model, tokenizer, training_pairs, lora_config, condition_name)
        except Exception as e:
            print(f"    TRL failed ({str(e)[:100]}...), using manual loop")

    return _finetune_manual(base_model, tokenizer, training_pairs, lora_config)

def _finetune_trl(base_model, tokenizer, training_pairs, lora_config, condition_name):
    import inspect
    from datasets import Dataset

    text_dataset = [{"text": build_prompt(p, q) + " " + a + tokenizer.eos_token}
                    for q, a, p in training_pairs]
    train_dataset = Dataset.from_list(text_dataset)

    output_dir = OUTPUT_DIR / f"lora_{condition_name}_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs = dict(
        output_dir=str(output_dir), num_train_epochs=LORA_EPOCHS,
        per_device_train_batch_size=LORA_BATCH_SIZE,
        gradient_accumulation_steps=LORA_GRAD_ACCUM_STEPS,
        learning_rate=LORA_LR, lr_scheduler_type=LORA_LR_SCHEDULER,
        warmup_ratio=LORA_WARMUP_RATIO, weight_decay=LORA_WEIGHT_DECAY,
        max_grad_norm=LORA_MAX_GRAD_NORM, adam_beta2=LORA_ADAM_BETA2,
        bf16=True, logging_steps=1, save_strategy="no", report_to="none",
        seed=SEED, packing=False,
    )
    sig = inspect.signature(SFTConfig.__init__)
    params = set(sig.parameters.keys())
    if "max_length" in params:
        config_kwargs["max_length"] = 512
    elif "max_seq_length" in params:
        config_kwargs["max_seq_length"] = 512
    if "dataset_text_field" in params:
        config_kwargs["dataset_text_field"] = "text"

    sft_config = SFTConfig(**config_kwargs)
    try:
        trainer = SFTTrainer(model=base_model, args=sft_config, train_dataset=train_dataset,
                             peft_config=lora_config, processing_class=tokenizer)
    except TypeError:
        base_model = get_peft_model(base_model, lora_config)
        trainer = SFTTrainer(model=base_model, args=sft_config, train_dataset=train_dataset,
                             processing_class=tokenizer)

    t_start = time.time()
    trainer.train()
    print(f"    TRL training: {time.time()-t_start:.0f}s")
    trainer.model.eval()
    return trainer.model

def _finetune_manual(base_model, tokenizer, training_pairs, lora_config):
    """Manual loop with corrected optimizer (beta2=0.95, wd=0.1, cosine, warmup)."""
    from transformers import get_cosine_schedule_with_warmup
    model = get_peft_model(base_model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LORA_LR,
        betas=(0.9, LORA_ADAM_BETA2), weight_decay=LORA_WEIGHT_DECAY,
    )
    total_steps = max(1, (len(training_pairs) // (LORA_BATCH_SIZE * LORA_GRAD_ACCUM_STEPS)) * LORA_EPOCHS)
    warmup_steps = max(1, int(total_steps * LORA_WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    total_opt_steps = 0
    for epoch in range(LORA_EPOCHS):
        random.shuffle(training_pairs)
        accum = 0
        for batch_start in range(0, len(training_pairs), LORA_BATCH_SIZE):
            batch = training_pairs[batch_start:batch_start + LORA_BATCH_SIZE]
            for question, target, paragraph in batch:
                prompt = build_prompt(paragraph, question)
                prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                                       max_length=512, add_special_tokens=True)
                answer_text = " " + target + tokenizer.eos_token
                answer_ids = tokenizer(answer_text, return_tensors="pt",
                                       add_special_tokens=False)["input_ids"][0]
                full_ids = torch.cat([prompt_ids["input_ids"][0], answer_ids], dim=0).unsqueeze(0)
                inputs = {"input_ids": full_ids.to(model.device),
                          "attention_mask": torch.ones_like(full_ids).to(model.device)}
                prompt_len = prompt_ids["input_ids"].shape[1]
                outputs = model(**inputs)
                logits = outputs.logits[:, prompt_len-1:-1, :]
                labels = inputs["input_ids"][:, prompt_len:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id else -100,
                ) / LORA_GRAD_ACCUM_STEPS
                loss.backward()
            accum += 1
            if accum >= LORA_GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), LORA_MAX_GRAD_NORM)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                total_opt_steps += 1; accum = 0
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), LORA_MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_opt_steps += 1
        print(f"    Epoch {epoch+1}/{LORA_EPOCHS} done ({total_opt_steps} opt steps)")
    model.eval()
    return model

class TokenMaskProcessor(LogitsProcessor):
    def __init__(self, mask_ids):
        self.mask_ids = torch.tensor(list(mask_ids), dtype=torch.long)
    def __call__(self, input_ids, scores):
        self.mask_ids = self.mask_ids.to(scores.device)
        scores[:, self.mask_ids] = float("-inf")
        return scores

def measure_pass_k(model, tokenizer, holdout_pairs, k, condition_name, mask_token_ids=None):
    """Measure pass^k: P(all k samples correct). Also pass@k: P(>=1 correct)."""
    print(f"  Measuring pass^{k} on {len(holdout_pairs)} pairs ({condition_name})...")
    mask_proc = TokenMaskProcessor(mask_token_ids) if mask_token_ids else None
    pass_k_results, pass_at_k_results = [], []
    per_question = []

    for i, (paragraph, question, gold) in enumerate(holdout_pairs):
        samples = []
        for _ in range(k):
            prompt = build_prompt(paragraph, question)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024).to(model.device)
            with torch.no_grad():
                kwargs = dict(**inputs, max_new_tokens=GEN_MAX_NEW_TOKENS,
                              temperature=0.7, do_sample=True, top_p=0.95,
                              pad_token_id=tokenizer.pad_token_id)
                if mask_proc:
                    kwargs["logits_processor"] = LogitsProcessorList([mask_proc])
                outputs = model.generate(**kwargs)
            ans = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            samples.append(ans)

        correct_flags = [ground_truth_check(s, gold) for s in samples]
        n_correct = sum(correct_flags)
        pass_k_results.append(1.0 if n_correct == k else 0.0)
        pass_at_k_results.append(1.0 if n_correct > 0 else 0.0)
        per_question.append({"question": question, "gold": gold, "samples": samples,
                             "correct_flags": correct_flags, "n_correct": n_correct,
                             "pass_k": 1.0 if n_correct == k else 0.0,
                             "pass_at_k": 1.0 if n_correct > 0 else 0.0})
        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(holdout_pairs)}] pass@{k} so far: {np.mean(pass_at_k_results):.3f}")

    mean_pk = float(np.mean(pass_k_results))
    mean_pak = float(np.mean(pass_at_k_results))
    print(f"  {condition_name}: pass^{k}={mean_pk:.3f}, pass@{k}={mean_pak:.3f}")
    return {"condition": condition_name, "mean_pass_k": mean_pk,
            "mean_pass_at_k": mean_pak, "n_questions": len(holdout_pairs), "k": k,
            "per_question": per_question}

def collect_output_tokens(model, tokenizer, pairs, n_samples=2):
    counter = Counter()
    for paragraph, question, gold in pairs:
        for _ in range(n_samples):
            ans = generate(model, tokenizer, build_prompt(paragraph, question),
                           max_new_tokens=GEN_MAX_NEW_TOKENS, temperature=0.7, do_sample=True)
            counter.update(tokenizer.encode(ans, add_special_tokens=False))
    return counter

def compute_distribution_shift(baseline_tokens, mva_tokens, tokenizer):
    """KL divergence + top shifted tokens (filtering prompt-template tokens)."""
    all_tokens = set(baseline_tokens.keys()) | set(mva_tokens.keys())
    total_b = sum(baseline_tokens.values())
    total_m = sum(mva_tokens.values())
    kl = 0.0
    shifts = []
    for tid in all_tokens:
        p_b = max(baseline_tokens.get(tid, 0) / total_b, 1e-10)
        p_m = max(mva_tokens.get(tid, 0) / total_m, 1e-10)
        kl += p_m * math.log(p_m / p_b)
        shifts.append((tid, p_m - p_b, p_m, p_b))
    shifts.sort(key=lambda x: x[1], reverse=True)

    # Filter prompt-template tokens and pure punctuation
    PROMPT_TOKENS = {"question", "answer", "passage", "brief", "fact", "ual", "factual",
                     "what", "which", "who", "when", "where", "how", "why",
                     ":", "):", "(", ")", "?"}
    PUNCT = set(string.punctuation)
    def is_junk(s):
        if not s or s.lower() in PROMPT_TOKENS:
            return True
        if all(c in PUNCT for c in s):
            return True
        if len(s) <= 2 and not s.isalnum():
            return True
        return False

    top = []
    for tid, shift, p_m, p_b in shifts:
        if p_m < 0.001:
            continue
        s = tokenizer.decode([tid]).strip()
        if is_junk(s) or s in [tokenizer.eos_token, tokenizer.pad_token]:
            continue
        top.append({"token_id": tid, "token_str": s, "shift": shift,
                    "p_mva": p_m, "p_baseline": p_b})
        if len(top) >= CAUSAL_TOP_K_TOKENS:
            break
    return float(kl), top, [t["token_id"] for t in top]

def run_phase2(model, tokenizer, q1_results):
    print("\n" + "=" * 70)
    print("PHASE 2: MVA SELF-IMPROVEMENT TEST")
    print("=" * 70)

    if not PEFT_AVAILABLE:
        print("  [SKIP] peft unavailable")
        return None

    all_pairs = load_squad_pairs(N_PHASE2_TRAIN + N_PHASE2_HOLDOUT)
    if all_pairs is None:
        return None
    random.shuffle(all_pairs)
    train_pairs = all_pairs[:N_PHASE2_TRAIN]
    holdout_pairs = all_pairs[N_PHASE2_TRAIN:N_PHASE2_TRAIN + N_PHASE2_HOLDOUT]
    print(f"  Train: {len(train_pairs)}, Holdout: {len(holdout_pairs)}")

    # Step 1: Validate training set
    print("\n  STEP 1: Validation on training set")
    quiz_results = run_validation_on_train_set(model, tokenizer, train_pairs)
    n_val = sum(1 for r in quiz_results if r["validated"])
    n_correct = sum(1 for r in quiz_results if r["correct"])
    n_val_correct = sum(1 for r in quiz_results if r["validated"] and r["correct"])
    gate_desc = f"{VALIDATION_SIGNAL}>={CERTAINTY_THRESHOLD if VALIDATION_SIGNAL == 'certainty' else CONSENSUS_THRESHOLD}"
    print(f"\n  Validated ({gate_desc}): {n_val}/{len(quiz_results)} ({100*n_val/len(quiz_results):.1f}%)")
    print(f"  Correct (gold): {n_correct}/{len(quiz_results)} ({100*n_correct/len(quiz_results):.1f}%)")
    print(f"  Validation precision: {n_val_correct}/{n_val} = {100*n_val_correct/max(n_val,1):.1f}%")

    naive_train = prepare_training_data(quiz_results, "naive")
    mva_train = prepare_training_data(quiz_results, "mva")
    oracle_train = prepare_training_data(quiz_results, "oracle")
    print(f"\n  Training sizes: naive={len(naive_train)}, mva={len(mva_train)}, oracle={len(oracle_train)}")

    # Step 2: Baseline pass^5
    print("\n  STEP 2: Baseline pass^5")
    baseline = measure_pass_k(model, tokenizer, holdout_pairs, PASS_K, "baseline")

    # Step 3: Baseline tokens for distribution shift
    print("\n  STEP 3: Baseline tokens")
    baseline_tokens = collect_output_tokens(model, tokenizer, holdout_pairs, n_samples=2)

    results = {
        "validation_stats": {
            "n_total": len(quiz_results), "n_validated": n_val, "n_correct": n_correct,
            "n_validated_correct": n_val_correct,
            "validation_rate": n_val / len(quiz_results),
            "accuracy": n_correct / len(quiz_results),
            "validation_precision": n_val_correct / max(n_val, 1),
            "validation_signal": VALIDATION_SIGNAL,
            "certainty_threshold": CERTAINTY_THRESHOLD if VALIDATION_SIGNAL == "certainty" else None,
            "consensus_threshold": CONSENSUS_THRESHOLD if VALIDATION_SIGNAL == "consensus" else None,
        },
        "baseline": baseline, "conditions": {}, "seed": SEED,
    }

    # Step 4: Fine-tune + evaluate each condition
    for cond_name, cond_train in [("naive", naive_train), ("mva", mva_train), ("oracle", oracle_train)]:
        print(f"\n  STEP 4: Fine-tuning '{cond_name}'")
        if len(cond_train) == 0:
            print(f"    [SKIP] no training pairs")
            results["conditions"][cond_name] = {"n_train_pairs": 0,
                "mean_pass_k": baseline["mean_pass_k"], "mean_pass_at_k": baseline["mean_pass_at_k"],
                "skipped": True}
            continue

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        model, _ = load_model_and_tokenizer()
        model = lora_finetune(model, tokenizer, cond_train, cond_name)

        cond_result = measure_pass_k(model, tokenizer, holdout_pairs, PASS_K, cond_name)
        results["conditions"][cond_name] = {
            "n_train_pairs": len(cond_train),
            "mean_pass_k": cond_result["mean_pass_k"],
            "mean_pass_at_k": cond_result["mean_pass_at_k"],
            "per_question": cond_result["per_question"],
        }

        # Distribution shift + causal link on MVA only
        if cond_name == "mva":
            print(f"\n  STEP 5: Distribution shift (MVA vs baseline)")
            mva_tokens = collect_output_tokens(model, tokenizer, holdout_pairs, n_samples=2)
            kl, top_shifted, top_ids = compute_distribution_shift(baseline_tokens, mva_tokens, tokenizer)
            results["distribution_shift"] = {"kl_divergence": kl, "top_shifted_tokens": top_shifted}
            print(f"    KL = {kl:.4f}")
            print(f"    Top tokens: {[t['token_str'] for t in top_shifted[:10]]}")

            print(f"\n  STEP 6: Causal link (masking top-{len(top_ids)} tokens)")
            if top_ids:
                masked = measure_pass_k(model, tokenizer, holdout_pairs, PASS_K,
                                        "mva_masked", mask_token_ids=top_ids)
                results["causal_link"] = {
                    "mva_pass_k_unmasked": cond_result["mean_pass_k"],
                    "mva_pass_k_masked": masked["mean_pass_k"],
                    "baseline_pass_k": baseline["mean_pass_k"],
                    "drop": cond_result["mean_pass_k"] - masked["mean_pass_k"],
                    "masked_tokens": [t["token_str"] for t in top_shifted],
                }

    with open(OUTPUT_DIR / "results_q2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results

# ============================================================================
# FINAL VERDICT
# ============================================================================

def print_final_verdict(q1, q2):
    print("\n" + "=" * 70)
    print(f"FINAL VERDICT (SEED={SEED})")
    print("=" * 70)
    lines = [f"FINAL VERDICT (SEED={SEED})", "=" * 70, ""]

    # Q1
    lines.append("Q1: Does self-certainty have signal at 350M?")
    best = max(q1["correlations"].items(), key=lambda x: abs(x[1]["correlation"]))
    best_corr = abs(best[1]["correlation"])
    lines.append(f"  Best signal: {best[0]} (r={best_corr:+.4f})")
    lines.append(f"  Accuracy: {q1['accuracy']*100:.1f}%")
    lines.append(f"  Decision: {q1['decision']}")
    lines.append("")

    if q2 is None:
        lines.append("Q2: SKIPPED")
        print("\n".join(lines))
        with open(OUTPUT_DIR / "decision.txt", "w") as f:
            f.write("\n".join(lines))
        return

    # Validation gate
    vs = q2.get("validation_stats", {})
    sig = vs.get("validation_signal", "certainty")
    thresh = vs.get("certainty_threshold") or vs.get("consensus_threshold")
    lines.append(f"Q2: Validation gate ({sig} >= {thresh})")
    lines.append(f"  Validated: {vs.get('n_validated',0)}/{vs.get('n_total',0)} "
                 f"({vs.get('validation_rate',0)*100:.1f}%)")
    lines.append(f"  Precision: {vs.get('n_validated_correct',0)}/{vs.get('n_validated',0)} "
                 f"= {vs.get('validation_precision',0)*100:.1f}%")
    lines.append("")

    # pass^5
    baseline_pk = q2["baseline"]["mean_pass_k"]
    lines.append(f"(a) pass^{PASS_K}:")
    lines.append(f"    Baseline: {baseline_pk:.3f}")
    for cond in ["naive", "mva", "oracle"]:
        if cond in q2["conditions"]:
            c = q2["conditions"][cond]
            if not c.get("skipped"):
                lines.append(f"    {cond:<8}: {c['mean_pass_k']:.3f} ({c['mean_pass_k']-baseline_pk:+.3f})")
    mva_pk = q2["conditions"].get("mva", {}).get("mean_pass_k", baseline_pk)
    naive_pk = q2["conditions"].get("naive", {}).get("mean_pass_k", baseline_pk)
    pass5_yes = mva_pk > baseline_pk and mva_pk > naive_pk
    lines.append(f"    Verdict: {'YES' if pass5_yes else 'NO'} (MVA > baseline AND > naive)")
    lines.append("")

    # Distribution shift
    if "distribution_shift" in q2:
        ds = q2["distribution_shift"]
        lines.append(f"(b) Distribution shift:")
        lines.append(f"    KL(MVA||Baseline) = {ds['kl_divergence']:.4f}")
        lines.append(f"    Top tokens: {[t['token_str'] for t in ds['top_shifted_tokens'][:10]]}")
        lines.append(f"    Verdict: {'YES' if ds['kl_divergence'] > 0.001 else 'NO'}")
        lines.append("")

    # Causal link
    if "causal_link" in q2:
        cl = q2["causal_link"]
        lines.append(f"(c) Causal link (mask top-{CAUSAL_TOP_K_TOKENS} shifted tokens):")
        lines.append(f"    MVA pass^{PASS_K} unmasked: {cl['mva_pass_k_unmasked']:.3f}")
        lines.append(f"    MVA pass^{PASS_K} masked:   {cl['mva_pass_k_masked']:.3f}")
        lines.append(f"    Drop: {cl['drop']:+.3f}  (baseline ref: {cl['baseline_pass_k']:.3f})")
        lines.append(f"    Verdict: {'YES' if cl['drop'] > 0.02 else 'NO'}")
        lines.append("")

    all_yes = pass5_yes
    if "distribution_shift" in q2:
        all_yes = all_yes and q2["distribution_shift"]["kl_divergence"] > 0.001
    if "causal_link" in q2:
        all_yes = all_yes and q2["causal_link"]["drop"] > 0.02

    lines.append("OVERALL:")
    if q1["decision"] in ["PROCEED", "PROCEED_WITH_CAUTION"] and all_yes:
        lines.append("  SHIP MVA — all 3 criteria pass")
    elif q1["decision"] in ["PROCEED", "PROCEED_WITH_CAUTION"]:
        lines.append("  ITERATE — some criteria failed")
    else:
        lines.append("  FALLBACK — signal too weak")

    verdict = "\n".join(lines)
    print(verdict)
    with open(OUTPUT_DIR / "decision.txt", "w") as f:
        f.write(verdict)

# ============================================================================
# MAIN
# ============================================================================

def run_single_seed():
    """Run the full experiment for SEED=123. Returns (q1, q2) results."""
    print("\n" + "#" * 70)
    print(f"# SEED {SEED}")
    print("#" * 70)

    set_seed(SEED)
    model, tokenizer = load_model_and_tokenizer()

    q1 = run_phase1(model, tokenizer)
    if q1 is None:
        return None, None

    q2 = None
    if q1["decision"] in ["PROCEED", "PROCEED_WITH_CAUTION"] and PEFT_AVAILABLE:
        q2 = run_phase2(model, tokenizer, q1)
    else:
        print("\n[SKIP] Phase 2 skipped")

    print_final_verdict(q1, q2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return q1, q2


def main():
    print("=" * 70)
    print(f"SELF-IMPROVEMENT VALIDATION — Liquid LFM2.5-350M (SEED={SEED})")
    print("=" * 70)
    print(f"Model:         {MODEL_NAME}")
    print(f"Output dir:    {OUTPUT_DIR}")
    print(f"Validation:    {VALIDATION_SIGNAL} gate")
    print(f"Phase 1: {N_PHASE1_PAIRS}, Phase 2: {N_PHASE2_TRAIN} train / {N_PHASE2_HOLDOUT} holdout")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}, lr={LORA_LR}, epochs={LORA_EPOCHS}")
    print(f"LoRA targets: {LORA_TARGET_MODULES}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PEFT: {PEFT_AVAILABLE}, TRL: {TRL_AVAILABLE}, SciPy: {SCIPY_AVAILABLE}")
    print(f"\nEstimated runtime: ~90 min on T4")
    print()

    q1, q2 = run_single_seed()

    # Comparison with seed 42 (already known from v5)
    if q2 is not None:
        print("\n" + "=" * 70)
        print("COMPARISON WITH SEED 42 (v5 result)")
        print("=" * 70)
        s42 = {"baseline": 0.530, "naive": 0.590, "mva": 0.710, "oracle": 0.630}
        s123 = {
            "baseline": q2.get("baseline", {}).get("mean_pass_k", 0),
            "naive": q2.get("conditions", {}).get("naive", {}).get("mean_pass_k", 0),
            "mva": q2.get("conditions", {}).get("mva", {}).get("mean_pass_k", 0),
            "oracle": q2.get("conditions", {}).get("oracle", {}).get("mean_pass_k", 0),
        }
        print(f"{'Condition':<12} {'Seed 42':<12} {'Seed 123':<12} {'delta':<10}")
        print("-" * 50)
        for cond in ["baseline", "naive", "mva", "oracle"]:
            d = s123[cond] - s42[cond]
            print(f"{cond:<12} {s42[cond]:<12.3f} {s123[cond]:<12.3f} {d:+.3f}")

        gap_42 = s42["mva"] - s42["naive"]
        gap_123 = s123["mva"] - s123["naive"]
        print(f"\nMVA vs Naive gap: seed 42 = {gap_42:+.3f}, seed 123 = {gap_123:+.3f}")
        if gap_123 > 0.05:
            print("VERDICT: ROBUST — MVA beats naive by >5pp on seed 123")
        elif gap_123 > 0.02:
            print("VERDICT: BORDERLINE — MVA beats naive but margin is thin")
        else:
            print("VERDICT: NOT ROBUST — MVA does not beat naive on seed 123")

    print(f"\nResults: {OUTPUT_DIR}/results_q1.json, results_q2.json, decision.txt")


if __name__ == "__main__":
    main()
