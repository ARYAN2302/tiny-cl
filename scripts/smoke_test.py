"""Smoke test: train on 100 GSM8K examples, eval on 20. ~5 min, ~$0.10.
Tests if the 3-tuple training (full reasoning) fixes the 12% accuracy problem."""
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "build-essential")
    .pip_install(
        "torch==2.5.1", "transformers>=4.45.0", "peft>=0.13.0",
        "datasets>=3.0.0", "accelerate>=1.0.0", "numpy<2",
        "sentencepiece", "protobuf", "scipy", "packaging",
    )
    .env({"PYDEVD_DISABLE_FILE_VALIDATION": "1", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("avr-cl-smoke")
output_vol = modal.Volume.from_name("avr-cl-output", create_if_missing=True)

@app.function(image=image, gpu="A100-40GB", timeout=1800, volumes={"/root/output": output_vol})
def smoke_test():
    import torch, json, re, time, random, math, gc
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader

    MODEL_ID = "Qwen/Qwen3-1.7B"
    SEED = 42
    random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    DEVICE = "cuda"

    print(f"Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="eager")
    lora_config = LoraConfig(r=128, lora_alpha=128, lora_dropout=0.05,
                             target_modules=["q_proj","k_proj","v_proj","o_proj"],
                             bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    # Load 100 GSM8K train examples, 20 test
    print("Loading GSM8K...", flush=True)
    ds_tr = load_dataset("openai/gsm8k", "main", split="train")
    ds_te = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(SEED)
    tr_all = list(ds_tr); rng.shuffle(tr_all)
    te_all = list(ds_te); rng.shuffle(te_all)
    train_examples = tr_all[:100]
    test_examples = te_all[:20]

    def fmt(ex):
        q = ex["question"]
        a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        prompt_q = f"Solve the math problem step by step. End with '#### <final_number>'.\n\n{q}"
        return (prompt_q, a, gold)  # 3-tuple: prompt, full_reasoning, gold

    train_data = [fmt(ex) for ex in train_examples]
    test_data = [fmt(ex) for ex in test_examples]
    print(f"Train: {len(train_data)}, Test: {len(test_data)}", flush=True)

    # Print one training example to verify format
    print("\n=== SAMPLE TRAINING EXAMPLE ===", flush=True)
    p, ta, g = train_data[0]
    print(f"Prompt: {p[:200]}...", flush=True)
    print(f"Train answer: {ta[:300]}...", flush=True)
    print(f"Gold: {g}", flush=True)

    # Build training stream
    def format_prompt(question):
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def format_example(question, answer):
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False) + tokenizer.eos_token

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

    all_tokens = []
    for prompt_q, train_answer, gold in train_data:
        text = format_example(prompt_q, train_answer)
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    dataset = TextDataset(token_ids, 1024)
    print(f"\nTraining: {len(token_ids):,} tokens, {len(dataset)} chunks", flush=True)

    # Train 3 epochs
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
    gs, tl = 0, 0.0
    t0 = time.time()
    for epoch in range(3):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); tl += out.loss.item(); gs += 1
            if gs % 10 == 0:
                print(f"  step {gs} | loss={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)

    # Eval on 20 test questions (unbatched for simplicity)
    print(f"\n=== EVAL ===", flush=True)
    model.eval()
    # Disable grad checkpointing for inference
    try: model.gradient_checkpointing_disable()
    except: pass

    def normalize_math_answer(s):
        s = s.strip().replace('$','').replace('\\','').replace('!','').replace(',','').replace('{','').replace('}','').replace(' ','')
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except: return s.lower()

    def extract_answer(response):
        response = response.strip()
        matches = re.findall(r'\\boxed\{([^}]+)\}', response)
        if matches: return matches[-1].strip()
        m = re.search(r'####\s*(-?[\d,.]+)', response)
        if m: return m.group(1).replace(",", "").strip()
        m = re.search(r'(?:final answer|answer)\s*:?\s*\**\s*([^\n*]+)', response, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip('*').strip()
            nums = re.findall(r'-?[\d,.]+', candidate)
            if nums: return nums[-1].replace(",", "").strip()
        numbers = re.findall(r'-?[\d,.]+', response)
        if numbers: return numbers[-1].replace(",", "").strip()
        return response[:50] if response else ""

    correct = 0
    for i, (prompt_q, train_answer, gold) in enumerate(test_data):
        text = format_prompt(prompt_q)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id, temperature=1.0)
        gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        resp_ans = extract_answer(gen_text)
        score = 1.0 if normalize_math_answer(resp_ans) == normalize_math_answer(gold) else 0.0
        correct += score
        if i < 5:
            print(f"\n--- Q{i+1} ---", flush=True)
            print(f"Gold: {gold}", flush=True)
            print(f"Extracted: {resp_ans}", flush=True)
            print(f"Score: {score}", flush=True)
            print(f"Response (first 300 chars): {gen_text[:300]}", flush=True)

    acc = correct / len(test_data)
    print(f"\n=== SMOKE TEST RESULT ===", flush=True)
    print(f"GSM8K accuracy on 20 test questions: {correct}/{len(test_data)} = {acc:.3f}", flush=True)
    print(f"(Base model was ~30-50% with reasoning. If this is >25%, the fix works.)", flush=True)

    with open("/root/output/smoke_test_result.json", "w") as f:
        json.dump({"accuracy": acc, "n_train": 100, "n_test": 20, "model": MODEL_ID}, f, indent=2)
    output_vol.commit()

@app.local_entrypoint()
def main():
    smoke_test.remote()
