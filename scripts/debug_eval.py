"""Debug script — check what Qwen3-1.7B actually generates on GSM8K.
No training, no LoRA, just base model inference with the chat template.
Tells us if the eval pipeline is broken before we rerun anything."""
import modal

app = modal.App("avr-cl-debug")
output_vol = modal.Volume.from_name("avr-cl-output", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.45.0",
        "peft>=0.13.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "numpy<2",
        "sentencepiece",
        "protobuf",
        "scipy",
        "packaging",
    )
    .env({"PYDEVD_DISABLE_FILE_VALIDATION": "1", "TOKENIZERS_PARALLELISM": "false"})
)

@app.function(image=image, gpu="A100-40GB", timeout=1800, volumes={"/root/output": output_vol})
def debug_generations():
    import torch, json, re, time
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    MODEL_ID = "Qwen/Qwen3-1.7B"
    print(f"Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager")
    model.eval()
    print(f"Loaded. Vocab size: {len(tokenizer)}", flush=True)

    # Load 5 GSM8K test questions
    ds = load_dataset("openai/gsm8k", "main", split="test")
    samples = list(ds)[:5]

    print("\n" + "="*70, flush=True)
    print("TEST 1: With chat template, enable_thinking=False", flush=True)
    print("="*70, flush=True)
    for i, ex in enumerate(samples):
        q = ex["question"]
        a = ex["answer"]
        # Extract gold
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()

        messages = [{"role": "user", "content": f"Solve the math problem. Give the final number after '####'.\n\n{q}\n\nAnswer:"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, temperature=1.0)
        elapsed = time.time() - t0
        gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        print(f"\n--- Q{i+1} ---", flush=True)
        print(f"Question: {q[:100]}...", flush=True)
        print(f"Gold: {gold}", flush=True)
        print(f"Generated ({gen_tokens} tokens, {elapsed:.1f}s):", flush=True)
        print(f"  {repr(gen_text[:500])}", flush=True)

    print("\n" + "="*70, flush=True)
    print("TEST 2: With chat template, enable_thinking=True (default)", flush=True)
    print("="*70, flush=True)
    for i, ex in enumerate(samples[:2]):
        q = ex["question"]
        messages = [{"role": "user", "content": f"Solve: {q}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, temperature=1.0)
        gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n--- Q{i+1} (thinking mode) ---", flush=True)
        print(f"Generated: {repr(gen_text[:600])}", flush=True)

    print("\n" + "="*70, flush=True)
    print("TEST 3: Raw prompt, no chat template", flush=True)
    print("="*70, flush=True)
    for i, ex in enumerate(samples[:2]):
        q = ex["question"]
        prompt = f"{q}\n\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, temperature=1.0)
        gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n--- Q{i+1} (raw) ---", flush=True)
        print(f"Generated: {repr(gen_text[:400])}", flush=True)

    print("\nDONE.", flush=True)

@app.local_entrypoint()
def main():
    debug_generations.remote()
