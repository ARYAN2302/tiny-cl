"""Batched evaluation + scoring."""
import gc, re, time
import torch
from .model import format_prompt


def generate_batch(model, tokenizer, questions, max_new_tokens=200, batch_size=8, device="cuda"):
    results = []
    gc_was = getattr(model, "gradient_checkpointing", False)
    if gc_was:
        try: model.gradient_checkpointing_disable()
        except: pass
    model.eval()
    try:
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            texts = [format_prompt(tokenizer, q) for q in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                             max_length=1024, padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id, temperature=1.0)
            for out in outputs:
                input_len = inputs["input_ids"].shape[1]
                results.append(tokenizer.decode(out[input_len:], skip_special_tokens=True).strip())
    finally:
        if gc_was:
            try: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
            except: pass
    return results


def normalize_answer(s):
    s = s.strip().lower()
    s = re.sub(r'[^\w\s.-]', ' ', s)
    return ' '.join(s.split())


def default_scorer(response, gold):
    resp = normalize_answer(response)
    g = normalize_answer(gold)
    if resp == g: return 1.0
    if g in resp or resp in g: return 1.0
    g_spaces = g.replace('_', ' ')
    resp_spaces = resp.replace('_', ' ')
    if g_spaces in resp_spaces or resp_spaces in g_spaces: return 1.0
    return 0.0


def evaluate(model, tokenizer, eval_examples, task_name, scorer=None,
             max_questions=200, batch_size=8, device="cuda"):
    if scorer is None:
        scorer = default_scorer
    total = min(len(eval_examples), max_questions)
    examples = eval_examples[:total]
    questions = [ex[0] for ex in examples]
    golds = [ex[2] for ex in examples]
    print(f"    Eval {task_name} ({total} Qs)...", flush=True)
    correct = 0
    t0 = time.time()
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i+batch_size]
        batch_g = golds[i:i+batch_size]
        responses = generate_batch(model, tokenizer, batch_q, max_new_tokens=200,
                                   batch_size=len(batch_q), device=device)
        for r, g in zip(responses, batch_g):
            if scorer(r, g): correct += 1
    acc = correct / total
    print(f"    {task_name}: {correct}/{total} = {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return acc
