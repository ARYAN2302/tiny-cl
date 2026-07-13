"""LEARN phase: SFT training + two-stream consolidation."""
import torch, time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .model import format_example
from .repair import get_lora_state, set_lora_state


class _TextDataset(Dataset):
    def __init__(self, token_ids, ctx_len):
        self.token_ids = token_ids
        self.ctx_len = ctx_len
        self.n_chunks = max(1, len(token_ids) // ctx_len)
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        s = idx * self.ctx_len
        e = s + self.ctx_len
        chunk = self.token_ids[s:e]
        return {"input_ids": chunk, "labels": chunk.clone()}


def train_sft(model, tokenizer, examples, epochs=3, lr=2e-4, batch_size=4,
              grad_accum=4, ctx_len=512, device="cuda", tag="sft"):
    all_tokens = []
    for question, answer, gold in examples:
        text = format_example(tokenizer, question, answer)
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    dataset = _TextDataset(token_ids, ctx_len)
    print(f"    [{tag}] {len(token_ids):,} tokens, {len(dataset)} chunks", flush=True)

    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(device),
                       labels=batch["labels"].to(device))
            (out.loss / grad_accum).backward()
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                accum = 0
                gs += 1
                tl += out.loss.item()
                if gs % 50 == 0:
                    print(f"      [{tag}] step {gs} | loss={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)


def consolidate(model, tokenizer, hippo_state, neo_state, examples,
                epochs=1, lr=1e-4, batch_size=4, grad_accum=4, ctx_len=512, device="cuda"):
    print(f"    [consolid] KL distill ({epochs} epoch)", flush=True)
    all_tokens = []
    for question, answer, gold in examples:
        text = format_example(tokenizer, question, answer)
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    dataset = _TextDataset(token_ids, ctx_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            set_lora_state(model, hippo_state, device)
            model.eval()
            with torch.no_grad():
                hippo_logits = model(input_ids=input_ids).logits
                p_hippo = F.softmax(hippo_logits[..., :-1, :].contiguous().float(), dim=-1)
            del hippo_logits

            set_lora_state(model, neo_state, device)
            model.train()
            neo_logits = model(input_ids=input_ids).logits
            shift_neo = neo_logits[..., :-1, :].contiguous()
            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)
            kl_loss = F.kl_div(log_p_neo, p_hippo, reduction='batchmean', log_target=False)
            (kl_loss / grad_accum).backward()
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                accum = 0
                gs += 1
                tl += kl_loss.item()
                if gs % 50 == 0:
                    print(f"      [consolid] step {gs} | KL={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)
            del neo_logits, log_p_neo, p_hippo, kl_loss, shift_neo
            neo_state = get_lora_state(model)
    print(f"    [consolid] Done: {gs} steps, avg KL={tl/max(gs,1):.4f}", flush=True)
    return neo_state
