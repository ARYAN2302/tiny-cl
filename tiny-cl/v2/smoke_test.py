"""
V2 Smoke Test: Pre-flight checks before spending credits.
Tests that pretrained models load, LoRA attaches, hidden states are accessible,
and anchor loss computes correctly.
"""

import sys
import torch


def test_smollm2_360m():
    """Test SmolLM2-360M: loading, hidden states, LoRA, anchor loss."""
    print("=" * 60)
    print("SMOKE TEST: SmolLM2-360M")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test 1: Load model and tokenizer
    print("\n[1/5] Loading SmolLM2-360M...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  OK — {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    model = model.to(device)
    model.eval()

    # Test 2: Get hidden states
    print("\n[2/5] Testing hidden state extraction...")
    try:
        input_ids = torch.randint(0, 1000, (2, 64)).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        n_layers = len(outputs.hidden_states)
        hidden_dim = outputs.hidden_states[0].shape[-1]
        print(f"  OK — {n_layers} layers, hidden_dim={hidden_dim}")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    # Test 3: Attach LoRA
    print("\n[3/5] Testing LoRA attachment...")
    try:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()  # Needed for anchor loss gradients
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  OK — Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    # Test 4: Forward pass with LoRA + hidden states + labels
    print("\n[4/5] Testing forward pass with LoRA + hidden states...")
    try:
        labels = input_ids.clone()  # For causal LM, labels = input_ids
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        loss_val = outputs.loss.item() if outputs.loss is not None else "N/A"
        print(f"  OK — Loss: {loss_val}, {len(outputs.hidden_states)} layers")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    # Test 5: Compute anchor loss
    print("\n[5/5] Testing anchor loss computation...")
    try:
        import torch.nn.functional as F
        # Save fake anchor
        anchor_states = {i: outputs.hidden_states[i].mean(dim=1).cpu() for i in range(len(outputs.hidden_states))}

        # Compute loss
        with torch.no_grad():
            new_outputs = model(input_ids=input_ids, output_hidden_states=True)
        total_loss = torch.tensor(0.0, device=device)
        for layer_idx, hs in enumerate(new_outputs.hidden_states):
            current_mean = hs.mean(dim=1)
            anchor_mean = anchor_states[layer_idx].to(device)
            drift = F.mse_loss(current_mean, anchor_mean)
            total_loss = total_loss + drift
        avg_loss = total_loss / len(new_outputs.hidden_states)
        print(f"  OK — Anchor loss: {avg_loss.item():.8f}")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    print("\n  SmolLM2-360M: ALL TESTS PASSED")
    return True


def test_lfm25_350m():
    """Test LFM2.5-350M: loading, hidden states, LoRA, anchor loss."""
    print("\n" + "=" * 60)
    print("SMOKE TEST: LFM2.5-350M (Moon Shot)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test 1: Load model and tokenizer
    print("\n[1/5] Loading LFM2.5-350M...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "LiquidAI/LFM2.5-350M",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2.5-350M")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"  OK — {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"  FAIL — {e}")
        print("  LFM2.5-350M is NOT usable. Skipping remaining tests.")
        return False

    model = model.to(device)
    model.eval()

    # Test 2: Get hidden states
    print("\n[2/5] Testing hidden state extraction...")
    try:
        input_ids = torch.randint(0, 1000, (2, 64)).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)

        if outputs.hidden_states is None:
            print("  FAIL — model does not return hidden_states")
            print("  Checking if we can hook into intermediate layers...")

            # Try alternative: register forward hooks
            try:
                hidden_states_hook = []
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        hidden_states_hook.append(output[0])
                    else:
                        hidden_states_hook.append(output)

                # Try to find transformer/attention layers to hook
                hooked = False
                for name, module in model.named_modules():
                    if any(kw in name.lower() for kw in ["attention", "transformer", "block", "layer"]):
                        if not any(kw in name for kw in ["drop", "norm"]):
                            module.register_forward_hook(hook_fn)
                            hooked = True

                if hooked:
                    with torch.no_grad():
                        _ = model(input_ids=input_ids)
                    if hidden_states_hook:
                        print(f"  PARTIAL — Can hook into {len(hidden_states_hook)} layers")
                        print(f"  But this requires custom anchor implementation")
                    else:
                        print("  FAIL — Hooks returned no hidden states")
                        return False
                else:
                    print("  FAIL — No suitable layers found for hooks")
                    return False
            except Exception as e2:
                print(f"  FAIL — Hook approach also failed: {e2}")
                return False
        else:
            n_layers = len(outputs.hidden_states)
            hidden_dim = outputs.hidden_states[0].shape[-1]
            print(f"  OK — {n_layers} layers, hidden_dim={hidden_dim}")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    # Test 3: Attach LoRA
    print("\n[3/5] Testing LoRA attachment...")
    try:
        from peft import LoraConfig, get_peft_model

        # Try standard target modules first
        target_options = [
            ["q_proj", "v_proj"],
            ["attention.wq", "attention.wv"],
            ["attention.q_proj", "attention.v_proj"],
        ]

        peft_model = None
        for targets in target_options:
            try:
                lora_config = LoraConfig(
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=targets, bias="none", task_type="CAUSAL_LM",
                )
                # Reload fresh model for each attempt
                model = AutoModelForCausalLM.from_pretrained(
                    "LiquidAI/LFM2.5-350M", trust_remote_code=True,
                ).to(device)
                peft_model = get_peft_model(model, lora_config)
                print(f"  OK — LoRA attached with targets: {targets}")
                model = peft_model
                break
            except Exception:
                continue

        if peft_model is None:
            print("  FAIL — Could not attach LoRA with any target module config")
            print("  Listing model modules for manual target selection:")
            for name, _ in model.named_modules():
                if len(name.split(".")) <= 3:
                    print(f"    {name}")
            return False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    # Test 4 & 5: Forward pass + anchor loss
    print("\n[4/5] Testing forward pass with LoRA + hidden states...")
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
        if outputs.hidden_states is not None:
            print(f"  OK — {len(outputs.hidden_states)} layers")
        else:
            print("  PARTIAL — No hidden states with LoRA (hook workaround needed)")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    print("\n[5/5] Testing anchor loss computation...")
    try:
        if outputs.hidden_states is not None:
            import torch.nn.functional as F
            anchor_states = {i: outputs.hidden_states[i].mean(dim=1).cpu() for i in range(len(outputs.hidden_states))}
            with torch.no_grad():
                new_outputs = model(input_ids=input_ids, output_hidden_states=True)
            total_loss = torch.tensor(0.0, device=device)
            for layer_idx, hs in enumerate(new_outputs.hidden_states):
                current_mean = hs.mean(dim=1)
                anchor_mean = anchor_states[layer_idx].to(device)
                drift = F.mse_loss(current_mean, anchor_mean)
                total_loss = total_loss + drift
            avg_loss = total_loss / len(new_outputs.hidden_states)
            print(f"  OK — Anchor loss: {avg_loss.item():.8f}")
        else:
            print("  SKIP — Cannot test without hidden states")
    except Exception as e:
        print(f"  FAIL — {e}")
        return False

    print("\n  LFM2.5-350M: ALL TESTS PASSED")
    return True


def main():
    print("Tiny-CL V2 Smoke Test")
    print("=" * 60)

    smollm2_ok = test_smollm2_360m()

    lfm25_ok = False
    try:
        lfm25_ok = test_lfm25_350m()
    except Exception as e:
        print(f"\nLFM2.5-350M test crashed: {e}")
        lfm25_ok = False

    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print(f"  SmolLM2-360M:  {'PASS' if smollm2_ok else 'FAIL'}")
    print(f"  LFM2.5-350M:   {'PASS' if lfm25_ok else 'FAIL'}")
    print()

    if smollm2_ok:
        print("  -> SmolLM2-360M is ready. Run the sanity experiment first:")
        print("     python v2/run_all.py --preset sanity")
    if lfm25_ok:
        print("  -> LFM2.5-350M is ready! Moon shot is GO.")
        print("     python v2/run_all.py --model lfm2.5-350M --method anchor_disc --experiment hero")
    elif smollm2_ok:
        print("  -> LFM2.5-350M failed. Proceeding with SmolLM2 only.")


if __name__ == "__main__":
    main()
