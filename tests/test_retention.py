import torch

from continual_pt.retention import get_lora_state, repair_toward_anchor, set_lora_state


class TinyLoRAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.lora_B = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.base = torch.nn.Parameter(torch.tensor([9.0]))


def test_avr_repair_moves_only_lora_parameters_toward_anchor():
    model = TinyLoRAModel()
    anchor = get_lora_state(model)
    model.lora_A.data.add_(10.0)
    model.base.data.add_(10.0)

    touched = repair_toward_anchor(model, anchor, alpha=0.25)

    assert touched == 2
    assert torch.allclose(model.lora_A.detach(), torch.tensor([8.5, 9.5]))
    assert torch.allclose(model.base.detach(), torch.tensor([19.0]))


def test_restoring_anchor_is_exact():
    model = TinyLoRAModel()
    anchor = get_lora_state(model)
    model.lora_A.data.zero_()
    set_lora_state(model, anchor)
    assert torch.equal(model.lora_A.detach().cpu(), anchor["lora_A"])
