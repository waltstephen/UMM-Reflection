"""LoRA utilities for BAGEL training.

Injects LoRA adapters into the Qwen2 language model while keeping
vae2llm, llm2vae, connector as fully trainable modules.
"""
import torch
from peft import LoraConfig, get_peft_model, TaskType


def inject_lora(model, rank=64, alpha=128, target_modules=None):
    """Inject LoRA into the BAGEL model's language_model.

    Args:
        model: Bagel model instance
        rank: LoRA rank (64-128 recommended for generation tasks)
        alpha: LoRA alpha (typically 2× rank)
        target_modules: list of module names to apply LoRA to

    Returns:
        model with LoRA injected, list of fully trainable module names
    """
    if target_modules is None:
        # Qwen2MoT attention + MLP (both und and gen experts)
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to language_model only
    model.language_model = get_peft_model(model.language_model, lora_config)

    # PEFT wraps the model: language_model.model.X → language_model.base_model.model.model.X
    # But BAGEL's forward accesses language_model.model.embed_tokens directly.
    # Fix: make language_model.model point to the inner model for attribute access.
    peft_lm = model.language_model
    if hasattr(peft_lm, 'base_model') and hasattr(peft_lm.base_model, 'model'):
        # PeftModel.base_model.model is the original Qwen2ForCausalLM
        inner_model = peft_lm.base_model.model
        # Forward the .model attribute so BAGEL code still works
        if hasattr(inner_model, 'model'):
            peft_lm.model = inner_model.model

    # Freeze everything first, then unfreeze LoRA + bridge modules
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    for name, param in model.language_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    # Unfreeze bridge modules (MSE gradient entry/exit)
    fully_trainable = ["vae2llm", "llm2vae", "connector"]
    for module_name in fully_trainable:
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA injected: {trainable/1e6:.1f}M trainable / {total/1e9:.2f}B total "
          f"({trainable/total*100:.2f}%)")

    # Cast LoRA params to match base model dtype (bf16) for FSDP compatibility
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)

    return model
