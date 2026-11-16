"""Shared, exact-checkpoint Gemma-4-31B-IT inference helpers."""

MODEL_ID = 'google/gemma-4-31B-it'


def load_model(device='cuda', *, attn_implementation='sdpa'):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Gemma-4-31B-IT BF16 suite requires CUDA')
    if torch.cuda.get_device_properties(0).total_memory < 85 * 1024**3:
        raise ValueError('Gemma-4-31B-IT BF16 suite requires a 96 GB GPU')
    revision = HfApi().model_info(MODEL_ID).sha
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.bfloat16,
        attn_implementation=attn_implementation).to(device).eval()
    return model, processor, revision


def text_layers(model):
    text = model.model.language_model
    layers = text.layers
    if len(layers) != 60:
        raise ValueError(f'Expected 60 Gemma-4 text blocks, got {len(layers)}')
    return layers


def render(processor, messages):
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
    inputs = processor.apply_chat_template(messages, tokenize=True,
                                           return_dict=True, return_tensors='pt',
                                           add_generation_prompt=True,
                                           enable_thinking=False)
    if processor.tokenizer.encode(prompt, add_special_tokens=False) != inputs['input_ids'][0].tolist():
        raise ValueError('Rendered Gemma-4 prompt does not match processor tokens')
    return prompt, inputs
