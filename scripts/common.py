"""Shared helpers: loading models on the Mac (MPS) and turning dataset items into chat prompts."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except ValueError:
        # Gemma-3 4B+ checkpoints are image-text models; text-only prompts work through this class.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    model.to(device()).eval()
    return tok, model


def prompt_text(tok, item):
    """Chat-formatted prompt ending where the assistant's reply begins.

    Templates without a system role (e.g. Gemma) fold the system prompt into the first user turn themselves.
    """
    return tok.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def slug(model_name):
    return model_name.split("/")[-1].lower()
