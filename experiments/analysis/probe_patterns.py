"""Diagnose the expanded model's extension failures with my own probe prompts.

Inference only: loads a saved model.pt, prints the top next tokens, never trains or
writes to corpus/. Probes vary one thing at a time: a taught object vs an object never
seen in that sentence frame, to separate "pattern not learned" from "pattern tied to
familiar objects". Probe prompts are mine, not eval cases.

    .venv/bin/python experiments/analysis/probe_patterns.py experiments/expanded/run/model.pt
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_evals import load_model, word_tokens  # noqa: E402

PROBES = {
    "negation, taught object": ["the kite is not green . it is pink . the kite is",
                                "the hat is not white . it is gray . the hat is"],
    "negation, object never in a negation story": ["the crate is not green . it is pink . the crate is",
                                                   "the bench is not white . it is gray . the bench is"],
    "negation, answer word never in the corrected slot": ["the kite is not green . it is blue . the kite is",
                                                           "the gate is not closed . it is open . the gate is"],
    "negation, denied-slot word never denied": ["the kite is not red . it is green . the kite is"],
    "left/right, taught objects": ["the sofa is left of the piano . the piano is to the",
                                   "the tent is right of the bench . the bench is to the"],
    "left/right, objects never in that frame": ["the key is left of the jar . the jar is to the",
                                                "the coin is right of the pen . the pen is to the"],
    "opposite frame, taught pair": ["the opposite of big is", "the opposite of warm is"],
    "opposite frame, contrast-only pair": ["the opposite of full is", "the opposite of quiet is"],
}


@torch.inference_mode()
def top(model, vocab, prompt, k=5):
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = [stoi["<BOS>"]] + [stoi.get(t, 0) for t in word_tokens(prompt)]
    probs = torch.softmax(model(torch.tensor([ids]))[0][0, -1], -1)
    values, indices = probs.topk(k)
    return [(vocab[i], round(v.item(), 3)) for v, i in zip(values, indices)]


def main():
    model, vocab, _ = load_model(sys.argv[1])
    results = {}
    for group, prompts in PROBES.items():
        print("##", group)
        for prompt in prompts:
            unknown = [t for t in word_tokens(prompt) if t not in vocab]
            results[prompt] = {"group": group, "unknown": unknown, "top5": top(model, vocab, prompt)}
            print(f"  {prompt!r} -> {results[prompt]['top5']}" + (f"  UNKNOWN {unknown}" if unknown else ""))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
