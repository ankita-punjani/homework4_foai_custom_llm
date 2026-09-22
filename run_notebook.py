"""Execute a copy of custom_llm.ipynb with my section-1 choices and prediction.

The starter notebook is never modified. Only the three section-1 settings, the
prediction markdown and the section-10 chat prompt are replaced in the copy, then
the whole notebook runs top to bottom (Run All) and is saved with every output.

    .venv/bin/python run_notebook.py --steps 3000 --prediction experiments/starter/prediction.md \
        --output experiments/starter/custom_llm_starter.executed.ipynb
"""
import argparse
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", default="0.001")
    parser.add_argument("--corpus", default="classroom", choices=["classroom", "folder"])
    parser.add_argument("--prediction", type=Path, required=True, help="Markdown written before training")
    parser.add_argument("--chat-prompt", default="the customer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    notebook = nbformat.read(ROOT / "custom_llm.ipynb", as_version=4)
    replaced = set()
    for cell in notebook.cells:
        src = cell.source
        if cell.cell_type == "code" and src.startswith('CORPUS = "classroom"'):
            src = src.replace('CORPUS = "classroom"', f'CORPUS = "{args.corpus}"')
            src = src.replace("TRAINING_STEPS = 3000", f"TRAINING_STEPS = {args.steps}")
            src = src.replace("LEARNING_RATE = 0.001", f"LEARNING_RATE = {args.lr}")
            replaced.add("settings")
        elif cell.cell_type == "markdown" and src.startswith("### My prediction"):
            head, _, rest = src.partition("## 2. Load the tools and network")
            src = "### My prediction\n" + args.prediction.read_text().strip() + "\n\n## 2. Load the tools and network" + rest
            replaced.add("prediction")
        elif cell.cell_type == "code" and src.startswith('CHAT_PROMPT = "the customer"'):
            src = src.replace('CHAT_PROMPT = "the customer"', f"CHAT_PROMPT = {args.chat_prompt!r}")
            replaced.add("chat")
        cell.source = src
    if replaced != {"settings", "prediction", "chat"}:
        raise SystemExit(f"Starter notebook layout changed; only found {sorted(replaced)}.")

    ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    print("Executed notebook:", args.output)


if __name__ == "__main__":
    main()
