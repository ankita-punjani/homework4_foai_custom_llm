"""Stricter eval-separation check for my added corpus files (reads evals/, writes nothing to corpus/).

The notebook rejects passages that contain an exact eval prompt. That misses near-copies,
so this script also checks every passage it would import for:
  1. an exact normalized eval prompt (same rule as the notebook);
  2. any shared run of N or more consecutive tokens with an eval prompt or explanation;
  3. all four answer choices of one case appearing together (a pasted answer list);
  4. names that appear in eval prompts;
and reports which content words of the untaught control categories reached the corpus.

    .venv/bin/python check_leakage.py --output experiments/expanded/leakage_check.json
"""
import argparse
import json
import re
from pathlib import Path

from run_evals import load_suite, matching_cases, word_tokens

ROOT = Path(__file__).resolve().parent
EVAL_NAMES = {"ava", "maya", "leo", "ella", "finn", "omar", "nina", "sara", "noah", "emma", "luca", "nora"}
TARGETED = {"grammar", "opposites", "negation", "spatial_relations"}
FUNCTION_WORDS = {"a", "an", "the", "is", "it", "to", "of", "we", "was", "in", "on", "that", "who",
                  "then", "into", "from", "after", "before", "the", ".", "person", "uses"}


def chunk_text(text, max_tokens=47):  # identical to the notebook's splitter
    chunks = []
    for unit in re.split(r"(?<=[.!?])\s+|\n+", text):
        tokens = word_tokens(unit)
        chunks.extend(" ".join(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens))
    return chunks


def ngrams(tokens, n):
    return {tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument("--n", type=int, default=5, help="Flag shared runs of this many tokens")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = load_suite(ROOT / "evals/language_evals.json")
    cases = suite["cases"]

    passages = []
    for path in sorted(args.corpus.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"} and path != args.corpus / "README.md":
            passages += [(str(path.relative_to(args.corpus)), p) for p in chunk_text(path.read_text("utf-8"))]

    eval_grams = {}
    for case in cases:
        for field in ("prompt", "reason"):
            for gram in ngrams(word_tokens(case[field]), args.n):
                eval_grams.setdefault(gram, set()).add(f"{case['id']}:{field}")

    exact, shared, answer_lists, names = [], [], [], []
    for source, passage in passages:
        tokens = word_tokens(passage)
        if (hits := matching_cases(passage, suite)):
            exact.append({"file": source, "passage": passage, "cases": hits})
        for gram in ngrams(tokens, args.n) & eval_grams.keys():
            shared.append({"file": source, "passage": passage, "ngram": " ".join(gram),
                           "matches": sorted(eval_grams[gram])})
        token_set = set(tokens)
        for case in cases:
            if set(case["choices"]) <= token_set:
                answer_lists.append({"file": source, "passage": passage, "case": case["id"]})
        if token_set & EVAL_NAMES:
            names.append({"file": source, "passage": passage, "names": sorted(token_set & EVAL_NAMES)})

    corpus_vocab = {t for _, p in passages for t in word_tokens(p)}
    control = {}
    for case in cases:
        if case["group"] == "extend_corpus" and case["category"] not in TARGETED:
            words = set(word_tokens(case["prompt"])) | set(case["choices"])
            control[case["id"]] = sorted((words - FUNCTION_WORDS) & corpus_vocab)

    report = {"method": f"normalized tokens; exact prompt match plus any shared {args.n}-token run with "
                        "an eval prompt or explanation; four-choice co-occurrence; eval names",
              "files": sorted({s for s, _ in passages}), "passages_checked": len(passages),
              "exact_prompt_matches": exact, f"shared_{args.n}gram_runs": shared,
              "all_four_choices_in_one_passage": answer_lists, "eval_names": names,
              "control_category_content_words_in_corpus": control}
    from run_evals import suite_hash
    report["suite_sha256"] = suite_hash(suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {len(passages)} passages from {len(report['files'])} files.")
    print(f"exact prompt matches: {len(exact)} | shared {args.n}-token runs: {len(shared)} | "
          f"four-choice lists: {len(answer_lists)} | eval names: {len(names)}")
    for case_id, words in control.items():
        if words:
            print(f"control {case_id}: corpus contains {words}")
    for row in shared[:20]:
        print("SHARED:", row["ngram"], "|", row["matches"], "|", row["passage"])
    if exact or shared or answer_lists or names:
        raise SystemExit("Leakage check FAILED. Rewrite the flagged passages.")
    print("Leakage check passed.")


if __name__ == "__main__":
    main()
