# Building a Custom LLM: nanoGPT on a starter and an expanded corpus

Class 4 assignment, From Zero to AI Agents (Fall 26) · Ankita Punjani

I trained Karpathy's actual nanoGPT transformer ([`nanogpt_model.py`](nanogpt_model.py), unchanged, commit `3adf61e`) from scratch twice. The first run used the supplied classroom corpus. The second used that corpus plus my own teaching text for four eval categories. I ran the fixed 48-case language eval suite before and after training in both experiments, and I chat with the trained model through a terminal interface. This is a **tiny word-level language model**. It continues short sentences from a narrow corpus. It is not a chatbot and does not understand language in general.

The starter code comes from [pepealonso95/custom-llm](https://github.com/pepealonso95/custom-llm). Its original README is kept as [STARTER_README.md](STARTER_README.md) and the assignment text as [ASSIGNMENT.md](ASSIGNMENT.md). I deleted its `examples/` and `legacy/` reference runs so that nobody mistakes them for my results.

## Results at a glance

| Experiment | Stage | Correct / 48 | Scorable / 48 | Accuracy among scorable cases | Full results |
|---|---|---|---|---|---|
| Starter corpus | Untrained | 9 (18.8%) | 24 | 37.5% | [CSV](experiments/starter/run/language_evals/untrained/eval_results.csv) · [JSON](experiments/starter/run/language_evals/untrained/eval_results.json) · [summary](experiments/starter/run/language_evals/untrained/eval_summary.json) |
| Starter corpus | Trained (3,000 steps) | **20 (41.7%)** | 24 | 83.3% | [CSV](experiments/starter/run/language_evals/final/eval_results.csv) · [JSON](experiments/starter/run/language_evals/final/eval_results.json) · [summary](experiments/starter/run/language_evals/final/eval_summary.json) |
| Expanded corpus | Untrained | 8 (16.7%) | 35 | 22.9% | [CSV](experiments/expanded/run/language_evals/untrained/eval_results.csv) · [JSON](experiments/expanded/run/language_evals/untrained/eval_results.json) · [summary](experiments/expanded/run/language_evals/untrained/eval_summary.json) |
| Expanded corpus | Trained (3,000 steps) | **30 (62.5%)** | 35 | 85.7% | [CSV](experiments/expanded/run/language_evals/final/eval_results.csv) · [JSON](experiments/expanded/run/language_evals/final/eval_results.json) · [summary](experiments/expanded/run/language_evals/final/eval_summary.json) |

- **Executed notebooks:** [starter](experiments/starter/custom_llm_starter.executed.ipynb) · [expanded](experiments/expanded/custom_llm_expanded.executed.ipynb) · [10-step smoke test](experiments/smoke_test/custom_llm_smoke10.executed.ipynb)
- **Complete run folders:** [starter](experiments/starter/run/) · [expanded](experiments/expanded/run/). Each folder has also been zipped whole: [starter ZIP](experiments/starter/run.zip) · [expanded ZIP](experiments/expanded/run.zip).
- **Chat evidence:** [transcript](experiments/expanded/chat/chat_transcript.json) · [screenshot](experiments/expanded/chat/chat_screenshot.png) · [terminal log](experiments/expanded/chat/terminal_session.txt)

The honest summary:
- **Extension categories.** The added corpus made 11 of the 12 targeted extension cases scorable (the 12 control cases stayed unscorable). The model gets **grammar 3/3** and **spatial 2/3** reliably. **Opposites (1/3) and negation (0/2)** are fragile.
- **New wording.** 4/8 → 8/8 looks like a gain, but a seed check shows it is mostly seed noise ([below](#is-it-real-a-seed-check)).
- **Limitations.** My probes show that the model learned slot-specific patterns: it copies or inverts words it has seen in that slot before. It did not learn general rules.

---

## 1. Corpus, sources and permissions

| | Experiment 1: starter | Experiment 2: expanded |
|---|---|---|
| Mode | `CORPUS = "classroom"`, empty `corpus/` | `CORPUS = "classroom"` + 4 files in [`corpus/extension/`](corpus/extension/) |
| Source | Synthetic classroom sentences generated in notebook section 3 | The same classroom sentences, plus 1,200 synthetic teaching passages I generated with [`make_extension_corpus.py`](make_extension_corpus.py) |
| Classroom passages withheld because they contain an eval prompt | 160 | 160 |
| Unique passages (after dedup) | 4,592 | 5,792 (1,200 new) |
| Train / validation split | 4,132 / 460 (90/10) | 5,212 / 580 (90/10) |
| Vocabulary (incl. `<UNK>`, `<BOS>`, `<EOS>`) | 136 | 410 |
| Training / held-out unknown-token rate | 0.00% / 0.00% | 0.00% / 0.045% |
| Manifest / vocabulary report | [manifest](experiments/starter/run/corpus_manifest.json) · [vocab](experiments/starter/run/vocabulary_report.json) | [manifest](experiments/expanded/run/corpus_manifest.json) · [vocab](experiments/expanded/run/vocabulary_report.json) |

**Permissions.** Everything is synthetic. The classroom sentences come from the starter's generator. The extension files are my own generated text with no third-party material, so I am free to publish them. They are un-ignored in `.gitignore` for that reason. I used **no PDFs**, so there was no PDF extraction to check. Instead I checked every imported file's passage count and preview in notebook section 3, and read the saved `corpus.txt`. The notebook produced **no extraction warnings** and ignored no files.

**The 0.045% held-out UNK rate** comes from three of my teaching sentences that the random split sent to validation. Their words *visits*, *climbs* and *put* therefore never entered the training vocabulary. Every other word the model needed survived: 407 training types, all kept, well under the 509-type cap.

## 2. My three choices and prediction

- **Corpus:** the starter corpus first, then the same corpus plus my four extension files. Only the data changed between experiments.
- **Training steps: 3,000.** One step is one AdamW update on 32 passages. 3,000 steps is about 96,000 passage samples, or about 23 passes over the starter training set. That is enough to fit fixed templates without mostly measuring over-training. A 10-step run checked the setup first ([smoke test](experiments/smoke_test/)): loss went 4.93 → 4.21, and every cell ran.
- **Learning rate: 0.001**, with the notebook's 100-step warmup and cosine decay to 10%. A much larger rate can overshoot: loss spikes or goes non-finite, and early updates can wreck the random initialization. A much smaller rate leaves the model close to random after 3,000 updates.

I wrote my predictions before each run. They are in the notebooks and in [experiments/starter/prediction.md](experiments/starter/prediction.md) and [experiments/expanded/prediction.md](experiments/expanded/prediction.md).

| I expected… | I observed |
|---|---|
| Loss from ≈ ln(136) = 4.9 to below 1.0, with train ≈ validation | 4.93 → **0.68 train / 0.71 val**. Held-out loss followed training loss, as the shared templates predict |
| Starter patterns mostly pass; new wording partly; extension 0/24 because of missing vocabulary | **16/16**, **4/8**, and **0/24, all 24 unscorable** |
| *customer* moves toward *client / buyer / shopper* | Final cosine neighbours: **shopper 0.978, client 0.977, buyer 0.977** (initially bus, educator, helped at about 0.2) |
| Expanded: 11 of 12 targeted cases scorable, controls still unscorable | **Exactly that.** 35/48 scorable. The only targeted case still unscorable is *ava*, because I used no eval names |
| Expanded: copy/inverse patterns learnable; opposites the hardest | Spatial and grammar yes; opposites were hard as predicted. **Negation failed, which I did not expect.** The cause is explained in [section 6](#what-the-model-actually-learned-probes) |

## 3. The runs

| | Starter | Expanded |
|---|---|---|
| Completed steps | 3,000 of 3,000 (not interrupted) | 3,000 of 3,000 (not interrupted) |
| Training time | 16.2 s | 23.5 s |
| Parameters | 111,872 | 129,408 (a larger vocabulary means more embedding rows) |
| Hardware | Apple M4 MacBook Air, **CPU** (`DEVICE="cpu"`), macOS 15.6.1, Python 3.13.15, PyTorch 2.14.0 | same |
| Config / summary | [config.json](experiments/starter/run/config.json) · [training_summary.json](experiments/starter/run/training_summary.json) · [training.csv](experiments/starter/run/training.csv) | [config.json](experiments/expanded/run/config.json) · [training_summary.json](experiments/expanded/run/training_summary.json) · [training.csv](experiments/expanded/run/training.csv) |

Model (both runs): 2 transformer blocks, 4 attention heads, 64-number embeddings, a 48-token context, and tied input/output embeddings, trained with AdamW (β = 0.9/0.95, weight decay 0.01), seed 42 and batch size 32. There were no failed or interrupted runs. The runs I did besides the two experiments were the 10-step smoke test and the seed check in section 5.

## 4. Evidence from training

### Loss

These are **fixed panels of 20 training and 20 validation passages** (at most 20 documents each). Each value is the mean over every non-padding next-token target in the panel. They are small estimates, not full-corpus losses. Losses from the two corpora are **not directly comparable**, because the vocabularies and data differ.

| Step | Starter train | Starter val | Expanded train | Expanded val |
|---:|---:|---:|---:|---:|
| 0 | 4.9263 | 4.9275 | 6.0182 | 6.0275 |
| 1500 | 0.6821 | 0.7182 | 0.7070 | 0.8451 |
| 3000 | 0.6783 | 0.7061 | 0.6810 | 0.8263 |

Sources: [starter history.json](experiments/starter/run/history.json) · [expanded history.json](experiments/expanded/run/history.json)

| Starter | Expanded |
|---|---|
| ![starter loss](experiments/starter/run/training_curves.svg) | ![expanded loss](experiments/expanded/run/training_curves.svg) |

Both step-0 losses equal ln(vocabulary size): ln 136 = 4.91 and ln 410 = 6.02. At initialization the model spreads its probability almost evenly over every token. Almost all of the learning happens before step 1,500; the second half changes loss by less than 0.02. Loss stays near 0.7 rather than 0, because many slots are genuinely unpredictable. After *"the customer"*, six different verbs are equally correct. The expanded run's validation panel sits higher (0.83). Its random 20 validation passages include longer teaching stories, and those contain a few unpredictable choices (which colour, which object).

### Samples: untrained, halfway, final

All samples use the same settings: temperature 0.8, sampling seed 2026, start token `<BOS>`. Complete files, with nothing omitted:
- starter: [0](experiments/starter/run/samples/step_0000.txt) · [1500](experiments/starter/run/samples/step_1500.txt) · [3000](experiments/starter/run/samples/step_3000.txt)
- expanded: [0](experiments/expanded/run/samples/step_0000.txt) · [1500](experiments/expanded/run/samples/step_1500.txt) · [3000](experiments/expanded/run/samples/step_3000.txt)

| Step | Starter (first 2 of 4 samples) | Expanded (first 2 of 4) |
|---|---|---|
| 0 | `pear professor bond doctor course harvest team physician journey checking buyer … <UNK> taste recommended …`<br>`kitchen purchase journey product question discussion journey service . nurse local` | `near selected talks harvest update wore dim merchandise winter slow sad stool …`<br>`farm weak after sad painted i gray open likes again yesterday … ball frogs <UNK>` |
| 1500 | `our school has a question about the new educator and lesson .`<br>`a review of risk helped us understand the different deposit .` | `today the office focused on data and the local system .`<br>`the new application was mentioned in the security report yesterday .` |
| 3000 | `our school has a question about the new educator and lesson .`<br>`a review of risk helped us understand the different deposit .` | `today the office focused on data and the local system .`<br>`the new application was mentioned in the code report yesterday .` |

**Visible change.** At step 0, the text is a random walk through the vocabulary. The untrained model even emits `<UNK>`, because UNK is just another row it can sample. By step 1,500, every sample is a well-formed classroom template ending in `.`, and the model has learned to stop with `<EOS>`. Between 1,500 and 3,000, the starter samples are almost identical. That matches the flat loss: the model was already fitted.

**Something that didn't change.** At temperature 0.8, the expanded model's samples still all come from the classroom templates, which make up 79% of its data. My extension sentences appear only at higher temperature (next section).

### Temperature: same model, same seed, no weight updates

Source: [starter temperature_comparison.json](experiments/starter/run/temperature_comparison.json) · [expanded temperature_comparison.json](experiments/expanded/run/temperature_comparison.json). The first of 4 samples at each temperature:

| T | Starter | Expanded |
|---|---|---|
| 0.3 | `our school has a question about the new educator and lesson .` | `the important credit was mentioned in the return report yesterday .` |
| 0.8 | `our school has a question about the new educator and lesson .` | `today the office focused on data and the local system .` |
| 1.2 | `our school has a question about the new educator and lesson .` | **`near and sour mean opposite things .`** |

Temperature divides the logits before the softmax: p ∝ exp(logit / T). A low T sharpens the distribution towards the top token; a high T flattens it. No weights change; only sampling changes. The starter model is so confident on its templates that the first sample is identical at all three temperatures. Its other samples differ only in single slot words: *investment* vs *deposit*, *consumer* vs *offering*. At T = 1.2, the expanded model wandered into my opposites frame and **invented `near and sour mean opposite things .`**. That sentence is not in the corpus; the taught pairs are near/far and sweet/sour. It is a fluent-looking but false recombination: a tiny example of a language model "hallucinating".

### One word, end to end (starter run)

Sources: [tokenization.json](experiments/starter/run/tokenization.json) · [inspection.json](experiments/starter/run/inspection.json) · [checkpoint.json](experiments/starter/run/checkpoint.json)

1. **Text → tokens → IDs.** The first training passage is `today the school focused on lesson and the local professor .` It becomes the IDs `[1, 121, 118, 101, 42, 74, 61, 7, 118, 63, 88, 3, 2]`. `1` is `<BOS>`, `2` is `<EOS>` and `118` is *the*, which appears twice. The IDs are arbitrary row numbers from an alphabetical sort. The model is trained on input → target pairs shifted by one: `<BOS>`→*today*, *today*→*the*, … *professor*→*.*, *.*→`<EOS>`.
2. **ID → vector.** *customer* has ID **28**. Its embedding is row 28 of the 136 × 64 table `wte`. Before training, its first six numbers were `[-0.0576, -0.0048, 0.0426, 0.0193, 0.0156, -0.0288]` (random, std 0.02). After training they are `[0.0366, -0.0182, 0.1330, 0.1060, 0.0630, 0.0189]`. All 64 numbers are in `inspection.json`. The vector moved by an L2 distance of 0.661. No coordinate has a named meaning. What changed is its **position relative to other words**: its nearest cosine neighbours went from *bus, educator, helped* (≈ 0.2, random) to ***shopper 0.978, client 0.977, buyer 0.977***. Those words fill the same template slots, so training pushed their vectors together. That is distributional learning, not human-style meaning.
3. **One real gradient and update.** At step 0, the loss gradient for `wte[28][0]` was **+0.000693**. A positive gradient means increasing this number would raise the loss. At the first step, the warmup learning rate was 1e-5 (0.001 × 1/100). AdamW's first step is roughly −lr × sign(gradient), because its momentum and variance estimates start from this single gradient, plus a tiny weight-decay term. The observed change was −0.0575919 → **−0.0576019**, a change of −0.0000100 = −lr. So the size of the gradient barely matters for Adam's first step; its sign does.
4. **Probabilities before vs after, for the prefix "the customer".** Before training, the top prediction was *customer* at 1.6%, which is close to uniform (1/136 = 0.7%). After training: ***reviewed* 17.8%, *recommended* 17.1%, *ordered* 16.9%, *selected* 16.3%, *compared* 16.0%**. These are almost exactly 1/6 each. The corpus template is `the {buyer-word} {one of 6 verbs} the {product} after checking the price .`, so the model learned both *which* words can follow and *how evenly* they are spread.
5. **Attention.** For the same prefix, block 1, head 1 gives the *customer* position these weights: **0.485 on `<BOS>`, 0.423 on *the*, 0.092 on itself**. Each position builds its next-token prediction from a weighted mix of the vectors at itself and **earlier** positions only. A causal mask sets future positions to −∞ before the softmax, so a prediction can never peek at the word it is supposed to predict. The expanded model's head instead puts 0.764 on *customer* itself. The same architecture learned a different use for this head.

## 5. The 48 fixed language evals

**Suite and runner (unchanged):**
- [`evals/language_evals.json`](evals/language_evals.json) (SHA-256 `e8affcd7…`, recorded in every summary) and [`run_evals.py`](run_evals.py)
- The notebook verifies both hashes at start-up, and would refuse to run if either had been edited.
- The suite is 16 `starter_patterns`, 8 `starter_transfer` (new wording) and 24 `extend_corpus` cases.

**Scoring.** Only the prompt goes into the model, never the choices or the answer key. The runner reads the model's next-token probabilities for the four candidate words. The highest one is the model's answer: 1 if correct, 0 if wrong or tied. If the prompt or any choice contains a word outside the vocabulary, the case is `out_of_vocabulary`. It is not scored, and it counts as 0 in all-case success. Random guessing would average 25%. Separately, the runner saves a free continuation (temperature 0.8, fixed per-case seed, 24 tokens), which is **not** scored.

### Scores by category (correct / scorable / total)

| Group · category | Starter untrained | Starter trained | Expanded untrained | Expanded trained |
|---|---|---|---|---|
| starter · domain_context | 3/8/8 | **8/8/8** | 2/8/8 | **8/8/8** |
| starter · domain_place | 3/8/8 | **8/8/8** | 1/8/8 | **8/8/8** |
| transfer · new_wording | 3/8/8 | 4/8/8 | 2/8/8 | **8/8/8** |
| extend · **grammar** (targeted) | 0/0/3 | 0/0/3 | 1/3/3 | **3/3/3** |
| extend · **opposites** (targeted) | 0/0/3 | 0/0/3 | 1/3/3 | 1/3/3 |
| extend · **negation** (targeted) | 0/0/3 | 0/0/3 | 0/2/3 | 0/2/3 |
| extend · **spatial_relations** (targeted) | 0/0/3 | 0/0/3 | 1/3/3 | **2/3/3** |
| extend · reference (control) | 0/0/3 | 0/0/3 | 0/0/3 | 0/0/3 |
| extend · sequence (control) | 0/0/3 | 0/0/3 | 0/0/3 | 0/0/3 |
| extend · everyday_knowledge (control) | 0/0/3 | 0/0/3 | 0/0/3 | 0/0/3 |
| extend · categories_and_analogies (control) | 0/0/3 | 0/0/3 | 0/0/3 | 0/0/3 |
| **All 48** | **9 / 24 / 48** | **20 / 24 / 48** | **8 / 35 / 48** | **30 / 35 / 48** |
| Coverage | 50.0% | 50.0% | 72.9% | 72.9% |

The untrained rows show how much luck is possible with four choices: 22–38% of scorable cases. Coverage is identical before and after training in each experiment, because it depends only on the vocabulary, which is fixed before training. **More training can never raise coverage. Only new data can.**

### Where the improvement came from: vocabulary, patterns, or both?

- **Starter patterns (16/16 in both):** learned patterns. These are the classroom templates with the exact prefixes withheld, so the model sees the same sentence frame with other nouns. The free continuations show it: `the report about the customer explains the` → *service in detail .*, and `the team discussed the mango and the juice at the` → *kitchen .*
- **New wording (starter 4/8):** a partly learned pattern. The model overfits to templates. For `yesterday the school discussed the educator and the` it picks *harvest*, not *student*. Its free continuation is `local lecturer .`: it treats the slot as another noun of the same kind, because in the templates *"and the"* is usually followed by a context word from the **same** sentence type. Passes like `our hospital discussed the nurse and the` → *health* work because *nurse* and *health* co-occur heavily.
- **Extension, starter run (0/24):** pure vocabulary. Every case contained a word the model had never seen, such as *not*, *opposite* or *below*, so it could not be scored. This is the clearest demonstration of *"more training on the original corpus cannot supply missing vocabulary"*.
- **Extension, expanded run (0 → 6 correct):** **both.**
  - **Vocabulary:** 11 targeted cases became scorable, but the untrained expanded model already got 3 of them right by luck.
  - **Learned patterns:** training added 3 more wins, and the probability margins are large, not lucky:

  | Case | Correct choice | Next-best choice | Free continuation |
  |---|---|---|---|
  | `one bird` | *is* 0.975 | *are* 0.002 | `is ready .` |
  | `the dogs` | *are* 0.967 | *is* 0.001 | `are sleeping .` |
  | `the lamp is above the desk . the desk is` | *below* 0.941 | *inside* 0.026 | `below the sock .` |

  None of these exact phrases was taught. *one bird*, *the dogs*, *yesterday she* and lamp/desk in a vertical relation were deliberately kept out of the corpus. So the model generalized the pattern to new fillers.
- **Controls (0 → 0, all unscorable):** the four untaught categories gained nothing, as intended. Any change in the targeted categories comes from the targeted teaching material, not from simply training on more data.

### Failures, with actual outputs (expanded, trained)

| Case | Picked (prob) | Correct (prob) | Free continuation | Why (see probes below) |
|---|---|---|---|---|
| `the opposite of hot is` | warm (0.080) | cold (0.006) | `cool .` | Hot/cold was taught only in contrast sentences. The model answers "opposite of" with a word from that frame's own vocabulary, here the warm/cool pair |
| `the opposite of noisy is` | late (0.035) | quiet (0.016) | `round .` | Same cause |
| `the opposite of empty is` ✓ | full (0.059) | – | `clean .` | Correct, but with a small margin: likely partly luck |
| `the box is not red . it is blue . the box is` | red (0.005) | blue (0.002) | `application .` | *blue* never appeared as the corrected colour in my stories |
| `the door is not open . it is closed . the door is` | open (0.007) | closed (0.003) | `rich .` | open/closed never appeared in a negation story |
| `the ball is left of the box . the box is to the` | left (0.394) | right (0.024) | `right of the truck .` | The inversion only works reliably for objects that appeared in that frame |
| `ava did not buy tea . she bought milk . ava bought` | unscorable | – | `. she ordered the pear .` | *ava*, *tea*, *milk* etc. are not in the vocabulary, by my choice |

The left/right case is a good illustration of **the four-choice score and the free text measuring different things**. The scored choice is wrong (*left* 0.394). But the sampled continuation happened to be `right of the truck .`, because sampling at T = 0.8 sometimes picks a lower-probability token. Neither result alone tells the full story.

### What the model actually learned: probes

To separate "pattern not learned" from "pattern tied to familiar words", I probed the expanded model with **my own prompts, not eval cases**. The probes are inference only: [`experiments/analysis/probe_patterns.py`](experiments/analysis/probe_patterns.py) → [`probe_results.json`](experiments/analysis/probe_results.json).

| Probe | Top prediction |
|---|---|
| `the kite is not green . it is pink . the kite is` (familiar object, colours) | **pink 0.915** |
| `the crate is not green . it is pink . the crate is` (object never in a negation story) | **pink 0.896** |
| `the kite is not green . it is blue . the kite is` (*blue* never in the corrected slot) | white 0.319, blue only 0.111 |
| `the gate is not closed . it is open . the gate is` (open/closed never in the stories) | inside 0.336 |
| `the sofa is left of the piano . the piano is to the` (familiar objects) | **right 0.932** |
| `the key is left of the jar . the jar is to the` (objects never in this frame) | right 0.497, south 0.202 |
| `the opposite of big is` (pair taught in this frame) | **small 0.53** |
| `the opposite of full is` (pair taught only in contrasts) | cool 0.399, empty 0.106 |

**Conclusion:** the model did learn the correction and inversion patterns, and they carry over to new *objects*. But they do **not** carry over to a new *answer word*. It copies colours it has seen in the corrected slot, not "whatever word follows *it is*". It also didn't combine two kinds of sentence: it never linked "hot and cold contrast in scenes" with "the opposite of X is Y". Ironically, the negation gap was created by my leakage precautions. I kept *red → blue* and *open → closed* out of the stories so that no 5-word run matched a test, and those were exactly the fillers the tests needed.

### Is it real? A seed check

A single seed can't separate a real change from luck, so I retrained both corpora with seeds 1, 2 and 3. [`experiments/analysis/seed_check.py`](experiments/analysis/seed_check.py) runs the unmodified `custom_llm.py` with only `SEED` changed; that also changes the split and the batches. Results: [seed_check_summary.json](experiments/analysis/seed_check/seed_check_summary.json).

| Correct (seeds 42 / 1 / 2 / 3) | Starter corpus | Expanded corpus |
|---|---|---|
| starter_patterns /16 | 16 / 16 / 16 / 16 | 16 / 16 / 16 / 16 |
| starter_transfer /8 | 4 / 6 / 7 / 7 | 8 / 7 / 8 / 6 |
| extend_corpus /24 | 0 / 0 / 0 / 0 | 6 / 9 / 7 / 9 |
| grammar · opposites · negation · spatial | – | 3·1·0·2 / 3·3·1·2 / 3·1·1·2 / 3·2·2·2 |
| **total /48** | **20 / 22 / 23 / 23** | **30 / 32 / 31 / 31** |

- **Real:** the extension gain (0 → 6–9 under every seed), grammar 3/3 and spatial 2/3 every time.
- **Noise:** the new-wording jump. Seed 42's starter run (4/8) was simply its worst seed, and the ranges overlap (4–7 vs 6–8). I don't credit my corpus for it.
- **Fragile:** opposites (1–3) and negation (0–2) swing with the seed, which fits the probes.
- **Coverage luck:** in seed 2, *door* ended up only in validation and dropped out of the vocabulary, so the door case became unscorable. My corpus has only one sentence containing *door*, and only one containing *book*.

### Keeping the exam out of the textbook

- **The notebook's own checks** ([starter](experiments/starter/run/eval_separation.json) · [expanded](experiments/expanded/run/eval_separation.json) `eval_separation.json`):
  - It withheld the 160 classroom sentences that contained an eval prompt, before the split and before building the vocabulary.
  - It would have rejected any imported passage containing an exact prompt.
  - It refuses a corpus folder that contains `evals/`.
  - `CORPUS_FOLDER` stayed `corpus/`.
- **My stricter check** ([`check_leakage.py`](check_leakage.py) → [leakage_check.json](experiments/expanded/leakage_check.json)) tested all 1,200 imported passages for four things. It found **0 exact prompt matches**, **0 shared runs of 5+ consecutive tokens** with any eval prompt *or explanation*, **0 passages containing all four choices of a case**, and **0 eval names**. At 4 tokens, the only overlap is the phrase `is left of the`: the relation being taught, not a test item.
- **How I wrote the corpus:**
  - [`make_extension_corpus.py`](make_extension_corpus.py) never opens `evals/`.
  - It uses a different cast of names (mia, sam, zoe…).
  - It never places a test's objects in the relation that test asks about: no book-in-bag, lamp-above-desk, ball-left-of-box or box/door negation, and never the phrases *one bird*, *the dogs* or *yesterday she*.
  - It keeps tested opposite pairs out of the `the opposite of … is` frame.
  - It avoids content words from the four control categories. Two words overlap unavoidably: *book* (needed for spatial) and *bird* (needed for grammar) also appear in one control case each, and both of those cases remain unscorable.
- **Nothing else entered training:** no eval outputs, chat transcripts or README text went into `corpus/`.
- **The limits:** exact-match and n-gram checks can't detect paraphrases or semantic overlap. I also looked at the eval categories while writing the teaching material, and re-inspected results afterwards. So this is a **public development benchmark**, not an unseen test of generalization. A generalization claim would need fresh tests that never guided my choices.

**Why multi-sentence stories have no space after internal periods.** The notebook splits passages at every `.` followed by whitespace. Written normally, `The kite is not green. It is pink.` would become two separate one-sentence passages, and the model would never see a correction inside one context. Writing `the kite is not green.it is pink.the kite is pink.` keeps the story in one passage. It tokenizes to exactly the same tokens as `green . it`, which you can check in [corpus.txt](experiments/expanded/run/corpus.txt).

## 6. Chat interface

The interface is [`chat.py`](chat.py), the supplied terminal loop, unchanged. It loads `model.pt` together with its saved vocabulary, and each prompt starts fresh, with no conversation memory. Every reply is generated by my trained nanoGPT at temperature 0.8, with at most 24 tokens and seed 2026 + turn number. The script prints unknown words and warns when a prompt is longer than the 48-token context. It never trains on the chat, and nothing it produces goes into `corpus/`.

**Model used:** the expanded-corpus run, [`experiments/expanded/run/model.pt`](experiments/expanded/run/model.pt), 3,000 steps. Its weights have SHA-256 `aafa9d5a…`, recorded in the transcript.

![chat screenshot](experiments/expanded/chat/chat_screenshot.png)

Transcript: [chat_transcript.json](experiments/expanded/chat/chat_transcript.json). Four real interactions:

| You | Model | Note |
|---|---|---|
| `the report about the nurse` | `explains the treatment in detail .` | Continues a classroom template |
| `the pen is not purple . it is green . the pen is` | `green .` | The negation pattern, with familiar colours |
| `the shelf is above the sofa . the sofa is` | `below the shelf .` | The inverse relation, and it even names the right object |
| `what is the capital of france ?` | `.` (unknown words: `?`, capital, france, what) | **Failure/limitation**: it isn't a question answerer. It has never seen these words, so it sees `<UNK> is the <UNK> of <UNK> <UNK>` and just ends the sentence |

The prompts were typed into the live, interactive `chat.py` by [`chat_demo.exp`](experiments/expanded/chat/chat_demo.exp), a small `expect` script, so the session can be replayed exactly. The replies are the model's real output ([terminal log](experiments/expanded/chat/terminal_session.txt)). The notebook's section 10 chat cell also ran once in each executed notebook, and its transcript is in each run folder as `chat_transcript.json`.

## 7. Reproduce everything

Requirements: Python 3.10+ (I used 3.13), plus `torch`, `pypdf`, `numpy`, and `jupyter`/`nbconvert` for the notebooks. CPU is enough; no API keys or pretrained weights are needed.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt numpy nbconvert
```

**Chat with the saved model** (type `/quit` to exit; use a new transcript name each time):

```bash
.venv/bin/python chat.py --model experiments/expanded/run/model.pt --transcript results/my-chat.json
```

**Rerun the 48 evals on saved weights.** These use inference only and write to a fresh folder. Use the same commands with `experiments/starter/run/...` for the starter experiment.

```bash
.venv/bin/python run_evals.py --model experiments/expanded/run/model.pt --output results/expanded-final
```

```bash
.venv/bin/python run_evals.py --model experiments/expanded/run/model_untrained.pt --stage untrained --output results/expanded-untrained
```

Rerunning reproduces the saved scores exactly, and the model hashes match those in the saved summaries.

**Retrain both experiments from scratch.** [`run_notebook.py`](run_notebook.py) copies the starter notebook, sets section 1 and the prediction, then runs all cells. Experiment 1 must run with the extension files moved out of `corpus/`:

```bash
mv corpus/extension /tmp/extension && .venv/bin/python run_notebook.py --steps 3000 --prediction experiments/starter/prediction.md --output results/starter.ipynb && mv /tmp/extension corpus/extension
```

```bash
.venv/bin/python make_extension_corpus.py && .venv/bin/python check_leakage.py --output results/leakage.json && .venv/bin/python run_notebook.py --steps 3000 --prediction experiments/expanded/prediction.md --output results/expanded.ipynb
```

Each run writes a new timestamped folder in `llm_runs/`, which is git-ignored. I moved mine to `experiments/`. The notebook also still opens in Colab or Jupyter: set section 1 and Run All. To view embeddings, open [`embedding-viewer.html`](embedding-viewer.html), choose **Open your checkpoint**, and load either run's `checkpoint.json`.

## 8. What I learned

1. **Corpus.** The corpus is the model's only source of knowledge. The starter corpus teaches eight topic families in eight fixed sentence frames, and nothing else: no negation, no spatial words, no *bird*. Held-out passages keep me honest about memorization. But since validation reuses the training templates, the low validation loss (0.71) shows fitting *within* templates, not general language ability.
2. **Token, ID, vector, embedding.** A token is a unit of text (*customer*). Its ID is an arbitrary row number (28). Its vector is the 64 numbers stored in that row. "Embedding" is the name for the learned table of those vectors. Only the vector carries learned information, and even then only in relation to other vectors (*customer* ≈ *shopper* ≈ *client*).
3. **Neural network and learning.** The weights (111,872 of them) turn a sequence of vectors into next-token scores, using attention, feed-forward layers with GELU, residual connections and LayerNorm. Loss is the average −log probability of the true next token. Backpropagation gives every weight a gradient, and AdamW nudges each weight against its gradient. I saw one such nudge: −0.0000100 on `wte[28][0]`. That repeated 3,000 times drove loss from 4.93 to 0.68.
4. **Attention.** It mixes information from earlier positions, and a causal mask blocks future positions. That is why `the lamp is above the desk . the desk is` can use *above* from six tokens back to produce *below*.
5. **Probabilities → text.** The final layer gives a probability for every vocabulary word. Generation samples one word, appends it and repeats until `<EOS>`. Temperature reshapes those probabilities at sampling time without touching the weights. Low T gave the same confident template three times; T = 1.2 produced a novel but false sentence.
6. **Prediction vs outcome.** My starter predictions held. For the extension, the model learned *narrow* patterns: agreement and spatial inversion worked on new fillers, but negation and opposites failed whenever the answer word hadn't been seen in the answer slot. I can honestly conclude that new data added vocabulary and some patterns. I can't conclude that the model understands negation or opposites.

## 9. One limitation and my next experiment

**Limitation (observed).** The model's "rules" are tied to the words it has seen in each slot. It copies *pink* in a negation story at 0.9 probability, but it can't copy *blue*, because *blue* never filled that slot in training. And it answered `the opposite of hot is` with *warm*. In the same way, it answered `what is the capital of france ?` with a lone `.`. It has no way to represent words it has never seen, and no knowledge beyond its 5,792 passages.

**Next experiment.**
- **Change:** a third corpus that varies the **answer slot** as well as the objects. Every colour and state (including open/closed) would appear on both sides of the negation stories. More pairs would get both a contrast sentence *and* an "opposite of" sentence. The frequency of rare words like *door* and *book* would go up. The eval stories would still be kept out, using the same leakage check.
- **Prediction:** negation and opposites should become stable across seeds, while starter scores stay at 16/16. To test *generalization* rather than development-set fitting, I would first write a separate set of fresh probe cases with new words and never look at them while writing the corpus.

## Repository map

```text
custom_llm.ipynb / custom_llm.py    starter notebook (unexecuted source) and its script form
nanogpt_model.py                    Karpathy's nanoGPT model.py, unchanged (MIT, NANOGPT_LICENSE)
run_evals.py, evals/                fixed eval runner and 48-case suite, unchanged
chat.py                             terminal chat interface, unchanged
run_notebook.py                     mine: executes a copy of the notebook with my settings
make_extension_corpus.py            mine: generates corpus/extension/*.txt
check_leakage.py                    mine: stricter eval-separation check
corpus/extension/                   my 1,200 teaching passages (4 files)
experiments/smoke_test/             10-step setup run
experiments/starter/                experiment 1: prediction, executed notebook, run folder + ZIP
experiments/expanded/               experiment 2: same, plus leakage report and chat evidence
experiments/analysis/               probes and seed check
```
