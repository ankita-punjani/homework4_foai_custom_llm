"""Write my corpus-extension teaching files into corpus/extension/.

Targets four extension-eval categories: grammar, opposites, negation, spatial relations.
The other four (reference, sequence, everyday knowledge, categories/analogies) are left
untaught as controls.

This script never opens evals/. It was written by hand from the category descriptions,
and follows these rules (checked independently by check_leakage.py):
- no eval names (ava, maya, leo, ...): stories use a different cast;
- no eval test story: the objects each test uses are never placed in the relation that
  test asks about (no book-in-bag, lamp-above-desk, ball-left-of-box, box/door
  negation, "one bird", "the dogs", "yesterday she");
- opposite pairs that are tested (hot/cold, empty/full, noisy/quiet) appear only in
  everyday contrast sentences, never in the "the opposite of ... is ..." frame;
- control-category content words (cup, wash, lunch, umbrella, kitten, ...) are avoided.

Multi-sentence stories are written without a space after internal periods
("green.it is pink"): the notebook splits passages at "." followed by whitespace, and
tokenizes "green.it" exactly like "green . it". This keeps each story in one passage.

Each file samples evenly across its sentence frames, so a large frame cannot drown out
a small one. Fixed seed: rerunning writes identical files.

    .venv/bin/python make_extension_corpus.py
"""
import random
from itertools import permutations, product
from pathlib import Path

OUT = Path(__file__).resolve().parent / "corpus" / "extension"
PER_FILE = 300
rng = random.Random(7)


def story(*sentences):
    return ".".join(sentences) + "."


def balanced(always, frames, total=PER_FILE):
    """Keep every `always` line (so no word is lost to sampling), then take an equal
    share of the remaining budget from every frame (all of a frame if it is smaller)."""
    always = sorted(set(always))
    frames = [sorted(set(f) - set(always)) for f in frames]
    chosen, remaining = list(always), total - len(always)
    for i, frame in enumerate(sorted(frames, key=len)):
        share = remaining // (len(frames) - i)
        picked = rng.sample(frame, min(share, len(frame)))
        chosen += picked
        remaining -= len(picked)
    rng.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------- grammar
def grammar():
    nouns = [("bird", "birds"), ("frog", "frogs"), ("rabbit", "rabbits"), ("lion", "lions"),
             ("bee", "bees"), ("cow", "cows"), ("owl", "owls"), ("girl", "girls"),
             ("boy", "boys"), ("farmer", "farmers"), ("pilot", "pilots"), ("baker", "bakers")]
    states = ["sleeping", "small", "here", "outside", "happy", "busy", "hiding", "ready"]
    verbs = ["walk", "jump", "cook", "paint", "clean", "play", "visit", "climb", "talk", "help", "kick", "look"]
    singular_people = ["he", "the boy", "the girl", "the farmer", "my friend", "the pilot", "our baker"]
    people = singular_people + ["they", "we", "i"]
    singular, plural, past_tense, other_tense, she = [], [], [], [], []
    for (one, many), state in product(nouns, states):
        if one != "bird":                       # never "one bird"
            singular.append(f"one {one} is {state} .")
        singular += [f"a {one} is {state} .", f"the {one} was {state} ."]
        plural += [f"two {many} are {state} .", f"many {many} were {state} .", f"the {many} are {state} ."]
    for state in states:                        # "dogs" appears, never "the dogs"
        plural += [f"two dogs are {state} .", f"our dogs were {state} ."]
        singular.append(f"i am {state} .")
    for verb, who in product(verbs, people):
        does = "s" if who in singular_people else ""
        be = "is" if does else ("am" if who == "i" else "are")
        past_tense.append(f"yesterday {who} {verb}ed .")
        other_tense += [f"today {who} {verb}{does} .", f"right now {who} {be} {verb}ing ."]
    for verb in verbs:                          # "she" forms, never "yesterday she"
        she += [f"she {verb}ed yesterday .", f"every day she {verb}s .", f"she is {verb}ing now .",
                f"she likes to {verb} ."]
    always = [f"a {one} is {states[i % len(states)]} ." for i, (one, _) in enumerate(nouns)]
    always += [f"two {many} are {states[(i + 3) % len(states)]} ." for i, (_, many) in enumerate(nouns)]
    return always, [singular, plural, past_tense, other_tense, she]


# ---------------------------------------------------------------- opposites
def opposites():
    # Only untested pairs use the explicit "opposite" frame.
    framed = [("big", "small"), ("fast", "slow"), ("tall", "short"), ("old", "young"),
              ("early", "late"), ("soft", "hard"), ("thick", "thin"), ("happy", "sad"),
              ("strong", "weak"), ("clean", "dirty"), ("bright", "dim"), ("sweet", "sour"),
              ("near", "far"), ("warm", "cool"), ("rich", "poor")]
    # Tested pairs appear only as everyday contrasts, exactly like the untested pairs do.
    contrast = framed + [("hot", "cold"), ("full", "empty"), ("noisy", "quiet"), ("loud", "quiet")]
    things = {"hot": "soup", "full": "jar", "noisy": "street", "loud": "radio"}
    times = [("in the morning", "at night"), ("in summer", "in winter"), ("at dawn", "at dusk"),
             ("before noon", "after sunset")]
    places = ["park", "market", "garden", "street", "pool", "hall", "road", "field"]
    explicit, scene, short = [], [], []
    for a, b in framed:
        explicit += [f"the opposite of {a} is {b} .", f"the opposite of {b} is {a} .",
                     f"{a} and {b} mean opposite things ."]
    # Scenes use only pairs that describe a place; warm/cool, bright/dim and clean/dirty
    # bridge the scene frame and the explicit "opposite" frame.
    scene_pairs = [("warm", "cool"), ("bright", "dim"), ("clean", "dirty"), ("hot", "cold"),
                   ("full", "empty"), ("noisy", "quiet"), ("loud", "quiet")]
    for (a, b), (t1, t2), place in product(scene_pairs, times, places):
        scene += [f"the {place} was {a} {t1} but {b} {t2} .",
                  f"{t1} the {place} felt {a} , and {t2} it felt {b} ."]
    for a, b in contrast:
        short += [f"one {things.get(a, 'box')} was {a} while the other was {b} .",
                  f"it was not {a} at all ; it was {b} ."]
    short += ["the suitcase was heavy and hard to carry .", "the coin is round and thin .",
              "the heavy bag was slow to move .", "a round plate is not square ."]
    always = short[-4:]
    return always, [explicit, scene, short]


# ---------------------------------------------------------------- negation
def negation():
    objects = ["kite", "hat", "shirt", "wall", "chair", "flower", "bowl", "pen", "sock",
               "scarf", "balloon", "boat", "fence", "jacket"]
    colors = ["red", "blue", "green", "yellow", "white", "black", "pink", "brown", "purple", "gray"]
    pairs = [(a, b) for a, b in permutations(colors, 2) if a != "red" and b != "blue"]
    present, past, people = [], [], []
    for obj, (c1, c2) in product(objects, pairs):
        present.append(story(f"the {obj} is not {c1}", f"it is {c2}", f"the {obj} is {c2}"))
        past.append(story(f"we thought the {obj} was {c1} , but it was not {c1}", f"it was {c2}",
                          f"the {obj} was {c2}"))
    cast = [("mia", "she"), ("sam", "he"), ("zoe", "she"), ("ben", "he"), ("kai", "he"),
            ("ivy", "she"), ("raj", "he"), ("lucy", "she"), ("dev", "he"), ("jade", "she")]
    acts = [("pick", "picked"), ("choose", "chose"), ("order", "ordered"), ("wear", "wore"),
            ("paint", "painted"), ("bring", "brought")]
    items = ["the pear", "the peach", "the banana", "the soup", "the scarf", "the jacket",
             "the kite", "the hat", "the green shirt", "the black boat"]
    for (name, pron), (base, did), (x, y) in product(cast, acts, permutations(items, 2)):
        people.append(story(f"{name} did not {base} {x}", f"{pron} {did} {y}", f"{name} {did} {y}"))
    words = ["the shop is open every sunday .", "the shop was closed that monday .", "the river is wide .",
             "one sock is missing .", "they painted the door green .", "the gate stayed open all day .",
             "the road is wide and the gate is closed .", "the pen was missing again ."]
    return words, [present, past, people]


# ---------------------------------------------------------------- spatial relations
def spatial():
    small = ["key", "coin", "pen", "shell", "ring", "letter", "toy", "card", "ball", "stone"]
    holders = ["jar", "basket", "drawer", "pocket", "envelope", "bowl", "bucket", "tin", "crate", "purse"]
    stacked = ["clock", "mirror", "picture", "shelf", "window", "fan", "chair", "table", "rug", "bed"]
    row = ["chair", "table", "plant", "bench", "stool", "sofa", "piano", "tent", "clock", "bed"]
    places = ["park", "river", "school", "market", "bridge", "farm", "hill", "station"]
    inside, vertical, beside, sideways, compass = [], [], [], [], []
    for s, h in product(small, holders):
        inside += [story(f"the {s} is inside the {h}", f"the {h} contains the {s}"),
                   story(f"the {h} contains the {s} because the {s} is inside it")]
    for a, b in permutations(stacked, 2):
        vertical += [story(f"the {a} is above the {b}", f"the {b} is below the {a}")]
        beside += [story(f"the {a} is beside the {b}", f"the {b} is beside the {a}")]
    for a, b in permutations(row, 2):
        sideways += [story(f"the {a} is left of the {b}", f"the {b} is to the right of the {a}"),
                     story(f"the {a} is right of the {b}", f"the {b} is to the left of the {a}")]
    for a, b in permutations(places, 2):
        compass += [story(f"the {a} is north of the {b}", f"the {b} is to the south of the {a}"),
                    story(f"the {a} is south of the {b}", f"the {b} is to the north of the {a}")]
    beside += ["the lamp is beside the bed .", "the desk is beside the window .", "a book is beside the table .",
               "she put the bag beside the chair .", "the lamp and the desk are new .",
               "the box is beside the bench ."]
    return beside[-6:], [inside, vertical, beside, sideways, compass]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in [("grammar", grammar), ("opposites", opposites), ("negation", negation), ("spatial", spatial)]:
        lines = balanced(*build())
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"corpus/extension/{name}.txt: {len(lines)} passages")


if __name__ == "__main__":
    main()
