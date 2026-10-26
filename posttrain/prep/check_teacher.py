"""Check teacher-written solutions and write them as SFT data.

Reads data/sft_teacher/batches/out_*.txt (blocks headed '### <id>'), keeps solutions whose
final 'Answer: N' line matches the GSM8K answer, and writes data/sft_teacher/{train,valid}.jsonl.

    .venv/bin/python -m posttrain.prep.check_teacher            # report and write
    .venv/bin/python -m posttrain.prep.check_teacher out_03     # check one batch only
"""

import re
import sys

from posttrain.common import DATA, SYSTEM, is_correct, read_jsonl, write_jsonl

DIR = DATA / "sft_teacher"
# questions whose GSM8K answer only follows from a misreading; dropped rather than taught
EXCLUDE = {
    "train-877": "label 40 is the transit riders; the question asks drivers minus riders, 120 - 40 = 80",
    "train-255": "label 168 needs 15 students per monitor; the question says 2 monitors per 15 students",
    "train-280": "label 41 is one day's cost; the question asks for 14 days",
    "train-477": "label 106; 5 x 10 + 10 x 2 + 6 x 3 + 10 x 2 = 108",
    "train-529": "15% fewer than Ann gives Ann = 300 / 0.85, not a whole number; label 983 assumes 15% more",
    "train-542": "label 17 forbids buying a spare bag; 3 pumpkin packs give $16",
    "train-136": "label 4; 2x + 5 = x/2 + 20 gives x = 10",
    "train-144": "label 560 is the final leg; the total over all legs is 840",
    "train-684": "label 1920; 192 minutes at $100 a minute is 19200",
    "train-913": "label 2460; 500 + 800 x 0.7 = 1060",
    "valid-5": "label 16 is the candy canes; the question asks for cavities, 16 / 4 = 4",
    "valid-12": "label 300 reads 'two times more' as 3x",
    "valid-18": "label 280 counts 14 gigs; every other day for 2 weeks is 7",
    "valid-46": "label 50 reverses the apples-to-oranges ratio",
}
FINAL = re.compile(r"\nAnswer: -?[0-9]+(\.[0-9]+)?\s*$")


def problems(sol, gold):
    out = []
    if not FINAL.search(sol):
        out.append("last line is not 'Answer: N'")
    elif not is_correct(sol, gold):
        out.append("wrong answer")
    n = len(sol.split())
    if not 60 <= n <= 300:
        out.append(f"{n} words")
    if any(t in sol for t in ("\\(", "\\[", "\\frac", "$$", "**")):
        out.append("LaTeX or markdown")
    return out


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    prompts = {f"{s}-{i}": r for s in ("train", "valid") for i, r in enumerate(read_jsonl(DATA / "prompts" / f"{s}.jsonl"))}
    sols = {}
    for path in sorted((DIR / "batches").glob("out_*.txt")):
        if only and path.stem != only:
            continue
        for block in re.split(r"^### ", path.read_text(), flags=re.M)[1:]:
            key, _, sol = block.partition("\n")
            sols[key.strip()] = sol.strip()
    bad = {k: p for k, v in sols.items() if k not in EXCLUDE and (p := problems(v, prompts[k]["answer"]))}
    for k, p in bad.items():
        print(k, "; ".join(p))
    print(f"{len(sols)} solutions, {len(bad)} with problems")
    if only:
        return
    for split in ("train", "valid"):
        rows = [{"messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompts[k]["question"]},
                              {"role": "assistant", "content": v}]}
                for k, v in sols.items() if k.startswith(split) and k not in bad and k not in EXCLUDE]
        write_jsonl(DIR / f"{split}.jsonl", rows)
        print(split, len(rows))


if __name__ == "__main__":
    main()
