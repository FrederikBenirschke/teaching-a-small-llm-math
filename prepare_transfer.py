"""Write transfer evaluation prompts (not used for training or selection).

data/prompts/svamp.jsonl   SVAMP test split, 300 problems, {"question", "answer"}
"""

from datasets import load_dataset

from common import DATA, write_jsonl


def main():
    ds = load_dataset("ChilleD/SVAMP")["test"]
    rows = [{"question": f"{r['Body'].strip().rstrip('.')}. {r['Question'].strip()}", "answer": str(r["Answer"]).strip()}
            for r in ds]
    write_jsonl(DATA / "prompts" / "svamp.jsonl", rows)
    print(len(rows), rows[0])


if __name__ == "__main__":
    main()
