import argparse
import json
from pathlib import Path


def convert_split(in_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row["text"]
            answers = []
            for quad in row["quadruples"]:
                a0, a1 = quad["aspect_span"]
                o0, o1 = quad["opinion_span"]
                answers.append(
                    [
                        quad["category"],
                        f"{a0},{a1}",
                        f"{o0},{o1}",
                        str(quad["sentiment"]),
                    ]
                )
            fout.write(f"{text}####{json.dumps(answers, ensure_ascii=False)}\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("D:/Python_main/ASQP/data/processed/Restaurant-ACOS"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("D:/Python_main/One_ASQP/One_ASQP/data/Restaurant-ACOS"),
    )
    args = parser.parse_args()

    expected = {"train": 1530, "dev": 171, "test": 583}
    observed = {}
    for split in ("train", "dev", "test"):
        in_path = args.input_dir / f"{split}.jsonl"
        out_path = args.output_dir / f"{split}.txt"
        observed[split] = convert_split(in_path, out_path)

    print(json.dumps({"expected": expected, "observed": observed}, indent=2, ensure_ascii=False))
    mismatch = [k for k, v in expected.items() if observed.get(k) != v]
    if mismatch:
        raise SystemExit(f"Count mismatch for splits: {mismatch}")


if __name__ == "__main__":
    main()
