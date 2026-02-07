from pathlib import Path


def dump(start: int, end: int):
    text = Path("cortexltm/summaries.py").read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(text, 1):
        if start <= idx <= end:
            print(f"{idx:04d}: {line}")


if __name__ == "__main__":
    dump(130, 220)
