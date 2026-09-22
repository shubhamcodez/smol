"""Tokenize FineWeb-Edu parquet into uint16 train/val memmaps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from tokenizer_gpt2 import EOT_TOKEN, encode_text
from token_dataset import default_shard_paths


def _bar(fraction: float, width: int = 8) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def resource_postfix() -> str:
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    parts = [
        f"CPU {_bar(cpu / 100.0)} {cpu:4.0f}%",
        f"RAM {_bar(ram.percent / 100.0)} {ram.used / 2**30:.1f}/{ram.total / 2**30:.0f}G",
    ]
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            used_b = total_b - free_b
            try:
                util = float(torch.cuda.utilization(0))
            except Exception:
                util = 0.0
            parts.append(
                f"GPU {_bar(util / 100.0)} {util:4.0f}% "
                f"{used_b / 2**30:.1f}/{total_b / 2**30:.0f}G"
            )
            _ = props
    except Exception:
        pass
    return " | ".join(parts)


def iter_texts(parquet_files: list[Path], desc: str):
    file_bar = tqdm(parquet_files, desc=f"{desc} files", unit="file", leave=False)
    for path in file_bar:
        file_bar.set_postfix_str(path.name)
        table = pq.read_table(path, columns=["text"])
        column = table.column("text")
        for index in tqdm(
            range(column.length()),
            desc=f"{desc} rows",
            unit="doc",
            leave=False,
            total=column.length(),
        ):
            text = column[index].as_py()
            if text and text.strip():
                yield text


def write_tokens(
    texts,
    output: Path,
    max_tokens: int,
    *,
    desc: str,
    buffer_tokens: int = 1_000_000,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    mmap = np.memmap(temporary, dtype=np.uint16, mode="w+", shape=(max_tokens,))
    written = 0
    buffered: list[int] = []
    last_stats = 0.0

    def flush() -> None:
        nonlocal written, buffered
        if not buffered:
            return
        chunk = np.asarray(buffered, dtype=np.uint16)
        end = written + chunk.size
        mmap[written:end] = chunk
        written = end
        buffered = []

    # Prime CPU percent so the first sample isn't 0.0.
    psutil.cpu_percent(interval=None)

    token_bar = tqdm(
        total=max_tokens,
        desc=desc,
        unit="tok",
        unit_scale=True,
        dynamic_ncols=True,
    )
    try:
        for text in texts:
            if written + len(buffered) >= max_tokens:
                break
            tokens = encode_text(text) + [EOT_TOKEN]
            remaining = max_tokens - written - len(buffered)
            if remaining <= 0:
                break
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
            buffered.extend(tokens)
            if len(buffered) >= buffer_tokens:
                before = written
                flush()
                mmap.flush()
                token_bar.update(written - before)
            now = time.perf_counter()
            if now - last_stats >= 0.5:
                token_bar.set_postfix_str(resource_postfix(), refresh=False)
                last_stats = now
        before = written
        flush()
        mmap.flush()
        token_bar.update(written - before)
        token_bar.set_postfix_str(resource_postfix())
    finally:
        token_bar.close()

    del mmap
    # Shrink to actual size.
    final = np.memmap(temporary, dtype=np.uint16, mode="r")
    actual = np.asarray(final[:written], dtype=np.uint16).copy()
    del final
    actual.tofile(output)
    temporary.unlink(missing_ok=True)
    return written


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu token shards")
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data" / "fineweb-edu" / "sample" / "10BT",
    )
    parser.add_argument("--train-tokens", type=int, default=50_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    files = sorted(args.source.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {args.source}")

    shards = default_shard_paths(root)
    start = time.perf_counter()
    print(f"tokenizing from {len(files)} parquet files -> {shards.train.parent}")

    # First file(s) feed validation so the held-out set is fixed and never reused for train.
    val_files = files[:1]
    train_files = files[1:] if len(files) > 1 else files

    val_count = write_tokens(
        iter_texts(val_files, "val"),
        shards.val,
        args.val_tokens,
        desc="val tokens",
    )
    train_count = write_tokens(
        iter_texts(train_files, "train"),
        shards.train,
        args.train_tokens,
        desc="train tokens",
    )

    meta = {
        "tokenizer": "tiktoken-gpt2",
        "dtype": "uint16",
        "eot_token": EOT_TOKEN,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "train_source_files": [str(path.name) for path in train_files],
        "val_source_files": [str(path.name) for path in val_files],
        "seed": args.seed,
        "seconds": time.perf_counter() - start,
    }
    shards.meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"WROTE_TRAIN={shards.train.resolve()}")
    print(f"WROTE_VAL={shards.val.resolve()}")


if __name__ == "__main__":
    main()
