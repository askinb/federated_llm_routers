"""Rebuild the dataset parquet files from the chunks in data/chunks/.

Each experiment script calls ensure_dataset() on startup, so this runs automatically the first
time a dataset is used. To rebuild all datasets up front instead:

    python data/reassemble.py
"""
import hashlib
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")
DATASETS = ["routerbench_0shot_w4emb.parquet", "proxrouter_train.parquet", "sprout_w4emb.parquet"]


def _expected_sha256(name):
    with open(os.path.join(DATA_DIR, "SHA256SUMS")) as f:
        for line in f:
            digest, fname = line.split()
            if fname == name:
                return digest
    raise KeyError(f"{name} is not listed in data/SHA256SUMS")


def ensure_dataset(name):
    """Return the path of data/<name>, reassembling it from its chunks if it does not exist yet."""
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return path
    parts = sorted(p for p in os.listdir(CHUNK_DIR) if p.startswith(name + ".part"))
    if not parts:
        raise FileNotFoundError(f"No chunks for {name} in {CHUNK_DIR}")
    print(f"Reassembling {name} from {len(parts)} chunks...")
    sha = hashlib.sha256()
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as out:
        for p in parts:
            with open(os.path.join(CHUNK_DIR, p), "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    sha.update(block)
                    out.write(block)
    if sha.hexdigest() != _expected_sha256(name):
        os.remove(tmp)
        raise RuntimeError(f"Checksum mismatch for {name}: data/chunks/ is incomplete or corrupted")
    os.replace(tmp, path)
    return path


if __name__ == "__main__":
    for name in sys.argv[1:] or DATASETS:
        print(ensure_dataset(name))
