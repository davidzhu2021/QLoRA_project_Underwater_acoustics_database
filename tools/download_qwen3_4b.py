from pathlib import Path

from modelscope import snapshot_download


MODEL_ID = "Qwen/Qwen3-4B"
CACHE_DIR = Path("D:/model_cache/modelscope")
TARGET_DIR = CACHE_DIR / "hub" / "models" / "Qwen" / "Qwen3-4B"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(MODEL_ID, cache_dir=str(CACHE_DIR))
    print(f"Downloaded model to: {path}")
    print(f"Expected project config path: {TARGET_DIR}")


if __name__ == "__main__":
    main()

