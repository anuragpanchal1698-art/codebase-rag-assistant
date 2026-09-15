"""Clones a GitHub repo and loads code + doc files into memory."""
import os
from pathlib import Path
from git import Repo
from config import GITHUB_REPO, REPO_CLONE_DIR

CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".next"}


def clone_repo(repo_slug: str = None, dest: str = None) -> str:
    """Clone repo if not already present. repo_slug like 'langchain-ai/langchain'."""
    repo_slug = repo_slug or GITHUB_REPO
    dest = dest or REPO_CLONE_DIR

    if os.path.exists(dest) and os.path.exists(os.path.join(dest, ".git")):
        print(f"Repo already cloned at {dest}, pulling latest...")
        try:
            Repo(dest).remotes.origin.pull()
        except Exception as e:
            print(f"Pull failed (continuing with existing clone): {e}")
        return dest

    url = f"https://github.com/{repo_slug}.git"
    print(f"Cloning {url} -> {dest} (depth=1, this may take a minute)...")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Repo.clone_from(url, dest, depth=1)
    return dest


def load_files(repo_path: str) -> list[dict]:
    """Walk repo, return list of {path, content, type} for code and doc files."""
    results = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

        for fname in files:
            ext = Path(fname).suffix.lower()
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_path)

            file_type = None
            if ext in CODE_EXTENSIONS:
                file_type = "code"
            elif ext in DOC_EXTENSIONS:
                file_type = "doc"
            else:
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if not content.strip():
                    continue
                # Skip huge generated/lock-like files
                if len(content) > 200_000:
                    continue
                results.append({
                    "path": rel_path,
                    "content": content,
                    "type": file_type,
                    "extension": ext,
                })
            except Exception as e:
                print(f"Skipping {fpath}: {e}")

    print(f"Loaded {len(results)} files "
          f"({sum(1 for r in results if r['type']=='code')} code, "
          f"{sum(1 for r in results if r['type']=='doc')} docs)")
    return results


if __name__ == "__main__":
    path = clone_repo()
    files = load_files(path)
    print(files[0] if files else "No files found")
