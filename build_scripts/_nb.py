"""Notebook construction helpers (nbformat v4)."""
from __future__ import annotations
import nbformat as nbf
from pathlib import Path

# Public GitHub repo from which the notebooks fetch assets when opened standalone on Colab.
# >>> SWAP THIS BEFORE DISTRIBUTION <<<
# Replace with the real public URL of the repo holding this package, e.g.
#   "https://github.com/ai-res/federico-ii-lab.git"
REPO_URL = "https://github.com/LucaGiamattei/ai_res_lab.git"
REPO_BRANCH = "main"
# The directory created by `git clone` inside Colab's /content/.
REPO_DIR_NAME = "ai_res_lab"


def bootstrap_cell(challenge_dir: str, required_files: list[str]) -> str:
    """Return a code-cell source string that:
      1. Detects Colab.
      2. If running standalone (no shared/ dir found by walking parents), git-clones REPO_URL
         into /content/REPO_DIR_NAME (or cwd) and chdirs into the challenge directory.
      3. Verifies that `required_files` are present after the bootstrap; raises FileNotFoundError
         otherwise with a clear Italian message.

    `challenge_dir` is the package-relative path of the current challenge, e.g. "challenge_3_llm_redteam".
    `required_files` are paths relative to the package root (e.g. ["challenge_3_llm_redteam/prompts/bank_assistant_system.txt"]).
    """
    files_repr = ", ".join(repr(f) for f in required_files)
    return f'''
# --- Bootstrap (auto-fetch assets su Colab quando si apre solo il .ipynb) ---
import os, subprocess, shutil
from pathlib import Path

REPO_URL    = {REPO_URL!r}      # NB: docente, sostituire con l'URL pubblico reale del repo
REPO_BRANCH = {REPO_BRANCH!r}
REPO_DIR    = {REPO_DIR_NAME!r}
CHALLENGE_DIR = {challenge_dir!r}
REQUIRED_FILES = [{files_repr}]

try:
    import google.colab  # noqa: F401
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

def _walk_to_package_root(start: Path) -> Path | None:
    p = start.resolve()
    for _ in range(8):
        if (p / "shared").is_dir() and (p / CHALLENGE_DIR).is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None

_root = _walk_to_package_root(Path.cwd())
if _root is None:
    if _IN_COLAB:
        target = Path("/content") / REPO_DIR
        if not target.exists():
            print(f"Cloning {{REPO_URL}} (branch={{REPO_BRANCH}}) ...")
            res = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(target)],
                capture_output=True, text=True,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    "Clone fallito. Controllare REPO_URL nella cella di bootstrap.\\n"
                    f"stderr: {{res.stderr}}"
                )
        os.chdir(target / CHALLENGE_DIR)
        _root = target
    else:
        raise FileNotFoundError(
            "Pacchetto non trovato. In locale, aprire il notebook dalla cartella del pacchetto.\\n"
            "Su Colab, viene clonato automaticamente; verificate REPO_URL."
        )
else:
    os.chdir(_root / CHALLENGE_DIR)

PACKAGE_ROOT = _root
print(f"PACKAGE_ROOT = {{PACKAGE_ROOT}}")
print(f"CWD          = {{Path.cwd()}}")

# Sanity check.
missing = [f for f in REQUIRED_FILES if not (PACKAGE_ROOT / f).is_file()]
if missing:
    raise FileNotFoundError("File mancanti dopo il bootstrap:\\n  - " + "\\n  - ".join(missing))
print("Bootstrap OK.")
'''


def make_nb(cells: list, kernel: str = "python3") -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": kernel},
        "language_info": {"name": "python", "version": "3.11"},
    })
    return nb


def make_nb(cells: list, kernel: str = "python3") -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": kernel},
        "language_info": {"name": "python", "version": "3.11"},
    })
    return nb


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.lstrip("\n"))


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.lstrip("\n"))


def write(nb: nbf.NotebookNode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
