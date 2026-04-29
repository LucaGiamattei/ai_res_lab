#!/usr/bin/env python3
"""verify_setup.py — Script di validazione end-to-end del pacchetto AI Lab.

Esegue una serie di controlli:
  1. Versione Python (>= 3.11)
  2. Import di tutte le librerie richieste
  3. C1: caricamento German Credit, training baseline, calcolo di una metrica fairlearn
  4. C2: caricamento MobileNetV2, predizione su immagine campione
  5. C3: istanziazione del client Groq; se GROQ_API_KEY è impostata, una chiamata trivial

Stampa un report PASS/FAIL al termine.

Uso:
    python verify_setup.py            # senza chiamata live a Groq
    GROQ_API_KEY=gsk_... python verify_setup.py   # con chiamata live a Groq

Idempotente. Output in italiano.
"""
from __future__ import annotations
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ANSI colors (disabilitati se stdout non è un TTY).
ISTTY = sys.stdout.isatty()
GREEN = "\033[92m" if ISTTY else ""
RED = "\033[91m" if ISTTY else ""
YELLOW = "\033[93m" if ISTTY else ""
BOLD = "\033[1m" if ISTTY else ""
RESET = "\033[0m" if ISTTY else ""


class Result:
    def __init__(self):
        # status: "pass", "fail", "warn"
        self.checks: list[tuple[str, str, str]] = []

    def add(self, name: str, ok: bool, detail: str = ""):
        status = "pass" if ok else "fail"
        self.checks.append((name, status, detail))
        marker = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{marker}] {name}" + (f"  — {detail}" if detail else ""))

    def warn(self, name: str, detail: str = ""):
        self.checks.append((name, "warn", detail))
        print(f"  [{YELLOW}WARN{RESET}] {name}" + (f"  — {detail}" if detail else ""))

    def summary(self) -> int:
        passed = sum(1 for _, s, _ in self.checks if s == "pass")
        warned = sum(1 for _, s, _ in self.checks if s == "warn")
        failed = sum(1 for _, s, _ in self.checks if s == "fail")
        total = len(self.checks)
        print()
        print(f"{BOLD}=== Riepilogo: {passed}/{total} pass, {warned} warn, {failed} fail ==={RESET}")
        if failed == 0:
            if warned == 0:
                print(f"{GREEN}{BOLD}TUTTI I CONTROLLI SONO PASSATI.{RESET}")
            else:
                print(f"{GREEN}{BOLD}PASS{RESET} — alcuni controlli opzionali non eseguiti (vedi WARN).")
            return 0
        print(f"{RED}{BOLD}ATTENZIONE: {failed} controlli falliti.{RESET}")
        for name, status, detail in self.checks:
            if status == "fail":
                print(f"  - {name}: {detail}")
        return 1


def check_python(r: Result):
    print(f"\n{BOLD}1. Versione Python{RESET}")
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    r.add(
        f"Python >= 3.11 (rilevato {v.major}.{v.minor}.{v.micro})",
        ok,
        "Si raccomanda Python 3.11; versioni 3.10–3.12 dovrebbero funzionare." if not ok else "",
    )


def check_imports(r: Result):
    print(f"\n{BOLD}2. Import librerie{RESET}")
    libs = [
        ("numpy",        "Algebra lineare"),
        ("pandas",       "DataFrame"),
        ("sklearn",      "scikit-learn — modelli baseline"),
        ("fairlearn",    "Fairness audit"),
        ("matplotlib",   "Plotting"),
        ("PIL",          "Image I/O"),
        ("torch",        "PyTorch"),
        ("torchvision",  "Modelli ImageNet pre-addestrati"),
        ("groq",         "Client API Groq (per C3)"),
    ]
    for mod, desc in libs:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "n/a")
            r.add(f"import {mod}", True, f"{desc} — versione {v}")
        except Exception as e:
            r.add(f"import {mod}", False, f"{desc} — errore: {e}")


def check_c1(r: Result):
    print(f"\n{BOLD}3. C1 — Fairness audit (German Credit){RESET}")
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        from fairlearn.metrics import (
            MetricFrame, demographic_parity_difference, selection_rate,
        )
        from sklearn.metrics import accuracy_score
    except Exception as e:
        r.add("C1 setup imports", False, str(e))
        return

    try:
        ds = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
        X = ds.data.copy()
        y = (ds.target == "good").astype(int)
        # Quick sex-from-personal_status mapping (mirrors what students implement).
        ps_to_sex = {
            "male single": "male",
            "male div/sep": "male",
            "male mar/wid": "male",
            "female div/dep/mar": "female",
        }
        sex = X["personal_status"].map(ps_to_sex)
        # Drop personal_status from features to avoid leakage; keep only numeric features for speed.
        Xn = X.drop(columns=["personal_status"]).select_dtypes(include="number")
        Xtr, Xte, ytr, yte, str_, ste = train_test_split(
            Xn, y, sex, test_size=0.25, stratify=y, random_state=42
        )
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ytr)
        yp = clf.predict(Xte_s)
        acc = accuracy_score(yte, yp)
        mf = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=yte, y_pred=yp, sensitive_features=ste,
        )
        dpd = demographic_parity_difference(yte, yp, sensitive_features=ste)
        ok = acc > 0.6 and 0.0 <= abs(dpd) <= 1.0
        r.add(
            "Caricamento, training, fairness check su German Credit",
            ok,
            f"accuracy={acc:.3f}, demographic parity diff={dpd:.3f}, gruppi={list(mf.by_group.index)}"
        )
    except Exception as e:
        traceback.print_exc()
        r.add("C1 end-to-end", False, str(e))


def check_c2(r: Result):
    print(f"\n{BOLD}4. C2 — MobileNetV2 + immagine campione{RESET}")
    try:
        import torch
        from PIL import Image
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    except Exception as e:
        r.add("C2 setup imports", False, str(e))
        return

    img_path = ROOT / "challenge_2_adversarial" / "data" / "images" / "panda.jpg"
    cls_path = ROOT / "challenge_2_adversarial" / "data" / "imagenet_classes.txt"
    if not img_path.exists():
        r.add("C2 immagine campione presente", False, f"manca {img_path}")
        return
    if not cls_path.exists():
        r.add("C2 imagenet_classes.txt presente", False, f"manca {cls_path}")
        return

    try:
        classes = [l.strip() for l in cls_path.read_text().splitlines() if l.strip()]
        if len(classes) != 1000:
            r.add("C2 numero classi ImageNet", False, f"atteso 1000, trovato {len(classes)}")
            return
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        model = mobilenet_v2(weights=weights).eval()
        preprocess = weights.transforms()
        img = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            logits = model(preprocess(img).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
        top5_idx = torch.topk(probs, 5).indices.tolist()
        # 388 = giant panda
        ok = 388 in top5_idx
        top1_name = classes[top5_idx[0]]
        r.add(
            "Predizione MobileNetV2 su panda.jpg (388 nei top-5)",
            ok,
            f"top-1={top1_name!r}, top-5 ids={top5_idx}"
        )
    except Exception as e:
        traceback.print_exc()
        r.add("C2 end-to-end", False, str(e))


def check_c3(r: Result):
    print(f"\n{BOLD}5. C3 — Client Groq{RESET}")
    try:
        import groq
    except Exception as e:
        r.add("C3 import groq", False, str(e))
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        r.warn(
            "GROQ_API_KEY non impostata",
            "Imposta GROQ_API_KEY per testare la connessione live (controllo opzionale)."
        )
        # Non blocking: client instantiation con chiave fittizia (deve fallire in modo controllato).
        try:
            _ = groq.Groq(api_key="gsk_dummy_key_for_smoke_test")
            r.add("Istanziazione client Groq con chiave dummy", True, "client costruito; nessuna chiamata effettuata.")
        except Exception as e:
            r.add("Istanziazione client Groq con chiave dummy", False, str(e))
        return

    try:
        client = groq.Groq(api_key=api_key)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Reply with the single word: PONG."}],
            max_tokens=10,
            temperature=0.0,
        )
        dt = time.perf_counter() - t0
        content = (resp.choices[0].message.content or "").strip()
        ok = "PONG" in content.upper()
        r.add(
            "Chiamata live a Groq llama-3.1-8b-instant",
            ok,
            f"risposta={content!r} (latency={dt:.2f}s)"
        )
    except Exception as e:
        traceback.print_exc()
        r.add("Chiamata live a Groq", False, str(e))


def check_package_files(r: Result):
    print(f"\n{BOLD}6. File del pacchetto{RESET}")
    expected = [
        "README.md",
        "requirements.txt",
        "docs/EVIDENCE_TEMPLATE.md",
        "docs/AI_ACT_MAPPING.md",
        "shared/evidence_template.csv",
        "challenge_1_fairness/README.md",
        "challenge_1_fairness/starter.ipynb",
        "challenge_1_fairness/solution.ipynb",
        "challenge_2_adversarial/README.md",
        "challenge_2_adversarial/starter.ipynb",
        "challenge_2_adversarial/solution.ipynb",
        "challenge_2_adversarial/data/imagenet_classes.txt",
        "challenge_2_adversarial/data/images/panda.jpg",
        "challenge_2_adversarial/data/images/school_bus.jpg",
        "challenge_2_adversarial/data/images/golden_retriever.jpg",
        "challenge_2_adversarial/data/images/traffic_light.jpg",
        "challenge_2_adversarial/data/images/espresso.jpg",
        "challenge_3_llm_redteam/README.md",
        "challenge_3_llm_redteam/starter.ipynb",
        "challenge_3_llm_redteam/solution.ipynb",
        "challenge_3_llm_redteam/solution_cheatsheet.md",
        "challenge_3_llm_redteam/prompts/bank_assistant_system.txt",
        "challenge_3_llm_redteam/prompts/injection_email_1.txt",
        "challenge_3_llm_redteam/prompts/injection_email_2.txt",
        "challenge_4_llm_defense/README.md",
        "challenge_4_llm_defense/starter.ipynb",
        "challenge_4_llm_defense/solution.ipynb",
        "challenge_4_llm_defense/prompts/judge_system.txt",
        "challenge_4_llm_defense/prompts/bank_assistant_hardened.txt",
        "challenge_4_llm_defense/prompts/jailbreak_seed_prompts.txt",
    ]
    for rel in expected:
        p = ROOT / rel
        r.add(f"presente: {rel}", p.exists(), "" if p.exists() else "FILE MANCANTE")


def main() -> int:
    print(f"{BOLD}=== verify_setup.py — Validazione pacchetto AI Lab Federico II ==={RESET}")
    print(f"Root: {ROOT}")
    res = Result()
    check_python(res)
    check_imports(res)
    check_package_files(res)
    check_c1(res)
    check_c2(res)
    check_c3(res)
    return res.summary()


if __name__ == "__main__":
    sys.exit(main())
