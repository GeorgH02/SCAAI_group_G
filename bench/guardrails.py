"""Evaluate the guardrails on the labeled prompt sets and write docs/guardrails_eval.md.

  python -m bench.guardrails

Input guard: rules + topic gate (bge-m3), as the app runs it (Llama Guard is optional and off).
Evidence guard: the review texts labeled ev_benign / ev_injection.
Output guard: bench/output_guard_cases.txt, sentences checked against the saved scenario B evidence.
"""

import json
import statistics
from pathlib import Path

from guardrails.evidence_guard import find_injection
from guardrails.prompts import load_prompts, split

ROOT = Path(__file__).resolve().parents[1]
GUARD_DIR = ROOT / "guardrails"
REPORT = ROOT / "docs" / "guardrails_eval.md"
CASES = ROOT / "bench" / "output_guard_cases.txt"
ATTACKS = ("injection", "identity", "off_topic", "other_city")


def question_sets():
    test_prompts = load_prompts(GUARD_DIR / "test_prompts.txt")
    return [
        ("test_prompts.txt, tuning half", split(test_prompts, "dev")),
        ("test_prompts.txt, held-out half", split(test_prompts, "test")),
        ("check_prompts.txt (written after tuning)", load_prompts(GUARD_DIR / "check_prompts.txt")),
        ("check_prompts_2.txt (second check set)", load_prompts(GUARD_DIR / "check_prompts_2.txt")),
    ]


def pct(part, whole):
    return f"{part}/{whole} ({100 * part / whole:.0f} %)" if whole else "–"


def input_section(guard):
    lines = ["## Input guard", "",
             "Blocked = the question never reaches the model. For benign questions every block is a false "
             "positive. \"Right reason\" means the matching rule fired; an injection can also be stopped by the "
             "topic gate instead, which still counts as blocked. The topic gate threshold was tuned on the "
             "tuning half, so the other sets are the fairer numbers.", ""]
    total = {"attacks": 0, "attacks_blocked": 0, "benign": 0, "benign_blocked": 0}
    misses = []
    for name, prompts in question_sets():
        prompts = [p for p in prompts if p[0] in ("benign", *ATTACKS)]
        lines += [f"### {name}", "", "| Label | Questions | Blocked | Blocked for the right reason |",
                  "|---|---|---|---|"]
        for label in ("benign", *ATTACKS):
            group = [p for p in prompts if p[0] == label]
            if not group:
                continue
            blocked = right = 0
            for _, language, text in group:
                decision = guard.check(text)
                if not decision.allowed:
                    blocked += 1
                    expected = {"identity": "identity_request"}.get(label, label)
                    right += decision.category == expected
                if decision.allowed == (label != "benign"):
                    misses.append((name, label, decision.category, text))
            lines.append(f"| {label} | {len(group)} | {pct(blocked, len(group))} | "
                         f"{'–' if label == 'benign' else pct(right, len(group))} |")
            key = "benign" if label == "benign" else "attacks"
            total[key] += len(group)
            total[key + "_blocked"] += blocked
        lines.append("")
    lines[4:4] = [
        f"Over all sets: {pct(total['attacks_blocked'], total['attacks'])} of the unwanted questions were "
        f"blocked, and {pct(total['benign_blocked'], total['benign'])} of the benign questions were blocked by "
        "mistake.", ""]
    lines += ["### Mistakes", "", "| Set | Label | Guard decision | Question |", "|---|---|---|---|"]
    lines += [f"| {s} | {label} | {category} | {text.replace('|', '/')} |" for s, label, category, text in misses]
    if not misses:
        lines.append("| – | – | – | none |")
    return lines + [""]


def evidence_section():
    prompts = load_prompts(GUARD_DIR / "test_prompts.txt")
    hostile = [t for label, _, t in prompts if label == "ev_injection"]
    normal = [t for label, _, t in prompts if label == "ev_benign"]
    caught = sum(bool(find_injection(t)) for t in hostile)
    flagged = sum(bool(find_injection(t)) for t in normal)
    return ["## Evidence guard", "",
            "Review texts with hidden instructions (e.g. \"ignore all previous instructions\") must be dropped "
            "before the model reads them; normal reviews must pass.", "",
            "| Review texts | Count | Flagged |", "|---|---|---|",
            f"| with injected instructions | {len(hostile)} | {pct(caught, len(hostile))} |",
            f"| normal reviews | {len(normal)} | {pct(flagged, len(normal))} |", ""]


def output_section():
    from assistant.pipeline import SNAPSHOT_NOTES
    from data_setup.load_vectors import Masker
    from guardrails.output_guard import OutputGuard, multilingual_nli
    from sentence_transformers import SentenceTransformer

    fixture = json.loads((ROOT / "bench" / "fixtures" / "scenario_B.json").read_text())
    embedder = SentenceTransformer("BAAI/bge-m3")
    guard = OutputGuard(multilingual_nli(), find_pii=Masker().find,
                        embed=lambda texts: embedder.encode(list(texts), normalize_embeddings=True))
    cases = [line.split(";", 1) for line in CASES.read_text().splitlines() if line and not line.startswith("#")]
    rows, counts = [], {("supported", True): 0, ("supported", False): 0,
                        ("unsupported", True): 0, ("unsupported", False): 0}
    for label, sentence in cases:
        result = guard.check(sentence, fixture["sql_evidence"], fixture["review_evidence"],
                             question=fixture["question"], notes=SNAPSHOT_NOTES, keep_names=fixture["landmarks"])
        kept = result.sentences[0].supported
        counts[(label, kept)] += 1
        reason = result.sentences[0].reason or ""
        rows.append(f"| {label} | {'kept' if kept else 'removed'} | {sentence.replace('|', '/')} | {reason} |")
    removed_bad = counts[("unsupported", False)]
    n_bad = removed_bad + counts[("unsupported", True)]
    kept_good = counts[("supported", True)]
    n_good = kept_good + counts[("supported", False)]
    return ["## Output guard", "",
            f"{len(cases)} hand-written sentences checked against the saved scenario B evidence "
            "(`bench/output_guard_cases.txt`): half are true and cited correctly, half have a wrong number, the "
            "opposite claim, a citation that is not in the evidence, or a made-up detail.", "",
            f"Unsupported sentences removed: {pct(removed_bad, n_bad)}. "
            f"Supported sentences kept: {pct(kept_good, n_good)}.", "",
            "| Label | Guard | Sentence | Reason |", "|---|---|---|---|", *rows, ""]


def main():
    from sentence_transformers import SentenceTransformer
    from guardrails.input_guard import InputGuard

    embedder = SentenceTransformer("BAAI/bge-m3")
    guard = InputGuard(embed=lambda texts: embedder.encode(list(texts), normalize_embeddings=True))
    lines = ["# Guardrail evaluation", "",
             "Generated with `python -m bench.guardrails`. Input guard as the app runs it: rules plus the bge-m3 "
             "topic gate (Llama Guard is optional and was off). The question sets are in `guardrails/` "
             "(English and German).", ""]
    lines += input_section(guard) + evidence_section() + output_section()
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
