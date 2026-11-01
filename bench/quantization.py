"""Benchmark the answer model at FP16, INT8 (GGUF Q8_0) and INT4 (GGUF Q4_K_M) on the same engine (Ollama/llama.cpp).

Every model writes the grounded answer for the same saved evidence (scenarios A, B, C of the use case), so
latency and quality are compared on identical prompts. Steps:

  python -m bench.quantization capture   # run the pipeline once per scenario and save its evidence (fixtures)
  python -m bench.quantization run       # stream answers from each model; TTFT, prompt and generation tokens/s
  python -m bench.quantization guard     # check every answer with the output guard (quality)
  python -m bench.quantization report    # Markdown table -> docs/benchmarks.md

Raw results go to derived/benchmarks/. Stop other users of Ollama first: it serves one request at a time.
"""

import argparse
import json
import os
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "bench" / "fixtures"
RESULTS = ROOT / "derived" / "benchmarks"
REPORT = ROOT / "docs" / "benchmarks.md"
FINDINGS = ROOT / "bench" / "findings.md"  # hand-written conclusions, included in the report
OLLAMA = os.getenv("OLLAMA_URL", "http://127.0.0.1:21434")

SCENARIOS = {
    "A": ("Entire apartments in Neubau for 4 guests under €150, free on New Year's Eve, "
          "where reviews say it's quiet at night.", 2026),
    "B": ("How does the average price of superhost listings in Leopoldstadt compare to other hosts, "
          "and what do guests praise about superhosts?", None),
    "C": ("Summarize guest feedback on listing 48604277 over the last year. Do the host's other "
          "listings share the same complaints?", None),
}
MODELS = {
    "qwen2.5:7b-instruct-fp16": "FP16",
    "qwen2.5:7b-instruct-q8_0": "INT8 (Q8_0)",
    "qwen2.5:7b": "INT4 (Q4_K_M)",
}
# Same settings the app uses: server context length and max_tokens of local_model().
OPTIONS = {"temperature": 0, "num_ctx": 16384, "num_predict": 1024}


def planned_scenario_a(connection):
    """The tool calls scenario A needs. The 7B model's own calls fail here (it passes "New Year's Eve" as a
    landmark and invents an amenity instead of checking the calendar), which would leave no evidence."""
    from langchain_agent.queries import check_availability, search_listings

    listings = search_listings(connection, district="Neubau", room_type="Entire home/apt", accommodates=4,
                               max_price=150, limit=20)
    ids = [row["listing_id"] for row in listings["rows"]]
    return [listings, check_availability(connection, ids, "2026-12-31", "2027-01-01")]


PLANNED = {"A": planned_scenario_a}


class _Captured(Exception):
    pass


def capture(names):
    """Run the real pipeline up to answer generation and save the model's input for each scenario."""
    from assistant import pipeline

    FIXTURES.mkdir(parents=True, exist_ok=True)
    original = pipeline.grounded_answer
    for name in names:
        question, year = SCENARIOS[name]
        saved = {}

        def save(model, clean_question, sql_evidence, review_evidence, on_token=None):
            saved.update(question=clean_question, sql_evidence=sql_evidence, review_evidence=review_evidence)
            raise _Captured

        pipeline.grounded_answer = save
        original_invoke = pipeline.EvidenceWorkflow.invoke
        if name in PLANNED:

            def planned_invoke(self, q, plan=PLANNED[name]):
                with pipeline.connect() as connection:
                    return {"status": "evidence_ready", "results": plan(connection)}

            pipeline.EvidenceWorkflow.invoke = planned_invoke
        started = time.perf_counter()
        try:
            result = pipeline.run_question(question, year)
            print(f"{name}: no evidence captured (status {result.get('status')})")
            continue
        except _Captured:
            pass
        finally:
            pipeline.grounded_answer = original
            pipeline.EvidenceWorkflow.invoke = original_invoke
        with pipeline.connect() as connection:
            saved["landmarks"] = [row[0] for row in connection.execute(
                "SELECT name FROM vienna_stay_assistant.landmarks")]
        saved["scenario"], saved["original_question"] = name, question
        saved["evidence_source"] = "planned tool calls" if name in PLANNED else "model tool calls"
        (FIXTURES / f"scenario_{name}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1, default=str))
        print(f"{name}: saved {len(saved['sql_evidence'])} SQL results and {len(saved['review_evidence'])} reviews "
              f"in {time.perf_counter() - started:.0f} s")


def load_fixtures():
    fixtures = [json.loads(path.read_text()) for path in sorted(FIXTURES.glob("scenario_*.json"))]
    if not fixtures:
        raise SystemExit("No fixtures yet: run `python -m bench.quantization capture` first.")
    return fixtures


def ollama_messages(fixture, nonce):
    from assistant.pipeline import answer_messages

    system, user = answer_messages(fixture["question"], fixture["sql_evidence"], fixture["review_evidence"])
    # A unique first line stops Ollama from reusing the cached prompt of an earlier run.
    return [{"role": "system", "content": f"Benchmark run {nonce}.\n{system.content}"},
            {"role": "user", "content": user.content}]


def stream_chat(client, model, messages, fmt):
    """Stream one answer; returns client-side timings plus Ollama's own token counts and durations."""
    started, first, text, final = time.perf_counter(), None, "", {}
    body = {"model": model, "messages": messages, "stream": True, "format": fmt, "options": OPTIONS,
            "keep_alive": "30m"}
    with client.stream("POST", f"{OLLAMA}/api/chat", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece and first is None:
                first = time.perf_counter()
            text += piece
            if chunk.get("done"):
                final = chunk
    ended = time.perf_counter()
    ns = 1e9
    prompt_s, eval_s = final.get("prompt_eval_duration", 0) / ns, final.get("eval_duration", 0) / ns
    return {
        "ttft_s": round((first or ended) - started, 2),
        "total_s": round(ended - started, 2),
        "prompt_tokens": final.get("prompt_eval_count"),
        "prompt_eval_s": round(prompt_s, 2),
        "prompt_tok_s": round(final.get("prompt_eval_count", 0) / prompt_s, 1) if prompt_s else None,
        "output_tokens": final.get("eval_count"),
        "eval_s": round(eval_s, 2),
        "gen_tok_s": round(final.get("eval_count", 0) / eval_s, 1) if eval_s else None,
        "done_reason": final.get("done_reason"),
        "raw": text,
    }


def unload_all(client):
    for model in client.get(f"{OLLAMA}/api/ps").json().get("models", []):
        client.post(f"{OLLAMA}/api/generate", json={"model": model["name"], "keep_alive": 0})


def run(models, repeats, label):
    from assistant.pipeline import ANSWER_FORMAT, GroundedAnswer

    fixtures = load_fixtures()
    fmt = ANSWER_FORMAT["json_schema"]["schema"]
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"quantization_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    installed = {m["name"] for m in httpx.get(f"{OLLAMA}/api/tags").json()["models"]}
    with httpx.Client(timeout=httpx.Timeout(1800, connect=10)) as client, out.open("w") as log:
        for model in models:
            if model not in installed and f"{model}:latest" not in installed:
                print(f"skip {model}: not pulled (ollama pull {model})")
                continue
            unload_all(client)
            started = time.perf_counter()
            client.post(f"{OLLAMA}/api/generate", json={"model": model, "keep_alive": "30m",
                                                       "options": {"num_ctx": OPTIONS["num_ctx"]}})
            load_s = time.perf_counter() - started
            loaded = next((m for m in client.get(f"{OLLAMA}/api/ps").json()["models"] if m["name"].startswith(model)), {})
            size, vram = loaded.get("size", 0), loaded.get("size_vram", 0)
            # Warm-up (first GPU run compiles shaders); not measured.
            client.post(f"{OLLAMA}/api/chat", json={"model": model, "stream": False, "options": {
                "num_predict": 4, "num_ctx": OPTIONS["num_ctx"]}, "messages": [{"role": "user", "content": "Say OK"}]})
            print(f"{model}: loaded in {load_s:.1f} s, {size / 2**30:.1f} GiB, {vram / max(size, 1):.0%} on GPU")
            for repeat in range(repeats):
                for fixture in fixtures:
                    stats = stream_chat(client, model, ollama_messages(fixture, uuid.uuid4().hex[:8]), fmt)
                    try:
                        answer, valid = GroundedAnswer.model_validate_json(stats["raw"]).answer, True
                    except ValueError:
                        answer, valid = None, False
                    row = {"label": label, "model": model, "quant": MODELS.get(model, model),
                           "scenario": fixture["scenario"], "repeat": repeat, "load_s": round(load_s, 1),
                           "size_gib": round(size / 2**30, 2), "gpu_share": round(vram / max(size, 1), 2),
                           "json_valid": valid, "answer": answer, **stats}
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    log.flush()
                    print(f"  {fixture['scenario']}#{repeat}: TTFT {stats['ttft_s']:6.1f} s | prompt "
                          f"{stats['prompt_tokens']} tok @ {stats['prompt_tok_s']} tok/s | output "
                          f"{stats['output_tokens']} tok @ {stats['gen_tok_s']} tok/s | json {'ok' if valid else 'INVALID'}")
        unload_all(client)
    print(f"results: {out}")
    return out


def latest_results():
    files = sorted(RESULTS.glob("quantization_*.jsonl"))
    if not files:
        raise SystemExit("No results yet: run `python -m bench.quantization run` first.")
    return files


def guard(path):
    """Check each answer with the output guard (NLI + numbers): how many sentences survive."""
    from data_setup.load_vectors import Masker
    from guardrails.output_guard import OutputGuard, multilingual_nli
    from assistant.pipeline import SNAPSHOT_NOTES
    from sentence_transformers import SentenceTransformer

    fixtures = {f["scenario"]: f for f in load_fixtures()}
    embedder = SentenceTransformer("BAAI/bge-m3")
    output_guard = OutputGuard(multilingual_nli(), find_pii=Masker().find,
                               embed=lambda texts: embedder.encode(list(texts), normalize_embeddings=True))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    cache = {}
    for row in rows:
        if not row["answer"]:
            continue
        key = (row["scenario"], row["answer"])
        if key not in cache:  # temperature 0: repeats usually give the same answer
            fixture = fixtures[row["scenario"]]
            started = time.perf_counter()
            check = output_guard.check(row["answer"], fixture["sql_evidence"], fixture["review_evidence"],
                                       question=fixture["question"], notes=SNAPSHOT_NOTES,
                                       keep_names=fixture["landmarks"])
            cache[key] = {"sentences": len(check.sentences), "supported": sum(s.supported for s in check.sentences),
                          "removed": sum(not s.supported for s in check.sentences), "verified": check.verified,
                          "guard_s": round(time.perf_counter() - started, 1)}
            print(f"{row['model']} {row['scenario']}: {cache[key]}")
        row["guard"] = cache[key]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=1):
    return "–" if value is None else f"{value:.{digits}f}"


def _ordered(groups):
    order = list(MODELS)
    return sorted(groups.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)


def _run_section(path, rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["model"], []).append(row)
    scenarios = sorted({r["scenario"] for r in rows})
    lines = [
        f"## {rows[0]['label']}",
        "",
        f"{len({r['repeat'] for r in rows})} repeat(s) × {len(scenarios)} scenarios ({', '.join(scenarios)}) per "
        f"precision, medians. Raw results: `derived/benchmarks/{path.name}`.",
        "",
        "| Precision | Model size | On GPU | Load | TTFT | Prompt tok/s | Generation tok/s | End-to-end answer | "
        "Valid JSON | Hit token limit | Sentences verified by output guard |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, group in _ordered(groups):
        checked = [r["guard"] for r in group if r.get("guard")]
        kept = (f"{sum(g['supported'] for g in checked)}/{sum(g['sentences'] for g in checked)} "
                f"(in {len(checked)} valid answers)" if checked else "–")
        capped = sum(1 for r in group if (r["output_tokens"] or 0) >= OPTIONS["num_predict"])
        lines.append(
            f"| {group[0]['quant']} | {group[0]['size_gib']:.1f} GiB | {group[0]['gpu_share']:.0%} | "
            f"{group[0]['load_s']:.0f} s | {fmt(median(r['ttft_s'] for r in group))} s | "
            f"{fmt(median(r['prompt_tok_s'] for r in group))} | {fmt(median(r['gen_tok_s'] for r in group))} | "
            f"{fmt(median(r['total_s'] for r in group))} s | "
            f"{sum(r['json_valid'] for r in group)}/{len(group)} | {capped}/{len(group)} | {kept} |")
    lines += ["", "Per scenario (median TTFT / end-to-end answer, prompt tokens → output tokens):", "",
              "| Precision | " + " | ".join(scenarios) + " |", "|---|" + "---|" * len(scenarios)]
    for model, group in _ordered(groups):
        cells = []
        for scenario in scenarios:
            s = [r for r in group if r["scenario"] == scenario]
            cells.append(f"{fmt(median(r['ttft_s'] for r in s))} s / {fmt(median(r['total_s'] for r in s))} s "
                         f"({fmt(median(r['prompt_tokens'] for r in s), 0)} → "
                         f"{fmt(median(r['output_tokens'] for r in s), 0)})")
        lines.append(f"| {group[0]['quant']} | " + " | ".join(cells) + " |")
    return lines, groups


def report(paths):
    """Write docs/benchmarks.md: one section per results file (oldest first), answers of the newest run."""
    lines = [
        "# Quantization benchmark",
        "",
        "Qwen2.5-7B-Instruct at FP16, INT8 (GGUF Q8_0) and INT4 (GGUF Q4_K_M) on the same engine, Ollama "
        "(llama.cpp). Each model writes the app's grounded answer (schema-constrained JSON with citations) for the "
        "same saved evidence of the use-case scenarios A, B and C (`bench/fixtures/`). Temperature 0, "
        f"{OPTIONS['num_ctx']} context, {OPTIONS['num_predict']}-token output limit; a unique first line per run "
        "stops Ollama from reusing a cached prompt. TTFT is measured by the client, token rates come from "
        "Ollama's own timings. Quality = sentences the output guard verifies against the evidence (NLI + numbers), "
        "counted over the answers with valid JSON.",
        "",
        "Generated by `python -m bench.quantization report`; do not edit by hand (conclusions live in "
        "`bench/findings.md`).",
        "",
    ]
    if FINDINGS.exists():
        lines += [FINDINGS.read_text().strip(), ""]
    groups = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        section, groups = _run_section(path, rows)
        lines += section + [""]
    lines += [f"## Answers of the last run (first repeat)", ""]
    for model, group in _ordered(groups):
        for row in sorted((r for r in group if r["repeat"] == 0), key=lambda r: r["scenario"]):
            answer = row["answer"] or f"(invalid JSON, {row['output_tokens']} tokens)"
            lines += [f"**{row['quant']}, scenario {row['scenario']}**", "", "> " + answer.replace("\n", "\n> "), ""]
    text = "\n".join(lines) + "\n"
    REPORT.write_text(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("capture")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p = sub.add_parser("run")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--label", default="vulkan-780m", help="hardware/backend name stored with the results")
    p = sub.add_parser("guard")
    p.add_argument("--results", type=Path, help="results file (default: newest)")
    p = sub.add_parser("report")
    p.add_argument("--results", type=Path, nargs="+", help="results files, oldest first (default: newest)")
    args = parser.parse_args()
    if args.command == "capture":
        capture(args.scenarios)
    elif args.command == "run":
        run(args.models, args.repeats, args.label)
    elif args.command == "guard":
        guard(args.results or latest_results()[-1])
    else:
        report(args.results or latest_results()[-1:])


if __name__ == "__main__":
    main()
