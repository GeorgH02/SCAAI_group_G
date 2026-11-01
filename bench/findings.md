## Findings (2026-11-01, Radeon 780M iGPU via Vulkan)

**Recommendation: use INT8 (Q8_0) for the demo** (`OLLAMA_MODEL=qwen2.5:7b-instruct-q8_0`). With the fixed
prompt (run 2) it had the most verified sentences (31/34, 91%), valid JSON in 8/9 runs, and writes 1.8× faster
than FP16. INT4 writes another 1.5× faster but produced valid JSON in only 4/9 runs, and 3/13 of its sentences
were verified.

- **Speed follows model size.** Generation: FP16 5.3, INT8 9.6, INT4 14.5 tokens/s. Size × speed is
  75–79 GiB/s for all three, so generation on this iGPU is limited by memory bandwidth.
- **Time to first token barely depends on precision** (32–37 s): almost all of it is reading the 7,400–8,000
  prompt tokens at about 230 tokens/s.
- **The prompt mattered more than precision for reliability.** Run 1 listed every allowed citation ID in the
  prompt; models copied it out until the 1,024-token limit (INT4 in 6/9 runs). Removing the list and asking
  for at most three citations per sentence, no tables and under 200 words brought FP16 and INT8 answers
  down to about 100–200 tokens (INT8 scenario A: 569). Valid JSON rose from 7 to 9 (FP16) and from 2 to 4
  (INT4) out of 9; INT8 stayed at 8.
- **INT4 still loops.** It ignores the three-citation rule and lists every review or listing ID in one
  sentence (scenarios B and C). Its scenario A answer was also wrong: it called the two free listings
  "availability unknown", so the output guard removed every sentence.
- **Lower precision did not mean more false facts at INT8.** The output guard verified 91% of INT8 sentences
  and 82% of FP16 sentences. Both rates come from small samples (34 and 39 sentences).

Changes after run 2 (not yet measured over all scenarios):
- The answer schema now caps the answer at 1,500 characters (`maxLength`, enforced while decoding). In a
  single test of INT4 on scenarios B and C, both answers were valid JSON; B still spent most of its length on
  IDs.
- Still open: the price comparison in scenario B carries every listing ID of each group (hundreds) into the
  answer prompt; sending a few example IDs and a count should stop the remaining ID lists.

Bugs found while benchmarking (fixed): the 1,024-token output limit never reached Ollama (LangChain sends it
as `max_completion_tokens`, which Ollama ignores), and the privacy masker turned some district names
(Margareten, Favoriten, Alsergrund) into `<PERSON_n>` inside tool arguments, so those district searches
returned nothing.
