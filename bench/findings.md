## Hardware and result

Tested on 1 November 2026 on a laptop with an AMD Radeon 780M (integrated GPU, Vulkan backend). All three
models fit completely into GPU memory.

We use INT8 for the demo (`OLLAMA_MODEL=qwen2.5:7b-instruct-q8_0`). In run 2 it had the best share of
verified sentences (31 of 34) and valid JSON in 8 of 9 runs, and it generates about 1.8 times faster than
FP16. INT4 is faster again (about 1.5 times INT8), but only 4 of its 9 answers were valid JSON and only 3 of
13 sentences were verified.

## Observations

Generation speed is roughly proportional to model size: 5.3 tokens/s for FP16, 9.6 for INT8 and 14.5 for
INT4. Size times speed gives about the same value for all three (75 to 79 GiB/s), so memory bandwidth is
the limit on this iGPU.

Time to first token is between 32 and 37 s for every precision. Most of it is spent reading the prompt
(7,400 to 8,000 tokens at about 230 tokens/s), so quantization does not help much there. Shorter prompts
would.

The prompt had a bigger effect on reliability than the precision. In run 1 the prompt contained a list of
all allowed citation IDs, and the models often copied that list into the answer until they hit the token
limit (INT4 in 6 of 9 runs). For run 2 we removed the list and asked for at most three citations per
sentence, no tables and less than 200 words. FP16 and INT8 answers then mostly stayed between 100 and 200
tokens (one exception: INT8 on scenario A with 569). Valid JSON went from 7 to 9 out of 9 for FP16 and from
2 to 4 for INT4. INT8 stayed at 8.

INT4 kept looping in run 2. It ignores the three citation rule and puts every review or listing ID into one
sentence (scenarios B and C). Its answer to scenario A was also wrong: it said the availability of the two
free listings was unknown, and the output guard removed all of its sentences.

We did not see INT8 make more mistakes than FP16. The guard verified 91 % of the INT8 sentences and 82 % of
the FP16 sentences, but with 34 and 39 sentences the samples are small.

## Changes after run 2

These were only tested on single scenarios, not with a full benchmark run.

- The JSON schema limits the answer to 1,500 characters. llama.cpp enforces this while generating, so the
  JSON is always closed. With the limit, INT4 produced valid JSON for scenarios B and C in a single test.
- In the prompt each price group now has one ID to cite (`cite_as`) instead of 20 example listing IDs. The
  output guard also understands citation lists like `[review 1, review 2]` now and keeps at most five per
  sentence. We ran scenario B three times per model: INT8 and INT4 were valid in 3 of 3 runs (run 2: 2 of 3
  and 0 of 3), and all sentences were verified.
- Still open: in the sentence about review themes the models list around 20 review IDs, which costs about
  500 output tokens or 70 to 90 s. Short labels for the reviews in the prompt would probably fix this.

## Bugs found during the benchmark

- The 1,024 token limit never reached Ollama. LangChain sends it as `max_completion_tokens`, which Ollama
  ignores, so a looping answer ran until the 400 s timeout.
- The PII masker treated some district names (Margareten, Favoriten, Alsergrund) as person names and
  replaced them in the tool arguments, so searches in those districts found nothing.

Both are fixed.
