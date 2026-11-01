# Quantization benchmark

We ran the answer model, Qwen2.5-7B-Instruct, in three precisions on the same engine (Ollama, which uses llama.cpp): FP16, INT8 (GGUF Q8_0) and INT4 (GGUF Q4_K_M). Every model got the same saved evidence for the three use case scenarios A, B and C (`bench/fixtures/`) and had to write the answer the app produces, as JSON with citations.

Settings: temperature 0, 16,384 token context, at most 1,024 output tokens. Each run starts with a different first line, so Ollama cannot reuse a cached prompt. Time to first token (TTFT) is measured on the client, the token rates come from Ollama. For quality we count the sentences the output guard could verify against the evidence, over the answers with valid JSON.

This file is generated with `python -m bench.quantization report`. The text below comes from `bench/findings.md`.

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

## Run 1: original prompt

3 repeats of each scenario (A, B, C) per precision. All values are medians. Raw results: `derived/benchmarks/quantization_20261101_144736.jsonl` (not in git).

| Precision | Model size | On GPU | Load | TTFT | Prompt tok/s | Generation tok/s | Total time | Valid JSON | Hit token limit | Sentences verified by output guard |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 14.2 GiB | 100% | 8 s | 41.6 s | 208.8 | 5.1 | 96.0 s | 7/9 | 2/9 | – |
| INT8 (Q8_0) | 8.2 GiB | 100% | 4 s | 37.6 s | 232.4 | 9.4 | 65.2 s | 8/9 | 1/9 | – |
| INT4 (Q4_K_M) | 5.2 GiB | 100% | 3 s | 37.6 s | 231.8 | 14.7 | 106.8 s | 2/9 | 6/9 | – |

Per scenario: median TTFT / total time (prompt tokens → output tokens).

| Precision | A | B | C |
|---|---|---|---|
| FP16 | 41.6 s / 78.3 s (8686 → 195) | 44.6 s / 265.5 s (9012 → 1024) | 39.5 s / 96.0 s (8261 → 289) |
| INT8 (Q8_0) | 37.6 s / 137.5 s (8686 → 940) | 39.5 s / 55.5 s (9011 → 152) | 35.8 s / 65.2 s (8262 → 276) |
| INT4 (Q4_K_M) | 37.6 s / 107.0 s (8686 → 1024) | 39.3 s / 89.3 s (9012 → 736) | 35.7 s / 104.6 s (8263 → 1024) |

## Run 2: revised prompt

3 repeats of each scenario (A, B, C) per precision. All values are medians. Raw results: `derived/benchmarks/quantization_20261101_162443.jsonl` (not in git).

| Precision | Model size | On GPU | Load | TTFT | Prompt tok/s | Generation tok/s | Total time | Valid JSON | Hit token limit | Sentences verified by output guard |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 14.2 GiB | 100% | 8 s | 37.1 s | 205.3 | 5.3 | 67.3 s | 9/9 | 0/9 | 32/39 (in 9 valid answers) |
| INT8 (Q8_0) | 8.2 GiB | 100% | 4 s | 32.4 s | 239.0 | 9.6 | 50.5 s | 8/9 | 1/9 | 31/34 (in 8 valid answers) |
| INT4 (Q4_K_M) | 5.2 GiB | 100% | 3 s | 33.5 s | 228.2 | 14.5 | 100.8 s | 4/9 | 5/9 | 3/13 (in 4 valid answers) |

Per scenario: median TTFT / total time (prompt tokens → output tokens).

| Precision | A | B | C |
|---|---|---|---|
| FP16 | 39.1 s / 77.2 s (7995 → 193) | 37.1 s / 57.2 s (7605 → 98) | 36.3 s / 67.3 s (7403 → 164) |
| INT8 (Q8_0) | 33.6 s / 91.9 s (7993 → 569) | 31.8 s / 42.6 s (7604 → 107) | 32.3 s / 49.3 s (7404 → 161) |
| INT4 (Q4_K_M) | 35.2 s / 46.0 s (7995 → 156) | 33.5 s / 104.2 s (7605 → 1024) | 32.9 s / 100.8 s (7404 → 1024) |

## Answers from the last run (first repeat)

**FP16, scenario A**

> For a quiet stay in Neubau on New Year's Eve, December 31, 2026, the following listings are available and suitable for 4 guests under €150: listing 781643931771873585 and listing 781659185752510771, both priced at €134.87 and €115.86 respectively, run by superhosts. Reviews mention the area is quiet and safe, though some issues with WiFi and mold were reported. Listing 781643931771873585 is described as spacious with a nice garden area, while listing 78165918575251071 is praised for its central location and cleanliness.

**FP16, scenario B**

> The average price of superhost listings in Leopoldstadt is 166.04 EUR, which is higher than the average price of non-superhost listings at 149.04 EUR. Guests praise superhosts for their kind hospitality, detailed recommendations, and proactive assistance. Superhosts often provide excellent locations, thoughtful amenities, and helpful instructions for navigating the city. They are generally responsive and friendly, enhancing the overall stay experience.

**FP16, scenario C**

> Guest feedback on listing 48604277 over the last year includes mixed reviews. Some guests appreciated the cleanliness, modern amenities, and central location, while others faced issues with noise, lack of hot water, and communication with the host. Specific complaints included uncomfortable behavior in the surrounding area, delayed check-in, and a lack of a guest umbrella. The host's other listings, such as 48604725 and 48604901, generally received positive reviews, with guests praising the cleanliness, location, and comfort. However, some reviews mentioned similar issues like noise, lack of hot water, and communication problems with the host. Overall, both listings share some common complaints, such as noise and communication issues, but also have positive aspects like cleanliness and central location.

**INT8 (Q8_0), scenario A**

> For a quiet stay in Neubau for 4 guests under €150 on New Year's Eve, listing 781643931771873585 and listing 781659185752510771 are available and within budget. Both listings are managed by the same host and are rated highly for cleanliness and location. However, there are no specific reviews mentioning quietness at night. Listing 781643931771873585 has a score of 0.133264 and listing 781659185752510771 has a score of 0.051218. [listing 781643931771873585] [listing 781659185752510771]

**INT8 (Q8_0), scenario B**

> The average price of superhost listings in Leopoldstadt is 166.04 EUR, which is higher than the average price of non-superhost listings at 149.04 EUR. Guests praise superhosts for their kind hospitality, detailed recommendations, and proactive assistance. Superhosts often provide excellent locations, thoughtful amenities, and responsive communication. For example, one guest highlighted the helpfulness and friendliness of the hosts, while another appreciated the apartment's size and cleanliness.

**INT8 (Q8_0), scenario C**

> Guest feedback on listing 48604277 over the last year includes mixed reviews. Some guests appreciated the cleanliness, modern amenities, and central location, while others faced issues with noise, lack of hot water, and communication with the host. Notable complaints include uncomfortable behavior in the surrounding area, delayed check-in, and the absence of a guest umbrella. The host's other listings, such as 48604725 and 48604901, generally received positive reviews, with guests praising the cleanliness, location, and comfort. However, some reviews mentioned similar issues like communication problems and minor inconveniences, suggesting that these complaints might not be unique to listing 48604277.

**INT4 (Q4_K_M), scenario A**

> - There are no entire apartments in Neubau for 4 guests under €150 that are confirmed to be available on New Year's Eve 2026. The closest match is listing [781643931771873585] and [781659185752510771] at €134.87 and €115.86 respectively, but availability is unknown for these dates. 
> - Reviews suggest that the area is quiet at night, with one review stating the apartment felt quiet and safe [781643931771873585].

**INT4 (Q4_K_M), scenario B**

> (invalid JSON, 1024 tokens)

**INT4 (Q4_K_M), scenario C**

> (invalid JSON, 1024 tokens)

