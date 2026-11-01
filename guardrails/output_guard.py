"""Output guard: checks a generated answer before it is shown.

1. Citation mapping: each sentence must cite [listing N] or [review N] from the retrieved evidence.
   For a sentence without a citation the guard searches the evidence itself and adds the citation it finds.
2. Faithfulness: the cited evidence must entail the sentence (multilingual DeBERTa-v3 NLI), and every
   number in the sentence must appear in that evidence. Sentences that fail are removed.
3. PII leak check: the remaining text is masked again (names, phones, emails) before display.
"""

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

NLI_MODEL = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
DEFAULT_THRESHOLD = float(os.getenv('OUTPUT_NLI_THRESHOLD', '0.5'))
NOTHING_VERIFIED = ('No statement in the generated answer could be verified against the retrieved evidence, '
                    'so it was withheld. Please check the evidence below or rephrase the question.')

CITATION = re.compile(r'\[(?:(listing|review)\s+)?(\d+)\]', re.IGNORECASE)  # [listing 1], [review 2] or [3]
NUMBER = re.compile(r'\d+(?:[.,]\d+)*')
LINE_PREFIX = re.compile(r'^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)?)')
SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=[\[A-ZÄÖÜ0-9"„(*])')
MARKDOWN = re.compile(r'[*_`]+')
PLACEHOLDER = re.compile(r'<[A-Z_]+_\d+>')
TRIM = set(' \t\n,.;:!?()[]"\'')  # left around a span after protected parts are cut out
EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')

LISTING_FIELDS = {
    'neighbourhood': 'It is in the district {}.',
    'room_type': 'Room type: {}.',
    'accommodates': 'It accommodates {} guests.',
    'bedrooms': 'It has {} bedrooms.',
    'price': 'Its snapshot base price is €{} per night.',
    'host_id': 'Its host has host ID {}.',
    'distance_m': 'It is {} m from the requested landmark.',
    'lnight_road_band': 'Its night-time road noise (Lnight) band is {} dB.',
    'lnight_rail_band': 'Its night-time rail noise (Lnight) band is {} dB.',
    'snapshot_date': 'Data from the snapshot of {}.',
}
AVAILABILITY = {
    'available': 'is available on every night of the stay in the calendar snapshot',
    'unavailable': 'is not available for the stay in the calendar snapshot',
    'unknown': 'has availability that could not be confirmed from the calendar snapshot',
    'not_found': 'was not found in the data',
}


def multilingual_nli(model_name=NLI_MODEL, device=None, batch_size=16):
    """Return entail(pairs) -> entailment probabilities for (premise, hypothesis) pairs."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    index = next(i for i, label in model.config.id2label.items() if label.lower().startswith('entail'))

    def entail(pairs):
        scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = tokenizer([p for p, _ in batch], [h for _, h in batch], truncation='only_first',
                               max_length=512, padding=True, return_tensors='pt').to(device)
            with torch.inference_mode():
                scores.extend(model(**inputs).logits.softmax(-1)[:, index].tolist())
        return scores
    return entail


def numbers(text):
    """Numbers in text as (value, decimals); '1,234.50', '12,5' and '120' are understood."""
    found = []
    for raw in NUMBER.findall(text):
        if ',' in raw and '.' in raw:
            decimal = max(raw.rfind(','), raw.rfind('.'))
            raw = re.sub(r'[.,]', '', raw[:decimal]) + '.' + raw[decimal + 1:]
        elif ',' in raw:
            parts = raw.split(',')
            raw = ''.join(parts) if all(len(p) == 3 for p in parts[1:]) else '.'.join(parts)
        if raw.count('.') > 1:
            found.extend((float(p), 0) for p in raw.split('.') if p)
            continue
        found.append((float(raw), len(raw.split('.')[1]) if '.' in raw else 0))
    return found


def missing_numbers(sentence, sources):
    """Numbers in the sentence that no source contains, allowing rounding and truncation."""
    available = [value for value, _ in numbers(sources)]
    missing = []
    for value, decimals in numbers(sentence):
        if not any(abs(round(p, decimals) - value) < 1e-9 or (decimals == 0 and int(p) == value)
                   for p in available):
            missing.append(value)
    return missing


def _text(value):
    return f'{value:.2f}' if isinstance(value, float) else str(value)


class EvidenceIndex:
    """Turns the evidence shown to the model into premises, keyed by citation."""

    def __init__(self, sql_evidence, review_evidence, notes=(), per_citation=10):
        self.reviews = defaultdict(list)
        self.listings = defaultdict(list)
        self.facts = []  # (kind, id, premise) for every piece of evidence, used to cite uncited sentences
        self.per_citation = per_citation
        notices = [n for item in sql_evidence for n in item.get('notices', [])]
        self.notes = ' '.join(dict.fromkeys([*notes, *notices]))
        for review in review_evidence:
            self.reviews[str(review['review_id'])].append(review.get('text', ''))
        for review_id, texts in list(self.reviews.items()):
            review = next(r for r in review_evidence if str(r['review_id']) == review_id)
            premise = (f'Review {review_id} of listing {review.get("listing_id")} '
                       f'({review.get("review_date")}): ' + ' '.join(texts))
            self.reviews[review_id] = [premise]
            self.facts.append(('review', review_id, premise))
            if review.get('listing_id') is not None:
                self.listings[str(review['listing_id'])].append(premise)
        for item in sql_evidence:
            for row in item.get('rows', []):
                if row.get('listing_id') is not None:
                    self.listings[str(row['listing_id'])].insert(0, self._listing(row))
                    self.facts.append(('listing', str(row['listing_id']), self._listing(row)))
            groups = item.get('groups', [])
            comparison = self._comparison(groups, item.get('district'))
            for group in groups:
                premise = self._group(group, item.get('district'))
                for listing_id in group.get('listing_ids', []):
                    self.listings[str(listing_id)].insert(0, premise)
                    if comparison:
                        self.listings[str(listing_id)].insert(0, comparison)
                if group.get('listing_ids'):
                    self.facts.append(('listing', str(group['listing_ids'][0]), premise))
            if comparison:
                ids = [str(g['listing_ids'][0]) for g in groups if g.get('listing_ids')
                       and g.get('host_is_superhost') in (True, False)]
                self.facts.append(('combined', [('listing', i) for i in ids], comparison))
            for stay in item.get('availability', []):
                if stay.get('listing_id') is not None:
                    self.listings[str(stay['listing_id'])].insert(0, self._stay(item, stay))
                    self.facts.append(('listing', str(stay['listing_id']), self._stay(item, stay)))

    @staticmethod
    def _listing(row):
        parts = [f'Listing {row["listing_id"]}.']
        for key, value in row.items():
            if key == 'listing_id' or value is None:
                continue
            if key == 'host_is_superhost':
                parts.append('Its host is a superhost.' if value else 'Its host is not a superhost.')
            else:
                parts.append(LISTING_FIELDS.get(key, key.replace('_', ' ') + ': {}.').format(_text(value)))
        return ' '.join(parts)

    @staticmethod
    def _group(group, district=None):
        status = group.get('host_is_superhost')
        label = 'Superhost listings' if status is True else 'Listings of other hosts' if status is False else (
            'Listings whose host status is unknown')
        average = group.get('average_base_price')
        average = f'€{float(average):.2f}' if average is not None else 'unknown'
        where = f' in {district}' if district else ''
        return (f'{label}{where}: {group.get("listing_count")} listings, {group.get("priced_listing_count")} with a price '
                f'and {group.get("missing_price_count")} without; their average snapshot base price is {average}.')

    @staticmethod
    def _comparison(groups, district=None):
        """State the superhost price comparison in words: NLI models cannot compare two numbers."""
        prices = {g.get('host_is_superhost'): g.get('average_base_price') for g in groups}
        if prices.get(True) is None or prices.get(False) is None:
            return None
        superhost, other = float(prices[True]), float(prices[False])
        where = f' in {district}' if district else ''
        relation = ('higher than' if superhost > other else 'lower than' if superhost < other else 'the same as')
        cheaper = 'more expensive' if superhost > other else 'cheaper' if superhost < other else 'equally priced'
        return (f'On average, superhost listings{where} are {cheaper} than listings of other hosts (hosts who are not '
                f'superhosts): their average snapshot base price of €{superhost:.2f} is {relation} the €{other:.2f} '
                f'average of non-superhost listings, a difference of €{abs(superhost - other):.2f}.')

    @staticmethod
    def _stay(item, stay):
        dates = f' from {item["check_in"]} to {item["check_out"]}' if item.get('check_in') else ''
        text = f'Listing {stay["listing_id"]} {AVAILABILITY.get(stay.get("status"), "has unknown availability")}{dates}.'
        reasons = sorted({f.get('reason') for f in stay.get('failures') or [] if f.get('reason')})
        return text + (f' Reasons: {", ".join(reasons)}.' if reasons else '')

    def text(self):
        return ' '.join([self.notes, *(premise for _, _, premise in self.facts)])

    def kind_of(self, identifier):
        """Resolve a bare [123] citation: a review ID if one matches, otherwise a listing ID."""
        return 'review' if identifier in self.reviews else 'listing'

    def premises(self, kind, identifier):
        source = self.reviews if kind == 'review' else self.listings
        return source.get(identifier, [])[:self.per_citation]


@dataclass
class CheckedSentence:
    text: str
    citations: list
    supported: bool
    reason: str | None = None
    score: float | None = None
    evidence: list = field(default_factory=list)


@dataclass
class OutputCheck:
    answer: str
    sentences: list
    pii_redacted: bool
    verified: bool

    @property
    def removed(self):
        return [s for s in self.sentences if not s.supported]

    @property
    def citation_map(self):
        return [{'sentence': s.text, 'citations': [f'{k} {i}' for k, i in s.citations], 'evidence': s.evidence}
                for s in self.sentences if s.supported and s.citations]


class OutputGuard:
    """Check an answer against the evidence that produced it.

    entail(pairs) returns entailment probabilities for (premise, hypothesis) pairs.
    find_pii(text) returns (start, end, kind) spans of names, phones and emails (Masker.find); None skips
    the PII check.
    embed(texts) returns normalized vectors; it picks the evidence to try for sentences without a
    citation (word overlap is used without it). A supported uncited sentence gets the citation added.
    """

    def __init__(self, entail, find_pii=None, threshold=DEFAULT_THRESHOLD, embed=None, candidates=12, combine=4):
        self.entail, self.find_pii, self.threshold = entail, find_pii, threshold
        self.embed, self.candidates, self.combine = embed, candidates, combine

    def check(self, answer, sql_evidence, review_evidence, *, question='', notes=(), keep_names=()):
        index = EvidenceIndex(sql_evidence, review_evidence, notes)
        lines = [self._split(line) for line in answer.strip().splitlines()]
        claims = [s for _, parts in lines for s in parts or ()]
        self._verify(claims, index, question)

        output = []
        for prefix, parts in lines:
            if parts is None:
                output.append(('structure', prefix))
                continue
            kept = [s.text for s in parts if s.supported]
            if kept:
                output.append(('content', prefix + ' '.join(kept)))
        text = re.sub(r'\n{3,}', '\n\n', '\n'.join(_drop_empty_sections(output))).strip()
        verified = any(s.supported for s in claims)
        if not verified:
            text = NOTHING_VERIFIED
        redacted = False
        if self.find_pii is not None and verified:
            keep_numbers = set(NUMBER.findall(index.text() + ' ' + question))
            text, redacted = redact_pii(text, self.find_pii, keep_names, keep_numbers)
        return OutputCheck(text, claims, redacted, verified)

    @staticmethod
    def _split(line):
        prefix = LINE_PREFIX.match(line).group(1)
        body = line[len(prefix):].strip()
        plain = MARKDOWN.sub('', CITATION.sub('', body)).strip()
        if not re.search(r'\w', plain) or prefix.lstrip().startswith('#') or (
                plain.endswith(':') and not CITATION.search(body) and not re.search(r'\d', plain)):
            return line, None
        sentences = []
        for piece in SENTENCE_BREAK.split(body):
            if sentences and not MARKDOWN.sub('', CITATION.sub('', piece)).strip(' .'):
                sentences[-1] += ' ' + piece  # a citation written after the full stop
            else:
                sentences.append(piece)
        return prefix, [CheckedSentence(s, [(k.lower(), i) for k, i in CITATION.findall(s)], False)
                        for s in sentences]

    def _verify(self, claims, index, question):
        pairs, owners = [], []
        for claim in claims:
            hypothesis = re.sub(r'\s+', ' ', MARKDOWN.sub('', CITATION.sub('', claim.text))).strip()
            hypothesis = re.sub(r'\s+([.,;:!?])', r'\1', hypothesis)
            if not claim.citations:
                candidates = self._candidates(hypothesis, index, question)
                if not candidates:
                    claim.reason = 'no citation, and no retrieved evidence contains its numbers'
                for candidate in candidates:
                    pairs.append((candidate[2], hypothesis))
                    owners.append((claim, candidate))
                continue
            premises = []
            claim.citations = [(kind or index.kind_of(identifier), identifier) for kind, identifier in claim.citations]
            claim.text = CITATION.sub(lambda m: f'[{m.group(1) or index.kind_of(m.group(2))} {m.group(2)}]', claim.text)
            for kind, identifier in claim.citations:
                found = index.premises(kind, identifier)
                if not found:
                    claim.reason = f'cites {kind} {identifier}, which is not in the retrieved evidence'
                    break
                premises.extend(found)
            if claim.reason:
                continue
            if len(claim.citations) > 1:
                premises.append(' '.join(dict.fromkeys(premises)))
            claim.evidence = list(dict.fromkeys(premises))
            missing = missing_numbers(hypothesis, ' '.join(premises) + ' ' + question)
            if missing:
                claim.reason = 'numbers not found in the cited evidence: ' + ', '.join(f'{m:g}' for m in missing)
                continue
            for premise in claim.evidence:
                pairs.append((premise, hypothesis))
                owners.append((claim, None))
        best = {}
        for (claim, candidate), score in zip(owners, self.entail(pairs) if pairs else []):
            if score > (claim.score or 0.0):
                claim.score = float(score)
                best[id(claim)] = candidate
        for claim in claims:
            if claim.reason:
                continue
            claim.supported = (claim.score or 0.0) >= self.threshold
            if not claim.supported:
                kind = 'cited evidence' if claim.citations else 'any retrieved evidence (no citation given)'
                claim.reason = f'not entailed by {kind} (score {claim.score or 0.0:.2f})'
            elif not claim.citations:
                kind, identifier, premise = best[id(claim)]
                claim.evidence = [premise]
                if kind != 'note':
                    claim.citations = identifier if kind == 'combined' else [(kind, identifier)]
                    claim.text = claim.text.rstrip() + ' ' + ' '.join(f'[{k} {i}]' for k, i in claim.citations)

    def _candidates(self, hypothesis, index, question):
        """Evidence to try for a sentence without a citation.

        Returns notices, the closest single facts, and one combined premise of the top facts, because a
        summary ("guests praise X and Y", "A costs 166 vs 149 for B") is often supported only jointly.
        Every candidate must contain all numbers of the sentence.
        """
        wanted = {value for value, _ in numbers(hypothesis)} - {value for value, _ in numbers(question)}
        facts = [f for f in index.facts if not wanted or set(v for v, _ in numbers(f[2])) & wanted
                 or not missing_numbers(hypothesis, f[2] + ' ' + question)]
        if facts and self.embed is not None:
            vectors = self.embed([hypothesis, *(premise for _, _, premise in facts)])
            ranked = sorted(zip(vectors[1:] @ vectors[0], range(len(facts))), reverse=True)
            facts = [facts[i] for _, i in ranked[:self.candidates]]
        elif facts:
            words = set(re.findall(r'\w{4,}', hypothesis.lower()))
            facts = sorted(facts, key=lambda f: -len(words & set(re.findall(r'\w{4,}', f[2].lower()))))
            facts = facts[:self.candidates]
        candidates = [f for f in facts if not missing_numbers(hypothesis, f[2] + ' ' + question)]
        # For a sentence with numbers, combine the facts that hold those numbers; otherwise the closest facts.
        with_numbers = [f for f in facts if wanted & {v for v, _ in numbers(f[2])}]
        top = (with_numbers or facts)[:self.combine]
        if len(top) > 1:
            combined = ' '.join(premise for _, _, premise in top)
            if not missing_numbers(hypothesis, combined + ' ' + question):
                cited = [c for kind, identifier, _ in top
                         for c in (identifier if kind == 'combined' else [(kind, identifier)])]
                candidates.append(('combined', list(dict.fromkeys(cited)), combined))
        if index.notes and not missing_numbers(hypothesis, index.notes + ' ' + question):
            candidates.insert(0, ('note', None, index.notes))
        return candidates


def _drop_empty_sections(lines):
    """Keep headings and lead-in lines only when content follows them."""
    kept = []
    for position, (kind, text) in enumerate(lines):
        if kind == 'content':
            kept.append(text)
            continue
        following = next((k for k, t in lines[position + 1:] if k == 'content' or t.lstrip().startswith('#')), None)
        if following == 'content' or not text.strip():
            kept.append(text)
    return kept


def redact_pii(text, find, keep_names=(), keep_numbers=()):
    """Mask names, phones and emails found by find(text) -> [(start, end, kind)].

    Citations, existing placeholders, public names (landmarks) and numbers taken from the evidence are cut
    out of every detected span, because the recognizers mistake listing IDs and prices for phone numbers.
    Name detections without an uppercase letter or digit ("helpfulness", "which") are ignored.
    Returns (text, redacted) where redacted says whether anything was masked.
    """
    keep_numbers = set(keep_numbers)
    protected = [m.span() for p in (CITATION, PLACEHOLDER) for m in p.finditer(text)]
    protected += [m.span() for n in keep_names if n and n.strip()
                  for m in re.finditer(r'(?<!\w)' + re.escape(n.strip()) + r'(?!\w)', text, re.IGNORECASE)]
    protected += [m.span() for m in NUMBER.finditer(text) if m.group(0) in keep_numbers]
    found = [*find(text), *((m.start(), m.end(), 'EMAIL_ADDRESS') for m in EMAIL.finditer(text))]

    pieces = []
    for start, end, kind in found:
        for low, high in sorted(protected):
            if low < end and high > start:
                if low > start:
                    pieces.append((start, low, kind))
                start = max(start, high)
        if start < end:
            pieces.append((start, end, kind))
    spans = []
    for start, end, kind in sorted(pieces):
        piece = text[start:end]
        while start < end and text[start] in TRIM:
            start += 1
        while end > start and text[end - 1] in TRIM:
            end -= 1
        if start == end or (kind == 'PERSON' and not re.search(r'[A-ZÄÖÜ0-9]', piece)):
            continue
        if spans and start < spans[-1][1]:
            spans[-1] = (spans[-1][0], max(end, spans[-1][1]), spans[-1][2])
        else:
            spans.append((start, end, kind))
    for index, (start, end, kind) in reversed(list(enumerate(spans, 1))):
        text = text[:start] + f'<{kind}_{index}>' + text[end:]
    return text, bool(spans)
