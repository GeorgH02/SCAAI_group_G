"""Input guard: checks a question before any model sees it.

Order: clean the text, block requests for personal data, block injection attempts,
reject questions that are not about Vienna accommodation, then ask Llama Guard (optional).
"""

import json
import re
import time
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

EXAMPLES = Path(__file__).with_name('domain_examples.txt')
OFF_TOPIC_EXAMPLES = Path(__file__).with_name('off_topic_examples.txt')
TOPIC_MARGIN = 0.025  # chosen on the tuning half of test_prompts.txt

# a few words between two terms
GAP = r'(?:\W+\w+){0,6}\W+'
PEOPLE = (r'(?:reviewers?|guests?|hosts?|owners?|landlords?|tenants?|gastgeber\w*|vermieter\w*|g[aä]ste?n?|'
          r'bewerter\w*|rezensenten|verfasser\w*)')
PERSONAL = (r'(?:names?|namen|e-?mails?|phone(?: numbers?)?|telefon\w*|addresses|adresse\w*|surnames?|'
            r'nachnamen?|vornamen?|identit\w+|kontaktdaten|contact (?:details|data|info\w*))')

IDENTITY_RULES = [
    ('personal data of reviewers or hosts', rf'\b{PERSONAL}\b{GAP}{PEOPLE}\b'),
    ('personal data of reviewers or hosts', rf'\b{PEOPLE}\b{GAP}{PERSONAL}\b'),
    ('asks who wrote a review', r"\bwho (?:wrote|posted|left)\b|\bwer hat\b(?:\W+\w+){0,6}\W+(?:geschrieben|bewertet|verfasst)\b"),
    ('asks for real names', r'\b(?:real|full|true) names?\b|\b(?:echten?|vollst[aä]ndigen?) namen\b'),
    ('de-anonymisation', r'\bde-?anonymi\w*|\bre-?identif\w*'),
    ('asks for first or last names',
     r'\b(?:first|last|sur|family|full)[- ]?names?\b|\b(?:vornamen?|nachnamen?|familiennamen?)\b|\bwie hei[ßs]en\b'),
    ('asks who a person is',
     r'\bwho (?:owns|hosts|manages|is behind)\b|\bwho (?:the )?(?:owner|landlord|host|author|reviewer)\b'
     r'|\bwho (?:is|are|was)\b(?:\W+\w+){0,3}?\W+the (?:owner|host|superhost|landlord|reviewer|author)s?\b'
     r'|\bwer\b(?:\W+\w+){0,4}\W+(?:gastgeber|vermieter|besitzer|eigent[üu]mer|autor|verfasser|inhaber|superhosts?|steckt)\w*'
     r'|\bwem\b(?:\W+\w+){0,4}\W+geh[öo]rt\b|\b(?:which|what) (?:person|people)\b|\bperson\b(?:\W+\w+){0,3}\W+behind\b'),
    ('asks for the author of a review',
     r'\b(?:author|writer)s?\b(?:\W+\w+){0,5}\W+(?:reviews?|comments?)\b'
     r'|\b(?:autor|verfasser|schreiber|urheber)\w*\b(?:\W+\w+){0,5}\W+(?:bewertung|rezension|kommentar)\w*'),
    ('asks for the profile of a person',
     r'\bprofiles?\b(?:\W+\w+){0,5}\W+(?:person|reviewer|guest|host|author)\w*'
     r'|\bprofil\w*\b(?:\W+\w+){0,5}\W+(?:person|gast|bewerter|gastgeber)\w*'),
    ('asks where a person lives',
     r'\bwhere\b(?:\W+\w+){0,6}\W+(?:live|lives|lived|reside|resides)\b|\bwo\s+(?:wohnt|lebt)\b'
     r'|\b(?:home|house|address)\b(?:\W+\w+){0,4}\W+(?:of|from)\W+(?:\w+\W+){0,2}(?:person|host|owner|landlord|reviewer|guest)\w*'
     r'|\b(?:zuhause|wohnort|wohnadresse|privatadresse|wohnsitz)\b(?:\W+\w+){0,4}\W+(?:person|gastgeber|vermieter|g[aä]ste?|bewerter|besitzer)\w*'),
    ('asks to contact a person',
     r'\bhow (?:can|do|could) i (?:contact|reach|call|email|message)\b(?:\W+\w+){0,3}\W+(?:them|him|her|the (?:reviewers?|hosts?|guests?|owners?|landlords?))\b'
     r'|\bwie (?:kann|könnte) ich\b(?:\W+\w+){0,4}\W+(?:kontaktieren|erreichen)\b|\bwie erreiche ich\b'),
    ('asks to list guests or reviewers',
     r'\b(?:name|list|identify|nenne)\b(?:\W+\w+){0,2}\W+(?:all )?(?:the |die )?(?:guests|reviewers|g[aä]ste|bewerter)\b'),
]

CITIES = ('Berlin Munich München Paris Rome Rom London Barcelona Madrid Prague Prag Budapest Amsterdam Hamburg Salzburg '
          'Graz Innsbruck Linz Zurich Zürich Geneva Genf Milan Mailand Venice Venedig Lisbon Lissabon Brussels Brüssel '
          'Copenhagen Kopenhagen Stockholm Warsaw Warschau Athens Athen Istanbul Tokyo Dubai Bratislava Ljubljana Zagreb '
          'Krakow Krakau Frankfurt Cologne Köln Dresden').split() + ['New York']
# "in Berlin", "near Prague" ...; "from Berlin" is not a search for a place there
OTHER_CITY = r'\b(?:in|near|around|at|for|bei|nahe|um|für|rund um)\s+(?:the\s+|der\s+|dem\s+|die\s+)?(?:' + '|'.join(CITIES) + r')\b'
VIENNA = (r'\b(?:vienna|wien|wiener|innere stadt|leopoldstadt|landstra\w+|wieden|margareten|mariahilf|neubau|josefstadt|'
          r'alsergrund|favoriten|simmering|meidling|hietzing|penzing|rudolfsheim\w*|ottakring|hernals|währing|döbling|'
          r'brigittenau|floridsdorf|donaustadt|liesing)\b')

MESSAGES = {
    'identity_request': {'en': "I can't share personal data about hosts or reviewers.",
                         'de': 'Ich kann keine persönlichen Daten von Gastgebern oder Bewertenden nennen.'},
    'injection': {'en': "I can't follow that request.", 'de': 'Dieser Aufforderung kann ich nicht folgen.'},
    'other_city': {'en': 'I only search accommodation in Vienna.', 'de': 'Ich suche nur Unterkünfte in Wien.'},
    'off_topic': {'en': 'I can only help with questions about accommodation in Vienna.',
                  'de': 'Ich kann nur Fragen zu Unterkünften in Wien beantworten.'},
    'unsafe_content': {'en': "I can't help with that request.", 'de': 'Dabei kann ich nicht helfen.'},
}

INJECTION_RULES = [
    ('override instructions',
     r'\b(?:ignore|disregard|forget|override|bypass)\b(?:\W+\w+){0,3}\W+(?:previous|prior|above|earlier|all|any|your|the|these|those|system)'
     r'\W+(?:\w+\W+){0,2}(?:instructions?|rules|prompts?|guidelines|polic(?:y|ies))\b'),
    ('override instructions (de)',
     r'\b(?:ignorier\w*|vergiss\w*|missachte\w*|umgehe\w*|[uü]berschreibe\w*)\b(?:\W+\w+){0,4}\W+'
     r'(?:anweisungen|regeln|instruktionen|vorgaben|richtlinien|befehle)\b'),
    ('asks for the system prompt', r'\b(?:system|developer)[- ]?(?:prompt|message)\b'),
    ('asks for hidden instructions',
     r'\b(?:reveal|show|print|repeat|output|leak|display|tell)\b(?:\W+\w+){0,5}\W+(?:your|the|initial|hidden)\W+(?:\w+\W+)?'
     r'(?:instructions|rules|prompt)\b'),
    ('asks for hidden instructions (de)',
     r'\b(?:zeig\w*|nenne\w*|gib\w*|verrate\w*|wiederhole\w*)\b(?:\W+\w+){0,5}\W+(?:deine|deinen|die)\W+(?:\w+\W+)?'
     r'(?:anweisungen|regeln|prompt)\b'),
    ('role play or mode switch',
     r'\b(?:you are now|from now on you|pretend (?:to be|you(?:\'re| are))|developer mode|dan mode|jailbreak\w*|'
     r'do anything now|you have no (?:restrictions|rules|limits))\b'
     r'|\b(?:du bist (?:jetzt|nun)|ab jetzt bist du|tu so als|verhalte dich wie|entwicklermodus|'
     r'du hast keine (?:einschr[aä]nkungen|regeln|grenzen))\b'),
    # only when it is aimed at the assistant; "ohne Einschränkungen" alone is normal in a review
    ('asks to drop the rules',
     r'\b(?:you (?:are|must|should|will)|answer|respond|reply|act|behave|antworte\w*|verhalte\w*|du (?:bist|musst|sollst|darfst))\b'
     r'(?:\W+\w+){0,6}\W+(?:without|ohne|no|keine?n?)\W+(?:any |jegliche[nr]? )?'
     r'(?:restrictions|filters|limits|rules|einschr[aä]nkungen|filter|regeln)\b'),
    ('fake system markup',
     r'<\|?\s*(?:system|im_start|im_end|assistant|user)\s*\|?>|\[/?(?:inst|system)\]|^\s*(?:system|assistant)\s*:|#{2,}\s*(?:system|instruction)'),
    ('database command', r'\bdrop\s+table\b|\btruncate\s+table\b|\bdelete\s+from\b|\binsert\s+into\b|\bunion\s+select\b|;\s*--|\balter\s+table\b'),
    # base64 has digits and both letter cases; a long list like "Austria/Germany/Japan/..." does not
    ('long encoded text', r'(?=[A-Za-z0-9+/]*\d)(?=[A-Za-z0-9+/]*[A-Z])(?=[A-Za-z0-9+/]*[a-z])[A-Za-z0-9+/]{60,}={0,2}'),
]


@dataclass(frozen=True)
class Decision:
    allowed: bool
    category: str  # ok, identity_request, injection, other_city, off_topic or unsafe_content
    reason: str
    seconds: float
    message: str = ''  # short reply for the user when blocked


def clean(text):
    """Normalise Unicode, drop invisible characters and collapse whitespace."""
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(c for c in text if unicodedata.category(c) not in ('Cf', 'Cc') or c in '\n\t')
    return re.sub(r'\s+', ' ', text).strip()


def reply(category, text):
    """Short answer for a blocked question, in German if the question looks German."""
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
    try:
        language = 'de' if detect(text) == 'de' else 'en'
    except LangDetectException:
        language = 'en'
    return MESSAGES[category][language]


def other_city(text):
    return bool(re.search(OTHER_CITY, text, re.IGNORECASE) and not re.search(VIENNA, text, re.IGNORECASE))


def first_match(rules, text):
    for reason, pattern in rules:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return reason
    return None


def load_examples(path=EXAMPLES):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]


def bge_m3_embedder(device=None):
    """Return embed(texts) -> normalised vectors, using the project's embedding model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)


def llama_guard(url='http://127.0.0.1:21434', model='llama-guard3:1b', timeout=30):
    """Return safety(text) -> (is_safe, categories), asking Llama Guard 3 through a local Ollama."""
    if urlparse(url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Llama Guard must run on this machine')

    def safety(text):
        body = json.dumps({'model': model, 'stream': False, 'options': {'temperature': 0},
                           'messages': [{'role': 'user', 'content': text}]}).encode()
        request = urllib.request.Request(url + '/api/chat', body, {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            words = json.load(response)['message']['content'].split()
        categories = [c for word in words[1:] for c in word.split(',') if c]
        return words[:1] == ['safe'], categories  # anything unexpected counts as unsafe
    return safety


class TopicGate:
    """Compares a question with in-domain and off-topic examples.

    The score is the best similarity to an in-domain example minus the best similarity
    to an off-topic example. A question is on topic if the score reaches the margin.
    """

    def __init__(self, embed, examples, off_topic_examples, margin=TOPIC_MARGIN):
        if not examples or not off_topic_examples:
            raise ValueError('The topic gate needs in-domain and off-topic example questions')
        self.embed, self.margin = embed, margin
        self.domain = embed(examples)
        self.other = embed(off_topic_examples)

    def score(self, text):
        vector = self.embed([text])[0]
        return float(max(self.domain @ vector) - max(self.other @ vector))


class InputGuard:
    """Without embed only the rules run; embed adds the topic gate, safety adds Llama Guard."""

    def __init__(self, embed=None, examples=None, off_topic_examples=None, margin=TOPIC_MARGIN, safety=None):
        self.safety = safety
        self.gate = TopicGate(embed, examples or load_examples(),
                              off_topic_examples or load_examples(OFF_TOPIC_EXAMPLES), margin) if embed else None

    def check(self, text):
        start = time.perf_counter()
        text = clean(text)

        def decision(allowed, category, reason):
            message = '' if allowed else reply(category, text)
            return Decision(allowed, category, reason, time.perf_counter() - start, message)

        reason = first_match(IDENTITY_RULES, text)
        if reason:
            return decision(False, 'identity_request', reason)
        reason = first_match(INJECTION_RULES, text)
        if reason:
            return decision(False, 'injection', reason)
        if other_city(text):
            return decision(False, 'other_city', 'asks about another city')
        if self.gate and self.gate.score(text) < self.gate.margin:
            return decision(False, 'off_topic', 'not about Vienna accommodation')
        if self.safety:
            is_safe, categories = self.safety(text)
            if not is_safe:
                return decision(False, 'unsafe_content', 'Llama Guard: ' + ','.join(categories))
        return decision(True, 'ok', '')

    def __call__(self, text):
        return self.check(text).allowed
