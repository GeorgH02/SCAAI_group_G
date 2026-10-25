"""Labelled prompts for testing the guards. See test_prompts.txt for the format."""

from pathlib import Path

PROMPTS = Path(__file__).with_name('test_prompts.txt')
QUESTION_LABELS = ('benign', 'off_topic', 'injection', 'identity', 'other_city')
EVIDENCE_LABELS = ('ev_benign', 'ev_injection')


def load_prompts(path=PROMPTS):
    """Return a list of (label, language, text). Every second prompt per label is for tuning (dev)."""
    prompts = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            label, language, text = line.split(';', 2)
            prompts.append((label, language, text))
    return prompts


def split(prompts, part):
    """part is 'dev' or 'test'; alternate prompts within each label so both parts cover every group."""
    seen, chosen = {}, []
    for label, language, text in prompts:
        index = seen[(label, language)] = seen.get((label, language), -1) + 1
        if (index % 2 == 0) == (part == 'dev'):
            chosen.append((label, language, text))
    return chosen
