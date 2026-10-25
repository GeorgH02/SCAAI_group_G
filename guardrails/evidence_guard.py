"""Evidence guard: removes rows whose text tries to give instructions to the model.

Reviews and descriptions are written by strangers, so a row can contain an injection.
Rows with such text are dropped from the tool result; the other rows are kept.
"""

import re

from guardrails.input_guard import INJECTION_RULES, clean, first_match

# instructions aimed at the model, not typical for a review
EVIDENCE_RULES = [
    ('addresses the model', r'\b(?:tell|ask|instruct|order)\s+the\s+(?:user|reader|assistant|model)\b'
                            r'|\bwhen you (?:summari[sz]e|answer|respond)\b|\bin your (?:answer|response|summary)\b'
                            r'|\b(?:dear|hey|hi|hello)\s+(?:assistant|chatbot|ai|llm|language model)\b'),
    ('addresses the model (de)', r'\bsag(?:e)?\s+(?:dem\s+)?nutzer\b|\bwenn du (?:diese|die) bewertung\b|\bin deiner (?:antwort|zusammenfassung)\b'
                                 r'|\b(?:lieber|liebe|hallo|hey)\s+(?:assistent|chatbot|sprachmodell|ki)\b'),
]
IDENTIFIERS = ('review_id', 'listing_id', 'host_id')


def find_injection(value):
    """Return the reason if any text inside value (string, list or dict) looks like an injection."""
    if isinstance(value, str):
        text = clean(value)
        return first_match(INJECTION_RULES, text) or first_match(EVIDENCE_RULES, text)
    items = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    for item in items:
        reason = find_injection(item)
        if reason:
            return reason
    return None


def remove_hostile_rows(result):
    """Return the tool result without rows that contain an injection, and say what was removed."""
    rows = result.get('rows') if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return result
    kept, removed = [], []
    for row in rows:
        reason = find_injection(row)
        if reason is None:
            kept.append(row)
        else:
            key = next((k for k in IDENTIFIERS if isinstance(row, dict) and k in row), None)
            removed.append({key: row[key], 'reason': reason} if key else {'reason': reason})
    if not removed:
        return result
    notice = f'{len(removed)} row(s) were removed because their text looked like instructions to the model.'
    return {**result, 'rows': kept, 'rows_removed_by_guard': removed, 'notices': [*result.get('notices', []), notice]}
