"""Render L2 search text without concatenating the same predicate twice."""

def render_assertion(assertion):
    sentence = str(assertion.get('retrieval_text') or '').strip()
    fields = {
        'subject': assertion.get('subject'),
        'relation': assertion.get('surface_relation'),
        'value': (assertion.get('answer_value') or {}).get('value'),
        'event': (assertion.get('roles') or {}).get('event'),
        'time': (assertion.get('scope') or {}).get('time_expression'),
    }
    # Labels preserve structured values absent from the generated sentence.
    # Only exact text inclusion is removed; no semantic equivalence is guessed.
    parts = [sentence] if sentence else []
    for label, value in fields.items():
        if value is None or not str(value).strip():
            continue
        value = str(value).strip()
        if value.casefold() not in ' '.join(parts).casefold():
            parts.append(f'{label}: {value}')
    return ' | '.join(parts)
