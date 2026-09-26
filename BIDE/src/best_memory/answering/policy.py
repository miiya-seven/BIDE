"""Deterministic validation and selection for answer-aware completion."""
import re

MONTHS={'january','february','march','april','may','june','july','august','september','october','november','december'}


def answer_literals(text):
    words=re.findall(r"[a-z]+|\d+",text.casefold())
    return {word for word in words if word.isdigit() or word in MONTHS}


def direct_literals_supported(claim, citations):
    required=answer_literals(claim)
    if not required:return True
    quoted=' '.join(c.get('quote','') for c in citations).casefold()
    present=set(re.findall(r"[a-z]+|\d+",quoted))
    return required<=present


def valid_citations(citations, packet):
    valid = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        raw_id, quote = citation.get('raw_id'), citation.get('quote')
        if raw_id not in packet or not isinstance(quote, str) or not quote.strip():
            continue
        sources = (packet[raw_id].get('original_text', ''), packet[raw_id].get('image_caption', ''))
        if any(' '.join(quote.split()) in ' '.join(source.split()) for source in sources):
            valid.append({'raw_id': raw_id, 'quote': quote})
    return valid


def validate_claims(claims, packet):
    if not isinstance(claims, list) or not claims:
        raise ValueError('empty_answer_claims')
    result = []
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get('claim'), str) or not claim['claim'].strip():
            raise ValueError('answer_claim_schema')
        citations = claim.get('citations')
        if not isinstance(citations, list):
            raise ValueError('answer_citation_schema')
        support_type=claim.get('support_type')
        derivation=claim.get('derivation')
        if support_type not in {'DIRECT','DERIVED'} or not isinstance(derivation,str):
            raise ValueError('answer_support_schema')
        valid = valid_citations(citations, packet)
        certificate_valid=(len(valid)==len(citations) and bool(valid) and
                           (support_type=='DIRECT' or bool(derivation.strip())) and
                           (support_type!='DIRECT' or direct_literals_supported(claim['claim'],valid)))
        result.append({'claim': claim['claim'].strip(), 'citations': valid,'support_type':support_type,
                       'derivation':derivation.strip(),'citations_valid':certificate_valid})
    return result


def should_retrieve(check):
    return check['decision'] == 'RETRIEVE' and bool(check['gaps'])


def select_answer(direct, revised, check, added):
    """Fail closed on unsupported revisions; never claim this guarantees QA correctness."""
    if not added or revised is None or check is None:
        return direct, 'no_qualified_new_evidence'
    if check['decision'] != 'USE_REVISED' or check['answer_support'] != 'SUPPORTED':
        return direct, 'revision_not_verified'
    if check['target_alignment'] != 'ALIGNED' or check['conflict_status'] != 'NONE':
        return direct, 'revision_target_or_conflict'
    if check['requirement_coverage'] != 'COMPLETE':
        return direct, 'revision_incomplete'
    claims = revised.get('claims') or []
    if not claims or any(not c['citations'] or not c['citations_valid'] for c in claims):
        return direct, 'revision_citations_invalid'
    if not any(c['raw_id'] in added for claim in claims for c in claim['citations']):
        return direct, 'revision_has_no_new_support'
    if not check['resolved_gaps']:
        return direct, 'no_resolved_gap'
    return revised, 'verified_revision'
