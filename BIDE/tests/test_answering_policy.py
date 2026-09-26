from best_memory.answering.policy import select_answer, should_retrieve, validate_claims


def answer(raw_id='r1'):
    return {'pred_answer':'six months','claims':[{'claim':'It lasted six months.',
            'citations':[{'raw_id':raw_id,'quote':'six months'}],'support_type':'DIRECT',
            'derivation':'','citations_valid':True}], 'scope':{}}


def check(decision='USE_REVISED'):
    return {'decision':decision,'answer_support':'SUPPORTED','requirement_coverage':'COMPLETE',
            'target_alignment':'ALIGNED','conflict_status':'NONE','resolved_gaps':['g1']}


def test_only_actionable_gaps_trigger_retrieval():
    assert should_retrieve({'decision':'RETRIEVE','gaps':[{'gap_id':'g1'}]})
    assert not should_retrieve({'decision':'RETRIEVE','gaps':[]})
    assert not should_retrieve({'decision':'KEEP','gaps':[{'gap_id':'g1'}]})


def test_revision_requires_support_from_added_evidence():
    direct=answer('r1');revised=answer('r2')
    chosen,reason=select_answer(direct,revised,check(),['r2'])
    assert chosen is revised and reason=='verified_revision'
    chosen,reason=select_answer(direct,revised,check(),['r3'])
    assert chosen is direct and reason=='revision_has_no_new_support'


def test_unverified_revision_preserves_direct():
    direct=answer('r1');revised=answer('r2')
    chosen,reason=select_answer(direct,revised,check('RETRIEVE'),['r2'])
    assert chosen is direct and reason=='revision_not_verified'


def test_claim_citations_are_exactly_validated():
    packet={'r1':{'original_text':'It lasted six months exactly.','image_caption':''}}
    claims=validate_claims([{'claim':'duration','citations':[{'raw_id':'r1','quote':'six months'}],
                             'support_type':'DIRECT','derivation':''}],packet)
    assert claims[0]['citations_valid']


def test_direct_date_must_appear_in_citation():
    packet={'r1':{'original_text':'We organized a charity tournament.','image_caption':''}}
    claims=validate_claims([{'claim':'It happened on 8 May 2022.','citations':[{'raw_id':'r1','quote':'We organized a charity tournament.'}],
                             'support_type':'DIRECT','derivation':''}],packet)
    assert not claims[0]['citations_valid']
