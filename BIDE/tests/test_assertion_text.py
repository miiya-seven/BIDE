from best_memory.retrieval.assertion_text import render_assertion


def test_preserves_unstated_structured_time_and_event():
    a = {'subject': 'Caroline', 'surface_relation': 'attended',
         'answer_value': {'value': 'a support group'},
         'retrieval_text': 'Caroline attended a support group.',
         'roles': {'event': 'support group visit'},
         'scope': {'time_expression': 'yesterday'}}
    text = render_assertion(a)
    assert text == ('Caroline attended a support group. | event: support group visit'
                    ' | time: yesterday')


def test_preserves_negation_in_sentence_and_zero_value():
    a = {'subject': 'Sam', 'surface_relation': 'owns',
         'answer_value': {'value': 0}, 'retrieval_text': 'Sam does not own a car.'}
    text = render_assertion(a)
    assert 'does not own' in text
    assert 'value: 0' in text
