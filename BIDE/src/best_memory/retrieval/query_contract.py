"""Publish a retrieval query without losing its nested plan or exposing gold."""
from copy import deepcopy

def publish_query(record):
    q={key:deepcopy(record[key]) for key in (
        'sample_id','conversation_id','question','query_text','query_v41',
        'schema_version','query_arm','target_entities','relation_family',
        'answer_type','evidence_shape','required_roles') if key in record}
    q['conversation_id']=q.get('conversation_id') or q['sample_id'].split('__')[0].split('::')[0]
    plan=q.get('query_v41')
    if plan is not None:
        if not isinstance(plan,dict) or not plan.get('relation_queries'):
            raise ValueError('Incomplete query plan: '+q['sample_id'])
        for key in ('target_entities','owner_roles','relation_queries','value_queries',
                    'event_queries','temporal_queries','lexical_expansions','required_evidence_acts'):
            if not isinstance(plan.get(key),list):
                raise ValueError('Invalid query plan field: '+key)
        q['target_entities']=deepcopy(plan['target_entities'])
        q['required_roles']=deepcopy(plan['owner_roles'])
        if not q.get('query_text'):
            parts=[q['question']]
            for key in ('target_entities','relation_queries','value_queries','event_queries',
                        'temporal_queries','lexical_expansions','owner_roles','required_evidence_acts'):
                parts.extend(plan[key])
            q['query_text']=' | '.join(str(x) for x in parts if x)
        q['schema_version']='v41-query-rebuilt-v2'
    elif not q.get('query_text'):
        parts=[q['question'],*q.get('target_entities',[]),q.get('relation_family'),
               q.get('answer_type'),*q.get('required_roles',[])]
        q['query_text']=' '.join(str(x) for x in parts if x)
        q['schema_version']='v41-query-v1'
    q['gold_visible']=False
    return q
