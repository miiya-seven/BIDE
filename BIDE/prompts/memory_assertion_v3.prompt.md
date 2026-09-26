# Clause-complete Memory Writer V3 contract

Input contains one immutable Original Raw plus deterministic clause spans. Produce one JSON object conforming to `memory_assertion_v3.schema.json`.

Do not use questions, answers, Gold, ranks, or known failures. Original Raw is the only authority.

Rules:

1. Process every clause. Record ASSERTED, NON_PROPOSITIONAL with reason, or UNRESOLVED. Silence is invalid.
2. Split coordinated propositions. Preserve evaluative/preference/negative/causal result clauses, including the second half after but/because/and.
3. One assertion must be an independently searchable proposition: name subject, canonical relation, typed value, semantic roles, polarity and relevant time/location owner. Avoid pronouns in `retrieval_text` when antecedents are available.
4. Scope negation narrowly. “not recently X but likes X” produces a negative recent-occurrence assertion and a positive preference assertion.
5. Explicit speech evaluation binds observer/speaker and evaluated_subject; do not collapse both to a generic person topic.
6. DIRECT assertions cite source clause(s). Derived assertions never masquerade as DIRECT and require an allowed rule_id and full derivation path.
7. Populate list_keys with answer value, occurrence, participant and side when applicable. These are protection keys, not ranking scores.
8. Do not propagate relevance across a whole Episode. Bridges are created only by the separate bridge compiler under enumerated rules.
