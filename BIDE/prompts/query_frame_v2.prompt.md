# QueryFrame V2 compiler contract

Convert one question into exactly one JSON object conforming to `query_frame_v2.schema.json`.

Do not answer the question and do not inspect memories, reference answers, Gold IDs, retrieval ranks, or failure lists. Express only constraints licensed by the question surface.

Rules:

1. Separate `subject`, `observer`, and `evaluated_subject`. “What might Alice say Bob is like?” binds observer=Alice and evaluated_subject=Bob.
2. `answer_role` says what the answer must be, not the topic. A city/country endpoint is LOCATION; “who supported” is PERSON.
3. Encode relation family and direction. Do not turn an inverse-role question into topical keyword search.
4. Negation has a scope. “has not hiked recently” does not negate a stable liking for hiking.
5. `evidence_shape`, not a route label, controls cardinality. Sets preserve different answer values; comparisons preserve both sides; causal questions preserve both ends.
6. Inference is closed-world: emit only listed `allowed_inference`; otherwise use DIRECT_ONLY/PARAPHRASE. Never invent world knowledge.
7. Populate `list_protection` so a later Top30 selector cannot replace an independent set member, comparison side, occurrence, polarity counterexample, or requested role merely because another Raw is topically similar.
