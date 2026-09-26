Compile one question into a minimal retrieval key. Do not answer it. Use only
the supplied closed ontology. Preserve the original question verbatim.

relation_family is a broad executable family, never a paraphrase of the whole
question. answer_type is the value requested, never SUMMARY. required_roles may
contain only ontology roles. target_entities contains only entities explicitly
licensed by the question. Separate observer from evaluated subject. Modality,
polarity, time and location constraints are requirements on acceptable evidence;
use modality_constraint=ANY and null scope when the question does not license
them. evidence_shape controls
evidence cardinality. Emit no memory IDs, answer guesses, or world knowledge.
