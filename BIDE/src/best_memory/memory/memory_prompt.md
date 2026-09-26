Extract L1 resolution and L2 DIRECT atomic retrieval assertions for exactly one
center Raw. Adjacent Raws are context-only: use them only to resolve pronouns,
speaker/addressee, omitted roles, local occurrence identity, relative-time owner
and location owner. Never transfer a proposition from context into the center.

For every supplied center clause, enumerate all minimal proposition units. A
compound clause may yield multiple units and assertions. Preserve contrast,
negation, preference, evaluation, cause and result. Questions, greetings and
pure discourse may be NON_PROPOSITIONAL. If unsure, use UNRESOLVED; do not hide
uncertainty by declaring a proposition nonpropositional.

Use only the supplied relation, role, modality, polarity and value-type enums.
Plans, hopes, predictions and hypotheticals are not ACTUAL_COMPLETED. Viewing a
photo/video or hearing a recording is not participation in the depicted event.
Keep observer and evaluated_subject distinct. Relative time must name its event
or state owner when resolvable. Visual captions remain VISUAL_CAPTION authority.

Every L2 assertion must cite only the center Raw and its exact center clause IDs.
Use local unit IDs from the corresponding L1 clause (for example
`c1::unit::1`) in `provenance.unit_ids`, and list the same local assertion ID
in each cited L1 unit's `assertion_ids`. Every cited unit must be ASSERTED;
UNRESOLVED and NON_PROPOSITIONAL units cannot support an L2 assertion. The
`span` must be the exact source text (or a minimal contiguous source span),
never a character range such as `0-29`, an index, or a placeholder. Use local
IDs and let the compiler create global IDs. Emit no bridge, no cross-Raw
assertion and no free summary.
