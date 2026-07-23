# Refining a thesis

A generated thesis is a starting point. The refinement agent revises it in response to structured feedback - up to three rounds per thesis - and every round is inspectable after the fact.

<iframe width="100%" height="400" src="https://www.youtube-nocookie.com/embed/kFAswVOY0Qk"
  title="FinThesis: refining a thesis" frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen></iframe>

## Giving feedback

Below every unapproved thesis sits the **Refine Thesis** panel with a fixed set of feedback reasons:

- Too many risks, not enough opportunities
- Missing recent market trends
- Investment signals are too vague
- Opportunity score seems too low
- Analysis is too broad, be more specific
- Need stronger evidence for key themes

Select one or more reasons and click **Refine Thesis**. The reasons are deliberately structured rather than free text: each maps to revision behavior the agent actually knows how to execute against the retrieved evidence, which keeps refinement grounded in the same retrieved article pool as the original thesis. Some reasons (the evidence-seeking ones) re-focus which articles from that pool the rewrite reads, but never fetch anything new.

## What a round produces

When the round completes, the thesis re-renders in place and two panels record what happened:

- **Previous versions** - a snapshot of the thesis as it was before the round, with its score, recommendation, and the feedback reasons that drove the change. One entry per executed round.
- **Execution Trace** - which agent tool actually ran, its status, and the changes it reported. This is the audit trail for "what did the agent actually do".

A round can also come back with the thesis unchanged. The round still counts against the cap and still records a version and a trace entry: the history logs every round the agent ran, not only the ones that altered the thesis. Such a round is labeled "No changes made" in the Execution Trace and in the status note after refining.

Whether a no-op is likely depends on *why* the current thesis is the way it is:

- **Refining a [narrative-refused](../concepts/refusals.md) thesis often succeeds.** If the Raw Summary shows "the sources touch on related fintech topics but don't specifically address this query", that refusal was the model's own judgment call on that attempt - refinement reruns the narrative with your feedback added to the prompt, and for evidence-seeking feedback it also pulls more relevant articles from the retrieved pool, so a later pass can succeed where the first declined.
- **Refining a thin-evidence refusal is a guaranteed no-op.** If the Raw Summary instead says "the sources didn't give us enough to write a reliable narrative", that refusal came from a deterministic evidence floor, not the model's judgment. A refinement round on it skips the rewrite entirely - the floor reflects the retrieved pool, which is never re-fetched - so this outcome cannot change; the round will report "No changes made" every time. Broaden or rephrase the original query instead.

If the agent ever attempts to call a tool that does not exist, a **Hallucinations Detected** panel appears listing the invalid calls. Its absence is the normal case.

## The three-round cap

The counter in the panel header (`refinement 1/3`, `2 left`) tracks rounds used. After the third round the feedback options disappear and the panel shows:

> Max refinements reached (3/3). Please refine your original query for a fresh analysis.

This cap is enforced server-side.

## Approving

**Approve** finalizes the thesis. Approval is terminal: the refinement panel disappears, the thesis is marked approved in your library, and further refinement requests are rejected. Approving the same thesis twice is harmless (the API is idempotent about it).

!!! note "Concurrent edits"
    If the same thesis is approved or refined from another tab while a refinement round is running, the in-flight round is discarded rather than silently overwriting the newer state, and the app asks you to reload. You will only ever see the outcome of one writer.
