# Refining a thesis

A generated thesis is a starting point. The refinement agent revises it in response to structured feedback - up to three rounds per thesis - and every round is inspectable after the fact.

<!-- Clip: assets/clips/refine.mp4
<video controls muted playsinline width="100%">
  <source src="../assets/clips/refine.mp4" type="video/mp4">
</video>
-->

## Giving feedback

Below every unapproved thesis sits the **Refine Thesis** panel with a fixed set of feedback reasons:

- Too many risks, not enough opportunities
- Missing recent market trends
- Investment signals are too vague
- Opportunity score seems too low
- Analysis is too broad, be more specific
- Need stronger evidence for key themes

Select one or more reasons and click **Refine Thesis**. The reasons are deliberately structured rather than free text: each maps to revision behavior the agent actually knows how to execute against the retrieved evidence, which keeps refinement grounded in the same sources as the original thesis.

## What a round produces

When the round completes, the thesis re-renders in place and two panels record what happened:

- **Previous versions** - a snapshot of the thesis as it was before the round, with its score, recommendation, and the feedback reasons that drove the change. One entry per executed round.
- **Execution Trace** - which agent tool actually ran, its status, and the changes it reported. This is the audit trail for "what did the agent actually do".

A round can also come back with the thesis unchanged - most commonly when the narrative was refused and the sources still cannot support one, so the agent re-confirms the existing state rather than inventing a change. The round still counts against the cap and still records a version and a trace entry: the history logs every round the agent ran, not only the ones that altered the thesis. Such a round is labeled "No changes made" in the Execution Trace and in the status note after refining.

If the agent ever attempts to call a tool that does not exist, a **Hallucinations Detected** panel appears listing the invalid calls. Its absence is the normal case.

## The three-round cap

The counter in the panel header (`refinement 1/3`, `2 left`) tracks rounds used. After the third round the feedback options disappear and the panel shows:

> Max refinements reached (3/3). Please refine your original query for a fresh analysis.

This cap is enforced server-side.

## Approving

**Approve** finalizes the thesis. Approval is terminal: the refinement panel disappears, the thesis is marked approved in your library, and further refinement requests are rejected. Approving the same thesis twice is harmless (the API is idempotent about it).

!!! note "Concurrent edits"
    If the same thesis is approved or refined from another tab while a refinement round is running, the in-flight round is discarded rather than silently overwriting the newer state, and the app asks you to reload. You will only ever see the outcome of one writer.
