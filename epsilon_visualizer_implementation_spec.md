# Epsilon Visualizer — Implementation Specification

## Purpose

Build an interactive visualizer that converts pairwise empirical Turing-ordering results into a **compressed partial-order graph** at a chosen threshold `ε`.

At threshold `ε`, draw an edge `A → B` only when the **current evidence supports the claim** that `A ≥_ε B`.

The visualizer must **not** force the data into a fake linear ranking. It must:

- preserve **incomparability** when the evidence does not support an ordering,
- collapse **mutual indistinguishability / cycles** into equivalence clusters,
- make **missingness, instability, and uncertainty** visible,
- let the user change `ε` interactively and see how the structure changes.

This document is the implementation brief for the coding agent. Treat it as the **source of truth** for the feature unless the repository already contains a more specific compatible requirement.

---

## Research Context

The project defines a relational notion of model intelligence via adversarial indistinguishability.

For model types `A` and `B`, a fresh `B` acts as the distinguisher. It must decide whether it is talking to:

- a genuine fresh `B`, or
- an `A` instructed to imitate `B`.

The formal quantity of interest is:

```text
P_B^A(correct) = 1/2 · P[y = 1 | genuine B] + 1/2 · P[y = 0 | A → B]
```

and the relation is:

```text
A ≥_ε B  iff  P_B^A(correct) ≤ 1/2 + ε
```

Important constraints from the research context:

1. The project currently has **multiple experimental regimes**:
   - Condition A = `no_query_no_explanation`
   - Condition B = `yes_query_no_explanation`
   - Condition C = `no_query_yes_explanation`

2. Condition **B** is currently the most theoretically central one, because it is closest to the self-querying setting used in the existing transitivity argument.

3. The documents explicitly warn that the current finite prompt-shaped measurements are **not automatically identical** to the formal theoretical object. The visualizer must therefore show an **evidence graph**, not pretend to be a proof engine.

4. Self-pairs are primarily **calibration diagnostics**, not ranking data.

5. The visualizer is intended to help answer an interpretive bottleneck: how to turn noisy finite pairwise outcomes into a defensible ordering **without overclaiming**.

---

## Product Goal

The product should answer this question:

> Given the currently available empirical evidence, what partial ordering claims are supported at threshold `ε`, under a specific experimental condition?

The visualizer should support the following workflow:

1. choose condition,
2. choose `ε`,
3. inspect the resulting graph of supported claims,
4. see which models collapse into equivalence clusters,
5. inspect missing / unstable / borderline pairwise relations,
6. drill down into the evidence behind any edge or cluster.

---

## Non-Negotiable Design Principles

### 1) No fake total order
The default view must **not** output a linear ranking unless the supported graph actually induces one after compression.

### 2) Separate supported, tentative, unsupported, and missing
An absent edge can mean very different things:
- the claim is contradicted,
- the evidence is too weak,
- the pair is not computable,
- there is not enough valid data.

The UI must distinguish these cases.

### 3) Do not mix experimental conditions silently
Condition A, B, and C are distinct experimental objects. The graph must be condition-specific.

### 4) Collapse only what is strongly justified
Cycles / equivalence clusters should be formed from **supported** edges, not from weak or merely suggestive edges.

### 5) Make uncertainty first-class
The visualizer is not just a picture of point estimates. It must expose sample size, instability, filtering, and uncertainty.

### 6) Fail closed rather than fake formality
If the available data do not support a faithful estimate of the formal ordering quantity, the tool must say so clearly and either:
- switch to an explicitly labeled surrogate mode, or
- refuse to draw formal ordering edges.

---

## Required Modes

## Mode 1 — Formal / strict mode
Use this mode when the data are sufficient to estimate the formal distinguishability quantity for each ordered pair `(A, B)`.

Required inputs per trial (or enough information to reconstruct them):

- `actor_model`
- `distinguisher_model`
- `condition`
- branch label: genuine `B` vs imitation `A → B`
- parsed final answer `y ∈ {0,1}`
- validity flags / filtering metadata
- turn counts

In strict mode, for each ordered pair `(A, B)` compute:

```text
p_same_real_B  = P[y = 1 | genuine B]
p_diff_impostor = P[y = 0 | A → B]
p_correct = 1/2 * p_same_real_B + 1/2 * p_diff_impostor
```

Then the ordering claim `A ≥_ε B` is supported only when the evidence supports:

```text
p_correct ≤ 1/2 + ε
```

### Conservative support rule
Use a conservative one-sided uncertainty rule by default.

Recommended default:
- compute a one-sided upper confidence bound (or Bayesian upper credible bound) for `p_correct`,
- classify `A → B` as **supported** only if

```text
upper_bound(p_correct) ≤ 1/2 + ε
```

This prevents the graph from collapsing too aggressively on tiny samples.

### Borderline rule
Classify as **tentative / borderline** when:

```text
point_estimate(p_correct) ≤ 1/2 + ε
but
upper_bound(p_correct) > 1/2 + ε
```

### Unsupported rule
Classify as **unsupported** when the point estimate itself violates the threshold.

### Missing / uncomputable rule
Classify as **missing** when data are absent or filtered below a minimum valid-trial threshold.

---

## Mode 2 — Explicit surrogate mode
If strict mode is impossible because branch structure or equivalent required information is missing, implement a fallback mode **only if it is explicitly labeled as a surrogate**.

In surrogate mode:

- do **not** present the graph as a literal implementation of `A ≥_ε B`,
- do **not** reuse the formal symbol in the main title without qualification,
- label the view clearly as something like:
  - `Empirical surrogate view`, or
  - `Imitation-only evidence view`.

The surrogate mode is allowed to visualize thresholded pairwise fooling / distinguishability patterns, but the UI must make clear that this is a proxy rather than the formal ordering.

### Default behavior
Preferred order of behavior:

1. try strict mode,
2. if impossible, expose a warning and offer surrogate mode,
3. if neither is sound, show a clear error state instead of fabricating claims.

---

## Data Requirements and Ingestion

## Preferred inputs
The implementation should prefer **raw per-trial data** over aggregate summaries when both are available.

Potential sources in the repository may include:
- per-trial JSON files,
- `results.csv`,
- aggregate summary CSVs,
- condition-specific result folders.

## Minimum normalized record
Create an internal normalized schema like:

```ts
interface TrialRecord {
  actorModel: string;
  distinguisherModel: string;
  condition: 'A' | 'B' | 'C' | string;
  branch?: 'genuine_target' | 'imitation';
  answer?: 0 | 1;
  valid: boolean;
  error?: string | null;
  finishReason?: string | null;
  truncated?: boolean;
  answeredInOpening?: boolean;
  trialId?: string;
  turnCount?: number | null;
  specimenTurnCount?: number | null;
  rawPath?: string;
}
```

## Normalization tasks
The implementation must normalize:
- model aliases / nicknames,
- condition names,
- answer parsing,
- validity flags,
- turn metrics.

This is important because the research notes explicitly mention possible silent label mismatches and invalid-trial artifacts.

## Filtering policy
At minimum, expose toggles or a documented default policy for excluding:
- errored trials,
- truncated trials,
- invalid parse cases,
- `distinguisher_answered_in_opening` cases,
- other clearly broken runs.

The UI should always display:
- valid trial count used,
- excluded trial count,
- reason counts for exclusions.

---

## Pairwise Evidence Object

For each ordered pair `(A, B)` under a chosen condition, compute a pairwise evidence object.

Suggested internal structure:

```ts
interface PairEvidence {
  source: string; // A
  target: string; // B
  condition: string;
  mode: 'strict' | 'surrogate';

  nValid: number;
  nExcluded: number;

  estimate: number | null;          // main score used for thresholding
  lowerBound: number | null;
  upperBound: number | null;
  thresholdAtPoint?: number | null; // critical epsilon where point estimate flips
  thresholdAtUpper?: number | null; // critical epsilon where support becomes conservative

  statusAtEpsilon: 'supported' | 'tentative' | 'unsupported' | 'missing';

  details: {
    pSameRealB?: number | null;
    pDiffImpostor?: number | null;
    pCorrect?: number | null;
    averageTurnCount?: number | null;
    averageSpecimenTurnCount?: number | null;
    answerCounts?: Record<string, number>;
    exclusionCounts?: Record<string, number>;
  };
}
```

This should be cached for fast re-rendering when `ε` changes.

---

## Threshold Semantics

## Epsilon range
Use `ε ∈ [0, 0.5]` unless the repository or project code already defines a narrower valid range.

## Critical epsilon values
Do not treat `ε` as a meaningless free-floating decimal.

Precompute, for each pair:
- the `ε` at which the point estimate changes status,
- the `ε` at which the conservative upper bound changes status.

The slider should preferably:
- move continuously,
- but optionally **snap** to meaningful breakpoints,
- or at least show tick marks at meaningful breakpoints.

This makes the slider scientifically interpretable rather than decorative.

---

## Graph Construction

## Step 1 — Build thresholded directed graph
For the selected condition and `ε`, create a graph on model nodes.

### Supported graph `G_supported(ε)`
Include edge `A → B` iff pair status is `supported`.

### Tentative overlay graph `G_tentative(ε)`
Include edge `A → B` iff pair status is `tentative`.

Do **not** let tentative edges affect clustering.

## Step 2 — Collapse strongly connected components
Compute SCCs **only on `G_supported(ε)`**.

Each SCC becomes an equivalence cluster.

Rationale:
- If `A → B` and `B → A` are both strongly supported, they should collapse.
- If the apparent cycle depends only on tentative edges, it should remain visibly unstable rather than being promoted to equivalence.

## Step 3 — Build condensation graph
Condense SCCs into a DAG.

Each cluster node should contain:
- model names,
- cluster size,
- optional summary strength / uncertainty metadata,
- indicators for incident missing or tentative relations.

## Step 4 — Layer the DAG
Lay out the DAG as a hierarchy, with stronger / dominating clusters visually above weaker ones.

Possible layering strategies:
- topological layering,
- longest-path layering,
- or a deterministic DAG layout library.

The layout should be stable across nearby `ε` values as much as possible.

## Step 5 — Optional transitive reduction
Provide either:
- a toggle to show full supported DAG vs transitive reduction, or
- default to reduced DAG if readability is much better.

Do this **only after** SCC condensation and **only** when the graph is acyclic (which it should be after condensation).

Use caution: do not hide evidence in a way that makes incompleteness harder to see.

---

## Visual Requirements

## Main graph view
The main graph view must show:
- cluster nodes,
- supported edges,
- tentative edges (visually distinct),
- a clear notion of levels / partial order,
- obvious indication that unobserved / missing relations are not the same as supported incomparability.

### Recommended visual semantics
- **Supported edge**: solid
- **Tentative edge**: dashed / lighter
- **Missing / uncomputable**: not drawn as an edge in the main graph, but surfaced elsewhere clearly
- **Cluster node**: grouped card / bubble containing the member model labels

## Side matrix or evidence table (strongly recommended)
The graph alone cannot explain every absent edge. Implement a companion matrix or table that shows every ordered pair and its current status:
- supported
- tentative
- unsupported
- missing

This is important because the main graph intentionally suppresses clutter.

## Detail panel
Clicking an edge or pair should show:
- condition
- current mode (strict vs surrogate)
- point estimate
- uncertainty interval / bound
- valid / excluded trial counts
- exclusion breakdown
- average turn count
- average specimen-turn count if relevant
- relevant raw file identifiers / transcript links if available

Clicking a cluster should show:
- which models are inside,
- which supported edges induced the cluster,
- whether there are tentative edges internal or external to the cluster.

## Header controls
Required controls:
- condition selector (`A`, `B`, `C`, etc.)
- epsilon slider
- mode indicator (`strict` / `surrogate`)
- optional toggle for transitive reduction
- optional toggle for filtering policy

Optional but desirable:
- highlight a selected model,
- show only a selected tier or subset,
- pin node positions for stable comparison.

---

## Missingness and Instability

The UI must make incomplete structure visible.

Examples of incompleteness:
- ordered pair not run,
- too few valid trials,
- judge pathology makes pair effectively unusable,
- strict-mode data unavailable,
- heavy exclusions make inference unstable.

Recommended signals:
- a warning badge on the graph if any pair is missing,
- per-node badge summarizing incident missing pairs,
- companion matrix with explicit `missing` cells,
- summary text such as:
  - `Graph contains 11 supported edges, 4 tentative edges, 9 unsupported pairs, 6 missing pairs.`

This matters because the research goal is not just to show a hierarchy, but to show when the hierarchy is incomplete or fragile.

---

## Secondary Metrics

Turn metrics are not the primary ordering signal, but the research context explicitly treats them as useful secondary evidence.

At minimum, compute and expose:
- average distinguishing turns in Condition A,
- average distinguishing turns and specimen turns in Condition B,
- optional distributions if easy.

These should appear in the detail panel, not drive the ordering unless the project later changes the theory.

---

## Error Handling and Known Pathologies

The implementation should be robust to the pathologies already observed in the project:

- invalid / null content payloads,
- immediate answer-in-opening behavior,
- truncation,
- retry-loop selection effects,
- model alias mismatch,
- sparse and asymmetric coverage.

Specific requirements:

1. do not silently swallow problematic trials,
2. expose filtering and exclusion counts,
3. preserve the distinction between `unsupported` and `uncomputable`,
4. surface a warning when a pair depends on very few valid trials,
5. never infer equivalence from missing or filtered data.

---

## Suggested Technical Architecture

Use the repository’s existing stack if there is an obvious frontend/data-visualization framework already in place.

If there is no obvious existing stack, a sensible default is:

- **Data preparation**: Python or TypeScript script to parse CSV / JSON and produce a normalized JSON artifact.
- **Frontend**: React + TypeScript.
- **Graph/layout**: React Flow, D3, Dagre, ELK, or another deterministic graph layout library.
- **State**: lightweight local state; no need for heavy backend unless the repo already has one.

Recommended separation:

```text
1. parser / adapters
2. normalization and statistics layer
3. threshold classification layer
4. graph compression / layout layer
5. UI layer
```

### Example file structure (adapt to repo)

```text
/scripts/
  build_epsilon_visualizer_data.py

/src/
  /epsilon-visualizer/
    components/
      EpsilonVisualizer.tsx
      GraphView.tsx
      PairMatrix.tsx
      DetailPanel.tsx
      Controls.tsx
    lib/
      parsing.ts
      normalization.ts
      stats.ts
      thresholding.ts
      graph.ts
      layout.ts
      types.ts
    data/
      epsilon_visualizer_data.json
```

---

## Algorithms and Implementation Notes

## 1) Parsing / normalization
- discover result files,
- parse trial records,
- normalize model names,
- compute condition-specific trial sets,
- aggregate by ordered pair.

## 2) Statistical layer
Implement a small reusable statistics module.

At minimum support:
- point estimate calculation,
- conservative upper bound calculation,
- minimum-valid-trials gating,
- per-pair classification at `ε`.

Choose a method that is simple, documented, and robust.
Examples:
- Wilson interval,
- Clopper–Pearson,
- Beta-Binomial posterior interval.

Do **not** bury the choice; expose it in the code and, ideally, in a tooltip or short note.

## 3) Thresholding
Given pair evidence and `ε`, return one of:
- `supported`
- `tentative`
- `unsupported`
- `missing`

## 4) SCC / condensation
Implement or reuse a reliable SCC algorithm.

Important rule:
- use only **supported** edges for SCC formation.

## 5) Layout
Use a deterministic layout so that small epsilon changes do not completely scramble positions.

If possible:
- preserve node positions between nearby `ε` values,
- keep cluster motion smooth when clusters split or merge.

## 6) Performance
The graph size is likely modest, so clarity matters more than micro-optimization.
Still, cache pair evidence and recompute only what depends on `ε`.

---

## Acceptance Criteria

The implementation is complete when all of the following are true:

1. A user can choose an experimental condition and `ε`.
2. The app renders a directed graph of supported pairwise claims.
3. Mutual supported cycles collapse into equivalence clusters.
4. Tentative edges are visible but do not drive clustering.
5. Missing / uncomputable pairs are clearly distinguishable from unsupported ones.
6. The app never forces a total ranking when the graph is partial.
7. Clicking an edge or pair reveals the underlying evidence and filtering.
8. The app clearly indicates whether it is in strict mode or surrogate mode.
9. The code includes tests for threshold classification and SCC compression.
10. The implementation runs locally and survives basic manual interaction without obvious errors.

---

## Minimum Test Plan

Implement automated tests for at least the following synthetic cases.

### Case 1 — Clean chain
Evidence supports:
- `A → B`
- `B → C`
- `A → C`

Expected result:
- three singleton nodes,
- DAG layering with `A` above `B` above `C`.

### Case 2 — Strong equivalence pair
Evidence supports:
- `A → B`
- `B → A`
- `A/B → C`

Expected result:
- cluster `{A, B}` collapsed,
- cluster above `C`.

### Case 3 — Tentative reverse edge
Evidence supports:
- `A → B` strongly,
- `B → A` only tentatively.

Expected result:
- no cluster collapse,
- tentative reverse edge shown distinctly.

### Case 4 — Missing pair
Evidence missing for one or more directed pairs.

Expected result:
- graph renders,
- missingness surfaced in matrix / warnings,
- no fabricated relation.

### Case 5 — Epsilon sweep
At lower `ε`, graph is sparse; at higher `ε`, graph merges.

Expected result:
- slider changes statuses correctly,
- clusters form only when supported thresholds are crossed.

---

## Conservative Defaults

Unless the repository already defines better project-wide defaults, use the following:

- default condition: **B** (`yes_query_no_explanation`),
- default display: **supported graph + tentative overlay + pair matrix**,
- default threshold rule: **conservative upper bound**,
- default clustering: **supported edges only**,
- default behavior on insufficient formal data: **warning + explicit surrogate option**, not silent fallback.

---

## Open Questions the Code Should Not Hide

The implementation may need to proceed before the theory is fully settled. That is acceptable, but the UI should not pretend these open issues are resolved.

The code / UI should preserve visibility of the following unresolved questions:

1. whether the current data fully identify the formal `P_B^A(correct)` quantity,
2. whether `ε` should later depend on turn budget or other resources,
3. how to interpret results across different conditions,
4. whether some pathologies are model failures or harness artifacts,
5. whether third-party judge variants will later require a second visualizer mode.

A short note or help tooltip is enough; the important part is not to erase the uncertainty.

---

## Deliverables

The coding agent should produce:

1. the interactive epsilon visualizer,
2. the data parsing / normalization pipeline needed to feed it,
3. tests for threshold logic and graph compression,
4. a brief README section or inline documentation explaining:
   - the meaning of the graph,
   - strict vs surrogate mode,
   - the uncertainty rule used,
   - how to run the visualizer locally.

---

## Recommended Implementation Strategy

Implement in this order:

1. build normalized pair-evidence objects from available data,
2. implement threshold classification and tests,
3. implement supported-edge graph + SCC collapse,
4. implement the main graph view,
5. add epsilon slider and condition selector,
6. add pair matrix and detail panel,
7. add warnings / missingness / surrogate-mode handling,
8. run end-to-end manual debugging and clean up UX.

Do not start from styling. Start from evidence semantics and correctness.

---

## Source Context Used To Write This Spec

This specification was synthesized from the project documents discussing:
- the formal Turing-ordering framework,
- the role of Conditions A / B / C,
- the interpretive bottleneck around pairwise tables,
- the need for an epsilon-threshold visualizer,
- the requirement for compressed partial-order display,
- the importance of uncertainty, missingness, and non-forced ranking.

The intent of this spec is to turn that research requirement into an implementable engineering task without losing the conceptual constraints.
