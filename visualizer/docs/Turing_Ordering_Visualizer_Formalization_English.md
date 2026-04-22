---
lang: en
numbersections: true
fontsize: 11pt
geometry: margin=1in
header-includes:
  - |
    \usepackage{amsmath,amssymb,amsthm,mathtools}
    \usepackage{lmodern}
    \usepackage{xcolor}
    \usepackage{titlesec}
    \usepackage{fancyhdr}
    \usepackage{enumitem}
    \usepackage{microtype}
    \definecolor{accent}{HTML}{193A63}
    \titleformat{\section}{\Large\bfseries\color{accent}}{\thesection.}{0.6em}{}
    \titleformat{\subsection}{\large\bfseries\color{accent}}{\thesubsection}{0.6em}{}
    \pagestyle{fancy}
    \fancyhf{}
    \fancyhead[L]{\footnotesize Turing Ordering - Formalization for the visualizer}
    \fancyfoot[C]{\thepage}
    \setlength{\headheight}{14pt}
    \setlength{\parindent}{0pt}
    \setlength{\parskip}{0.45em}
    \setlist[itemize]{leftmargin=1.5em}
    \setlist[enumerate]{leftmargin=1.6em}
    \theoremstyle{definition}
    \newtheorem{definition}{Definition}
    \newtheorem{observation}{Observation}
    \theoremstyle{plain}
    \newtheorem{proposition}{Proposition}
    \newtheorem{corollary}{Corollary}
...

\begin{center}
{\LARGE\bfseries Mathematical formalization for the Turing Ordering visualizer\par}
\vspace{0.5em}
{\large\bfseries Definition of $\varepsilon$, empirical relation, transitivity diagnostic, and coherent graphical solution\par}
\vspace{0.4em}
{\itshape Internal working document - operational version for implementation\par}
Based on the project notes, the short memorandum, and the operational specification sent by the supervisor on 16/04/2026.
\end{center}

**Main result.** For the visualizer, $\varepsilon$ should not be "estimated" from the data: it should be treated as a threshold chosen by the user on the distinguisher's advantage. The rule consistent with the project formalism is

$$
A \succeq_{\varepsilon,\pi} B \iff d_{\pi}(A,B) \le \varepsilon,
$$

where $d_{\pi}(A,B)$ is judge $B$'s advantage against $A$ above chance. It follows that the most natural quantity to store and visualize is the *critical epsilon*

$$
\varepsilon^{\star}_{\pi}(A,B) := \max\{0,d_{\pi}(A,B)\},
$$

so that $A \succeq_{\varepsilon,\pi} B$ holds if and only if $\varepsilon^{\star}_{\pi}(A,B) \le \varepsilon$. This choice resolves the sign issue, cleanly separates theory and empirical evidence, and makes the visualizer a simple threshold on a weighted matrix. If cycles or transitivity violations emerge, the correct visualization is not a linear ranking but the directed graph, together with its quotient by strongly connected components.

# Mathematical objective of the visualizer

The visualizer must represent, for a threshold $\varepsilon$ chosen by the user, which relations of the form "$A$ is at least as intelligent as $B$" are supported by the data. The problem is not only graphical: it is first of all formal. One must decide which quantity is thresholded, how it is estimated from the existing tables, and how to treat cases in which the observed relation is not transitive.

To avoid collision with the numeric symbol $\ge$, in this note I use the notation

$$
A \succeq_{\varepsilon,\pi} B
$$

for the same relation that in the project documents appears as $A \ge_{\varepsilon} B$. The index $\pi$ reminds us that every relation depends on the fixed experimental protocol: condition (A/B/C), turn limit, any specimen-query budget, prompt template, validity filter, and in general all harness choices. The visualizer must not mix different protocols.

# Ideal formalism: the theoretical object

Let $M$ be the set of model types and let $\pi \in \Pi$ be a fixed protocol. For each ordered pair $(A,B) \in M^2$, consider the following ideal experiment:

- the distinguisher is a fresh instance of $B$;
- with probability $1/2$ the distinguisher talks to a genuine fresh instance of $B$;
- with probability $1/2$ the distinguisher talks to a fresh instance of $A$ instructed to imitate $B$; we write this case as $A \to B$;
- at the end the distinguisher emits $Y \in \{0,1\}$, where $Y=1$ means "I believe I am talking to a real $B$" and $Y=0$ means "I believe I am talking to an impostor".

\begin{definition}[Success probability of the distinguisher]
For protocol $\pi$ and pair $(A,B)$, define
\[
P^{A}_{B,\pi}(\mathrm{correct}) := \tfrac12 \Pr_{\pi}[Y=1 \mid B\ \text{genuine}] + \tfrac12 \Pr_{\pi}[Y=0 \mid A \to B].
\]
\end{definition}

\begin{definition}[Distinguisher advantage]
Define the advantage of distinguisher $B$ against actor $A$ as
\[
d_{\pi}(A,B) := P^{A}_{B,\pi}(\mathrm{correct}) - \tfrac12.
\]
\end{definition}

This is the theoretically natural quantity: it measures how much $B$ beats chance in distinguishing $A \to B$ from a real $B$.

By construction $d_{\pi}(A,B) \in [-1/2,1/2]$. If $d_{\pi}(A,B)$ is small, then $A$ passes well for $B$; if it is large, then $B$ distinguishes well and therefore $A$ is not credibly "at least as much as $B$" under that protocol.

\begin{definition}[Threshold ordering]
For $\varepsilon \ge 0$, define
\[
A \succeq_{\varepsilon,\pi} B \iff d_{\pi}(A,B) \le \varepsilon.
\]
In words: $A$ is at least as intelligent as $B$ at level $\varepsilon$ if $B$'s advantage against $A$ does not exceed $\varepsilon$.
\end{definition}

\begin{definition}[Strict relation]
Define
\[
A \succ_{\varepsilon,\pi} B \iff A \succeq_{\varepsilon,\pi} B\ \text{and}\ \neg(B \succeq_{\varepsilon,\pi} A).
\]
\end{definition}

\begin{observation}[Sign correction]
The supervisor's WhatsApp specification correctly describes the advantage as the distinguisher's "gap from 1/2", but it reverses the direction of the threshold in the sentence "if the advantage $\ge \varepsilon$, then $A \ge B$". That direction is incompatible with the project's definition. If the distinguisher's advantage increases, imitation gets worse, not better. Therefore the correct rule for the visualizer must be
\[
A \succeq_{\varepsilon,\pi} B \iff d_{\pi}(A,B) \le \varepsilon.
\]
\end{observation}

# Empirical object currently available: what can actually be computed from the tables

The current tables do not implement the ideal experiment literally as a single hidden genuine-vs-impostor branch for each ordered pair. Operationally, what we have are counts of valid trials for ordered actor-model / judge-model pairs. One must therefore formalize the operational estimator usable by the visualizer.

For each ordered pair $(A,B)$ and protocol $\pi$, let

$$
N^{(1)}_{A,B,\pi} := \#\{\text{valid trials with answer 1}\}, \qquad
N^{(0)}_{A,B,\pi} := \#\{\text{valid trials with answer 0}\},
$$

$$
N_{A,B,\pi} := N^{(1)}_{A,B,\pi} + N^{(0)}_{A,B,\pi}.
$$

From the available data one can construct two proportions:

$$
q^{b}_{\pi}(B) := \frac{N^{(1)}_{B,B,\pi}}{N_{B,B,\pi}}, \qquad
r^{b}_{\pi}(A,B) := \frac{N^{(0)}_{A,B,\pi}}{N_{A,B,\pi}},
$$

when the denominators are positive. Here:

- $q^{b}_{\pi}(B)$ is the frequency with which $B$ answers "same" in the self-pair $B$ vs $B$;
- $r^{b}_{\pi}(A,B)$ is the frequency with which $B$ answers "different" when the actor is $A$ imitating $B$.

The plug-in estimator coherent with the theoretical formula is then

$$
\widehat{P}^{A}_{B,\pi}(\mathrm{correct}) := \tfrac12 q^{b}_{\pi}(B) + \tfrac12 r^{b}_{\pi}(A,B),
$$

$$
\widehat{d}_{\pi}(A,B) := \widehat{P}^{A}_{B,\pi}(\mathrm{correct}) - \tfrac12 = \tfrac12\bigl(q^{b}_{\pi}(B) + r^{b}_{\pi}(A,B) - 1\bigr).
$$

**Crucial caveat.** This $\widehat{d}_{\pi}(A,B)$ is the correct object given the current dataset, but it coincides with the ideal theoretical advantage only if the self-pair $B$ vs $B$ is accepted as the available proxy for the "real $B$" branch of the ideal experiment. Thus the visualizer can already be formally coherent, but the methodological note should state explicitly that it is working with an operational empirical relation reconstructed from the current tables, rather than with the ideal experiment realized literally as one hidden randomized branch for each ordered pair.

# The right definition of $\varepsilon$ for the visualizer

## Correct interpretation

In the visualizer, $\varepsilon$ is not a quantity to be inferred automatically. It is a global tolerance threshold on distinguishability. The user chooses how small the judge's advantage must be in order to consider $A$ at least as much as $B$.

The natural range is

$$
0 \le \varepsilon \le \tfrac12.
$$

At $\varepsilon = 0$, one requires that the distinguisher have no positive advantage over chance; for larger $\varepsilon$, the relation widens and the graph progressively collapses.

## Critical epsilon

The most useful quantity to define and store for each pair is the *critical epsilon*

$$
\varepsilon^{\star}_{\pi}(A,B) := \inf\{\varepsilon \ge 0 : A \succeq_{\varepsilon,\pi} B\}.
$$

It follows immediately from the definition that

$$
\varepsilon^{\star}_{\pi}(A,B) = \max\{0,d_{\pi}(A,B)\}.
$$

Analogously, for the empirical object,

$$
\widehat{\varepsilon}^{\star}_{\pi}(A,B) := \max\{0,\widehat{d}_{\pi}(A,B)\}.
$$

\begin{proposition}[Threshold equivalence]
For every pair $(A,B)$ and every $\varepsilon \ge 0$,
\[
A \succeq_{\varepsilon,\pi} B \iff \varepsilon^{\star}_{\pi}(A,B) \le \varepsilon,
\]
and analogously
\[
A \widehat{\succeq}_{\varepsilon,\pi} B \iff \widehat{\varepsilon}^{\star}_{\pi}(A,B) \le \varepsilon.
\]
\end{proposition}

**Proof.** Saying that $A \succeq_{\varepsilon,\pi} B$ is equivalent, by definition, to saying that $d_{\pi}(A,B) \le \varepsilon$. If $d_{\pi}(A,B) \le 0$, the relation holds for every $\varepsilon \ge 0$ and therefore the minimum admissible value is $0$. If instead $d_{\pi}(A,B) > 0$, the minimum admissible value is precisely $d_{\pi}(A,B)$. Therefore $\varepsilon^{\star}_{\pi}(A,B)=\max\{0,d_{\pi}(A,B)\}$.

This is the cleanest solution for the visualizer: one precomputes a weighted matrix $\widehat{\varepsilon}^{\star}_{\pi}(A,B)$ and the $\varepsilon$ slider simply applies a threshold to that matrix.

# Empirical relation, transitivity, and correct visualization

Define the empirical relation observed at level $\varepsilon$ as

$$
A \widehat{\succeq}_{\varepsilon,\pi} B \iff \widehat{d}_{\pi}(A,B) \le \varepsilon \iff \widehat{\varepsilon}^{\star}_{\pi}(A,B) \le \varepsilon.
$$

## View 1: raw directed graph

For a fixed protocol $\pi$ and threshold $\varepsilon$, define the directed graph

$$
G_{\varepsilon,\pi} = (M,E_{\varepsilon,\pi}), \qquad E_{\varepsilon,\pi} := \{(A,B) : A \widehat{\succeq}_{\varepsilon,\pi} B\}.
$$

This is the first object to visualize. An edge $A \to B$ means that, at that threshold, the empirical evidence supports the claim "$A$ is at least as much as $B$".

Useful relations:

$$
A \widehat{\sim}^{\mathrm{pair}}_{\varepsilon,\pi} B \iff A \widehat{\succeq}_{\varepsilon,\pi} B \ \text{and}\ B \widehat{\succeq}_{\varepsilon,\pi} A,
$$

$$
A \widehat{\succ}_{\varepsilon,\pi} B \iff A \widehat{\succeq}_{\varepsilon,\pi} B \ \text{and}\ \neg(B \widehat{\succeq}_{\varepsilon,\pi} A).
$$

## Transitivity diagnostic

The condition to be checked is

$$
\forall A,B,C \in M, \qquad A \widehat{\succeq}_{\varepsilon,\pi} B \ \text{and}\ B \widehat{\succeq}_{\varepsilon,\pi} C \implies A \widehat{\succeq}_{\varepsilon,\pi} C.
$$

Equivalently, a transitivity violation is a triple $(A,B,C)$ such that

$$
A \widehat{\succeq}_{\varepsilon,\pi} B, \qquad B \widehat{\succeq}_{\varepsilon,\pi} C, \qquad \neg\bigl(A \widehat{\succeq}_{\varepsilon,\pi} C\bigr).
$$

A useful metric to display next to the graph is

$$
TV_{\varepsilon}(\pi) := \#\bigl\{(A,B,C) : A \widehat{\succeq}_{\varepsilon,\pi} B,\ B \widehat{\succeq}_{\varepsilon,\pi} C,\ \neg\bigl(A \widehat{\succeq}_{\varepsilon,\pi} C\bigr)\bigr\}.
$$

This quantity does not replace the graph, but it immediately signals whether the empirical order resembles a quasi-order or not.

## View 2: quotient by strongly connected components

If the graph contains cycles, a linear ranking can be misleading. The mathematically correct repair is to pass from individual entities to strongly connected components (SCCs). Define a graphical equivalence relation:

$$
A \equiv_{\varepsilon,\pi} B \iff A \ \text{and}\ B \ \text{belong to the same SCC of}\ G_{\varepsilon,\pi}.
$$

This equivalence coincides with the idea of "same bucket / same height" not only in the directly symmetric cases $A \leftrightarrow B$, but also in the presence of longer cycles such as $A \to B \to C \to A$.

\begin{proposition}[The SCC quotient is always a DAG]
If one collapses each SCC of $G_{\varepsilon,\pi}$ into a single meta-node, the resulting quotient graph
\[
\mathrm{Cond}(G_{\varepsilon,\pi})
\]
is acyclic.
\end{proposition}

**Proof.** If there were a cycle among distinct meta-nodes in the quotient, then all components along that cycle would be mutually reachable. But this contradicts the maximality of SCCs, which by definition already collapse all mutually reachable nodes. Therefore the quotient cannot contain cycles.

\begin{corollary}[Interpretation]
The SCC quotient always provides a coherent hierarchical visualization of the cyclic structure of the graph. It should not by itself be read as proving that the empirical relation is transitive: acyclic failures of transitivity may still be present in the raw graph. If the empirical relation does turn out to be transitive, the quotient coincides with the usual passage to equivalence classes of the quasi-order.
\end{corollary}

## Vertical axis and "height"

Once the SCC DAG has been obtained, one can define the height of a component $C$ formally as

$$
h(C) =
\begin{cases}
0 & \text{if } C \text{ has no outgoing edges},\\
1 + \max\{h(D) : C \to D\} & \text{otherwise.}
\end{cases}
$$

Drawing components with greater height higher realizes exactly the idea "$A$ is above $B$ if $A$ is strictly greater than $B$" in the quotient view. The implemented visualizer keeps the full quotient DAG rather than reducing redundant edges, so that all supported evidence remains visible.

# Complete recommended algorithm

**Minimal correct pipeline.**

1. Fix a protocol $\pi$; do not aggregate different conditions in the same graph.
2. For each $B$, compute $q^{b}_{\pi}(B) = N^{(1)}_{B,B,\pi}/N_{B,B,\pi}$.
3. For each pair $(A,B)$, compute $r^{b}_{\pi}(A,B) = N^{(0)}_{A,B,\pi}/N_{A,B,\pi}$.
4. Compute $\widehat{d}_{\pi}(A,B) = \tfrac12\bigl(q^{b}_{\pi}(B) + r^{b}_{\pi}(A,B) - 1\bigr)$.
5. Compute $\widehat{\varepsilon}^{\star}_{\pi}(A,B) = \max\{0,\widehat{d}_{\pi}(A,B)\}$.
6. Given the $\varepsilon$ slider, create the edge $A \to B$ if and only if $\widehat{\varepsilon}^{\star}_{\pi}(A,B) \le \varepsilon$.
7. Visualize both the raw graph $G_{\varepsilon,\pi}$ and the SCC quotient DAG.
8. Print the number of transitivity violations $TV_{\varepsilon}(\pi)$ and the list of problematic triples.

## Missing data and reliability

The visualizer must distinguish three logical states:

- **edge present**: the datum exists and the threshold is satisfied;
- **edge absent**: the datum exists but the threshold is not satisfied;
- **edge unknown**: there are not enough valid trials to estimate $\widehat{d}_{\pi}(A,B)$.

In particular, if $N_{A,B,\pi}=0$ or $N_{B,B,\pi}=0$, the relation for that pair must be marked as *undefined*, not as false. This is essential in order to prevent harness pathologies or API failures from being mistaken for negative evidence.

An additional practical threshold is a minimum number of valid trials $n_{\min}$. In that case one requires

$$
N_{A,B,\pi} \ge n_{\min}, \qquad N_{B,B,\pi} \ge n_{\min}
$$

before making a binary decision visible on the edge.

# Recommended extension: statistical uncertainty

For a first version of the visualizer, the point matrix $\widehat{\varepsilon}^{\star}_{\pi}$ is enough. However, since many counts are small, a more rigorous version should add an uncertainty band.

A simple and stable solution is to work with separate Wilson intervals for the two proportions:

$$
I_q(B) \ni q^{b}_{\pi}(B), \qquad I_r(A,B) \ni r^{b}_{\pi}(A,B).
$$

From these one obtains an interval for the advantage via interval addition:

$$
I_d(A,B) := \tfrac12 I_q(B) + \tfrac12 I_r(A,B) - \tfrac12.
$$

One can then adopt a three-valued logic:

- **supported**: $\sup I_d(A,B) \le \varepsilon$;
- **not supported**: $\inf I_d(A,B) > \varepsilon$;
- **indeterminate**: otherwise.

This is not necessary to get started, but it is the natural next step if the visualizer is to be a serious interpretive tool, not merely an illustrative one.

# Decisions that the document should fix explicitly

1. **Direction of the threshold.** It must be stated unambiguously that the ordering uses $d \le \varepsilon$, not $d \ge \varepsilon$.
2. **Indexing by protocol.** Every matrix and every graph must be indexed by $\pi$. There is no such thing as *the* empirical Turing ordering without specifying the experimental condition.
3. **Empirical object vs ideal object.** It must be declared that the visualizer works on the plug-in estimator built from the current tables, using the self-pair as a calibration component.
4. **Missing data.** Missing does not mean false. Cells without sufficient validity must remain undetermined.
5. **Standard visualization.** The standard view should show both the raw graph and the SCC DAG; a single linear scale should be used only if the resulting DAG is sufficiently clean.

# Conclusion

The mathematical problem of the visualizer is solved cleanly by introducing a distinction that was so far only implicit in the materials:

- $d_{\pi}(A,B)$ is the distinguisher's advantage;
- $\varepsilon$ is a global tolerance threshold on that advantage;
- $\varepsilon^{\star}_{\pi}(A,B)$ is the minimum threshold required for the relation to hold for a given pair.

From this the practical solution follows: precompute $\widehat{\varepsilon}^{\star}_{\pi}(A,B)$, then visualize the edges with $\widehat{\varepsilon}^{\star}_{\pi}(A,B) \le \varepsilon$.

This solution preserves the project's formalism, corrects the direction of the inequality, clearly separates theory from empirical estimation, and provides a natural mathematical treatment of cyclic structure through the SCC quotient. In brief: the visualizer does not need to "define" $\varepsilon$ from scratch; it needs to treat it correctly as a threshold parameter on a critical-epsilon matrix.

**Source materials used in this note.** (i) Turing Ordering project notes, with the definition of $A \ge_{\varepsilon} B$, the distinction between the theoretical object and the current harness, and the discussion of Conditions A/B/C; (ii) the short memorandum, with emphasis on the visualizer, SCCs, and open theoretical tasks on $\varepsilon$; (iii) the supervisor's WhatsApp message of 16/04/2026, used here as the operational specification for the visualizer and corrected in the direction of the inequality where necessary in order to realign it with the formalism.
