import assert from "node:assert/strict";
import test from "node:test";
import { buildCondensation } from "../src/ui/graphCompression";
import { actorFoolingAsSource, actorMeanMarginBySource, actorOutDegreeBySource,
  classifyAtEpsilon, defenderImitationStrictness, formatPct, pairFoolingRate } from "../src/ui/logic";
import { condensationHeights } from "../src/ui/transitivity";
import type { PairRow } from "../src/ui/types";

function pair(source: string, target: string, overrides: Partial<PairRow> = {}): PairRow {
  return { source, target, condition: "A", mode: "strict", nValid: 10, nExcluded: 0,
    estimate: .5, lowerBound: null, upperBound: null, thresholdAtPoint: 0,
    thresholdAtUpper: null, statusAtEpsilon: "supported",
    details: { pSameRealB: 1, pDiffImpostor: 0 }, ...overrides };
}

test("condensation cannot reintroduce excluded models", () => {
  const graph = buildCondensation({ models: ["a", "b"],
    pairs: [pair("a", "b"), pair("b", "c"), pair("c", "a")],
    condition: "A", epsilon: .05, minValid: 3 });
  assert.deepEqual(Object.keys(graph.memberToCluster).sort(), ["a", "b"]);
  assert.equal(graph.clusters.length, 2);
});

test("rankings only use selected models and measured branches", () => {
  const measured = pair("a", "b", { details: { pSameRealB: 1, pDiffImpostor: 1 } });
  const imputed = pair("a", "b", { details: { pSameRealB: 0, pDiffImpostor: 0,
    imitationDataProvenance: "imputed_uniform", genuineDataProvenance: "imputed_uniform" } });
  const pairs = [imputed, measured, pair("a", "c"), pair("c", "b")];
  assert.equal(actorFoolingAsSource(pairs, "A", ["a", "b"]).get("a"), 0);
  assert.equal(defenderImitationStrictness(pairs, "A", ["a", "b"]).get("b"), 1);
  assert.equal(actorMeanMarginBySource(pairs, "A", ["a", "b"], 3).get("a"), -.5);
  assert.deepEqual([...actorOutDegreeBySource(pairs, "A", ["a", "b"], 0, 3)], [["a", 0], ["b", 0]]);
});

test("nonfinite and out of range probabilities fail closed", () => {
  for (const value of [NaN, Infinity, -.1, 1.1]) {
    assert.equal(classifyAtEpsilon(pair("a", "b", { estimate: value }), .05, 3), "missing");
    assert.equal(pairFoolingRate(pair("a", "b", { details: { pDiffImpostor: value } }), 3), null);
    assert.equal(formatPct(value), "—");
  }
});

test("invalid condensation cycle reports a clear error", () => {
  assert.throws(() => condensationHeights(["a", "b"], [["a", "b"], ["b", "a"]]), /acyclic/);
});


test("rankings respect the minimum sample count", () => {
  const pairs = [pair("a", "b", { nValid: 2 })];
  assert.equal(actorFoolingAsSource(pairs, "A", ["a", "b"], 3).get("a"), 0);
  assert.equal(actorFoolingAsSource(pairs, "A", ["a", "b"], 1).get("a"), 1);
  assert.equal(defenderImitationStrictness(pairs, "A", ["a", "b"], 3).get("b"), 0);
});
