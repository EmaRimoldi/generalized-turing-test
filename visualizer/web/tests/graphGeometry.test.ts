import assert from "node:assert/strict";
import test from "node:test";
import { edgeGeometry, edgePathDag, resolveCircleCollisions } from "../src/ui/graphGeometry";

for (const reversed of [false, true]) {
  test(`separates overlapping circles, reversed=${reversed}`, () => {
    const nodes = [
      { id: "a", x: 250, y: 200, r: 40, label: "A" },
      { id: "b", x: 270, y: 200, r: 40, label: "B" },
    ];
    if (reversed) nodes.reverse();
    const before = structuredClone(nodes);
    const { nodes: solved } = resolveCircleCollisions(nodes);
    assert.ok(Math.abs(solved[0].x - solved[1].x) >= 114 - 0.001);
    assert.deepEqual(nodes, before);
    assert.equal(solved[0].y, nodes[0].y);
    assert.equal(solved[0].label, nodes[0].label);
  });
}

test("coincident circles remain finite and separate", () => {
  const { nodes } = resolveCircleCollisions([
    { id: "a", x: 0, y: 0, r: 30 }, { id: "b", x: 0, y: 0, r: 30 },
  ]);
  assert.ok(nodes.every((n) => Number.isFinite(n.x) && Number.isFinite(n.y)));
  assert.ok(Math.abs(nodes[0].x - nodes[1].x) >= 94);
});

test("export padding changes the canvas without moving nodes", () => {
  const nodes = [{ id: "a", x: 1300, y: 200, r: 30 }];
  const screen = resolveCircleCollisions(nodes);
  const image = resolveCircleCollisions(nodes, 120);
  assert.deepEqual(image.nodes, screen.nodes);
  assert.equal(image.width, screen.width + 120);
});

test("web labels and exported edges follow the same quadratic curve", () => {
  const nodes = [{ id: "a", x: 0, y: 0, r: 10 }, { id: "b", x: 100, y: 0, r: 10 }];
  const edge = { from: "a", to: "b", kind: "supported" as const };
  const geometry = edgeGeometry(edge, 0, nodes, [edge]);
  assert.deepEqual(geometry, {
    path: "M 13.00 0.00 Q 48.00 22.00 83.00 0.00", labelX: 48, labelY: 11,
  });
  assert.equal(edgePathDag(edge, 0, nodes, [edge]), geometry.path);
  assert.deepEqual(edgeGeometry(edge, 0, [], [edge]), { path: "", labelX: 0, labelY: 0 });
});
