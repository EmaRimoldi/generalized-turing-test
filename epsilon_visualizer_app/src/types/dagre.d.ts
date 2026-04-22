declare module "dagre" {
  export namespace graphlib {
    class Graph {
      constructor(options?: Record<string, unknown>);
      setGraph(label: Record<string, unknown>): this;
      setDefaultEdgeLabel(callback: () => Record<string, unknown>): this;
      setNode(id: string, label: Record<string, unknown>): this;
      setEdge(source: string, target: string): this;
      node(id: string): unknown;
      graph(): { width?: number; height?: number };
    }
  }

  export function layout(graph: graphlib.Graph): void;
}
