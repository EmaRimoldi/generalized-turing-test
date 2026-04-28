export function modelName(id: string): string {
  return id.split("/").pop() ?? id;
}

export function clusterLabel(members: string[]): string {
  if (members.length === 1) return modelName(members[0]!);
  return `${members.length} models`;
}

export function labelLines(label: string): string[] {
  const parts = label.split(/[-_\s]+/).filter(Boolean);
  if (parts.length <= 1) return [label];
  const lines: string[] = [];
  let line = "";
  for (const part of parts) {
    const next = line ? `${line}-${part}` : part;
    if (next.length > 13 && line) {
      lines.push(line);
      line = part;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  if (lines.length <= 3) return lines;
  return [lines[0]!, lines[1]!, `${lines.slice(2).join("-").slice(0, 14)}...`];
}

export function labelFontSize(label: string, memberCount: number): number {
  if (memberCount > 1) return 12;
  if (label.length <= 9) return 14;
  if (label.length <= 15) return 12;
  if (label.length <= 22) return 10;
  return 8;
}

export function nodeRadius(label: string, memberCount: number): number {
  if (memberCount > 1) return Math.min(76, 54 + memberCount * 3);
  if (label.length <= 10) return 48;
  if (label.length <= 18) return 56;
  return 64;
}
