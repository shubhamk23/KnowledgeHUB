import type { TOCItem } from "@/types";

export function extractTOC(html: string): TOCItem[] {
  // Match h1–h4 with an id attribute
  const headingRegex = /<h([1-4])[^>]*id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/gi;
  const flat: TOCItem[] = [];

  let match;
  while ((match = headingRegex.exec(html)) !== null) {
    const level = parseInt(match[1], 10);
    const id = match[2];
    // Strip inner HTML tags to get plain text
    const text = match[3].replace(/<[^>]+>/g, "").trim();
    if (text) {
      flat.push({ id, text, level, children: [] });
    }
  }

  return buildTree(flat);
}

function buildTree(flat: TOCItem[]): TOCItem[] {
  const root: TOCItem[] = [];
  const stack: TOCItem[] = [];

  for (const item of flat) {
    const node: TOCItem = { ...item, children: [] };

    while (stack.length > 0 && stack[stack.length - 1].level >= node.level) {
      stack.pop();
    }

    if (stack.length === 0) {
      root.push(node);
    } else {
      stack[stack.length - 1].children.push(node);
    }

    stack.push(node);
  }

  return root;
}
