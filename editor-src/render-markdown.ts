// Minimal, XSS-safe markdown → HTML for node display (matches the viewer, which
// renders markdown rather than showing raw `**...**`). Covers what model .canvas
// text uses: headings, bold, italic, inline code, line breaks.
//
// SAFETY: the source is HTML-escaped FIRST, so any literal markup in the text
// becomes inert; only the tags this function emits are real. No dep, no innerHTML
// of untrusted content.
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function inline(escaped: string): string {
  return escaped
    .replace(/`([^`]+)`/g, '<code class="md-code">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
}

export function renderMarkdown(src: string): string {
  return src
    .split("\n")
    .map((line) => {
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) return `<span class="md-h md-h${h[1].length}">${inline(escapeHtml(h[2]))}</span>`;
      return inline(escapeHtml(line));
    })
    .join("<br/>");
}
