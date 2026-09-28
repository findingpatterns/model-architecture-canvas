// Only http(s) links are rendered as anchors (never javascript: or data: URLs).
export const isHttpUrl = (s?: string): s is string => !!s && /^https?:\/\//i.test(s);
