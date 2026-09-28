// Dark/light theme shared with the gallery viewer and the canvas editor
// (same localStorage key, same html[data-theme] attribute).
import { useCallback, useEffect, useState } from "react";

const THEME_KEY = "modelcanvas-theme";
type Theme = "dark" | "light";

function readTheme(): Theme {
  try {
    return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  const toggle = useCallback(() => {
    setTheme((t) => {
      const next = t === "light" ? "dark" : "light";
      try { localStorage.setItem(THEME_KEY, next); } catch { /* private mode: theme just won't persist */ }
      return next;
    });
  }, []);
  return [theme, toggle];
}
