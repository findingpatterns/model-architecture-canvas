import { createContext, useContext } from "react";

// Lets any node/toolbar component flag unsaved edits without prop-drilling.
// App owns the actual dirty flag + the beforeunload / "← Preview" guards (red-team #9).
export const EditorContext = createContext<{ markDirty: () => void }>({ markDirty: () => {} });
export const useEditorContext = () => useContext(EditorContext);
