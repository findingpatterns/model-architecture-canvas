// Types for the zero-dependency validator so the TS 3D app can import it.
export declare const SCHEMA_VERSION: 1;
export declare const GEOM_TYPES: readonly string[];
export declare const SUMMARY_KEYS: Record<string, { label: string; unit: string | null; higherIsBetter: boolean | null }>;
export declare function validateChip(chip: unknown): string[];
export declare function allComponents(chip: unknown): { c: unknown; scene: string | null }[];
