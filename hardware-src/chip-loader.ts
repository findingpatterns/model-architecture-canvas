// Fetch + validate a published chip, and list the chips in the catalog.
import { validateChip } from "../scripts/chip-schema.mjs";
import type { Chip } from "./chip-types.ts";

const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export async function loadChip(id: string): Promise<Chip> {
  if (!idPattern.test(id)) throw new Error(`"${id}" is not a valid chip id.`);
  const r = await fetch(`../hardware-data/${id}.json`, { cache: "no-cache" });
  if (!r.ok) throw new Error(`Could not load chip "${id}": HTTP ${r.status}`);
  const data = await r.json();
  const errs = validateChip(data);
  if (errs.length) throw new Error(`Chip "${id}" is invalid: ${errs[0]}`);
  return data as Chip;
}

export interface ChipListing {
  id: string;
  name: string;
}

export async function listChips(): Promise<ChipListing[]> {
  const r = await fetch("../catalog.json", { cache: "no-cache" });
  if (!r.ok) return [];
  const entries = (await r.json()) as { id: string; name: string; kind?: string }[];
  return entries.filter((e) => e.kind === "hardware").map(({ id, name }) => ({ id, name }));
}
