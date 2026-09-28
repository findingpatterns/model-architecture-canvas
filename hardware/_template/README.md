# Hardware chip template

Copy this folder to `hardware/<chip-id>/` (kebab-case), then edit `chip.json` and add a `meta.json`
(`{ name, description, kind: "hardware", tags?, source?, author? }`). Run `npm run validate`.

You can also open any chip in the 3D view, switch to **Edit**, change it visually, and download the JSON.

## chip.json

| Field | Type | Notes |
|---|---|---|
| `schemaVersion` | `1` | Required. |
| `id`, `name`, `vendor`, `arch` | string | `id` is kebab-case and should match the folder. |
| `disclaimer` | string | Shown in the viewer. Geometry is illustrative. |
| `groups` | `{ name: "#rrggbb" }` | Color families (compute, memory, io…). |
| `components[]` | Component | The main scene. |
| `scenes` | `{ id: { anchor, offset?, components[] } }` | Drill-down sub-scenes, floating above `anchor` (a main component) at `offset`. |
| `flows[]` | `{ id, from, to, group, particles? }` | Animated particles between main components. |
| `steps[]` | `{ text, focus?, dot?, explode?, drill? }` | Guided animation. `drill`: scene id to open, or `null` to close. |

### Component

| Field | Type | Notes |
|---|---|---|
| `id` | string | Unique across the whole chip (including scenes). |
| `parent` | id | Optional; must be in the same scene. Used for the tree and the 2D diagram. |
| `name`, `group` | string | `group` must be declared in `groups`. |
| `color` | `#rrggbb` | Optional override of the group color. |
| `geom` | `{ type: "box" \| "cylinder", size: [x,y,z], pos: [x,y,z] }` | Position is the center (first instance when repeated). |
| `explode` | number | Upward lift at full explode. |
| `repeat` | `{ count: [nx,ny,nz], step: [dx,dy,dz], spread? }` | Grid of identical instances. `spread` = extra lift per y-layer when exploded. |
| `desc` | markdown | `**bold**`, `*italic*`, `` `code` ``. |
| `specs[]` | `{ label, value, source?, estimate? }` | Each spec needs an http(s) `source` **or** `estimate: true`. |
| `drill` | scene id | Adds an "Open" button that shows that scene. |
