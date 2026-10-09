// Guided-step visuals: a red dot that glides to the current step's `dot`
// component, and a camera rig that eases the orbit target toward the focus.
import { useRef } from "react";
import * as THREE from "three";
import { useFrame, useThree } from "@react-three/fiber";
import type { Chip, Step } from "../chip-types.ts";
import { worldCenter } from "../chip-geometry.ts";

function focusCenter(chip: Chip, step: Step | null, explode: number): THREE.Vector3 | null {
  if (!step) return null;
  const ids = step.focus?.length ? step.focus : step.dot ? [step.dot] : [];
  const pts = ids.map((id) => worldCenter(chip, id, explode)).filter(Boolean) as [number, number, number][];
  if (!pts.length) return null;
  const v = new THREE.Vector3();
  for (const p of pts) v.add(new THREE.Vector3(...p));
  return v.divideScalar(pts.length);
}

export function TravelDot({ chip, step, explode, instant }: { chip: Chip; step: Step | null; explode: number; instant: boolean }) {
  const ref = useRef<THREE.Mesh>(null!);
  const target = step?.dot ? worldCenter(chip, step.dot, explode) : null;
  useFrame((_, dt) => {
    if (!ref.current || !target) return;
    const goal = new THREE.Vector3(target[0], target[1] + 0.35, target[2]);
    if (instant) ref.current.position.copy(goal);
    else ref.current.position.lerp(goal, Math.min(1, dt * 4));
  });
  if (!target) return null;
  return (
    <mesh ref={ref} raycast={() => null}>
      <sphereGeometry args={[0.16, 16, 16]} />
      <meshBasicMaterial color="#ff3b3b" />
    </mesh>
  );
}

// Eases OrbitControls' target toward the active step's focus (no-op when no step).
export function CameraRig({ chip, step, explode, instant, scale }: { chip: Chip; step: Step | null; explode: number; instant: boolean; scale: number }) {
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  const goal = focusCenter(chip, step, explode)?.multiplyScalar(scale) ?? null; // scene is drawn inside a scaled group
  useFrame((_, dt) => {
    if (!controls || !goal) return;
    if (instant) controls.target.copy(goal);
    else controls.target.lerp(goal, Math.min(1, dt * 2.5));
    controls.update();
  });
  return null;
}
