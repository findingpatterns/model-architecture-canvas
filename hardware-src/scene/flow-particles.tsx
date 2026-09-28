// Animated data-flow particles for chip.flows: small spheres travelling (with a
// little arc) from every instance of `from` to the center of `to`.
import { useMemo, useRef, useLayoutEffect } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import type { Chip } from "../chip-types.ts";
import { groupPosition, instanceOffsets } from "../chip-geometry.ts";

interface Particle { from: THREE.Vector3; to: THREE.Vector3; phase: number; color: string }

const tmp = new THREE.Object3D();

export function FlowParticles({ chip, explode }: { chip: Chip; explode: number }) {
  const ref = useRef<THREE.InstancedMesh>(null!);
  const particles = useMemo(() => {
    const out: Particle[] = [];
    for (const f of chip.flows ?? []) {
      const src = chip.components.find((c) => c.id === f.from);
      const dst = chip.components.find((c) => c.id === f.to);
      if (!src || !dst) continue;
      const base = groupPosition(chip, src, null, explode);
      const to = new THREE.Vector3(...groupPosition(chip, dst, null, explode));
      const seen = new Set<string>();
      for (const { offset } of instanceOffsets(src)) {
        // One source per stack column (ignore stacked layers) keeps the effect readable.
        const key = `${offset[0]},${offset[2]}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const from = new THREE.Vector3(base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]);
        const n = f.particles ?? 2;
        for (let i = 0; i < n; i++)
          out.push({ from, to: to.clone().lerp(from, 0.35), phase: i / n, color: chip.groups[f.group] ?? "#ff4d4d" });
      }
    }
    return out;
  }, [chip, explode]);

  useLayoutEffect(() => {
    if (!ref.current) return;
    const c = new THREE.Color();
    particles.forEach((p, i) => ref.current.setColorAt(i, c.set(p.color).offsetHSL(0, 0.1, 0.15)));
    if (ref.current.instanceColor) ref.current.instanceColor.needsUpdate = true;
  }, [particles]);

  useFrame(({ clock }) => {
    if (!ref.current) return;
    const t = clock.getElapsedTime() / 1.6;
    particles.forEach((p, i) => {
      const f = (t + p.phase) % 1;
      tmp.position.lerpVectors(p.from, p.to, f);
      tmp.position.y += Math.sin(f * Math.PI) * 0.6;
      tmp.updateMatrix();
      ref.current.setMatrixAt(i, tmp.matrix);
    });
    ref.current.instanceMatrix.needsUpdate = true;
  });

  if (!particles.length) return null;
  return (
    <instancedMesh key={particles.length} ref={ref} args={[undefined, undefined, particles.length]} raycast={() => null}>
      <sphereGeometry args={[0.07, 8, 8]} />
      <meshBasicMaterial />
    </instancedMesh>
  );
}
