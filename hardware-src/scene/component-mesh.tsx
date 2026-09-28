// One chip component in 3D: a group at its world position holding either a
// single mesh or an InstancedMesh (for `repeat`). Handles hover/click picking,
// focus dimming during guided steps, and the edit-mode transform gizmo.
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TransformControls } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import type { Chip, Component, SceneKey } from "../chip-types.ts";
import { groupPosition, instanceOffsets } from "../chip-geometry.ts";
import { moveComponent, resizeComponent } from "../chip-edit-actions.ts";
import { useChipStore } from "../chip-store.tsx";

interface Props {
  chip: Chip;
  comp: Component;
  scene: SceneKey;
  dimmed: boolean;
  onHover: (name: string | null, e?: ThreeEvent<PointerEvent>) => void;
}

const tmp = new THREE.Object3D();

export function ComponentMesh({ chip, comp, scene, dimmed, onHover }: Props) {
  const { state, dispatch } = useChipStore();
  const group = useRef<THREE.Group>(null!);
  const inst = useRef<THREE.InstancedMesh>(null!);
  const [gizmoMode, setGizmoMode] = useState<"translate" | "scale">("translate");
  const selected = state.selected === comp.id;
  const offsets = useMemo(() => instanceOffsets(comp), [comp]);
  const [x, y, z] = groupPosition(chip, comp, scene, state.explode);
  const spread = scene === null ? (comp.repeat?.spread ?? 0) * state.explode : 0;
  const color = comp.color ?? chip.groups[comp.group] ?? "#888888";
  const isInstanced = offsets.length > 1;

  useLayoutEffect(() => {
    if (!isInstanced || !inst.current) return;
    offsets.forEach(({ offset, layer }, i) => {
      tmp.position.set(offset[0], offset[1] + layer * spread, offset[2]);
      tmp.updateMatrix();
      inst.current.setMatrixAt(i, tmp.matrix);
    });
    inst.current.instanceMatrix.needsUpdate = true;
    inst.current.computeBoundingSphere();
  }, [offsets, spread, isInstanced]);

  const [sx, sy, sz] = comp.geom.size;
  const geometry =
    comp.geom.type === "cylinder" ? <cylinderGeometry args={[sx / 2, sx / 2, sy, 32]} /> : <boxGeometry args={[sx, sy, sz]} />;
  const material = (
    <meshLambertMaterial
      color={color}
      emissive={selected ? "#553300" : "#000000"}
      transparent
      opacity={dimmed ? 0.18 : 1}
      depthWrite={!dimmed}
    />
  );
  const handlers = {
    onClick: (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      dispatch({ type: "set", patch: { selected: comp.id } });
    },
    onPointerOver: (e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); onHover(comp.name, e); },
    onPointerMove: (e: ThreeEvent<PointerEvent>) => { e.stopPropagation(); onHover(comp.name, e); },
    onPointerOut: () => onHover(null),
    // Resize is disabled for repeated parts: scaling the group would stretch the grid spacing too.
    onDoubleClick: () => state.edit && !isInstanced && setGizmoMode((m) => (m === "translate" ? "scale" : "translate")),
  };

  // Gizmo drag ends → convert the group transform back into chip.json geometry.
  const commitTransform = () => {
    const g = group.current;
    const eps = 1e-4;
    if (gizmoMode === "scale") {
      const { x: kx, y: ky, z: kz } = g.scale;
      g.scale.set(1, 1, 1);
      if (Math.abs(kx - 1) + Math.abs(ky - 1) + Math.abs(kz - 1) < eps) return; // click without drag
      dispatch({ type: "commit", chip: resizeComponent(chip, comp.id, [sx * kx, sy * ky, sz * kz]), external: true });
    } else {
      const d = [g.position.x - x, g.position.y - y, g.position.z - z];
      if (Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]) < eps) return;
      const p = comp.geom.pos;
      dispatch({ type: "commit", chip: moveComponent(chip, comp.id, [p[0] + d[0], p[1] + d[1], p[2] + d[2]]), external: true });
    }
  };

  const body = (
    <group ref={group} position={[x, y, z]}>
      {isInstanced ? (
        <instancedMesh ref={inst} args={[undefined, undefined, offsets.length]} {...handlers}>
          {geometry}
          {material}
        </instancedMesh>
      ) : (
        <mesh {...handlers}>
          {geometry}
          {material}
        </mesh>
      )}
    </group>
  );

  if (!(state.edit && selected)) return body;
  return (
    <>
      {body}
      <TransformControls object={group} mode={gizmoMode} size={0.8} onMouseUp={commitTransform} />
    </>
  );
}
