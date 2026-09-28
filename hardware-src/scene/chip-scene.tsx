// The whole 3D scene: lights, controls, main components, the open drill-down
// scene (with a leader line to its anchor), flows and guided-step visuals.
import { OrbitControls, Line } from "@react-three/drei";
import type { ThreeEvent } from "@react-three/fiber";
import { useChipStore } from "../chip-store.tsx";
import { groupPosition, sceneOrigin } from "../chip-geometry.ts";
import { ComponentMesh } from "./component-mesh.tsx";
import { FlowParticles } from "./flow-particles.tsx";
import { TravelDot, CameraRig } from "./step-visuals.tsx";

export function ChipScene({ onHover, reducedMotion }: { onHover: (n: string | null, e?: ThreeEvent<PointerEvent>) => void; reducedMotion: boolean }) {
  const { state, dispatch } = useChipStore();
  const { chip, explode, drill } = state;
  const step = state.step >= 0 ? chip.steps?.[state.step] ?? null : null;
  const focus = step?.focus?.length ? new Set(step.focus) : null;
  const scene = drill ? chip.scenes?.[drill] : undefined;
  const anchor = scene && chip.components.find((c) => c.id === scene.anchor);

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[6, 12, 8]} intensity={0.9} />
      <OrbitControls makeDefault enableDamping autoRotate={state.autoRotate && !reducedMotion} autoRotateSpeed={0.6} minDistance={3} maxDistance={60} />

      <group onPointerMissed={() => dispatch({ type: "set", patch: { selected: null } })}>
        {chip.components.map((c) => (
          <ComponentMesh key={c.id} chip={chip} comp={c} scene={null} dimmed={!!focus && !focus.has(c.id)} onHover={onHover} />
        ))}
        {scene &&
          drill &&
          scene.components.map((c) => (
            <ComponentMesh key={c.id} chip={chip} comp={c} scene={drill} dimmed={!!focus && !focus.has(c.id)} onHover={onHover} />
          ))}
      </group>

      {anchor && drill && (
        <Line
          points={[groupPosition(chip, anchor, null, explode), sceneOrigin(chip, drill, explode)]}
          color="#d04040"
          dashed
          dashSize={0.15}
          gapSize={0.1}
          lineWidth={1.5}
        />
      )}

      {state.flows && !reducedMotion && <FlowParticles chip={chip} explode={explode} />}
      <TravelDot chip={chip} step={step} explode={explode} instant={reducedMotion} />
      <CameraRig chip={chip} step={step} explode={explode} instant={reducedMotion} />
    </>
  );
}
