// Two synced 3D views on ONE WebGL canvas (drei <View>). Each view renders its
// chip from its own store. Both OrbitControls attach to the Canvas eventSource
// (the shared container) with identical starting cameras, so a drag/zoom
// anywhere moves both identically — sync by construction, no feedback loop.
// Step/drill/explode are driven by compare-app, not by each chip's own steps.
import { useRef, useState } from "react";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { View, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { ChipStoreBridge, type ChipStore } from "../chip-store.tsx";
import { ChipScene } from "../scene/chip-scene.tsx";

const CAMERA_POS: [number, number, number] = [14, 13, 16];

interface Props {
  stores: [ChipStore, ChipStore];
  highlightRole: string | null;
  reducedMotion: boolean;
  viewKey: number; // bump to reset both cameras
}

export function CompareViews({ stores, highlightRole, reducedMotion, viewKey }: Props) {
  const container = useRef<HTMLDivElement>(null!);
  const tracks = [useRef<HTMLDivElement>(null!), useRef<HTMLDivElement>(null!)];
  const [tip, setTip] = useState<{ name: string; x: number; y: number } | null>(null);
  const onHover = (name: string | null, e?: ThreeEvent<PointerEvent>) => {
    if (!name || !e) return setTip(null);
    const r = container.current.getBoundingClientRect();
    setTip({ name, x: e.nativeEvent.clientX - r.left, y: e.nativeEvent.clientY - r.top });
  };

  return (
    <div className="compare-stage" ref={container} onPointerLeave={() => setTip(null)}>
      {stores.map((store, i) => (
        <div className="compare-track" ref={tracks[i]} key={i}>
          <span className="compare-label mono">{store.state.chip.name}</span>
        </div>
      ))}
      <Canvas key={viewKey} className="compare-canvas" eventSource={container} dpr={[1, 2]}>
        {stores.map((store, i) => (
          <View track={tracks[i]} key={i}>
            <ChipStoreBridge store={store}>
              <PerspectiveCamera makeDefault position={CAMERA_POS} fov={40} />
              <OrbitControls makeDefault minDistance={3} maxDistance={60} />
              <ChipScene compare onHover={onHover} reducedMotion={reducedMotion} highlightRole={highlightRole} />
            </ChipStoreBridge>
          </View>
        ))}
      </Canvas>
      {tip && <div className="tip" style={{ left: tip.x + 14, top: tip.y + 14 }}>{tip.name}</div>}
    </div>
  );
}
