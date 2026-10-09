"""Synthetic cricket delivery generator for validation data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .pitch import BALL_RADIUS_M, PITCH_LENGTH_M, stump_landmarks_3d

GRAVITY_M_S2 = 9.81


@dataclass
class CameraModel:
    """Pinhole camera behind the bowler-end stumps."""

    position_m: np.ndarray
    look_at_m: np.ndarray
    fov_deg: float
    resolution: tuple[int, int]

    def intrinsic_matrix(self) -> np.ndarray:
        w, h = self.resolution
        f = (w / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))
        return np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def extrinsic_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Return world-to-camera rotation R and translation t.

        Image axes are right-handed: +X right, +Y down, +Z forward.
        ``det(R)`` is +1, and projected ``v`` increases downward.
        """
        forward = self.look_at_m - self.position_m
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= right_norm
        down = np.cross(forward, right)
        R = np.stack([right, down, forward], axis=0)
        t = -R @ self.position_m
        return R, t

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixel coordinates."""
        K = self.intrinsic_matrix()
        R, t = self.extrinsic_matrix()
        pts = np.asarray(points_3d, dtype=np.float64)
        cam = (R @ pts.T).T + t
        depth = cam[:, 2]
        depth = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
        uv_h = (K @ cam.T).T
        uv = uv_h[:, :2] / uv_h[:, 2:3]
        uv[depth <= 0.0] = np.nan
        return uv


@dataclass
class DeliveryParams:
    """Randomised parameters for one synthetic delivery."""

    release_pos_m: np.ndarray
    release_vel_m_s: np.ndarray
    beta_s: float
    restitution: float
    friction: float
    turn_m: float
    radar_speed_kmh: float


def default_camera(fps: int = 120) -> CameraModel:
    """Behind-bowler camera pose (~4 m back, ~1.66 m height)."""
    return CameraModel(
        position_m=np.array([0.0, -4.0, 1.66], dtype=np.float64),
        look_at_m=np.array([0.0, PITCH_LENGTH_M * 0.5, 0.8], dtype=np.float64),
        fov_deg=55.0,
        resolution=(1280, 720),
    )


def _random_delivery(rng: np.random.Generator) -> DeliveryParams:
    """Sample plausible fast-medium delivery parameters."""
    speed_kmh = float(rng.uniform(115.0, 145.0))
    speed_m_s = speed_kmh / 3.6
    release_y = float(rng.uniform(0.8, 1.5))
    release_z = float(rng.uniform(1.8, 2.2))
    release_x = float(rng.uniform(-0.15, 0.15))
    line_offset = float(rng.uniform(-0.25, 0.25))
    length_target = float(rng.uniform(4.0, 9.0))

    release_pos = np.array([release_x, release_y, release_z], dtype=np.float64)
    direction = np.array([line_offset, PITCH_LENGTH_M - length_target - release_y, -0.6], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    release_vel = direction * speed_m_s

    return DeliveryParams(
        release_pos_m=release_pos,
        release_vel_m_s=release_vel,
        beta_s=float(rng.uniform(-2.0, 2.0)),
        restitution=float(rng.uniform(0.65, 0.85)),
        friction=float(rng.uniform(0.05, 0.25)),
        turn_m=float(rng.uniform(-0.3, 0.3)),
        radar_speed_kmh=speed_kmh,
    )


def simulate_trajectory(
    params: DeliveryParams,
    dt: float = 1.0 / 240.0,
    max_time_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Simulate ballistic flight with swing, bounce, and post-bounce roll.

    Returns
    -------
    times, positions, bounce_index, bounce_position
    """
    pos = params.release_pos_m.copy()
    vel = params.release_vel_m_s.copy()
    times: list[float] = [0.0]
    positions: list[np.ndarray] = [pos.copy()]
    bounce_index = -1
    bounce_pos = np.zeros(3, dtype=np.float64)
    t = 0.0
    bounced = False

    while t < max_time_s:
        t += dt
        if not bounced:
            acc = np.array([params.beta_s, 0.0, -GRAVITY_M_S2], dtype=np.float64)
            vel = vel + acc * dt
            pos = pos + vel * dt
            if pos[2] <= BALL_RADIUS_M:
                pos[2] = BALL_RADIUS_M
                bounce_pos = pos.copy()
                bounce_index = len(positions)
                # Reflect vertical velocity with restitution; apply friction on horizontal.
                vel[2] = -params.restitution * vel[2]
                vel[0] *= 1.0 - params.friction
                vel[1] *= 1.0 - params.friction
                # Lateral turn after bounce.
                vel[0] += params.turn_m * 3.0
                bounced = True
        else:
            acc = np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)
            vel = vel + acc * dt
            pos = pos + vel * dt
            if pos[2] <= BALL_RADIUS_M:
                pos[2] = BALL_RADIUS_M
                vel[2] = -params.restitution * vel[2]
                vel[0] *= 1.0 - params.friction
                vel[1] *= 1.0 - params.friction

        positions.append(pos.copy())
        times.append(t)

        if bounced and pos[1] > PITCH_LENGTH_M + 1.0:
            break
        if not bounced and pos[1] > PITCH_LENGTH_M + 0.5 and pos[2] < 0.0:
            break

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        bounce_index,
        bounce_pos,
    )


def _bounce_labels(bounce_pos: np.ndarray) -> tuple[float, float]:
    """Compute bounce line and length from world bounce position."""
    bounce_line_m = float(bounce_pos[0])
    bounce_length_m = float(PITCH_LENGTH_M - bounce_pos[1])
    return bounce_line_m, bounce_length_m


def _draw_stumps(frame: np.ndarray, camera: CameraModel) -> None:
    """Draw stump sticks projected into the image."""
    landmarks = stump_landmarks_3d()
    h, w = frame.shape[:2]
    for key in sorted(landmarks):
        if not key.endswith("_top"):
            continue
        base_key = key.replace("_top", "_base")
        top = landmarks[key]
        base = landmarks[base_key]
        uv = camera.project(np.stack([base, top]))
        if np.any(np.isnan(uv)):
            continue
        p1 = (int(np.clip(uv[0, 0], 0, w - 1)), int(np.clip(uv[0, 1], 0, h - 1)))
        p2 = (int(np.clip(uv[1, 0], 0, w - 1)), int(np.clip(uv[1, 1], 0, h - 1)))
        cv2.line(frame, p1, p2, (200, 200, 200), 2)


def render_delivery_video(
    camera: CameraModel,
    positions: np.ndarray,
    times: np.ndarray,
    fps: int,
    out_path: Path,
    draw_stumps: bool = True,
) -> tuple[list[list[float]], int]:
    """Write MP4 with ball as red circle; return 2D track and frame count."""
    w, h = camera.resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    dt = 1.0 / fps
    track_2d: list[list[float]] = []
    frame_idx = 0
    max_frames = int(times[-1] * fps) + 1

    while frame_idx < max_frames:
        t = frame_idx * dt
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(positions) - 1))

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if draw_stumps:
            _draw_stumps(frame, camera)

        uv = camera.project(positions[idx : idx + 1])[0]
        track_2d.append([float(uv[0]), float(uv[1])])

        if not np.any(np.isnan(uv)):
            centre = (int(np.clip(uv[0], 0, w - 1)), int(np.clip(uv[1], 0, h - 1)))
            radius = max(4, int(BALL_RADIUS_M * camera.intrinsic_matrix()[0, 0] / max(positions[idx, 1], 1.0)))
            cv2.circle(frame, centre, radius, (0, 0, 255), -1)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    return track_2d, frame_idx


def generate_delivery(
    delivery_id: str,
    out_dir: Path,
    fps: int,
    rng: np.random.Generator,
    draw_stumps: bool = True,
) -> dict:
    """Generate one synthetic delivery (video + JSON annotation)."""
    params = _random_delivery(rng)
    times, positions, bounce_idx, bounce_pos = simulate_trajectory(params)
    camera = default_camera(fps=fps)

    video_path = out_dir / f"{delivery_id}.mp4"
    track_2d, frame_count = render_delivery_video(
        camera, positions, times, fps, video_path, draw_stumps=draw_stumps
    )

    bounce_line_m, bounce_length_m = _bounce_labels(bounce_pos)
    release_speed_kmh = float(np.linalg.norm(params.release_vel_m_s) * 3.6)
    bounce_time_s = float(times[bounce_idx]) if bounce_idx >= 0 else None
    bounce_frame = int(round(bounce_time_s * fps)) if bounce_time_s is not None else -1

    annotation = {
        "delivery_id": delivery_id,
        "video_path": str(Path("data") / "synthetic" / out_dir.name / f"{delivery_id}.mp4"),
        "radar_speed_kmh": round(params.radar_speed_kmh, 2),
        "bounce_line_m": round(bounce_line_m, 4),
        "bounce_length_m": round(bounce_length_m, 4),
        "release_speed_kmh": round(release_speed_kmh, 2),
        "release_position_m": params.release_pos_m.tolist(),
        "bounce_position_m": bounce_pos.tolist(),
        "beta_s": round(params.beta_s, 4),
        "turn_m": round(params.turn_m, 4),
        "waypoints_3d_m": positions.tolist(),
        "track_2d_px": track_2d,
        "fps": fps,
        "frame_count": frame_count,
        "bounce_time_s": bounce_time_s,
        "bounce_frame_index": bounce_frame,
        "bounce_sim_index": int(bounce_idx),
        "camera": {
            "position_m": camera.position_m.tolist(),
            "look_at_m": camera.look_at_m.tolist(),
            "fov_deg": camera.fov_deg,
            "resolution": list(camera.resolution),
        },
        "notes": "synthetic validation delivery",
    }

    json_path = out_dir / f"{delivery_id}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2)

    return annotation


def generate_batch(out_dir: Path, n: int, fps: int, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic deliveries and write summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for i in range(n):
        delivery_id = f"delivery_{i + 1:04d}"
        ann = generate_delivery(delivery_id, out_dir, fps, rng)
        rows.append(
            {
                "delivery_id": ann["delivery_id"],
                "video_path": ann["video_path"],
                "radar_speed_kmh": ann["radar_speed_kmh"],
                "bounce_line_m": ann["bounce_line_m"],
                "bounce_length_m": ann["bounce_length_m"],
                "notes": ann["notes"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "deliveries.csv", index=False)
    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic cricket delivery validation data")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--n", type=int, default=5, help="Number of deliveries")
    parser.add_argument("--fps", type=int, default=120, help="Video frame rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = generate_batch(args.out, args.n, args.fps, seed=args.seed)
    print(f"Generated {len(df)} deliveries in {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
