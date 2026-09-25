# Ground Truth Schema

Delivery-level annotations for cricket bowling analytics validation.

## Coordinate conventions

- **World frame**: origin at bowler-end middle stump base; +Y toward batter; +X off-side (bowler's right); +Z up. Units: metres.
- **Image frame**: origin top-left; u right, v down. Units: pixels.
- **Bounce length**: distance from batter-end popping crease measured toward the bowler (+Y).
- **Bounce line**: lateral offset from pitch centre at bounce (+X = off-side).

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `delivery_id` | string | Unique identifier, e.g. `session_001_d003` |
| `video_path` | string | Relative path to the capture MP4 |
| `radar_speed_kmh` | float | Radar-measured release speed (km/h) |
| `bounce_line_m` | float | Lateral bounce offset from centre (m) |
| `bounce_length_m` | float | Bounce distance from batter crease (m) |

## Extended fields (JSON annotation)

| Field | Type | Description |
|-------|------|-------------|
| `release_speed_kmh` | float | Video/physics-derived release speed |
| `release_position_m` | [x,y,z] | Ball position at release |
| `bounce_position_m` | [x,y,z] | World bounce point |
| `beta_s` | float | True seam swing parameter (lateral accel, m/s²) |
| `turn_m` | float | Post-bounce lateral deviation (m) |
| `waypoints_3d_m` | list[[x,y,z]] | Sampled 3D ball positions |
| `track_2d_px` | list[[u,v]] | Projected image coordinates per frame |
| `fps` | int | Video frame rate |
| `frame_count` | int | Total frames |
| `notes` | string | Free-text annotation notes |

## Files

- **JSON schema**: `data/ground_truth/schema.json`
- **CSV template**: `data/ground_truth/template.csv`

## CSV usage

The CSV is the session summary; one row per delivery. Extended kinematics live in per-delivery JSON files alongside the video.

```csv
delivery_id,video_path,radar_speed_kmh,bounce_line_m,bounce_length_m,notes
session_001_d001,data/raw/session_001/d001.mp4,132.5,0.12,6.8,good track
```

## JSON usage

Full annotation example:

```json
{
  "delivery_id": "synthetic_0001",
  "video_path": "data/synthetic/demo/delivery_0001.mp4",
  "radar_speed_kmh": 128.4,
  "bounce_line_m": 0.08,
  "bounce_length_m": 7.2,
  "release_speed_kmh": 128.4,
  "beta_s": 1.2,
  "turn_m": 0.15,
  "waypoints_3d_m": [[0.0, 0.5, 2.1], "..."],
  "track_2d_px": [[640, 360], "..."],
  "fps": 120,
  "frame_count": 48,
  "notes": "synthetic validation delivery"
}
```

## Validation rules

1. `radar_speed_kmh` > 0
2. `bounce_length_m` ∈ [0, pitch_length_m]
3. `bounce_line_m` typically ∈ [-1.0, 1.0] for legal deliveries
4. `track_2d_px` length equals `frame_count` when present
5. `waypoints_3d_m` timestamps align with video frames when both present
