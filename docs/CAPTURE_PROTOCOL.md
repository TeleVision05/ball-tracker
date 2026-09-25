# Capture Protocol

FullTrack-class behind-stumps video capture for cricket bowling analytics.

## Bringing your own videos

Place real delivery clips in `data/raw/real/` (`.mp4` or `.mov`). Then:

1. **Calibrate** stump landmarks once per session (or per clip if the camera moved):

   ```bash
   python scripts/calibrate_stumps.py \
     --video data/raw/real/your_clip.mp4 \
     --out data/raw/real/session.calib.json
   ```

2. **Analyze** each delivery:

   ```bash
   python scripts/analyze_delivery.py \
     --video data/raw/real/your_clip.mp4 \
     --calib data/raw/real/session.calib.json \
     --out /tmp/analysis.json
   ```

   Or use the Streamlit session app: `streamlit run scripts/run_session_app.py`

**Filming tips for real footage:**

- Prefer **120 fps** (or higher) from behind the bowler-end stumps (~4 m back).
- Keep **both stump sets** (bowler and batter ends) fully visible and upright.
- Short shutter / high shutter speed to reduce motion blur on the ball.
- Re-calibrate if the camera is bumped or reframed between sessions.

## Camera placement

| Parameter | Requirement |
|-----------|-------------|
| Position | ~4 m behind the bowler-end stumps, centred on the pitch |
| Height | ≥ 60 in (1.52 m) above ground; target ~1.66 m for standard tripod |
| Orientation | Level horizon; pitch axis centred in frame |
| Field of view | Both stump sets (bowler and batter ends) visible |
| Occlusion | No players, umpire, or equipment blocking the ball flight path |

## Recording settings

| Parameter | Requirement |
|-----------|-------------|
| Frame rate | 120–240 fps |
| Shutter speed | Short exposure (≤ 1/2000 s) to minimise motion blur |
| Resolution | ≥ 1080p preferred; 720p minimum at high FPS |
| Format | H.264/H.265 MP4 or lossless intermediate for processing |
| Audio | Optional; not used for ball tracking |

## Calibration

Before each session:

1. Confirm both stump sets are fully visible and upright.
2. Mark the pitch centre line and popping creases if not already visible.
3. Record a calibration clip with a known reference (measuring tape along the pitch).
4. Reject sessions where:
   - Either stump set is partially occluded or out of frame
   - Camera height is below 60 in
   - FPS drops below 120 during recording
   - Significant camera shake or auto-exposure flicker is present

## Ground-truth validation

Each delivery should be validated against independent measurements:

### Radar speed

- Use a calibrated radar gun or ball-tracking radar at release point.
- Record `radar_speed_kmh` per delivery.
- Acceptable agreement with video-derived speed: ±3 km/h at 120+ km/h.

### Bounce location

- Mark bounce point on the pitch with chalk or tape before the session.
- Measure from the batter-end popping crease (Y = pitch length):
  - **Bounce line** (`bounce_line_m`): lateral offset from pitch centre (+X = off-side).
  - **Bounce length** (`bounce_length_m`): distance from batter-end crease toward bowler (+Y from crease).

### Rejection criteria

Reject a delivery annotation when:

- Radar and video speed disagree by > 5 km/h
- Measured bounce is > 0.30 m from video-estimated bounce
- Ball is occluded for > 30 % of flight frames
- Stump calibration reprojection error exceeds 5 px RMS

## Session checklist

- [ ] Camera mounted ≥ 60 in, ~4 m behind bowler stumps
- [ ] Both stump sets visible, no occlusion
- [ ] 120–240 fps, short shutter confirmed
- [ ] Radar calibrated and logging
- [ ] Bounce chalk/tape markers prepared
- [ ] Calibration clip recorded and passed quality check
- [ ] `delivery_id` scheme agreed (session + sequence number)

## File naming

```
data/raw/{session_date}/{delivery_id}.mp4
data/ground_truth/{session_date}/deliveries.csv
```

See [GROUND_TRUTH.md](GROUND_TRUTH.md) for the annotation schema.
