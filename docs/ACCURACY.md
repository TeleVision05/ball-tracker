# Accuracy: synthetic deliveries only

> **This is synthetic-only validation.** Every number on this page compares
> BallTracker against its own physics simulator, not against real deliveries.
> No real delivery has been measured against radar or a marked bounce yet
> (see [GROUND_TRUTH.md](GROUND_TRUTH.md) for how that will be done).

**What this proves.** Given ball positions of the quality a phone detector
produces, it measures how well the tracker's outlier filter, the stump
calibration, the bounce-segmented fit and the metric formulas recover the
simulator's speed, bounce spot and turn, and how that changes with frame rate
and bowling style. The fast test in `tests/test_benchmark.py` re-runs a small
version of this and fails if the fit gets worse.

**What this does not prove.** It says nothing about real footage. The image
detector is not run here: detections are simulated as true pixel positions plus
noise (`balltracker demo` runs the real detector on a rendered clip). The
simulator shares the fit's assumptions: a pinhole camera with no lens
distortion and a known field of view, a flat pitch, no air drag, and a bounce
model from the same family as the one the fit expects. Real balls, lenses,
pitches and lighting will add error that this page cannot see.

## How this page was made

- Command: `balltracker benchmark --n 500 --seed 20261113`
- Seed: `20261113`
- Date: 2026-11-13
- Cases: 500 (0 where the fit could not run)
- Calibrations that needed fresh clicks: 0 of 500 (the app asks for new clicks above 8 px); 0 still exceeded it after five rounds.

Re-running the command with the same seed reproduces these numbers.

## Errors

Absolute error of each estimate against the simulator, over all cases. The first three columns are the full path a video takes after detection.

| Metric | Unit | Median error | 90th percentile | Median signed error | Median error, filter skipped | Median error, noise-free |
|---|---|---:|---:|---:|---:|---:|
| Release speed | km/h | 7.9 | 16.4 | 7.8 | 3.4 | 2.6 |
| Bounce line (sideways) | cm | 2.0 | 5.5 | 0.2 | 1.5 | 0.1 |
| Bounce length | cm | 94.0 | 198.1 | -93.8 | 20.6 | 10.2 |
| Swing before the bounce | deg | 0.7 | 1.5 | -0.1 | 0.7 | 0.7 |
| Turn off the pitch | deg | 1.3 | 5.6 | 0.1 | 0.9 | 0.5 |

### Median error by frame rate

| Frame rate (fps) | Cases | Release speed (km/h) | Bounce line (sideways) (cm) | Bounce length (cm) | Swing before the bounce (deg) | Turn off the pitch (deg) |
|---|---:|---:|---:|---:|---:|---:|
| 60 | 177 | 9.0 | 2.1 | 118.8 | 0.8 | 1.7 |
| 120 | 177 | 8.5 | 2.2 | 91.7 | 0.7 | 1.4 |
| 240 | 146 | 5.8 | 1.7 | 78.5 | 0.7 | 1.0 |

### Median error by bowling style

| Bowling style | Cases | Release speed (km/h) | Bounce line (sideways) (cm) | Bounce length (cm) | Swing before the bounce (deg) | Turn off the pitch (deg) |
|---|---:|---:|---:|---:|---:|---:|
| fast | 210 | 6.5 | 1.8 | 58.0 | 0.6 | 1.0 |
| medium | 119 | 9.3 | 1.7 | 97.6 | 1.1 | 1.2 |
| spin | 171 | 8.3 | 2.4 | 134.2 | 0.7 | 2.0 |

## What the numbers say

- **The tracker's outlier filter is the largest error source.** Sending the same detections straight to the fit lowers the median speed error from 7.9 to 3.4 km/h and the median length error from 94.0 to 20.6 cm. The filter fits one quadratic through the whole track, bounce included, so the frames around the bounce look like outliers and are dropped, and the fit then places the bounce from the frames either side of the gap.
- **Swing is not measured yet.** The pre-bounce fit has gravity and no sideways force, so its swing estimate is 0° in every case. The swing error above is the size of the simulated swing (median 0.7°). The missing swing also leaks into the turn estimate, because turn is measured against the fitted straight approach.
- **Length is the weakest bounce coordinate.** Looking down the pitch, a few pixels near the bounce move the bounce point a long way along the pitch: median length error 94.0 cm against 2.0 cm for line.
- **Frame rate.** Going from 60 to 240 fps lowers the median error for: release speed, bounce line (sideways), bounce length, turn off the pitch.

## How each case is built

- **Delivery.** Bowling style: fast 130 to 150 km/h (40%), medium 110 to 130 km/h (25%), spin 70 to 95 km/h (35%). Length in front of the batter's stumps 1.5 to 10 m (medium 1.5 to 9 m, spin 1 to 6 m), line 35 cm leg side to 45 cm off side, swing up to 3 m/s² sideways (spin 0.8), a sideways kick at the bounce of up to 2.7 m/s for spin and up to 0.45 m/s for seam, release 0.8 to 1.4 m in front of the bowler's stumps and up to 45 cm either side of them.
- **Flight.** `synthetic.simulate_trajectory` at 1 ms steps: gravity plus the constant swing force, a bounce with random restitution (0.65 to 0.85) and friction (0.05 to 0.25), then flight to 1 m past the batter's stumps.
- **Camera.** Behind the bowler: 4 m back, 1.66 m high, 55° field of view, 1280x720, tilted so both stump sets are fully in frame as [CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md) asks.
- **Capture.** Frame rate chosen from 60, 120, 240 fps with a random phase. Detection jitter 0.5 to 2 px (Gaussian), 0 to 15% of frames missed, half the cases also lose a burst of 1 to 4 frames in the second half of the flight, the first 0 to 3 frames after release are hidden, and 2% of detections are strays 5 to 15 px off.
- **Tracking.** The detections go through `tracking.filter_ballistic_segment`, the outlier filter every video tracking method ends with.
- **Calibration.** PnP on the stump points visible in the frame, as a person would click them, each off by 1.5 px (Gaussian), with the configured field of view.
- **Measurement.** `physics.fit_trajectory` then `metrics.compute_metrics`, with the physics settings from `config/default.yaml`, as the video pipeline calls them.
- **Filter-skipped column.** The same noisy detections sent straight to the fit.
- **Noise-free column.** The same deliveries and frame rates with perfect detections, perfect clicks and no filter: the error of the fit model alone.
