# Real footage: what the pipeline does with it

Measured on 2026-11-13. This page records every run of the pipeline on real
video that exists for this project, and what it showed. **It makes no accuracy
claim.** None of these clips has a radar speed or a marked bounce, so there is
nothing to measure error against. All accuracy numbers in this repo are still
synthetic only ([ACCURACY.md](ACCURACY.md)).

## Short version

- **No phone capture exists.** The Android app has never produced a clip and
  sidecar that reached this work, so `scripts/analyze_capture.py` has never run
  on a real phone recording.
- The only real footage is **two watermarked stock clips of net practice**,
  each holding one usable delivery, at a 24 fps playback rate.
- The pipeline ran on them 26 times: 17 runs finished, 9 refused with an
  error, none crashed. **None of the 17 results is believable.** Each fails at
  least one of the plain bounds below, and the one that fails only the
  frame-rate bound is wrong for a reason no check can see (it used the other
  clip's camera setup).
- Before this change, every finished result was printed or saved with no
  warning at all. Now each comes with plain-language checks (see "Fixed").
- Two contained defects showed up on real pixels and are fixed with synthetic
  regression tests. The rest are limits of the method and are in the
  [ROADMAP](../ROADMAP.md).

## What exists

| Item | Count | Details |
|---|---|---|
| Phone captures with a JSON sidecar | 0 | |
| Real video clips | 2 | Stock clips of outdoor nets, H.264, 900×474, 24 fps container rate. Clip A: 15.3 s, 367 frames. Clip B: 18.5 s, 445 frames. |
| Stump calibrations | 1 | One file for the whole folder: six bowler-end stump points, 0.75 px reprojection error. No batter-end stumps. |
| Saved app sessions | 2 | One per clip, both from balls marked by hand in the Streamlit app (7 and 9 clicks). They saved 29.5 and 30.0 km/h. |
| Labelled ground truth | 0 | `data/ground_truth/template.csv` is a header line. |
| MediaPipe pose model | 1 | `pose_landmarker_lite.task`, 5.5 MB, used by `--method auto` to find the release. |

What the clips show, from looking at the frames:

- **Clip A** is shot by a handheld camera that follows the bowler, with cuts
  between shots. The delivery is around frames 60 to 95: release near frame 68,
  the ball is a few pixels wide and is visible for only a handful of frames.
- **Clip B** is shot from behind the stumps of one lane in a multi-lane net,
  with a slow pan. The delivery in that lane is around frames 132 to 165:
  release near frame 139, with the ball near the top edge of the frame for the
  first few frames of its flight.
- The stump points in the calibration file sit where clip B's near stumps are
  (x ≈ 650 to 680 px), and about 25 px left of where they are at clip B's
  release, because the camera pans. In clip A's delivery shot the near stumps
  are at x ≈ 420 to 490 px and their bases are below the bottom of the frame.
  So the one calibration file matches neither delivery exactly, and clip A's
  saved session was computed with clip B's camera.
- At 24 fps a 130 km/h ball travels 1.5 m between frames. Whether either clip
  was slowed down for stock use is unknown, so even the frame rate the speed
  depends on is unknown.

## Bounds used to call a result believable

These are not accuracy targets. They only catch values no real delivery from
the bowler's end can have. They are the constants in
`src/balltracker/result_checks.py`.

| Quantity | Believable range |
|---|---|
| Release speed | 50 to 170 km/h |
| Bounce distance from the batter's stumps (`pitch_length_m`) | −2.0 to 20.12 m |
| Bounce line from middle stump (`pitch_line_m`) | within ±1.6 m |
| Average pixel gap between fitted path and tracked points | under 15 px |
| Release velocity | not stopped on the fit's search limit |
| Frame rate | 100 fps or more (the capture guide asks for 120) |

## What was run

All outputs went to a scratch folder outside the repo. No frame, clip or
session file was copied into the repo.

**1. `scripts/analyze_delivery.py`** on both clips, with the folder's
calibration file, every detector (`auto`, `hsv`, `white`, `motion`), over the
whole clip and over the delivery window (A: frames 60 to 95, B: 132 to 165).
That is 16 runs.

| Clip | Method | Window | Ball points (filled) | Speed km/h | Length m | Line m | Fit gap px | Believable? |
|---|---|---|---|---|---|---|---|---|
| A | auto | whole | 6 (0) | 199.1 | 12.15 | −7.17 | 84.0 | no: speed, line, fit, limit |
| A | auto | 60–95 | 8 (0) | 36.4 | 19.12 | −2.66 | 11.2 | no: speed, line, limit |
| A | hsv | whole | 5 (4) | 29.0 | 19.45 | −1.22 | 12.0 | no: speed, limit |
| A | hsv | 60–95 | 5 (4) | 28.9 | 19.45 | −1.52 | 6.2 | no: speed, limit |
| A | white | whole | 14 (0) | 116.1 | 0.48 | 8.61 | 6.4 | no: line |
| A | white | 60–95 | 33 (0) | 23.1 | 15.74 | −1.22 | 237.5 | no: speed, fit |
| A | motion | whole | 5 (4) | 29.0 | 19.45 | −1.18 | 10.9 | no: speed, limit |
| A | motion | 60–95 | 5 (4) | 199.2 | 9.54 | −1.08 | 15.1 | no: speed, fit, limit |
| B | auto | whole | 7 (0) | 34.8 | 19.12 | 1.68 | 15.8 | no: speed, line, fit, limit |
| B | auto | 132–165 | 10 (0) | 32.4 | 13.45 | 0.50 | 28.4 | no: speed, fit, limit |
| B | hsv | whole | 5 (4) | 31.4 | 19.45 | 0.89 | 6.5 | no: speed, limit |
| B | hsv | 132–165 | 0 | — | — | — | — | refused: no ball found |
| B | white | whole | 6 (0) | 30.1 | 18.45 | −3.31 | 6.7 | no: speed, line, limit |
| B | white | 132–165 | 7 (0) | 16.7 | 16.92 | −2.63 | 57.0 | no: speed, line, fit |
| B | motion | whole | 7 (0) | 64.7 | 10.44 | −3.84 | 78.2 | no: line, fit |
| B | motion | 132–165 | 14 (10) | 31.4 | 18.79 | −0.28 | 13.0 | no: speed, limit |

Every run also gets the low-frame-rate warning, since both clips play at 24 fps.

**2. MediaPipe release seeding** (`--method auto` uses it). Inside the
delivery windows it put the release at frame 70 in clip A and frame 140 in
clip B, within one to two frames of where the release looks to be by eye. Over
the whole clip it picked frames 34 and 36, which are a running bowler and a
different player, because it only looks at the first 90 frames of the window.
The ball track that followed the seed was still too short or too mixed with
other things to fit.

**3. `scripts/analyze_capture.py`.** With no phone capture, I trimmed clip B's
delivery (frames 120 to 170) to a scratch file and wrote a sidecar by hand:
the four stump bases read off frame 138, the focal length borrowed from the
calibration file, 24 fps and the clip's real presentation timestamps. I made
a second copy rotated 90° in the file with an orientation hint of 90, as a
phone held upright would record it. All 8 runs (two copies, four detectors)
refused with "no capture pose is behind the bowler and above the pitch": with
those marks and that focal length no camera position fits. That is the right
answer; it did not guess.

**4. The capture path's rotation and clock**, run directly on the rotated
copy. The picture came back the right size (900×474) and the presentation
timestamps were used and matched frame index ÷ 24 exactly. The pixels did not
come back intact; see "Fixed" below.

**5. The two saved app sessions**, refitted from their hand-marked clicks.

| Session | Saved speed | Refit with the fixed gap filler | Result checks after the fix |
|---|---|---|---|
| Clip B, 7 clicks | 29.5 km/h | 30.4 km/h | fit limit, speed, frame rate |
| Clip A, 9 clicks | 30.0 km/h | 78.2 km/h, bounce 7.2 m from the batter, 0.03 m off middle | frame rate only |

The clip A refit passes every bound except frame rate, and it is still not
trustworthy: it uses clip B's camera calibration on a handheld clip. No check
can see that, which is why tying a calibration to its clip is on the roadmap.

## What worked

- Nothing crashed. Every refusal was a typed error with a message.
- On clip B, inside the delivery window, `auto` found the ball in four
  consecutive frames (140 to 143), within 3 px of where it was marked by hand
  in the saved session. Its other six points were the bowler's hand and
  another player, so the fit still failed.
- MediaPipe found the release frame when given the delivery window.
- The capture path's rotation geometry and presentation-timestamp clock did
  what they should on a real clip.

## What did not

1. **No detector produced a ball track long enough to fit.** At 900×474 the
   ball is 2 to 5 px across. `auto` over clip A locked onto the bowler's shirt
   during a camera cut (frames 32 to 37) and reported 199 km/h, 7 m wide of the
   stumps. Inside clip A's window it followed something moving up and to the
   left from the release point, about 200 px from where the ball was marked by
   hand. `white` over clip A's window returned a point on all 33 frames, which
   no fit could join up (fit gap 237 px).
2. **The fit stopped on its own limit.** In 11 of the 15 finished runs the
   release velocity ended on the slowest value the fit searches (8 m/s along
   the pitch), which reads as about 29 km/h. Both saved sessions reported this
   same floor (29.5 and 30.0 km/h) as if it were a measurement.
3. **Nothing warned.** The command line printed the numbers alone, and the app
   showed them as the delivery's speed and bounce.
4. **One calibration per folder.** The app and the command line both reuse
   `session.calib.json` for every clip in a folder, so clip A was analysed with
   clip B's camera.
5. **Moving cameras.** Clip A is handheld with cuts and clip B pans. A
   calibration from one frame is wrong for the others; nothing detects this.
6. **24 fps.** The clips play far below the 120 fps the capture guide asks for,
   and the true capture rate is unknown. Nothing lets a user give the real rate.

## Fixed in this change

| Defect | Evidence on real footage | Fix | Regression test (synthetic) |
|---|---|---|---|
| Filling the frames between two clicked balls put them behind the last click | Clip A session: clicks moved 24 px a frame down the image, the first filled frame landed 34 px back up, and the fit read 30.0 km/h | Fill on the straight line between the clicks either side (a gap always has a click on both sides) | `test_filled_click_gap_stays_between_the_clicks_around_it` |
| Turning a phone clip upright made a lossy copy | Rotated clip B: average pixel change up to 4.7 grey levels per frame (0.5 when rotated in memory), and the detections changed (`hsv`: 5 points became 0) | Write the temporary upright copy losslessly (FFV1), falling back to the old codec only if FFV1 is unavailable | `test_rotation_keeps_every_pixel_of_a_small_ball` |
| Unbelievable results were shown with no warning | All 15 finished runs and both saved sessions | `result_checks` adds plain-language warnings to the result (`checks`), the command-line output and the app's result page. Numbers are unchanged. The fit now also reports when the release velocity stopped on its search limit (`within_fit_limits`). | `test_each_result_check_fires_on_its_own`, `test_fit_stopped_on_a_velocity_limit_is_reported`, `test_capture_with_a_slow_clock_is_flagged`, `test_clean_synthetic_deliveries_raise_no_checks` |

To make sure the checks do not cry wolf, they were run over 300 seeded
synthetic deliveries per benchmark mode. With perfect detections none fired.
With phone-like noise, 5 of 300 fired, and those were real misses: speeds of
174 to 199 km/h against a true maximum of 150, one fit gap over 15 px, and one
release on the velocity limit with a 50 km/h speed error. The fit's numbers
were compared before and after the change on 450 synthetic fits and were
identical.

The regression tests are in `tests/test_real_footage_regressions.py`. None of
them reads real footage.

## Not fixed (see the ROADMAP)

- A calibration that belongs to one clip, not to a folder.
- Detecting a moving or cut camera before trusting a one-frame calibration.
- A way to give the true capture frame rate when a file plays slower than it
  was filmed.
- Ball detection on real nets footage. This needs real phone captures at
  120 fps or more, filmed per [CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md).
- Real labelled deliveries (milestone M2), without which there is still no
  accuracy number for real footage.

## Rerunning this on your own clips

```bash
# A clip and a calibration made with scripts/calibrate_stumps.py on a frame of that clip:
python scripts/analyze_delivery.py --video CLIP.mp4 --calib CLIP.calib.json \
    --start-frame FIRST --end-frame LAST --method auto --out analysis.json

# A phone capture from the Android app:
python scripts/analyze_capture.py --sidecar CLIP.json --out analysis.json
```

Read the `Check this result:` lines, or the `checks` list in the JSON, before
reading the numbers. Keep clips, frames and session folders out of git.
