# README media

Both files are output of `balltracker demo` (default seed), made on 2026-11-13.
They show a synthetic delivery, not real footage.

| File | Made with |
|---|---|
| `demo.gif` | `ffmpeg -i out/demo/overlay.mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 docs/media/demo.gif` |
| `pitch_map.png` | copied from `out/demo/pitch_map.png` |

Rerun both after a change to the demo, the fit or the metrics, so the README
does not show numbers the code no longer produces.
