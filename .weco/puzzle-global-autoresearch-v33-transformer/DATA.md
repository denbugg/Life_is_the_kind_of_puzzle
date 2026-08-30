# Data

- Reuse all 60 V32 scene-group-isolated spatial cache files.
- Each real solver board has clean and two independently corrupted views with
  32 feature planes at 24x24; near-miss boards remain capped below 50%.
- Train groups: scenes 6700-6727 and 6957-6980.
- Locked validation: 6981-6988.
- No target image, tile ID, or canonical position is exposed as an input.
- All views and candidates from one source scene remain in one CV fold.
