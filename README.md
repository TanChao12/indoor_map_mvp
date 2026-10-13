# Indoor Map Detector — iPhone Camera MVP

A small experimental Streamlit app for testing monocular video reconstruction from an iPhone 15 (non-Pro) recording. It estimates relative camera motion from adjacent frames, triangulates sparse feature points, and exports:

- `indoor_sparse_reconstruction.ply` — sparse 3D point cloud
- `indoor_topdown_projection.png` — exploratory projection of points in the X-Z plane
- `camera_trajectory.csv` — estimated camera path

## Important limitations

This is a prototype to test the camera pipeline, not a finished automatic architectural floor-plan generator. Monocular video does not give absolute metric scale without a reference. This implementation uses approximate camera intrinsics and pairwise pose chaining, so drift and outliers are expected. The top-down projection is not wall geometry and must not be presented as a verified floor plan. It does not yet output a watertight mesh, CAD drawing, or semantically labelled rooms.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 recommended
- iPhone 15 video copied to the computer
- Internet for the first dependency installation

## Run on Windows

1. Extract this ZIP to a folder.
2. Open the folder in File Explorer.
3. Double-click `run_windows.bat`.
4. Wait for dependencies to install; the browser should open the local Streamlit app.
5. Upload a short `.MOV` or `.MP4` video and click **Reconstruct test map**.

If the `py` command is not found, install Python 3.11 from https://www.python.org/downloads/ and enable **Add Python to PATH**. Then reopen the terminal and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## iPhone capture guide

- Use rear 1× camera, 1080p / 30 fps.
- Walk slowly; do not whip-pan or digitally zoom.
- Keep textured details, corners, door frames and signs in view.
- Record a short corridor with a turn, not only a long blank wall.
- Keep people moving through the scene to a minimum.
- Test with 10–20 seconds before trying an entire floor.

## Suggested first test

Record a 10–20 second walk along a corridor with at least one corner or several door frames. In the app, try sampling every 10–15 frames. If few frame pairs track successfully, try a more textured route and smoother movement.

## Next development milestones

1. Read real camera intrinsics and use a calibrated camera model.
2. Replace pairwise monocular pose chaining with a maintained SLAM pipeline and loop closure.
3. Calibrate scale from a measured reference distance.
4. Extract dominant floor/wall planes and fit wall segments.
5. Build room polygons and door connectivity, with manual review.
6. Generate a true 3D mesh and editable SVG/GeoJSON floor plans.
