import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

st.set_page_config(page_title="Indoor Map Detector — Camera MVP", layout="wide")
st.title("Indoor Map Detector — iPhone Camera MVP")
st.caption("Experimental monocular video reconstruction. Outputs a sparse 3D point cloud and a top-down projection for testing.")

st.warning(
    "This is a research prototype, not a survey-grade floor-plan generator. An iPhone 15 (non-Pro) has no LiDAR. "
    "A normal video does not provide absolute scale, and textureless/repetitive corridors can cause tracking failure. "
    "The top-down view is a projection of reconstructed points, not automatically verified room/wall geometry."
)

with st.expander("How to record with iPhone 15", expanded=True):
    st.markdown("""
    1. Use the rear **1× main camera**; avoid cinematic/action mode and digital zoom.
    2. Record at **1080p / 30 fps**. Walk slowly and smoothly; keep the phone in landscape orientation if possible.
    3. Keep walls, corners, door frames, signs and other textured details in view. Avoid pointing only at blank white walls.
    4. Record a short test (10–20 seconds) first. Turn corners slowly and include overlapping views.
    5. Do not use this prototype for safety-critical measurements. Manually measure one known distance if you later need approximate scale.
    """)

uploaded = st.file_uploader("Upload an iPhone video", type=["mp4", "mov", "m4v", "avi"])
col_a, col_b, col_c = st.columns(3)
with col_a:
    frame_step = st.slider("Sample every N video frames", 5, 60, 15, help="At 30 fps, 15 samples about every half-second.")
with col_b:
    max_frames = st.slider("Maximum frames to process", 10, 100, 35)
with col_c:
    max_features = st.slider("ORB features per frame", 500, 4000, 2000, step=250)

min_matches = st.slider("Minimum feature matches between adjacent frames", 8, 80, 18)
run = st.button("Reconstruct test map", type="primary", disabled=uploaded is None)


def extract_frames(video_bytes, suffix, frame_step, max_frames):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(video_bytes)
    tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            h, w = frame.shape[:2]
            # Downscale large iPhone frames to keep processing reasonable.
            scale = min(1.0, 1280.0 / max(w, h))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return frames


def reconstruct(frames, max_features, min_matches):
    if len(frames) < 3:
        raise ValueError("Not enough frames extracted. Try a longer video or a smaller frame-sampling interval.")

    h, w = frames[0].shape[:2]
    # Approximate intrinsics only. Real iPhone camera calibration metadata is not read in this MVP.
    focal = 0.9 * max(w, h)
    K = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    keypoints, descriptors = [], []
    for rgb in frames:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)

    # World-to-camera pose: X_cam = Rcw * X_world + tcw
    Rcw = np.eye(3, dtype=np.float64)
    tcw = np.zeros((3, 1), dtype=np.float64)
    camera_centres = [np.zeros(3)]
    all_points = []
    pair_stats = []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    for i in range(len(frames) - 1):
        d1, d2 = descriptors[i], descriptors[i + 1]
        if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
            pair_stats.append((i, 0, 0, False))
            camera_centres.append((-Rcw.T @ tcw).reshape(3))
            continue
        raw = matcher.knnMatch(d1, d2, k=2)
        good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.72 * n.distance]
        if len(good) < min_matches:
            pair_stats.append((i, len(good), 0, False))
            camera_centres.append((-Rcw.T @ tcw).reshape(3))
            continue
        p1 = np.float32([keypoints[i][m.queryIdx].pt for m in good])
        p2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None:
            pair_stats.append((i, len(good), 0, False))
            camera_centres.append((-Rcw.T @ tcw).reshape(3))
            continue
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(p1), dtype=bool)
        p1i, p2i = p1[inlier_mask], p2[inlier_mask]
        if len(p1i) < min_matches:
            pair_stats.append((i, len(good), len(p1i), False))
            camera_centres.append((-Rcw.T @ tcw).reshape(3))
            continue
        try:
            _, R21, t21, pose_mask = cv2.recoverPose(E, p1i, p2i, K)
        except cv2.error:
            pair_stats.append((i, len(good), len(p1i), False))
            camera_centres.append((-Rcw.T @ tcw).reshape(3))
            continue
        valid = pose_mask.ravel().astype(bool) if pose_mask is not None else np.ones(len(p1i), dtype=bool)
        p1t, p2t = p1i[valid], p2i[valid]
        # Triangulate in camera-i coordinates. Translation magnitude is normalized by monocular pose recovery.
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R21, t21))
        if len(p1t) >= 8:
            Xh = cv2.triangulatePoints(P1, P2, p1t.T, p2t.T)
            X1 = (Xh[:3] / np.maximum(np.abs(Xh[3:4]), 1e-9)) * np.sign(Xh[3:4])
            X2 = R21 @ X1 + t21
            good_depth = (X1[2] > 0.05) & (X2[2] > 0.05) & np.isfinite(X1).all(axis=0)
            X1 = X1[:, good_depth]
            # Convert camera-i points to world coordinates using current pose.
            Xworld = Rcw.T @ (X1 - tcw)
            if Xworld.shape[1]:
                # Keep a bounded subset to avoid excessive memory.
                if Xworld.shape[1] > 2500:
                    Xworld = Xworld[:, np.linspace(0, Xworld.shape[1]-1, 2500).astype(int)]
                all_points.append(Xworld.T)
        # Compose world-to-camera pose for the next frame.
        Rcw = R21 @ Rcw
        tcw = R21 @ tcw + t21
        camera_centres.append((-Rcw.T @ tcw).reshape(3))
        pair_stats.append((i, len(good), len(p1i), True))

    points = np.vstack(all_points) if all_points else np.empty((0, 3), dtype=np.float64)
    centres = np.array(camera_centres)
    if len(points):
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        # Robustly trim extreme triangulation outliers based on coordinate percentiles.
        if len(points) > 30:
            lo, hi = np.percentile(points, [2, 98], axis=0)
            keep = np.all((points >= lo) & (points <= hi), axis=1)
            points = points[keep]
    return points, centres, pair_stats, (w, h)


def make_ply(points):
    buf = io.StringIO()
    buf.write("ply\nformat ascii 1.0\n")
    buf.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    for p in points:
        buf.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    return buf.getvalue().encode("utf-8")


def topdown_png(points, centres, width=1000, height=800):
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    if len(points) == 0:
        return Image.fromarray(canvas)
    # Show the X-Z plane as an exploratory top-down projection; axis convention may vary by recording.
    xy = points[:, [0, 2]]
    lo = np.percentile(xy, 2, axis=0)
    hi = np.percentile(xy, 98, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    margin = 35
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    pix = (xy - lo) * scale
    pix[:, 1] = height - margin - pix[:, 1]
    pix[:, 0] += margin + ((width - 2 * margin) - span[0] * scale) / 2
    for x, y in pix.astype(int):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 1, (35, 95, 180), -1)
    # Draw camera path as a separate red polyline, using X-Z projection.
    if len(centres) > 1:
        cxy = centres[:, [0, 2]]
        cp = (cxy - lo) * scale
        cp[:, 1] = height - margin - cp[:, 1]
        cp[:, 0] += margin + ((width - 2 * margin) - span[0] * scale) / 2
        cp = cp.astype(int)
        for j in range(len(cp) - 1):
            if np.linalg.norm(cp[j+1] - cp[j]) < 2000:
                cv2.line(canvas, tuple(cp[j]), tuple(cp[j+1]), (200, 70, 55), 2)
        if len(cp):
            cv2.circle(canvas, tuple(cp[0]), 6, (30, 150, 50), -1)
            cv2.circle(canvas, tuple(cp[-1]), 6, (180, 50, 180), -1)
    return Image.fromarray(canvas)


if run and uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with st.spinner("Extracting frames and reconstructing a sparse point cloud. This can take a few minutes..."):
        try:
            frames = extract_frames(uploaded.getvalue(), suffix, frame_step, max_frames)
            if len(frames) < 3:
                st.error("Could not extract enough frames. Try exporting the video as MP4 or use a longer clip.")
                st.stop()
            points, centres, pair_stats, image_size = reconstruct(frames, max_features, min_matches)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    successful_pairs = sum(1 for x in pair_stats if x[3])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames extracted", len(frames))
    m2.metric("Pairs tracked", f"{successful_pairs}/{len(pair_stats)}")
    m3.metric("Sparse 3D points", len(points))
    m4.metric("Camera path samples", len(centres))

    if len(points) < 100 or successful_pairs < 2:
        st.error("Reconstruction is weak. Record more textured surfaces, move more slowly, keep the camera stable, and try sampling every 5–10 frames.")

    left, right = st.columns(2)
    with left:
        st.subheader("Approximate top-down projection")
        st.image(topdown_png(points, centres), use_container_width=True)
        st.caption("Blue dots = reconstructed feature points. Red line = estimated camera path. This is not a verified floor plan.")
    with right:
        st.subheader("Sparse 3D reconstruction")
        if len(points):
            # Cap display points to keep browser rendering responsive.
            sample = points
            if len(sample) > 12000:
                sample = sample[np.linspace(0, len(sample)-1, 12000).astype(int)]
            fig = go.Figure(data=[go.Scatter3d(x=sample[:,0], y=sample[:,1], z=sample[:,2], mode="markers", marker=dict(size=2, color=sample[:,1], colorscale="Viridis", opacity=0.8), name="3D feature points")])
            fig.add_trace(go.Scatter3d(x=centres[:,0], y=centres[:,1], z=centres[:,2], mode="lines+markers", marker=dict(size=3, color="red"), line=dict(color="red", width=4), name="Camera path"))
            fig.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=0), scene=dict(aspectmode="data"), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No reliable triangulated points were produced for this clip.")

    st.subheader("Frame-pair tracking diagnostics")
    st.dataframe([{"Frame pair": f"{i} → {i+1}", "Ratio-test matches": matches, "RANSAC inliers": inliers, "Pose estimated": "Yes" if ok else "No"} for i, matches, inliers, ok in pair_stats], use_container_width=True)

    if len(points):
        st.download_button("Download sparse 3D point cloud (.ply)", data=make_ply(points), file_name="indoor_sparse_reconstruction.ply", mime="application/octet-stream")
        png_buf = io.BytesIO()
        topdown_png(points, centres).save(png_buf, format="PNG")
        st.download_button("Download top-down projection (.png)", data=png_buf.getvalue(), file_name="indoor_topdown_projection.png", mime="image/png")
        trajectory = "frame,x,y,z\n" + "\n".join(f"{i},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}" for i,p in enumerate(centres))
        st.download_button("Download estimated camera path (.csv)", data=trajectory.encode(), file_name="camera_trajectory.csv", mime="text/csv")

    st.info("Next engineering step: add plane fitting and room-topology reconstruction, then let a human correct the generated walls and openings. Monocular coordinates have arbitrary scale; use measured reference distances for approximate scale calibration.")
