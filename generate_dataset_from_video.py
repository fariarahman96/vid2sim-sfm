import json

import cv2
import numpy as np
import tqdm
from ultralytics import YOLO

# =========================
# Config
# =========================
VIDEO_IN = "example.mp4"

MODEL_WEIGHTS = "yolo11s.pt"
IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = "mps"  # use "cpu" if mps fails

REF_W, REF_H = 1920, 1080

left_pos = 320
top_pos = 700

# Two destination points in REF (1920x1080) coordinates
END_A = (850, top_pos)
END_B = (1500, REF_H - 30)

SRC_TL = (left_pos + 260, top_pos)
SRC_TR = (REF_W - left_pos, top_pos)
SRC_BR = (REF_W, REF_H - 30)
SRC_BL = (480, REF_H - 30)

SAMPLE_FPS = 30.0
DOT_RADIUS = 4

SAMPLE_FRAME_OUT = "sample_frame.jpg"
TRANSFORMED_SAMPLE_FRAME_OUT = "transformed_sample_frame.jpg"


GENERATE_DATA = True # True for generating False for verifying



DETECTION_VIDEO_OUT = "pedestrian_detection.mp4"
TOPDOWN_VIDEO_OUT = "pedestrian_topdown.mp4"
DATASET_JSON_OUT = "pedestrian_dataset.json"

# Top-down visual settings
GRID_STEP = 100
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =========================
# Helpers
# =========================
def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography(frame_w, frame_h):
    sx, sy = frame_w / float(REF_W), frame_h / float(REF_H)
    src = np.array([SRC_TL, SRC_TR, SRC_BR, SRC_BL], dtype=np.float32)
    src[:, 0] *= sx
    src[:, 1] *= sy
    src = order_points(src)

    w_top = np.linalg.norm(src[1] - src[0])
    w_bot = np.linalg.norm(src[2] - src[3])
    h_left = np.linalg.norm(src[3] - src[0])
    h_right = np.linalg.norm(src[2] - src[1])

    out_w = max(int(round(max(w_top, w_bot))), 2)
    out_h = max(int(round(max(h_left, h_right))), 2)

    dst = np.array(
        [[0, 0],
         [out_w - 1, 0],
         [out_w - 1, out_h - 1],
         [0, out_h - 1]],
        dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(src, dst)
    return M, (out_w, out_h), src


def point_in_quad(pt, quad):
    return cv2.pointPolygonTest(
        quad.astype(np.float32),
        (float(pt[0]), float(pt[1])),
        False
    ) >= 0


def transform_points(points_xy, M):
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, M).reshape(-1, 2)  # (u,v) in image coords
    return warped


def safe_int(x, lo=0, hi=None):
    xi = int(round(float(x)))
    if hi is not None:
        xi = max(lo, min(hi, xi))
    else:
        xi = max(lo, xi)
    return xi


def ref_to_frame_point(pt_ref_xy, frame_w, frame_h):
    """Scale a point from REF (1920x1080) to current frame size."""
    sx, sy = frame_w / float(REF_W), frame_h / float(REF_H)
    return (float(pt_ref_xy[0]) * sx, float(pt_ref_xy[1]) * sy)


# ---------- image <-> Cartesian conversions (origin bottom-left) ----------
def img_to_cart_points(uv_pts, td_h):
    """
    (u,v) image coords (origin top-left, y down) -> (x,y) Cartesian (origin bottom-left, y up)
    x = u, y = (H-1) - v
    """
    if len(uv_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    uv = np.asarray(uv_pts, dtype=np.float32).reshape(-1, 2)
    xy = uv.copy()
    xy[:, 1] = (td_h - 1) - uv[:, 1]
    return xy


def cart_to_img_points(xy_pts, td_h):
    """
    (x,y) Cartesian (origin bottom-left) -> (u,v) image coords (origin top-left)
    u = x, v = (H-1) - y
    """
    if len(xy_pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    xy = np.asarray(xy_pts, dtype=np.float32).reshape(-1, 2)
    uv = xy.copy()
    uv[:, 1] = (td_h - 1) - xy[:, 1]
    return uv


def draw_labeled_point(frame, pt_xy, label, color=(0, 0, 255), radius=7):
    x, y = pt_xy
    h, w = frame.shape[:2]
    xi = safe_int(x, 0, w - 1)
    yi = safe_int(y, 0, h - 1)
    cv2.circle(frame, (xi, yi), radius, color, -1)
    cv2.putText(frame, label, (min(w - 5, xi + 10), max(20, yi - 10)),
                FONT, 0.7, color, 2, cv2.LINE_AA)


def draw_topdown_canvas(td_w, td_h, people_cart_xy, t_sec, frame_idx,
                        endA_cart=None, endB_cart=None, draw_axes=True):
    """
    White top-down canvas (OpenCV image space),
    BUT points/labels come from Cartesian coords (origin bottom-left).
    We convert Cartesian -> image coords for drawing, so dots match the real warp.
    """
    canvas = np.full((td_h, td_w, 3), 255, dtype=np.uint8)

    # Border
    cv2.rectangle(canvas, (0, 0), (td_w - 1, td_h - 1), (0, 0, 0), 2)

    # Grid
    if GRID_STEP > 0:
        for x in range(GRID_STEP, td_w, GRID_STEP):
            cv2.line(canvas, (x, 0), (x, td_h - 1), (220, 220, 220), 1)
        for y in range(GRID_STEP, td_h, GRID_STEP):
            cv2.line(canvas, (0, y), (td_w - 1, y), (220, 220, 220), 1)

    # Axes label (origin)
    if draw_axes:
        O_xy = np.array([[0.0, 0.0]], dtype=np.float32)
        O_uv = cart_to_img_points(O_xy, td_h)[0]
        Oi = (safe_int(O_uv[0], 0, td_w - 1), safe_int(O_uv[1], 0, td_h - 1))
        cv2.putText(canvas, "(0,0)", (Oi[0] + 6, Oi[1] - 6 if Oi[1] > 20 else Oi[1] + 20),
                    FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Info line
    info = f"frame={frame_idx}  t={t_sec:.2f}s  n={len(people_cart_xy)} "
    cv2.putText(canvas, info, (10, td_h - 15), FONT, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw endpoints (red)
    def _draw_endpoint(cart_pt, name):
        if cart_pt is None:
            return
        cart_pt = np.asarray(cart_pt, dtype=np.float32).reshape(1, 2)
        uv = cart_to_img_points(cart_pt, td_h)[0]
        ui = safe_int(uv[0], 0, td_w - 1)
        vi = safe_int(uv[1], 0, td_h - 1)
        cv2.circle(canvas, (ui, vi), 10, (0, 0, 255), -1)
        cv2.putText(canvas, name, (min(td_w - 5, ui + 12), max(20, vi - 12)),
                    FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    _draw_endpoint(endA_cart, "Destination A")
    _draw_endpoint(endB_cart, "Destination B")

    # Draw people (red dots + labels)
    people_cart_xy = np.asarray(people_cart_xy, dtype=np.float32).reshape(-1, 2)
    people_uv = cart_to_img_points(people_cart_xy, td_h)

    for (x, y), (u, v) in zip(people_cart_xy, people_uv):
        ui = safe_int(u, 0, td_w - 1)
        vi = safe_int(v, 0, td_h - 1)
        cv2.circle(canvas, (ui, vi), DOT_RADIUS, (0, 0, 255), -1)

        label = f"({int(round(x))},{int(round(y))})"
        tx = min(td_w - 5, ui + 8)
        ty = min(td_h - 5, vi - 8 if vi > 20 else vi + 20)
        cv2.putText(canvas, label, (tx, ty), FONT, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    return canvas


# =========================
# Main
# =========================
def main():
    print("[INFO] Opening video:", VIDEO_IN)
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {VIDEO_IN}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    h0, w0 = frame0.shape[:2]
    print(f"[INFO] Video size: {w0}x{h0}, fps={video_fps:.3f}, total_frames={total_frames}")

    print("[INFO] Computing homography ...")
    M, (td_w, td_h), src_poly = compute_homography(w0, h0)

    print("[INFO] Homography matrix M (src -> topdown):\n", M)
    print(f"[INFO] Top-down size: {td_w}x{td_h}")
    print("[INFO] src points used (frame coords):", src_poly.tolist())

    # --- NEW: compute endpoints in frame coords, then topdown uv + cart coords ---
    endA_frame = ref_to_frame_point(END_A, w0, h0)
    endB_frame = ref_to_frame_point(END_B, w0, h0)

    end_uv = transform_points([endA_frame, endB_frame], M)        # (u,v)
    end_xy = img_to_cart_points(end_uv, td_h)                     # (x,y) cartesian
    endA_uv, endB_uv = end_uv[0], end_uv[1]
    endA_xy, endB_xy = end_xy[0], end_xy[1]

    # Save sample frames (with rectangle + endpoints)
    frame0_vis = frame0.copy()
    cv2.polylines(frame0_vis, [src_poly.astype(np.int32)], True, (0, 255, 255), 2)
    draw_labeled_point(frame0_vis, endA_frame, "Destination A", color=(0, 0, 255), radius=8)
    draw_labeled_point(frame0_vis, endB_frame, "Destination B", color=(0, 0, 255), radius=8)
    cv2.imwrite(SAMPLE_FRAME_OUT, frame0_vis)

    td0 = cv2.warpPerspective(frame0, M, (td_w, td_h), flags=cv2.INTER_LINEAR)

    # draw endpoints on transformed sample too (in IMAGE coords)
    td0_vis = td0.copy()
    for (u, v), name in [(endA_uv, "Destination A"), (endB_uv, "Destination B")]:
        ui = safe_int(u, 0, td_w - 1)
        vi = safe_int(v, 0, td_h - 1)
        cv2.circle(td0_vis, (ui, vi), 10, (0, 0, 255), -1)
        cv2.putText(td0_vis, name, (min(td_w - 5, ui + 12), max(20, vi - 12)),
                    FONT, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(TRANSFORMED_SAMPLE_FRAME_OUT, td0_vis)

    if not GENERATE_DATA:
        cap.release()
        print("[INFO] GENERATE_DATA=False -> exiting after saving sample frames.")
        print("[INFO] Endpoints transformed:")
        print(f"       A: uv={endA_uv.tolist()}  cart={endA_xy.tolist()}")
        print(f"       B: uv={endB_uv.tolist()}  cart={endB_xy.tolist()}")
        return

    print("[INFO] Loading YOLO model:", MODEL_WEIGHTS)
    model = YOLO(MODEL_WEIGHTS)

    stride = max(int(round(video_fps / float(SAMPLE_FPS))), 1)
    out_fps = float(video_fps) / float(stride)
    print(f"[INFO] Sampling: desired={SAMPLE_FPS} fps -> stride={stride} -> output_fps={out_fps:.3f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(DETECTION_VIDEO_OUT, fourcc, out_fps, (w0, h0))
    out_td = cv2.VideoWriter(TOPDOWN_VIDEO_OUT, fourcc, out_fps, (td_w, td_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    dataset = {
        "video_in": VIDEO_IN,
        "video_fps": float(video_fps),
        "sample_fps": float(out_fps),
        "frame_stride": int(stride),
        "ref_size": [int(REF_W), int(REF_H)],

        "src_points_ref": {
            "tl": list(SRC_TL),
            "tr": list(SRC_TR),
            "br": list(SRC_BR),
            "bl": list(SRC_BL),
        },
        "src_points_used": src_poly.tolist(),
        "homography_M": M.tolist(),
        "topdown_size": [int(td_w), int(td_h)],

        "topdown_coord_system": {
            "type": "cartesian",
            "origin": "bottom_left",
            "x_direction": "right",
            "y_direction": "up",
            "units": "topdown_pixels"
        },

        # --- NEW: store destination endpoints in json (ref -> frame -> topdown uv -> cart xy) ---
        "destinations": {
            "A": {
                "ref_xy": [float(END_A[0]), float(END_A[1])],
                "frame_xy": [float(endA_frame[0]), float(endA_frame[1])],
                "topdown_uv": [float(endA_uv[0]), float(endA_uv[1])],
                "topdown_xy": [float(endA_xy[0]), float(endA_xy[1])]
            },
            "B": {
                "ref_xy": [float(END_B[0]), float(END_B[1])],
                "frame_xy": [float(endB_frame[0]), float(endB_frame[1])],
                "topdown_uv": [float(endB_uv[0]), float(endB_uv[1])],
                "topdown_xy": [float(endB_xy[0]), float(endB_xy[1])]
            }
        },

        "frames": {}
    }

    print("[INFO] Destinations transformed (also written to JSON):")
    print(f"       A: uv={endA_uv.tolist()}  cart={endA_xy.tolist()}")
    print(f"       B: uv={endB_uv.tolist()}  cart={endB_xy.tolist()}")

    total_iters = (total_frames // stride) if total_frames > 0 else None
    print("[INFO] Starting detection loop ...")

    frame_idx = 0
    sampled = 0

    pbar = tqdm.tqdm(total=total_iters, desc="Processing sampled frames", unit="frame")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            ts = float(frame_idx) / float(video_fps)

            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                classes=[0],
                device=DEVICE,
                verbose=False,
            )

            people_xy = []
            r = results[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    x = float((x1 + x2) / 2.0)
                    y = float(y2)  # bottom center
                    if point_in_quad((x, y), src_poly):
                        people_xy.append((x, y))

            # 1) homography -> topdown IMAGE coords (u,v)
            td_uv = transform_points(people_xy, M)

            # 2) convert to CARTESIAN coords (x,y) with origin bottom-left
            td_xy = img_to_cart_points(td_uv, td_h)

            # Original-frame output
            frame_vis = frame.copy()
            cv2.polylines(frame_vis, [src_poly.astype(np.int32)], True, (0, 255, 255), 2)

            # draw endpoints on original frame too (static)
            draw_labeled_point(frame_vis, endA_frame, "Destination A", color=(0, 0, 255), radius=8)
            draw_labeled_point(frame_vis, endB_frame, "Destination B", color=(0, 0, 255), radius=8)

            for (x, y) in people_xy:
                cv2.circle(frame_vis,
                           (safe_int(x, 0, w0 - 1), safe_int(y, 0, h0 - 1)),
                           DOT_RADIUS, (0, 0, 255), -1)
            out_vid.write(frame_vis)

            # Top-down output: include endpoints (static) + people
            td_canvas = draw_topdown_canvas(
                td_w, td_h, td_xy, ts, frame_idx,
                endA_cart=endA_xy, endB_cart=endB_xy,
                draw_axes=True
            )
            out_td.write(td_canvas)

            # JSON: store CARTESIAN coords
            dataset["frames"][str(frame_idx)] = {
                "t": ts,
                "obj": [{"x": float(p[0]), "y": float(p[1])} for p in td_xy]
            }

            sampled += 1
            frame_idx += 1
            pbar.update(1)

            if sampled == 1:
                print("[INFO] First sampled frame processed.")
                print("       Example topdown UV (img coords):", td_uv[:5].tolist())
                print("       Example topdown XY (cart coords):", td_xy[:5].tolist())
    finally:
        pbar.close()
        cap.release()
        out_vid.release()
        out_td.release()

    print("[INFO] Writing dataset json:", DATASET_JSON_OUT)
    with open(DATASET_JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print("\n[DONE] Outputs:")
    print(" -", SAMPLE_FRAME_OUT)
    print(" -", TRANSFORMED_SAMPLE_FRAME_OUT)
    print(" -", DETECTION_VIDEO_OUT)
    print(" -", TOPDOWN_VIDEO_OUT)
    print(" -", DATASET_JSON_OUT)


if __name__ == "__main__":
    main()