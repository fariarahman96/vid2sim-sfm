#!/usr/bin/env python3
"""
Learn 4 SFM parameters (tau, A, B, v0) by trajectory matching:

- Track without IDs using distance + direction + prediction
- Build fixed-N windows (same tracked agents stay present in window)
- Roll out SFM forward for K steps (Euler integration)
- Loss = RMSE of positions (and optional velocity) over the window
- Adam + autograd (PyTorch)

Driving direction uses destinations A/B:
- per-agent choose A->B or B->A based on velocity alignment and distance trend

Also does sensitivity analysis after calibration:
- RMSE vs tau
- RMSE vs A
- RMSE vs B
- RMSE vs v0
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# =========================
# USER CONFIG
# =========================
DATASET_JSON = "pedestrian_dataset.json"

REAL_WIDTH_M = 70.0
MASS_KG = 80.0
RADIUS_M = 0.30
MAX_NEIGHBOR_DIST_M = 6.0

# Tracking (no IDs)
TRACK_MAX_DIST_M = 1.5
TRACK_MAX_MISSES = 2
W_DIST, W_ANGLE, W_PRED = 1.0, 0.7, 0.7

# Window building
K_STEPS = 15
STRIDE = 3
MIN_AGENTS = 2
MAX_WINDOWS = 20000

# Training
SEED = 1
DEVICE = "cpu"            # "cuda" if available
EPOCHS = 200
BATCH_WINDOWS = 128
LR = 0.03
PRINT_EVERY = 10

# Loss weights
W_POS = 1.0
W_VEL = 0.2               # set 0.0 to use only position RMSE

# Bounds for params
TAU_BOUNDS = (0.03, 1.0)        # s
A_BOUNDS   = (0.1, 10)     # N
B_BOUNDS   = (0.02, 0.5)       # m
V0_BOUNDS  = (0.2, 4.0)        # m/s

# Start values
START_TAU = 0.5
START_A   = 2.0
START_B   = 0.08
START_V0  = 1.34

# Sensitivity analysis
SENS_POINTS = 40
SENS_MAX_WINDOWS = 300        # use subset for speed
SENS_PLOT_FILE = "sensitivity_analysis.png"
TRAIN_LOSS_PLOT_FILE = "training_loss.png"


# =========================
# Utils
# =========================
EPS = 1e-9

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def unit_np(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v)
    return v / n

def angle_cost(v_pred: np.ndarray, disp: np.ndarray) -> float:
    nv = np.linalg.norm(v_pred)
    nd = np.linalg.norm(disp)
    if nv < 1e-6 or nd < 1e-6:
        return 0.5
    c = float(np.dot(v_pred, disp) / (nv * nd + 1e-12))
    return 0.5 * (1.0 - c)  # aligned->0, opposite->1

def inv_sigmoid(y: float) -> float:
    y = float(np.clip(y, 1e-6, 1 - 1e-6))
    return math.log(y / (1.0 - y))

def unscale01(x: float, lo: float, hi: float) -> float:
    x = float(np.clip(x, lo + 1e-9, hi - 1e-9))
    return (x - lo) / (hi - lo)

def rmse_masked(residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    residual: (..., 2)
    mask: (...) bool
    RMSE over components where mask is True
    """
    m = mask.float().unsqueeze(-1)
    r2 = (residual * residual) * m
    denom = torch.clamp(m.sum() * residual.shape[-1], min=1.0)
    return torch.sqrt(r2.sum() / denom + 1e-12)


# =========================
# Tracking
# =========================
@dataclass
class Track:
    last_frame: int
    last_t: float
    last_xy: np.ndarray
    last_v: np.ndarray
    misses: int
    traj: List[Tuple[int, float, float, float]]  # (frame, t, x, y)

def match_tracks_to_dets(tracks: Dict[int, Track], active_ids: List[int],
                         det_xy: List[np.ndarray], t_now: float):
    if len(active_ids) == 0 or len(det_xy) == 0:
        return {}, active_ids.copy(), list(range(len(det_xy)))

    pairs = []
    for tid in active_ids:
        tr = tracks[tid]
        dt = max(t_now - tr.last_t, 1e-6)
        x_pred = tr.last_xy + tr.last_v * dt
        for j, x_det in enumerate(det_xy):
            dist = float(np.linalg.norm(x_det - tr.last_xy))
            if dist > TRACK_MAX_DIST_M:
                continue
            disp = x_det - tr.last_xy
            ang = angle_cost(tr.last_v, disp)
            pred_dist = float(np.linalg.norm(x_det - x_pred))
            cost = W_DIST * dist + W_ANGLE * ang + W_PRED * pred_dist
            pairs.append((cost, tid, j))

    pairs.sort(key=lambda x: x[0])
    used_t, used_d = set(), set()
    matches = {}
    for cost, tid, j in pairs:
        if tid in used_t or j in used_d:
            continue
        matches[tid] = j
        used_t.add(tid)
        used_d.add(j)

    unmatched_tracks = [tid for tid in active_ids if tid not in used_t]
    unmatched_dets = [j for j in range(len(det_xy)) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets


# =========================
# A/B driving direction
# =========================
def choose_goal_dir_np(x: np.ndarray, v: np.ndarray, A_m: np.ndarray, B_m: np.ndarray) -> np.ndarray:
    toA = A_m - x
    toB = B_m - x
    dA = float(np.linalg.norm(toA))
    dB = float(np.linalg.norm(toB))
    uA = unit_np(toA)
    uB = unit_np(toB)

    sp = float(np.linalg.norm(v))
    if sp < 1e-4:
        return uB if dB < dA else uA

    denom = max(dA + dB, 1e-6)
    prefB = (dA - dB) / denom
    prefA = (dB - dA) / denom

    sB = 0.7 * float(np.dot(unit_np(v), uB)) + 0.3 * prefB
    sA = 0.7 * float(np.dot(unit_np(v), uA)) + 0.3 * prefA
    return uB if sB >= sA else uA


# =========================
# Build windows with fixed set of agents
# =========================
def build_windows(dataset: dict):
    td_w, _ = dataset["topdown_size"]
    s = REAL_WIDTH_M / float(td_w)

    A_td = np.array(dataset["destinations"]["A"]["topdown_xy"], dtype=np.float64)
    B_td = np.array(dataset["destinations"]["B"]["topdown_xy"], dtype=np.float64)
    A_m = A_td * s
    B_m = B_td * s

    frames = dataset["frames"]
    frame_ids = sorted([int(k) for k in frames.keys()])

    dets_xy = {}
    dets_t = {}
    for fid in frame_ids:
        f = frames[str(fid)]
        dets_t[fid] = float(f["t"])
        dets_xy[fid] = [np.array([p["x"] * s, p["y"] * s], dtype=np.float64) for p in f["obj"]]

    tracks: Dict[int, Track] = {}
    next_tid = 0
    active_ids: List[int] = []

    for fid in frame_ids:
        t_now = dets_t[fid]
        det_xy = dets_xy[fid]
        matches, un_tracks, un_dets = match_tracks_to_dets(tracks, active_ids, det_xy, t_now)

        new_active = []
        for tid, j in matches.items():
            tr = tracks[tid]
            x_new = det_xy[j]
            dt = max(t_now - tr.last_t, 1e-6)
            v_new = (x_new - tr.last_xy) / dt

            tr.last_frame = fid
            tr.last_t = t_now
            tr.last_xy = x_new
            tr.last_v = v_new
            tr.misses = 0
            tr.traj.append((fid, t_now, float(x_new[0]), float(x_new[1])))
            new_active.append(tid)

        for tid in un_tracks:
            tr = tracks[tid]
            tr.misses += 1
            if tr.misses <= TRACK_MAX_MISSES:
                new_active.append(tid)

        for j in un_dets:
            x0 = det_xy[j]
            tracks[next_tid] = Track(
                last_frame=fid, last_t=t_now, last_xy=x0,
                last_v=np.zeros(2, dtype=np.float64),
                misses=0,
                traj=[(fid, t_now, float(x0[0]), float(x0[1]))],
            )
            new_active.append(next_tid)
            next_tid += 1

        active_ids = new_active

    track_map: Dict[int, Dict[int, Tuple[float, np.ndarray]]] = {}
    for tid, tr in tracks.items():
        m = {}
        for fid, t, x, y in tr.traj:
            m[fid] = (t, np.array([x, y], dtype=np.float64))
        track_map[tid] = m

    frame_set = set(frame_ids)

    windows = []
    for start in frame_ids[::STRIDE]:
        end = start + K_STEPS
        if end not in frame_set:
            continue

        present = []
        for tid, m in track_map.items():
            ok = True
            for f in range(start, end + 1):
                if f not in m:
                    ok = False
                    break
            if ok:
                present.append(tid)

        if len(present) < MIN_AGENTS:
            continue

        X_obs = []
        T = []
        for f in range(start, end + 1):
            xs = []
            t = track_map[present[0]][f][0]
            for tid in present:
                _, x = track_map[present[tid == tid]][f] if False else track_map[tid][f]
                xs.append(x)
            X_obs.append(np.stack(xs, axis=0))
            T.append(t)

        X_obs = np.stack(X_obs, axis=0).astype(np.float32)  # (K+1, N, 2)
        T = np.array(T, dtype=np.float32)                   # (K+1,)
        dt = np.diff(T)                                     # (K,)

        v0 = (X_obs[1] - X_obs[0]) / max(float(dt[0]), 1e-6)
        v0 = v0.astype(np.float32)

        E = []
        for k in range(K_STEPS + 1):
            if k < K_STEPS:
                vk = (X_obs[k + 1] - X_obs[k]) / max(float(dt[k]), 1e-6)
            else:
                vk = (X_obs[k] - X_obs[k - 1]) / max(float(dt[k - 1]), 1e-6)
            ek = np.zeros_like(vk)
            for i in range(vk.shape[0]):
                ek[i] = choose_goal_dir_np(X_obs[k, i], vk[i], A_m, B_m)
            E.append(ek)
        E = np.stack(E, axis=0).astype(np.float32)

        windows.append((X_obs, v0, E, dt))
        if len(windows) >= MAX_WINDOWS:
            break

    if len(windows) == 0:
        raise RuntimeError("No usable windows. Reduce K_STEPS, reduce MIN_AGENTS, or relax tracking limits.")

    info = {
        "scale_m_per_px": s,
        "A_m": A_m,
        "B_m": B_m,
        "tracks": len(tracks),
        "windows": len(windows),
    }
    return windows, info


# =========================
# Torch SFM + rollout
# =========================
def sfm_accel_t(X, V, E, tau, A, B, v_des):
    """
    X,V,E: (N,2)
    return a: (N,2)
    """
    tau = torch.clamp(tau, min=1e-6)
    Bp = torch.clamp(B, min=1e-6)

    a_drv = (v_des * E - V) / tau

    N = X.shape[0]
    dvec = X.unsqueeze(1) - X.unsqueeze(0)            # (N,N,2)
    dist = torch.linalg.norm(dvec, dim=-1)            # (N,N)

    eye = torch.eye(N, dtype=torch.bool, device=X.device)
    valid = (~eye) & (dist > 1e-6) & (dist < MAX_NEIGHBOR_DIST_M)

    n_ij = dvec / (dist.unsqueeze(-1) + 1e-9)

    r_ij = 2.0 * RADIUS_M
    A_over_m = A / MASS_KG
    mag = A_over_m * torch.exp((r_ij - dist) / Bp)
    mag = mag * valid.float()

    a_rep = torch.sum(mag.unsqueeze(-1) * n_ij, dim=1)

    return a_drv + a_rep


def rollout_window(X0, V0, E_seq, dt_seq, params):
    """
    X0: (N,2) at t0
    V0: (N,2) at t0
    E_seq: (K+1,N,2)
    dt_seq: (K,)
    returns X_pred: (K+1,N,2), V_pred: (K+1,N,2)
    """
    tau, A, B, v_des = params
    K = dt_seq.shape[0]

    Xs = [X0]
    Vs = [V0]
    X = X0
    V = V0

    for k in range(K):
        dt = dt_seq[k]
        E = E_seq[k]

        a = sfm_accel_t(X, V, E, tau, A, B, v_des)
        V = V + dt * a
        X = X + dt * V

        Xs.append(X)
        Vs.append(V)

    return torch.stack(Xs, dim=0), torch.stack(Vs, dim=0)


# =========================
# Bounded parameters
# =========================
class ParamBox(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def u_from_x(x, lo, hi):
            x01 = unscale01(x, lo, hi)
            return inv_sigmoid(x01)

        self.u = torch.nn.Parameter(torch.tensor([
            u_from_x(START_TAU, *TAU_BOUNDS),
            u_from_x(START_A,   *A_BOUNDS),
            u_from_x(START_B,   *B_BOUNDS),
            u_from_x(START_V0,  *V0_BOUNDS),
        ], dtype=torch.float32))

    def forward(self):
        z = torch.sigmoid(self.u)
        tau = TAU_BOUNDS[0] + (TAU_BOUNDS[1] - TAU_BOUNDS[0]) * z[0]
        A   = A_BOUNDS[0]   + (A_BOUNDS[1]   - A_BOUNDS[0])   * z[1]
        B   = B_BOUNDS[0]   + (B_BOUNDS[1]   - B_BOUNDS[0])   * z[2]
        v0  = V0_BOUNDS[0]  + (V0_BOUNDS[1]  - V0_BOUNDS[0])  * z[3]
        return tau, A, B, v0


# =========================
# Loss evaluation
# =========================
def compute_window_loss(X_obs, V0, E_seq, dt, params, device):
    X0 = X_obs[0]
    X_pred, V_pred = rollout_window(X0, V0, E_seq, dt, params)

    pos_rmse = rmse_masked(
        (X_pred - X_obs).reshape(-1, 2),
        torch.ones((X_pred.numel() // 2,), device=device, dtype=torch.bool)
    )

    if W_VEL > 0:
        V_obs = (X_obs[1:] - X_obs[:-1]) / dt.view(-1, 1, 1)
        vel_rmse = rmse_masked(
            (V_pred[1:] - V_obs).reshape(-1, 2),
            torch.ones((V_obs.numel() // 2,), device=device, dtype=torch.bool)
        )
        loss = W_POS * pos_rmse + W_VEL * vel_rmse
    else:
        loss = pos_rmse

    return loss


@torch.no_grad()
def evaluate_param_set(windows, tau, A, B, v0, device, max_windows=None):
    if max_windows is not None and len(windows) > max_windows:
        idx = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
        eval_windows = [windows[i] for i in idx]
    else:
        eval_windows = windows

    params = (
        torch.tensor(float(tau), dtype=torch.float32, device=device),
        torch.tensor(float(A),   dtype=torch.float32, device=device),
        torch.tensor(float(B),   dtype=torch.float32, device=device),
        torch.tensor(float(v0),  dtype=torch.float32, device=device),
    )

    losses = []
    for X_obs_np, V0_np, E_np, dt_np in eval_windows:
        X_obs = torch.from_numpy(X_obs_np).to(device)
        V0_t  = torch.from_numpy(V0_np).to(device)
        E_seq = torch.from_numpy(E_np).to(device)
        dt    = torch.from_numpy(dt_np).to(device)

        loss = compute_window_loss(X_obs, V0_t, E_seq, dt, params, device)
        losses.append(float(loss.cpu().item()))

    return float(np.mean(losses))


def make_sensitivity_plots(windows, best_params, device):
    best_tau, best_A, best_B, best_v0 = best_params

    tau_vals = np.linspace(TAU_BOUNDS[0], TAU_BOUNDS[1], SENS_POINTS)
    A_vals   = np.linspace(A_BOUNDS[0],   A_BOUNDS[1],   SENS_POINTS)
    B_vals   = np.linspace(B_BOUNDS[0],   B_BOUNDS[1],   SENS_POINTS)
    v0_vals  = np.linspace(V0_BOUNDS[0],  V0_BOUNDS[1],  SENS_POINTS)

    tau_rmse = [
        evaluate_param_set(windows, t, best_A, best_B, best_v0, device, max_windows=SENS_MAX_WINDOWS)
        for t in tau_vals
    ]
    A_rmse = [
        evaluate_param_set(windows, best_tau, a, best_B, best_v0, device, max_windows=SENS_MAX_WINDOWS)
        for a in A_vals
    ]
    B_rmse = [
        evaluate_param_set(windows, best_tau, best_A, b, best_v0, device, max_windows=SENS_MAX_WINDOWS)
        for b in B_vals
    ]
    v0_rmse = [
        evaluate_param_set(windows, best_tau, best_A, best_B, v, device, max_windows=SENS_MAX_WINDOWS)
        for v in v0_vals
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(tau_vals, tau_rmse, linewidth=2)
    axes[0, 0].axvline(best_tau, linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Sensitivity: RMSE vs tau")
    axes[0, 0].set_xlabel("tau")
    axes[0, 0].set_ylabel("RMSE / loss")

    axes[0, 1].plot(A_vals, A_rmse, linewidth=2)
    axes[0, 1].axvline(best_A, linestyle="--", linewidth=1.5)
    axes[0, 1].set_title("Sensitivity: RMSE vs A")
    axes[0, 1].set_xlabel("A")
    axes[0, 1].set_ylabel("RMSE / loss")

    axes[1, 0].plot(B_vals, B_rmse, linewidth=2)
    axes[1, 0].axvline(best_B, linestyle="--", linewidth=1.5)
    axes[1, 0].set_title("Sensitivity: RMSE vs B")
    axes[1, 0].set_xlabel("B")
    axes[1, 0].set_ylabel("RMSE / loss")

    axes[1, 1].plot(v0_vals, v0_rmse, linewidth=2)
    axes[1, 1].axvline(best_v0, linestyle="--", linewidth=1.5)
    axes[1, 1].set_title("Sensitivity: RMSE vs v0")
    axes[1, 1].set_xlabel("v0")
    axes[1, 1].set_ylabel("RMSE / loss")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(SENS_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"[INFO] Sensitivity plot saved to: {SENS_PLOT_FILE}")


def plot_training_loss(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(loss_history) + 1), loss_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Calibration loss over epochs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRAIN_LOSS_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[INFO] Training loss plot saved to: {TRAIN_LOSS_PLOT_FILE}")


# =========================
# Train
# =========================
def main():
    set_seed(SEED)
    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")
    print("[INFO] device:", device)

    dataset = json.loads(Path(DATASET_JSON).read_text(encoding="utf-8"))
    windows, info = build_windows(dataset)

    print("[INFO] Built windows")
    print(f"       tracks={info['tracks']}, windows={info['windows']}")
    print(f"       scale(m/px)={info['scale_m_per_px']:.6f}")
    print(f"       A(m)={info['A_m']}, B(m)={info['B_m']}")

    model = ParamBox().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    prev_loss = None
    best_loss = float("inf")
    best_state = None
    loss_history = []

    W = len(windows)
    all_idx = np.arange(W)

    for epoch in range(1, EPOCHS + 1):
        if W <= BATCH_WINDOWS:
            idx = all_idx
        else:
            idx = np.random.choice(all_idx, size=BATCH_WINDOWS, replace=False)

        losses = []
        opt.zero_grad(set_to_none=True)

        for wi in idx:
            X_obs_np, V0_np, E_np, dt_np = windows[wi]

            X_obs = torch.from_numpy(X_obs_np).to(device)
            V0_t  = torch.from_numpy(V0_np).to(device)
            E_seq = torch.from_numpy(E_np).to(device)
            dt    = torch.from_numpy(dt_np).to(device)

            params = model()
            loss = compute_window_loss(X_obs, V0_t, E_seq, dt, params, device)

            loss.backward()
            losses.append(loss.detach().cpu().item())

        opt.step()

        L = float(np.mean(losses))
        loss_history.append(L)

        dL = None if prev_loss is None else (L - prev_loss)
        prev_loss = L

        if L < best_loss:
            best_loss = L
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % PRINT_EVERY == 0:
            tau, A, B, v0 = [float(x.detach().cpu().item()) for x in model()]
            if dL is None:
                print(f"epoch={epoch:4d}  loss={L:.6f}  tau={tau:.4f}  A={A:.1f}  B={B:.4f}  v0={v0:.3f}")
            else:
                print(f"epoch={epoch:4d}  loss={L:.6f}  dLoss={dL:+.6f}  tau={tau:.4f}  A={A:.1f}  B={B:.4f}  v0={v0:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    tau, A, B, v0 = [float(x.detach().cpu().item()) for x in model()]
    print("\n[RESULT] Best parameters:")
    print(f"  tau={tau:.6f}  A={A:.6f}  B={B:.6f}  v0={v0:.6f}")
    print(f"  best_loss={best_loss:.6f}")
    print("  loss = RMSE(position over rollout) + optional velocity RMSE")

    # Plot training loss
    plot_training_loss(loss_history)

    # Sensitivity analysis
    print("\n[INFO] Running sensitivity analysis ...")
    make_sensitivity_plots(
        windows=windows,
        best_params=(tau, A, B, v0),
        device=device
    )


if __name__ == "__main__":
    main()