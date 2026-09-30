#!/usr/bin/env python3
"""
polymerge.py -- merge multiple Polytopia screenshots of the same map into a
single composite showing the union of all players' explored (non-fog) area.

The pipeline is built on facts about the game's renderer (fixed isometric
camera, deterministic fog art, full NxN board, ...); they are listed with their
consequences under "What the game guarantees" in CLAUDE.md.

Usage:
  python3 polymerge.py shot1.jpg shot2.jpg shot3.jpg --map-size 20 -o merged.png \
      --debug-dir debug/

A wrong N puts every tile's fog art out of phase with the template, so the fog
test calls the whole board explored. Three checks catch it, strongest first:
check_board_size, --min-fog-lock, and CONFLICT_FRAC_SUSPECT.

--ui-mask takes per-file exclusion rectangles in each image's own pixels, for
chrome --top-crop/--bottom-crop don't cover:
  { "shot1.jpg": [[820,0,1850,225],[0,1700,2732,2048]] }

Requires: opencv-python >= 4.4, numpy
"""
import argparse, collections, contextlib, glob, itertools, json, os, time
import cv2
import numpy as np


# ------------------------------------------------------------------ timing ---
class Phases:
    """Wall-clock per pipeline phase, accumulated across repeat entries.

    A module-level instance rather than a parameter threaded through every
    call: anchoring's two interesting phases live inside anchor_to_template
    and joint_register, and this is a single-file CLI, so the global buys a
    real reduction in signature noise for no ambiguity about who owns it."""

    def __init__(self):
        self.t = collections.OrderedDict()
        self.n = collections.Counter()

    @contextlib.contextmanager
    def __call__(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
            self.n[name] += 1

    def report(self, wall):
        if not self.t:
            return
        print(f"\ntiming ({wall:.1f}s wall):")
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            calls = f" over {self.n[k]} calls" if self.n[k] > 1 else ""
            print(f"  {k:38s} {v:7.2f}s  {100 * v / wall:5.1f}%{calls}")
        acc = sum(self.t.values())
        print(f"  {'(unattributed)':38s} {wall - acc:7.2f}s  "
              f"{100 * (wall - acc) / wall:5.1f}%")


PHASES = Phases()


# ---------------------------------------------------------------- validity ---
def build_frame_mask(img, ui_rects, top_crop=0.0, bottom_crop=0.0):
    """Pixels within the photographed frame and not UI chrome. Unlike
    build_valid_mask it keeps dark pixels: this is the mask the paste uses, and
    a winning source's dark content (shadow, a tree trunk) is not a hole. See
    the mask taxonomy in CLAUDE.md.

    `top_crop`/`bottom_crop` are fractions of image height, the same for every
    input: the HUD is docked top and bottom in a band of consistent relative
    height across devices. `ui_rects` covers anything else (a mid-screen
    dialog)."""
    h, w = img.shape[:2]
    m = np.full((h, w), 255, np.uint8)
    if top_crop > 0:
        m[:int(round(h * top_crop)), :] = 0
    if bottom_crop > 0:
        m[h - int(round(h * bottom_crop)):, :] = 0
    for x0, y0, x1, y1 in ui_rects:
        m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 0
    return m


# Local std below which a neighborhood counts as featureless. 2.0, 4.0 and 8.0
# give identical results on every test set; 4.0 leaves room for JPEG noise.
SKY_SMOOTH_STD = 4.0


def sky_mask(img, std_thresh=SKY_SMOOTH_STD, k=9):
    """Pixels belonging to the empty space *behind* the board.

    The game's sunrise lightens the sky past --dark-thresh, so sky is found by
    being *empty* (local std ~0 against the board's 24-27), never by
    brightness. Smooth regions must also touch the image border, or flat
    interior terrain (sand, water, ice) is deleted too. See "Don't assume the
    sky is black" in CLAUDE.md."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (k, k))
    mean_sq = cv2.boxFilter(gray * gray, -1, (k, k))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    smooth = (std < std_thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(smooth, 8)
    h, w = smooth.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, cw, ch = stats[i, :4]
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            keep[i] = True
    return keep[lab]


def build_valid_mask(img, ui_rects, dark_thresh, erode_px, top_crop=0.0,
                     bottom_crop=0.0, drop_sky=False):
    """Pixels usable for witnessing/classification: not UI chrome, not empty
    sky, not too dark to trust for color-based fog/terrain judgment. See
    build_frame_mask for the different (and now separate) question of what
    pasting should be allowed to touch."""
    m = build_frame_mask(img, ui_rects, top_crop, bottom_crop)
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    m[v < dark_thresh] = 0
    # A last-resort fallback only (see sky_rebuild in anchor_to_template): it
    # also nibbles smooth rim and can shift a weakly supported edge.
    if drop_sky:
        m[sky_mask(img)] = 0
    if erode_px > 0:
        m = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
    return m


def _fill_enclosed_holes(mask):
    """`mask` with every fully enclosed hole filled (flood from a corner;
    whatever the flood cannot reach is a hole)."""
    h, w = mask.shape
    flood = mask.copy()
    ffm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffm, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def badge_halo(badge, found):
    """The badge mask grown to cover the glow around it, for geometry only.

    The glow passes --dark-thresh and grows a bump on the board silhouette;
    on star_change/oum2.png it cost the NW edge its support (58 px against 607
    without it) and put the shot a tile out. Dilated by the badge's own radius
    (the glow reaches ~0.7 of it), since the badge is HUD-sized per device.
    Eating a little real rim is harmless: edge_lines takes a trimmed median.
    The badge is a pin and can float over the sky beyond a rim tile, so never
    assume it overlaps the board."""
    r = max(int(round(float(np.sqrt(a / np.pi)))) for a, *_ in found)
    return cv2.dilate(badge.astype(np.uint8),
                      np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


def detect_capture_badges(img, valid, h_lo=90, h_hi=112, s_lo=70, s_hi=170,
                          v_lo=190, ring_v=235, ring_s=60,
                          min_area=1500, min_fill=0.45, min_aspect=0.55,
                          min_ring_frac=0.30):
    """Flag pixels of a capture HUD badge: a solid sky-blue disk in a glowing
    near-white ring, floated over a tile being captured. It is per-moment UI,
    so it is excluded like chrome, but it moves, so it is found by appearance.
    Thresholds are calibrated on the corpus's badges; an unseen variant may
    need them widened.

    Fog and small blue decoration icons share the hue, so color only
    nominates; shape decides:
      - `min_fill`, `min_aspect`: a near-solid disk. The white icon in its
        centre can split it into two crescents, which are rejoined by
        `_fill_enclosed_holes`, never by dilating -- a dilate large enough to
        close the notch also bridges to neighboring same-hued icons.
      - `min_ring_frac`: a near-white band around most of the perimeter.
      - `min_area`: real badges measured 3500-8300 px, the worst false
        positive (a decoration-icon cluster) 800-900 px."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    core = ((H >= h_lo) & (H <= h_hi) & (S >= s_lo) & (S <= s_hi) &
            (V >= v_lo) & (valid > 0)).astype(np.uint8) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    filled = _fill_enclosed_holes(core)
    ring = ((V >= ring_v) & (S <= ring_s)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    out = np.zeros(img.shape[:2], np.uint8)
    found = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        r = max(w, h) / 2
        fill = area / (np.pi * r * r)
        aspect = min(w, h) / max(w, h)
        if fill < min_fill or aspect < min_aspect:
            continue
        comp = (lab == i).astype(np.uint8)
        band = cv2.dilate(comp, np.ones((11, 11), np.uint8)) & ~cv2.dilate(comp, np.ones((3, 3), np.uint8))
        ring_frac = int((band & ring).sum()) / max(int(band.sum()), 1)
        if ring_frac >= min_ring_frac:
            out |= comp
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            found.append((int(area), int(x), int(y)))
    return out, found


# ------------------------------------------------------------- registration ---
def sift_features(img, mask, nfeatures, contrast):
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, mask)
    return np.array([k.pt for k in kp], np.float32), des


def pair_transform(pts_a, des_a, pts_b, des_b, ratio, reproj, terrain=None):
    """Similarity transform mapping image A coords -> image B coords, as
    (M, n_inliers, n_terrain_inliers).

    `terrain` is an optional (mask_a, mask_b) pair of boolean images marking
    pixels saturated enough that they cannot be the fog cube. When given, the
    third return value counts how many inliers land on terrain at *both* ends,
    and that is the number worth gating on -- see SIFT_TERRAIN_MIN_INLIERS.
    Without it the third value is None."""
    if des_a is None or des_b is None:
        return None, 0, None
    matches = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, 0, None
    src = np.float32([pts_a[m.queryIdx] for m in good])
    dst = np.float32([pts_b[m.trainIdx] for m in good])
    M, inl = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=reproj,
        maxIters=20000, confidence=0.999)
    if M is None or inl is None:
        return None, 0, None
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not (0.05 < scale < 20.0):        # degenerate RANSAC fit
        return None, 0, None
    n_terr = None
    if terrain is not None:
        ta, tb = terrain
        sel = inl.ravel().astype(bool)
        pa, pb = src[sel].astype(int), dst[sel].astype(int)
        ok_a = ta[np.clip(pa[:, 1], 0, ta.shape[0] - 1),
                  np.clip(pa[:, 0], 0, ta.shape[1] - 1)]
        ok_b = tb[np.clip(pb[:, 1], 0, tb.shape[0] - 1),
                  np.clip(pb[:, 0], 0, tb.shape[1] - 1)]
        n_terr = int((ok_a & ok_b).sum())
    return M, int(inl.sum()), n_terr


def to_h(M):
    H = np.eye(3)
    H[:2] = M
    return H


# ----------------------------------------------------------- board geometry ---
SIDE_CORNER_MIN_WALL = 0.5   # fraction of the local wall height a column must
                             # reach before it can define a side corner


def _side_corner(mask, x_ext, inward, look=20):
    """The top of the first full-height wall column at or inside `x_ext`.

    `inward` is +1 scanning right from the west extreme, -1 scanning left from
    the east. The reference height is the tallest column within `look` px of
    the extreme, which is the slab's wall -- anything under half of that is a
    fringe artifact rather than the board's side. See detect_corners."""
    cols = [np.count_nonzero(mask[:, x_ext + inward * d]) for d in range(look)]
    need = SIDE_CORNER_MIN_WALL * max(cols)
    d = next((d for d, h in enumerate(cols) if h >= need), 0)
    x = x_ext + inward * d
    return np.array([float(x), float(np.nonzero(mask[:, x])[0].min())])


def detect_corners(mask):
    """A board's 4 tile-grid vertices from a *complete* silhouette mask (the
    template's; a screenshot's can be clipped short of the vertex).

    The silhouette includes the slab's side wall, so a side corner is the *top*
    of the extreme column (its mean skews the lattice 3.7% vertically), taken
    from the first full-height wall column rather than an antialiasing spur
    (small-blank, tiny-blank). The near vertex is completed from the other
    three; `residual` is how far the measured one sits off horizontally."""
    ys, xs = np.where(mask > 0)
    y0 = ys.min(); top = np.array([xs[ys == y0].mean(), float(y0)])
    left = _side_corner(mask, xs.min(), +1)
    right = _side_corner(mask, xs.max(), -1)
    bottom = right + left - top
    y1 = ys.max()
    residual = float(abs(xs[ys == y1].mean() - bottom[0]))
    return top, right, bottom, left, residual


# The projection is fixed, so the edge directions are constants at every board
# size. The slope is 0.5986, NOT 3:5 (CLAUDE.md, "Don't feed the exact 3:5 angle
# into edge_lines"); only the step *lengths* come from each render's corners
# (build_lattice). The families meet at ~118 deg: never orthonormalize.
BOARD_EDGE_SLOPE = 0.5986            # rise/run of the dir_a family
BOARD_DIR_A = np.array([1.0, BOARD_EDGE_SLOPE]) / np.hypot(1.0, BOARD_EDGE_SLOPE)
BOARD_DIR_B = np.array([-BOARD_DIR_A[0], BOARD_DIR_A[1]])
# Pixel offset -> (a, b) coordinates.
BOARD_BASIS_INV = np.linalg.inv(np.stack([BOARD_DIR_A, BOARD_DIR_B], axis=1))


# A detached component is kept only if some stretch of its outline runs at a
# board angle for BOARD_COMPONENT_MIN_EDGE_RUN px: chrome is screen-aligned and
# board edges never are. The wide smoothing window is what makes it
# discriminate (the replay play button sits 1.1 deg off dir_a), so do not
# narrow it. 80 px sits between corpus chrome (<= 30) and severed board halves
# (>= 173). Alternatives tried and rejected are in CLAUDE.md
# (_board_component).
BOARD_COMPONENT_MIN_EDGE_RUN = 80    # px of outline at a board angle
BOARD_COMPONENT_MIN_AREA = 1000      # runtime prefilter: too small for such a run
BOARD_ANGLE_TOL = 5.0                # degrees
BOARD_ANGLE_SMOOTH = 35              # outline neighbors each side


def _longest_board_angle_run(one, dirs):
    """Longest contiguous stretch of `one`'s outline running at a board edge
    angle, in outline pixels. `one` is a single component's mask."""
    targets = [np.degrees(np.arctan2(d[1], d[0])) % 180.0 for d in dirs]
    k = BOARD_ANGLE_SMOOTH
    best = 0
    cs, _ = cv2.findContours(one, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cs:
        pts = c[:, 0, :].astype(np.float64)
        if len(pts) < 4 * k:
            continue
        # Tangent from neighbors k apart rather than adjacent pixels: a
        # rasterized 30.7 deg line is a staircase, so adjacent steps read 0 or 45.
        step = np.roll(pts, -k, axis=0) - np.roll(pts, k, axis=0)
        ang = np.degrees(np.arctan2(step[:, 1], step[:, 0])) % 180.0
        ok = np.zeros(len(ang), bool)
        for t in targets:
            gap = np.abs(ang - t)
            ok |= np.minimum(gap, 180.0 - gap) < BOARD_ANGLE_TOL
        if ok.all():
            best = max(best, len(ok))
            continue
        if not ok.any():
            continue
        # The outline is a closed loop, so a run can straddle index 0. Rotating
        # to start just past the last gap makes every run contiguous in the
        # rotated array, after which the runs are the spacings between gaps.
        cut = int(np.flatnonzero(~ok)[-1])
        seq = np.concatenate([ok[cut + 1:], ok[:cut + 1]])
        gaps = np.flatnonzero(~seq)
        runs = np.diff(np.concatenate([[-1], gaps])) - 1
        best = max(best, int(runs.max()))
    return best


def _board_component(m, dirs=None):
    """`m` with everything detached from the board's own silhouette removed.

    An edge line fits whatever lies furthest out, so chrome outside the board
    captures that edge and orphans the real one (fogless's turn-timeline strip,
    u_forest2's drawer tab, pol_archi_test/kick.png's banner glyphs). The
    largest component is always kept; see BOARD_COMPONENT_MIN_EDGE_RUN for the
    rest. Without `dirs` nothing is filtered (a template is one clean
    component)."""
    if dirs is None:
        return m
    nc, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if nc < 3:                            # background plus at most one region
        return m
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = [big]
    for k in range(1, nc):
        if k == big or stats[k, cv2.CC_STAT_AREA] < BOARD_COMPONENT_MIN_AREA:
            continue
        x, y, w, h = stats[k, :4]
        # The component's own padded bbox, which is what keeps this cheap.
        one = np.zeros((h + 2, w + 2), np.uint8)
        one[1:-1, 1:-1] = (lab[y:y + h, x:x + w] == k) * 255
        if _longest_board_angle_run(one, dirs) >= BOARD_COMPONENT_MIN_EDGE_RUN:
            keep.append(k)
    return np.where(np.isin(lab, keep), m, 0).astype(m.dtype)


# --- menu screenshots ----------------------------------------------------------
# A score screen is drawn over a dimmed map, anchors, and locks fog (NCC is
# illumination-invariant), so it merges silently. It is rejected because
# nothing on the board is horizontal while a menu is screen-aligned. Apply the
# HUD crop first: without it a real shot's banner and button row invert the
# margin. 0.35 is mid-gap (1.48x both ways, three score screens); do not tune
# it to catch mapless menus, which the anchor path rejects anyway. See
# board_angle_fraction in CLAUDE.md.
MENU_BOARD_ANGLE_FRAC = 0.35   # below this, the frame is not a view of the board
MENU_PROBE_WIDTH = 256         # long edge the probe downsamples to
MENU_EDGE_PCT = 92.0           # gradient magnitude percentile counted as an edge


def board_angle_fraction(img, dirs, top_crop=0.0, bottom_crop=0.0,
                         width=MENU_PROBE_WIDTH):
    """Of this frame's strong-edge energy running either at a board angle or
    with the screen, the share running at a board angle.

    ~0.6-0.9 for a screenshot of the board, ~0.2 for a menu drawn over one
    (see MENU_BOARD_ANGLE_FRAC). A ratio between the two families, so it does
    not move with how textured the shot is. ~7 ms per shot."""
    h = img.shape[0]
    sub = img[int(round(h * top_crop)): h - int(round(h * bottom_crop))]
    if sub.size == 0:
        return 1.0
    s = width / max(sub.shape[0], sub.shape[1])
    g = cv2.cvtColor(cv2.resize(sub, None, fx=s, fy=s,
                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    sel = mag >= max(float(np.percentile(mag, MENU_EDGE_PCT)), 1e-6)
    if int(sel.sum()) < 50:
        return 1.0     # nothing to judge; never reject on an absence of evidence
    # An edge runs perpendicular to its own gradient, hence the 90 deg turn.
    ang = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 90.0) % 180.0
    m = mag[sel]

    def energy_at(deg):
        gap = np.abs(ang - deg)
        return float(m[np.minimum(gap, 180.0 - gap) <= BOARD_ANGLE_TOL].sum())

    board = sum(energy_at(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
                for d in dirs)
    screen = energy_at(0.0) + energy_at(90.0)
    return board / max(board + screen, 1e-9)


def board_region(mask, dirs, open_px=15, close_px=15):
    """The board's own silhouette as a mask, chrome removed.

    board_boundary's own cleanup, exposed so SIFT looks only at the board: two
    shots of one replay have byte-identical UI, which otherwise matches
    perfectly and yields the identity transform."""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return _board_component(m, dirs)


def board_boundary(mask, dirs=None, frame_margin=10, open_px=15, close_px=15,
                   top_crop=0.0, bottom_crop=0.0):
    """The board's silhouette outline within one image, as an (N,2) point array.

    Interior holes are filled, detached chrome dropped (_board_component), and
    points along the image frame or the `top_crop`/`bottom_crop` cut lines
    dropped: those are where the photo ends, not the board, and would leave a
    small phantom support on an edge that was cropped away. Pass the same crop
    the mask was built with.

    Cleanup is an open and a close, never an erosion, which would pull the
    edge in by a zoom-dependent amount and bias the scale. The pad goes on
    after the morphology, or a close can bridge it and the flood fill leaks."""
    m = board_region(mask, dirs, open_px, close_px)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = pad.copy()
    cv2.floodFill(outside, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8),
                  (0, 0), 255)
    filled = (pad | cv2.bitwise_not(outside))[1:-1, 1:-1]
    bnd = cv2.subtract(filled, cv2.erode(filled, np.ones((3, 3), np.uint8)))
    h, w = mask.shape
    ys, xs = np.where(bnd > 0)
    top = int(round(h * top_crop))
    bot = h - int(round(h * bottom_crop))
    keep = ((xs > frame_margin) & (ys > top + frame_margin) &
            (xs < w - 1 - frame_margin) & (ys < bot - 1 - frame_margin))
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)


def _line_mode(vals, bw=2.0, smooth=3):
    """Offset of the sharpest straight line in `vals`: the peak of a 2px
    histogram. The side wall makes a ~30px band of boundary points, whose
    median would sit mid-band; the mode finds the crisp outer line, which is
    what the template's corners measure too."""
    n = int(np.ceil((vals.max() - vals.min()) / bw)) + 1
    hist, edges = np.histogram(vals, bins=n, range=(vals.min(), vals.min() + n * bw))
    hist = np.convolve(hist, np.ones(smooth) / smooth, mode="same")
    k = int(hist.argmax())
    sel = (vals >= edges[k] - bw) & (vals <= edges[k + 1] + bw)
    return float(np.median(vals[sel]))


def edge_lines(pts, windows=(40.0, 15.0, 8.0), tol=3.0):
    """Fit the four board edges as lines of known direction. Returns their
    offsets [a_min, a_max, b_min, b_max] in the oblique BOARD_DIR_A/BOARD_DIR_B
    basis, plus how many boundary points support each.

    Only the offset is unknown, estimated from the points on that edge (an
    area percentile would shrink the board a couple of percent). The support
    counts matter as much as the offsets: an edge not in frame still gets a
    phantom line, and only its support gives it away."""
    ab = pts @ BOARD_BASIS_INV.T
    coord = [ab[:, 0], ab[:, 0], ab[:, 1], ab[:, 1]]
    off = [np.percentile(ab[:, 0], 0.5), np.percentile(ab[:, 0], 99.5),
           np.percentile(ab[:, 1], 0.5), np.percentile(ab[:, 1], 99.5)]
    idx = np.arange(len(pts))
    for win in windows:                  # trimmed reassign-and-refit
        d = np.stack([np.abs(c - o) for c, o in zip(coord, off)], axis=1)
        near = d.argmin(1)
        on_edge = d[idx, near] < win
        for k in range(4):
            sel = (near == k) & on_edge
            if sel.sum() >= 20:
                off[k] = _line_mode(coord[k][sel])
    d = np.stack([np.abs(c - o) for c, o in zip(coord, off)], axis=1)
    near = d.argmin(1)
    support = [int(((near == k) & (np.abs(coord[k] - off[k]) < tol)).sum())
               for k in range(4)]
    return off, support


PIXEL_FOG_SAT = 110      # 98.8% of the all-fog template's pixels sit below this


def fogish_mask(valid, hsv, gray):
    """Pixels that *could* be fog by colour: bright and unsaturated.

    A nominator, never a classifier: mountains, snow, ice and sand pass too
    (83.5% of xizauh/pol.jpg). Use it only where a false positive costs work
    rather than an answer."""
    return (valid > 0) & (hsv[:, :, 1] < PIXEL_FOG_SAT) & (gray > 140)


# joint_register sums only the top JOINT_TOP_K tile scores, so it samples only
# tiles that could be fog by colour (JOINT_TILE_FOG_FRAC), topped up to
# JOINT_TILE_FLOOR so a fog-poor shot still has an objective. The keep-set is
# chosen once from the edge prior, never per candidate or per level. See the
# fog-colour tile prefilter and the top_k cap in CLAUDE.md.
JOINT_TOP_K = 60
JOINT_TILE_FOG_FRAC = 0.50
JOINT_TILE_FLOOR = 120


def _masked_shift_ncc(gray, fogish, dx, dy, min_overlap):
    """NCC between the image and itself shifted by (dx, dy), over pixels that
    are fog-ish at both ends of the shift."""
    h, w = gray.shape
    x0, x1 = max(0, dx), min(w, w + dx)
    y0, y1 = max(0, dy), min(h, h + dy)
    if x1 <= x0 or y1 <= y0:
        return None
    a = gray[y0:y1, x0:x1]
    b = gray[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    m = fogish[y0:y1, x0:x1] & fogish[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    if int(m.sum()) < min_overlap:
        return None
    af = a[m].astype(np.float32)
    bf = b[m].astype(np.float32)
    af -= af.mean()
    bf -= bf.mean()
    den = float(np.sqrt((af * af).sum() * (bf * bf).sum()))
    return float((af * bf).sum() / den) if den > 0 else 0.0


# How far an autocorrelation peak must rise above the valley before it to be a
# period rather than a smooth decay. Real fog peaks show 0.35+; the desert false
# positive (badland_test) had none.
PERIOD_MIN_PROMINENCE = 0.10

# The smallest peak scoring at least this is taken; a weaker earlier wobble
# does not stop the search. Every genuine period in the corpus scores >= 0.68
# (xizauh 0.68, basin_treaties 0.72, hood.png 0.92), including z2's fundamental
# that must still beat its 0.83 harmonic.
PERIOD_PEAK_MIN_NCC = 0.65


def _peak_parabola(ts, ss, k):
    if 0 < k < len(ss) - 1 and (ss[k - 1] - 2 * ss[k] + ss[k + 1]) < 0:
        return ts[k] + (ts[k] - ts[k - 1]) * (ss[k - 1] - ss[k + 1]) / (
            2 * (ss[k - 1] - 2 * ss[k] + ss[k + 1]))
    return float(ts[k])


def fog_period_scale(gray, valid, hsv, dir_a, tile_px,
                     lo=0.30, hi=3.75, min_ncc=0.50):
    """The image's zoom, read off the fog art's own repeat period.

    Fog is one fixed render per tile, so autocorrelating the fog region along
    dir_a peaks at one tile step in this image's pixels; zoom = template step /
    measured step. Never match fog patches against the template instead: a
    wrong scale correlates with a different repeat and returns a confident
    wrong answer. The smallest strong peak is taken, so a 2x harmonic cannot
    win while the fundamental is inside [lo, hi]. Those bounds come from the
    zoom extremes in tests/ (missized_test/z2.png at 40.4px,
    basin_treaties/q.png at 202px) with margin; `lo` also needs a few coarse
    steps of runway for the prominence test.

    Coarse sweep at quarter resolution, then a full-resolution parabolic
    refinement (~0.2% accurate). Used as the zoom prior only when a shot has no
    opposite edge pair; always used as the denominator of the board-size count.
    Returns (s_it, period_px, ncc), or None if no periodic fog is found."""
    fogish = fogish_mask(valid, hsv, gray)
    q = 4
    gq = cv2.resize(gray, (gray.shape[1] // q, gray.shape[0] // q),
                    interpolation=cv2.INTER_AREA)
    fq = cv2.resize(fogish.astype(np.uint8),
                    (gq.shape[1], gq.shape[0]),
                    interpolation=cv2.INTER_AREA) > 0
    t_lo, t_hi = tile_px * lo, tile_px * hi
    coarse = []
    for t in np.arange(t_lo / q, t_hi / q + 1e-9, 1.0):
        dx, dy = int(round(t * dir_a[0])), int(round(t * dir_a[1]))
        s = _masked_shift_ncc(gq, fq, dx, dy, min_overlap=2000)
        if s is not None:
            coarse.append((t * q, s))
    if not coarse:
        return None
    ts = np.array([t for t, _ in coarse])
    ss = np.array([s for _, s in coarse])
    if float(ss.max()) < min_ncc:
        return None
    # A genuine peak, not just a high score: a smooth region's autocorrelation
    # decays from the shortest shift (badland_test read 35px for 118px).
    peaks = [i for i in range(1, len(ss) - 1)
             if ss[i] >= ss[i - 1] and ss[i] > ss[i + 1] and ss[i] >= min_ncc
             and ss[i] - float(ss[:i].min()) >= PERIOD_MIN_PROMINENCE]
    if not peaks:
        return None
    # Smallest *strong* peak (PERIOD_PEAK_MIN_NCC), else the smallest of all.
    strong = [i for i in peaks if ss[i] >= PERIOD_PEAK_MIN_NCC]
    k = strong[0] if strong else peaks[0]  # fundamental, not a harmonic
    t0 = ts[k]
    fine = []
    for t in np.arange(t0 - q - 1, t0 + q + 1 + 1e-9, 1.0):
        dx, dy = int(round(t * dir_a[0])), int(round(t * dir_a[1]))
        s = _masked_shift_ncc(gray, fogish, dx, dy, min_overlap=30000)
        if s is not None:
            fine.append((t, s))
    if len(fine) >= 3:
        fts = np.array([t for t, _ in fine])
        fss = np.array([s for _, s in fine])
        period = _peak_parabola(fts, fss, int(fss.argmax()))
        score = float(fss.max())
    else:
        # The chosen peak's score, not ss.max(): the fundamental need not be
        # the strongest (z2.png: 0.68 against its harmonic's 0.83).
        period, score = float(t0), float(ss[k])
    if score < min_ncc:
        return None
    return tile_px / period, period, score


# ------------------------------------------------------- joint registration ---
def _tile_sample_grid(origin, u_col, u_row, n, inset, div, max_px=320):
    """Integer sample coordinates (X, Y arrays of shape (n*n, px)) covering
    every tile's inset interior, at 1/div resolution.

    One rasterized tile mask is reused for every tile, which is what makes
    searching hundreds of (zoom, dx, dy) candidates affordable. Each tile keeps
    at most max(max_px // div, 160) samples (160 / 160 / 320 at div 4 / 2 / 1):
    the cap must shrink with `div` or the coarse levels cost more than the fine
    one, and the 160 floor protects shots with little fog (CLAUDE.md,
    "Running it", timing)."""
    uc, ur, org = u_col / div, u_row / div, origin / div
    base = tile_poly(np.zeros(2), uc, ur, 0, 0, inset)
    corner = base.min(0)
    base = base - corner
    w = int(np.ceil(base[:, 0].max())) + 1
    h = int(np.ceil(base[:, 1].max())) + 1
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.round(base).astype(np.int32), 1)
    ry, rx = np.nonzero(m)
    cap = max(max_px // div, 160) if max_px else 0
    if cap and rx.size > cap:
        # Evenly spread, not every step'th: striding ties the count to the
        # render's px/tile instead of the board (CLAUDE.md, anchoring, "Cause 2").
        idx = np.linspace(0, rx.size - 1, cap).round().astype(np.int64)
        rx, ry = rx[idx], ry[idx]
    ij = np.array([(i, j) for i in range(n) for j in range(n)], np.float64)
    tops = org + ij[:, 0:1] * uc + ij[:, 1:2] * ur + corner
    X = np.round(tops[:, 0]).astype(np.int64)[:, None] + rx[None, :].astype(np.int64)
    Y = np.round(tops[:, 1]).astype(np.int64)[:, None] + ry[None, :].astype(np.int64)
    return X, Y


def _fog_alignment_score(img_flat, tmpl_vals, vmask_flat, base, shape, off,
                         top_k, min_valid):
    """Sum of the top_k per-tile correlations with the template's fog art.

    Only the most fog-like tiles count, since explored terrain scores ~0 and
    would bury the signal. This is the program's inner loop. `tmpl_vals` is
    the template pre-gathered at the sample points, which is the same for
    every call in a pyramid level.

    Gathers use a flat take (7.3x faster than a 2-D fancy index): `base` is
    the flat index at zero pan and a pan is the scalar `off` = dy*width + dx."""
    idx = base + off
    a = img_flat.take(idx).astype(np.float32).reshape(shape)
    b = tmpl_vals
    v = (vmask_flat.take(idx).reshape(shape) > 0).astype(np.float32)
    cnt = v.sum(1)
    keep = cnt >= min_valid * shape[1]
    if not keep.any():
        return -1.0
    a, b, v, cnt = a[keep], b[keep], v[keep], cnt[keep]
    ac = (a - ((a * v).sum(1) / cnt)[:, None]) * v
    bc = (b - ((b * v).sum(1) / cnt)[:, None]) * v
    den = np.sqrt((ac * ac).sum(1) * (bc * bc).sum(1))
    ncc = np.where(den > 1e-6, (ac * bc).sum(1) / np.maximum(den, 1e-6), 0.0)
    k = min(top_k, ncc.size)
    return float(np.sort(ncc)[-k:].sum())


def _fog_full_score(img_flat, bc_pre, bden, base, shape, off, top_k):
    """`_fog_alignment_score` for tiles with no invalid sample, in closed form.

    The masked form must re-centre the template side at every pan, because
    which samples are valid moves with the pan. For an all-valid tile the
    centring is pan-independent, so `bc_pre = b - mean(b)` and `bden =
    |bc_pre|` come precomputed per level, sum(ac*bc) reduces to sum(a*bc_pre),
    and sum(ac^2) to sum(a^2) - sum(a)^2/n: ~7x faster. Agrees with the masked
    form to ~1e-5 relative, not bit-for-bit. See joint_register for where it
    is used."""
    a = img_flat.take(base + off).astype(np.float32).reshape(shape)
    sa = a.sum(1)
    den = np.sqrt(np.maximum((a * a).sum(1) - sa * sa / shape[1], 0.0)) * bden
    ncc = np.where(den > 1e-6, (a * bc_pre).sum(1) / np.maximum(den, 1e-6), 0.0)
    k = min(top_k, ncc.size)
    return float(np.sort(ncc)[-k:].sum())


# (div, zoom half-span, zoom step, pan radius in div-px, pan step in div-px,
#  how many zoom candidates this level hands to the next).
# Each finer span is deliberately wider than the coarser level's step: the
# coarse optimum is not reliably within one step of the truth. The beam width
# tapers 3, 2, 1 because a coarse level cannot tell its best candidates apart;
# do not narrow it, least of all before div=1. See CLAUDE.md, "Don't drop or
# narrow the div=1 level" and the anchoring section's "Cause 1".
JOINT_LEVELS = ((4, 0.030, 0.010, 6, 2, 3),
                (2, 0.010, 0.003, 3, 1, 2),
                (1, 0.003, 0.001, 2, 1, 1))


def _prune_beam(cand, width, min_sep):
    """The `width` best candidates, no two within `min_sep` in zoom.

    Plain top-N would return N neighbors of one optimum."""
    kept = []
    for c in sorted(cand, reverse=True):
        if all(abs(c[1] - k[1]) >= min_sep - 1e-9 for k in kept):
            kept.append(c)
            if len(kept) >= width:
                break
    return kept or [max(cand)]


def _fogish_tiles(img, valid, gray, origin, u_col, u_row, n, Wt, Ht,
                  scale0, trans0):
    """Which of the n*n tiles joint_register should bother scoring.

    Kept: tiles with at least JOINT_TILE_FOG_FRAC of samples fog-ish under the
    edge-derived prior, topped up to JOINT_TILE_FLOOR. Returns (keep,
    n_fogish), the latter counted before the top-up. On 71 of 74 corpus shots
    no excluded tile goes on to lock fog; an excluded tile only loses its vote
    on the anchor, never its place in the merge."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fogish = fogish_mask(valid, hsv, gray).astype(np.uint8)
    M = np.array([[scale0, 0.0, trans0[0]], [0.0, scale0, trans0[1]]])
    warped = cv2.warpAffine(fogish, M, (Wt, Ht), flags=cv2.INTER_NEAREST)
    pad = 16
    warped = cv2.copyMakeBorder(warped, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, 0)
    X, Y = _tile_sample_grid(origin, u_col, u_row, n, 0.25, 1)
    Xc = np.clip(X + pad, 0, warped.shape[1] - 1)
    Yc = np.clip(Y + pad, 0, warped.shape[0] - 1)
    frac = warped[Yc, Xc].mean(1)
    keep = frac >= JOINT_TILE_FOG_FRAC
    n_fogish = int(keep.sum())
    if n_fogish < JOINT_TILE_FLOOR:
        keep = np.zeros(frac.size, bool)
        keep[np.argsort(frac)[-min(JOINT_TILE_FLOOR, frac.size):]] = True
    return keep, n_fogish


def joint_register(img, valid, tmpl_gray, origin, u_col, u_row, n,
                   scale0, trans0, top_k=JOINT_TOP_K, min_valid=0.5):
    """Refine zoom and pan *together*, coarse to fine, against the fog artwork.

    A zoom error shows up as a pan error and vice versa, so they are searched
    as one product, coarse to fine over JOINT_LEVELS. Returns the image ->
    template affine."""
    Ht, Wt = tmpl_gray.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keep, n_fogish = _fogish_tiles(img, valid, gray, origin, u_col, u_row, n,
                                   Wt, Ht, scale0, trans0)
    # Never sum more tiles than could be fog: on a fog-poor shot the rest are
    # noise terms that move with the candidate. The colour count is a generous
    # upper bound, so it errs toward leaving k alone.
    top_k = min(top_k, max(n_fogish, 1))
    # The zoom sweep pivots about the board's centre (see below).
    board_c = origin + (n / 2.0) * u_col + (n / 2.0) * u_row
    beam = [(-2.0, float(scale0), float(trans0[0]), float(trans0[1]))]
    for div, s_span, s_step, p_rad, p_step, emit in JOINT_LEVELS:
      # Timed per level: a single total would hide which level a change moved.
      with PHASES(f"anchor: refine pyramid div={div}"):
          tw, th = Wt // div, Ht // div
          tgt = tmpl_gray if div == 1 else cv2.resize(tmpl_gray, (tw, th),
                                                      interpolation=cv2.INTER_AREA)
          small = gray if div == 1 else cv2.resize(
              gray, (gray.shape[1] // div, gray.shape[0] // div),
              interpolation=cv2.INTER_AREA)
          small_v = valid if div == 1 else cv2.resize(
              valid, (valid.shape[1] // div, valid.shape[0] // div),
              interpolation=cv2.INTER_NEAREST)
          X, Y = _tile_sample_grid(origin, u_col, u_row, n, 0.25, div)
          # Same (i, j) row order at every div, so one boolean selects the same
          # tiles throughout.
          X, Y = X[keep], Y[keep]
          pad = p_rad + 4
          tgtp = cv2.copyMakeBorder(tgt, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
          Xp, Yp = X + pad, Y + pad
          # Invariant across every zoom and pan tried at this level; see
          # _fog_alignment_score.
          tvals = tgtp[Yp, Xp].astype(np.float32)
          # Flat sample addresses for this level; see _fog_alignment_score.
          # tgtp/wgp/wvp all share this padded width, so one base serves all.
          pw = tgtp.shape[1]
          base = (Yp.astype(np.int64) * pw + Xp.astype(np.int64)).astype(np.int32)
          shape = Xp.shape
          # Tiles with no invalid sample anywhere in this level's search are
          # scored in closed form (_fog_full_score). Coarse levels only: they
          # are ~69% of the scoring cost, and div=1 stays exact so the final
          # answer is chosen at full fidelity.
          fast = div > 1
          if fast:
              bc_pre = tvals - tvals.mean(1)[:, None]
              bden = np.sqrt((bc_pre * bc_pre).sum(1))
              pan_se = np.zeros((2 * p_rad + 1, 2 * p_rad + 1), np.uint8)
              pan_se[::p_step, ::p_step] = 1
          zooms = []
          seen = set()
          for _, s_cur, tx0, ty0 in beam:
              # This beam entry's board centre, in the shot's own pixels.
              piv = (board_c - np.array([tx0, ty0])) / s_cur
              # A zoom grid shared by every beam entry, so overlapping sweeps
              # hit identical zooms and the repeats are skipped below.
              lo = int(np.floor((s_cur - s_span) / s_step))
              hi = int(np.ceil((s_cur + s_span) / s_step))
              for q in range(lo, hi + 1):
                  s = q * s_step
                  # s * p + t scales about the image origin, which would swing
                  # the board further than the pan sweep reaches; hold the
                  # board centre fixed instead (CLAUDE.md, zoom-pivot decision).
                  tx = tx0 + (s_cur - s) * piv[0]
                  ty = ty0 + (s_cur - s) * piv[1]
                  # Same zoom *and* same pan origin means the identical set of
                  # (dx, dy) probes -- nothing new to learn from repeating it.
                  key = (q, round(tx, 3), round(ty, 3))
                  if key in seen:
                      continue
                  seen.add(key)
                  zooms.append((s, tx, ty))

          def _warp_valid(s, tx, ty):
              """This candidate's validity mask, in padded template space."""
              M = np.array([[s, 0.0, tx / div], [0.0, s, ty / div]])
              wv = cv2.warpAffine(small_v, M, (tw, th), flags=cv2.INTER_NEAREST)
              return cv2.copyMakeBorder(wv, pad, pad, pad, pad,
                                        cv2.BORDER_CONSTANT, 0)

          # The fast path's tile set is fixed once per level: the tiles valid
          # at every pan (erosion by the pan grid) under every zoom candidate.
          # Per candidate, a larger set would win on top-k for reasons
          # unrelated to alignment. borderValue 0 so off-canvas reads invalid.
          sel = None
          if fast:
              for s, tx, ty in zooms:
                  full = (cv2.erode(_warp_valid(s, tx, ty), pan_se,
                                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
                          .ravel().take(base).reshape(shape) > 0).all(1)
                  sel = full if sel is None else (sel & full)
                  # The intersection only shrinks, so stop once it is short.
                  if int(sel.sum()) < JOINT_TOP_K:
                      sel = None
                      break
          if sel is not None:
              # Reached only if at least JOINT_TOP_K (the constant, not this
              # shot's top_k) tiles qualify; otherwise the whole level stays on
              # the exact score. Never mix the two forms within a level.
              bcp_s, bden_s = bc_pre[sel], bden[sel]
              base_s, shape_s = base[sel], (int(sel.sum()), shape[1])

          cand = []
          for s, tx, ty in zooms:
              # p_template = s * p_image + t, for div-downscaled inputs
              M = np.array([[s, 0.0, tx / div], [0.0, s, ty / div]])
              wg = cv2.warpAffine(small, M, (tw, th), flags=cv2.INTER_LINEAR)
              wgp = cv2.copyMakeBorder(wg, pad, pad, pad, pad,
                                       cv2.BORDER_CONSTANT, 0)
              # ravel of a contiguous array is a view, so this is free
              wgp_f = wgp.ravel()
              # Only the exact path reads validity; re-warping is cheaper than
              # keeping every candidate's mask from the pre-pass.
              wvp_f = None if sel is not None else _warp_valid(s, tx, ty).ravel()
              for dx in range(-p_rad, p_rad + 1, p_step):
                  for dy in range(-p_rad, p_rad + 1, p_step):
                      off = dy * pw + dx
                      sc = (_fog_alignment_score(
                                wgp_f, tvals, wvp_f, base, shape,
                                off, top_k, min_valid)
                            if sel is None else
                            _fog_full_score(wgp_f, bcp_s, bden_s, base_s,
                                            shape_s, off, top_k))
                      # sampling the image at +dx means the matching content
                      # sits dx to the right, so the image moves by -dx
                      cand.append((sc, float(s),
                                   tx - dx * div, ty - dy * div))
          beam = _prune_beam(cand, emit, s_step)
    return np.array([[beam[0][1], 0.0, beam[0][2]],
                     [0.0, beam[0][1], beam[0][3]]])


# Span / fog period overstates N by the slab's side wall: [N+0.61, N+0.90] over
# 30 edge-pair shots, the same at 18x18 and 20x20.
BOARD_SPAN_WALL_TILES = 0.78

# The game's names for its board sizes, which the Overlays/ renders use.
BOARD_SIZE_NAMES = {11: "tiny", 14: "small", 16: "normal",
                    18: "large", 20: "huge", 30: "massive"}

# 30x30 is held back although its render works: its 2x harmonic lands near 14
# and 16 (CLAUDE.md, "30x30 is deferred").
MAP_SIZE_DEFERRED = (30,)
MAP_SIZE_CHOICES = tuple(n for n in sorted(BOARD_SIZE_NAMES)
                         if n not in MAP_SIZE_DEFERRED)


def size_list():
    """The board sizes, phrased for a player: every message using this reaches
    a Discord channel, so it must never name --map-size. Same comma-then-"or"
    shape as polybot's unrecognized-size reply."""
    return (", ".join(str(n) for n in MAP_SIZE_CHOICES[:-1])
            + f" or {MAP_SIZE_CHOICES[-1]}")


# The remedy every "could not work the size out" refusal ends with, kept in one
# place so they cannot drift. Not used where the size is known but unsupported.
# No full stop: one caller continues the sentence.
RESTATE_SIZE = "Please retry including the map size ({})"

# How far shots' measured board sizes may disagree before none is believed:
# catches two different boards in one merge. The worst intra-set spread is
# 0.36 (badland_test2), so the headroom is only 1.4x -- see the deferred item
# in CLAUDE.md on the period's phase sensitivity.
MAP_SIZE_DETECT_MAX_SPREAD = 0.5

# A measurement further than this from every real size describes no board and
# is dropped -- the number, not the shot, since the anchor comes from separate
# evidence. Honest readings sit <= 0.32 out, the z2 harmonic was 9.4; under 1.0
# keeps a neighboring size plausible so two boards still reach the spread check.
MAP_SIZE_PLAUSIBLE_TOL = 0.9


def plausible_sizes(implied):
    """Split {name: tiles-across} into the measurements that could describe a
    board and those that could not, as (kept, discarded).

    Callers report the discarded ones and carry on with `kept`."""
    kept, discarded = {}, {}
    for n, v in implied.items():
        near = min(abs(v - N) for N in MAP_SIZE_CHOICES)
        (kept if near <= MAP_SIZE_PLAUSIBLE_TOL else discarded)[n] = v
    return kept, discarded


def implausible_note(discarded):
    """One line naming measurements dropped by plausible_sizes, or ''."""
    if not discarded:
        return ""
    detail = "  ".join(f"{n}={v:.2f}" for n, v in sorted(discarded.items()))
    return (f"  ignoring {len(discarded)} measurement(s) too far from any real "
            f"board size to be one ({detail}); the shot(s) still merge -- the "
            f"count and the anchor are separate measurements")

# SIFT inliers, counted on *terrain* at both ends, before a match may supply a
# zoom, a pan offset or a whole anchor. Raw counts cannot separate fog matching
# the wrong repeat of itself (115 raw) from a genuine match (108); on terrain
# those read 0 and 100.
SIFT_TERRAIN_MIN_INLIERS = 60

# How close (in tiles, as --cross-check measures) a SIFT-corroborated shot must
# sit to its own anchor to escape drop_misanchored. Correct sets read well
# under it; the failures it catches are a few percent of the board. No corpus
# set reaches this path: exercise changes deliberately.
MISANCHOR_CORROBORATE_MAX_TILES = 0.20

# --cross-check flags a pair past this many tiles apart. Healthy sets sit
# under ~0.05; see CLAUDE.md for the pairs that exceed it without being wrong.
CROSS_CHECK_FLAG_TILES = 0.10

# How far the shots' own tile count may sit from a *stated* --map-size before
# the run is refused as the wrong size. Every honest measurement in the corpus
# lands within 0.32 of the truth and neighbouring sizes are 2 apart.
MAP_SIZE_STATED_TOL = 0.4

# How far two opposite edge pairs may disagree on the zoom before neither is
# believed. Corpus spreads run 0.00-1.24%, the one outlier 25.99%; anything in
# ~2-10% behaves identically.
EDGE_PAIR_MAX_SPREAD = 0.03


# A lone pair with a weak side is believed only if its span matches the board's
# observed extent the other way (the board is square; the extent is a lower
# bound). Clears the nearest genuine pair by 3.5% and nearest phantom by 8%, on
# only four genuine samples. Consulted only when min_support already failed.
LONE_PAIR_EXTENT_LO = 0.95
LONE_PAIR_EXTENT_HI = 1.15


def anchor_to_template(img, mask, valid, hsv, tmpl_gray, t_off, dir_a, dir_b,
                       origin, u_col, u_row, n_tiles, min_support, label,
                       refine=True, min_scale_support=30, zoom_hint=None,
                       sky_rebuild=None, pan_hint=None, cache=None,
                       top_crop=0.0, bottom_crop=0.0):
    """Map one image onto the template.

    Board edges give the prior for zoom and pan; joint_register refines both
    against the fog art. Zoom comes from an opposite edge pair (a pair checks
    itself, hence `min_scale_support` << `min_support`), else `zoom_hint`
    (SIFT), else fog_period_scale. Each edge pins one of the two pan offsets,
    so pan needs one edge per *direction*; an opposite pair pins the same one
    twice, and fog cannot supply it (it is periodic). `pan_hint`, a SIFT affine
    from an anchored shot, supplies the missing direction only; the caller must
    already have cleared SIFT_TERRAIN_MIN_INLIERS, the only gate on it.

    Returns (affine, zoom_source, implied_n, prior): zoom_source is "edges",
    "sift" or "fog-period"; implied_n is this shot's own board-size count or
    None (check_board_size); `prior` is the unrefined transform, or None when
    refinement was skipped, kept so fog_lock_guards can undo a refinement.

    `top_crop`/`bottom_crop` must match what `mask` was built with (see
    board_boundary)."""
    # t_off is the template's edges fitted the same way, so the side wall
    # cancels -- exactly only when this shot's rim is also fog.
    span = [t_off[1] - t_off[0], t_off[3] - t_off[2]]

    # A private cache still dedups this call's own repeats; see ShotCache.
    cache = cache if cache is not None else ShotCache()

    tags = ["a-min", "a-max", "b-min", "b-max"]
    basis_inv = np.linalg.inv(np.stack([dir_a, dir_b], axis=1))

    def fit_edges(m):
        with PHASES("anchor: board outline + edge fit"):
            pts = cache.boundary(label, m, (dir_a, dir_b), top_crop, bottom_crop)
            if len(pts) < 100:
                return pts, None, None, False
            off, support = edge_lines(pts)
        have = [c >= min_support for c in support]
        print(f"  board edges ({len(pts)} boundary px): " + "  ".join(
            f"{t}={'ok' if h else 'MISSING'}({c})"
            for t, h, c in zip(tags, have, support)))
        return pts, off, support, (have[0] or have[1]) and (have[2] or have[3])

    pts, off, support, pan_ok = fit_edges(mask)
    # Sunrise fallback (sky_mask), only on failure: it costs a black-sky shot
    # nothing here, and applied to every shot it regressed test_ss_2.
    if not pan_ok and sky_rebuild is not None:
        print("  no usable board edge from brightness alone -- retrying with "
              "the sunrise-sky test")
        mask, valid = sky_rebuild()
        # The masks just changed, so earlier measurements are stale.
        cache.invalidate(label)
        pts, off, support, pan_ok = fit_edges(mask)
    if off is None:
        raise SystemExit(f"{label}: no board outline found (only {len(pts)} "
                         f"boundary px) -- is any board edge actually in frame?")
    have = [c >= min_support for c in support]
    have_dir = [have[0] or have[1], have[2] or have[3]]
    # Borrowing still needs one real edge: with none, the shot's whole position
    # would come from another shot.
    can_borrow = pan_hint is not None and any(have_dir)
    if not pan_ok and not can_borrow:
        seen = " ".join(t for t, h in zip(tags, have) if h) or "none"
        raise SystemExit(f"{label}: pan needs one board edge from each direction, "
                         f"only found [{seen}] -- merge it with a shot that shows "
                         f"more of the board.")
    # True only when an offset is actually taken. Every shot re-anchored in
    # anchor_all's second pass gets a hint, and one with both directions must
    # keep its own edges and refinement (conflating them once cost
    # pol_archi_test/kick.png its whole fog lock).
    borrow_pan = can_borrow and not pan_ok

    have_scale = [c >= min_scale_support for c in support]
    edge_scales = [(off[hi] - off[lo]) / span[k]
                   for k, (lo, hi) in enumerate([(0, 1), (2, 3)])
                   if have_scale[lo] and have_scale[hi]]
    # A lone pair has nothing to cross-check it, so a weak side must pass the
    # extent check (LONE_PAIR_EXTENT_LO/HI; CLAUDE.md, "A lone edge pair needs
    # both sides properly supported").
    if len(edge_scales) == 1:
        lo, hi = (0, 1) if (have_scale[0] and have_scale[1]) else (2, 3)
        weak = min(support[lo], support[hi]) < min_support
        ratio = None
        if weak:
            k = 0 if lo == 0 else 1
            reg = board_region(mask, (dir_a, dir_b))
            ys, xs = np.where(reg > 0)
            if len(xs) >= 100:
                ab = np.stack([xs, ys], axis=1) @ basis_inv.T
                other = float(ab[:, 1 - k].max() - ab[:, 1 - k].min())
                if other > 1.0:
                    ratio = (off[hi] - off[lo]) / other
        corroborated = (ratio is not None
                        and LONE_PAIR_EXTENT_LO <= ratio <= LONE_PAIR_EXTENT_HI)
        if weak and not corroborated:
            why = ("no board extent to check it against" if ratio is None
                   else f"its span is {ratio:.3f} of the board's width the "
                        f"other way, which a square board rules out")
            print(f"  sole edge pair too weak to trust on its own "
                  f"(support {support[lo]}/{support[hi]}, need {min_support}; "
                  f"{why}) -- looking for another zoom source")
            edge_scales = []
        elif weak:
            print(f"  sole edge pair believed on a weak side (support "
                  f"{support[lo]}/{support[hi]}): its span is {ratio:.3f} of "
                  f"the board's width the other way, and a square board makes "
                  f"those match")
    elif len(edge_scales) == 2:
        # Disagreeing pairs mean a phantom edge; averaging would split the
        # difference (test_ss_elyruins/hood.png: 25.99%).
        spread = abs(edge_scales[0] / edge_scales[1] - 1)
        if spread > EDGE_PAIR_MAX_SPREAD:
            print(f"  two edge pairs disagree by {100 * spread:.2f}% "
                  f"(max {100 * EDGE_PAIR_MAX_SPREAD:.0f}%) -- at least one "
                  f"edge is a phantom, looking for another zoom source")
            edge_scales = []
    implied_n = None
    if edge_scales:
        zoom_source = "edges"
        s = float(np.mean(edge_scales))
        print(f"  zoom prior from {len(edge_scales)} edge pair(s): {s:.4f}"
              + (f" (pair spread {100 * abs(edge_scales[0] / edge_scales[1] - 1):.2f}%)"
                 if len(edge_scales) == 2 else ""))
        # A shot spanning the board can count its tiles: span / fog period
        # (check_board_size).
        with PHASES("anchor: board-size check"):
            got = cache.period(label, img, valid, hsv, dir_a,
                               np.linalg.norm(u_col))
        if got is not None:
            spans = [off[hi] - off[lo] for k, (lo, hi) in enumerate([(0, 1), (2, 3)])
                     if have_scale[lo] and have_scale[hi]]
            implied_n = float(np.mean(spans)) / got[1] - BOARD_SPAN_WALL_TILES
            print(f"  board size implied by span/fog-period: {implied_n:.2f} tiles")
    elif zoom_hint is not None:
        # SIFT against an anchored shot (~0.2px) beats a period that needs a
        # real expanse of fog.
        zoom_source = "sift"
        s = 1.0 / zoom_hint
        print(f"  zoom prior from SIFT against an anchored shot "
              f"(no edge pair available): {s:.4f}")
    else:
        zoom_source = "fog-period"
        with PHASES("anchor: fog-period zoom fallback"):
            got = cache.period(label, img, valid, hsv, dir_a,
                               np.linalg.norm(u_col))
        if got is None:
            raise SystemExit(f"{label}: no opposite edge pair, and no periodic "
                             f"fog found to read the zoom off -- pale terrain "
                             f"(sand, snow, mountains, city roofs) is bright "
                             f"and unsaturated like fog but does not repeat "
                             f"tile-to-tile the way fog does.")
        s_from_period, period, ncc = got
        s = 1.0 / s_from_period
        print(f"  zoom prior from fog repeat period ({period:.1f}px at "
              f"ncc {ncc:.2f}, no edge pair available): {s:.4f}")

    # s is image px per template px; the image -> template transform scales by
    # its reciprocal.
    s_it = 1.0 / s
    # The hint's translation as per-direction offsets. The caller passes the
    # same transform's scale as zoom_hint, so it is already at s_it.
    if borrow_pan:
        hint_shift = basis_inv @ np.array([pan_hint[0, 2], pan_hint[1, 2]])
    # Blend a pair's edges by support, although the bottom lip is more accurate
    # on its own: preferring it made the merge worse (CLAUDE.md, "A better prior
    # is not a better answer").
    shift = []
    for d, (lo, hi) in enumerate(((0, 1), (2, 3))):
        cand = [(support[k], t_off[k] - s_it * off[k]) for k in (lo, hi) if have[k]]
        if cand:
            shift.append(sum(c[0] * c[1] for c in cand) / sum(c[0] for c in cand))
        else:
            shift.append(float(hint_shift[d]))
            print(f"  no {'ab'[d]}-direction board edge in frame -- taking that "
                  f"one offset from the SIFT hint, keeping the "
                  f"{'ba'[d]}-direction edge")
    if borrow_pan:
        # Free corroboration: compare the kept edge with the hint.
        for d in range(2):
            if have_dir[d]:
                print(f"  {'ab'[d]}-direction edge sits "
                      f"{shift[d] - hint_shift[d]:+.1f}px from where the SIFT "
                      f"hint puts it")
    trans = shift[0] * dir_a + shift[1] * dir_b
    if borrow_pan or not refine:
        # Never refine a borrowed pan: such a shot has barely any fog, so
        # refining walks a good SIFT prior into noise (badland_test3/cym.png:
        # 0.006 -> 0.212 tiles).
        why = ("SIFT geometry, unrefined -- too little fog to refine against"
               if borrow_pan else "edges only, unrefined")
        print(f"  {label} -> template: scale={s_it:.4f} ({why})")
        return (np.array([[s_it, 0.0, trans[0]], [0.0, s_it, trans[1]]]),
                zoom_source, implied_n, None)
    prior = np.array([[s_it, 0.0, trans[0]], [0.0, s_it, trans[1]]])
    M = joint_register(img, valid, tmpl_gray, origin, u_col, u_row, n_tiles,
                       s_it, trans)
    moved = float(np.hypot(M[0, 2] - trans[0], M[1, 2] - trans[1]))
    print(f"  {label} -> template: scale={M[0, 0]:.4f} "
          f"({100 * (M[0, 0] / s_it - 1):+.2f}% vs prior), pan moved {moved:.1f}px "
          f"(rotation not fitted -- the camera never rotates)")
    return M, zoom_source, implied_n, prior


# ------------------------------------------------------------ alignment QA ---
# Fraction of *comparable* tiles (two witnesses; not the whole board, which
# dilutes it) that conflict, above which a stated size is probably wrong: an
# out-of-phase lattice puts fog on other sources' terrain. It only ever rules a
# size out, so a low value gets no message. Populations in CLAUDE.md (the fog
# lock adjudication item).
CONFLICT_FRAC_SUSPECT = 0.18
CONFLICT_FRAC_MIN_COMPARABLE = 20


def cross_check(anchors, sift_edges, t_corners, tile_px):
    """How far independently-anchored shots disagree with each other, in tiles.

    Anchor A and B separately, hop A -> B -> template via SIFT (good to
    ~0.2px), and measure how far that lands from A's own anchor at the
    template corners. Any gap is anchor error. Meaningless for shots whose
    explored regions do not overlap (see CLAUDE.md)."""
    dst = np.float32(t_corners).reshape(-1, 1, 2)
    out = []
    for (a, b), (M_ab, n_inl, _terr) in sift_edges.items():
        if M_ab is None or a not in anchors or b not in anchors:
            continue
        back = np.linalg.inv(to_h(anchors[a]))            # template -> a
        via = to_h(anchors[b]) @ to_h(M_ab) @ back        # template -> a -> b -> template
        got = cv2.transform(dst, via[:2]).reshape(-1, 2)
        err = float(np.max(np.linalg.norm(got - np.float32(t_corners), axis=1)))
        out.append((a, b, n_inl, err, err / tile_px))
    return out


# ------------------------------------------------------ tile grid & sampling ---
def build_lattice(top, right, left, n):
    """The two tile step vectors, and the board's north corner as the origin.

    Directions from the fixed basis, lengths from the corners: the lattice and
    the silhouette share one angle, so the grid must not take its own from
    corner pixels."""
    u_col = BOARD_DIR_A * (np.linalg.norm(right - top) / n)
    u_row = BOARD_DIR_B * (np.linalg.norm(left - top) / n)
    return top, u_col, u_row


def tile_of_point(xy, origin, u_col, u_row):
    """The (i, j) of the tile containing a canvas point (tile_poly inverted)."""
    ij = np.linalg.inv(np.stack([u_col, u_row], axis=1)) @ (np.asarray(xy, float) - origin)
    return int(np.floor(ij[0])), int(np.floor(ij[1]))


def tile_poly(origin, u_col, u_row, i, j, inset=0.0):
    c = np.float32([origin + i * u_col + j * u_row,
                    origin + (i + 1) * u_col + j * u_row,
                    origin + (i + 1) * u_col + (j + 1) * u_row,
                    origin + i * u_col + (j + 1) * u_row])
    if inset > 0:
        center = c.mean(0)
        c = center + (c - center) * (1 - inset)
    return c


def tile_top_wedge(origin, u_col, u_row, i, j, frac=0.55):
    """The triangular fraction of a tile nearest its north (screen-up) vertex.

    Tall sprites occlude the tile north of their own from its south side, so a
    fogged tile hidden that way still shows clean fog in this corner."""
    top = origin + i * u_col + j * u_row
    right, left = top + u_col, top + u_row
    return np.float32([top, top + frac * (right - top), top + frac * (left - top)])


def poly_mask_bbox(poly, W, H):
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, np.round(poly - [x0, y0]).astype(np.int32), 1)
    return m, (x0, y0, x1, y1)


def tile_mask_bbox(origin, u_col, u_row, i, j, W, H, inset=0.0):
    """tile_poly + poly_mask_bbox: one tile's mask and slice bounds on the
    canvas, or None if it falls entirely outside it."""
    return poly_mask_bbox(tile_poly(origin, u_col, u_row, i, j, inset), W, H)


def _region_ncc(warped_img, tmpl_gray, wmask, poly, min_px=40):
    """Normalized correlation between one warped region and the same-shaped
    patch of template, for the whole-tile and top-wedge fog tests. The shot is
    already warped into template space, so one bbox indexes both."""
    r = poly_mask_bbox(poly, wmask.shape[1], wmask.shape[0])
    if r is None:
        return None
    m, (x0, y0, x1, y1) = r
    vm = (m > 0) & (wmask[y0:y1, x0:x1] > 0)
    if vm.sum() < min_px:
        return None
    a = cv2.cvtColor(warped_img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)[vm]
    b = tmpl_gray[y0:y1, x0:x1].astype(np.float32)[vm]
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def sample_tile(warped_img, wmask, tmpl_gray, poly, fog_ncc, min_valid_frac,
                wedge_poly=None, fog_wedge_ncc=0.80):
    """Classify one tile as fog or explored by correlating it against the
    template's fog art at the same spot.

    Never by saturation: mountains, snow and ice are as pale as fog. Real fog
    scores a median 0.88 and explored tiles 0.02, with nothing in 0.16-0.54.

    `wedge_poly` adds a second test for fog partly hidden by a sprite to its
    south (tile_top_wedge); a big enough city reaches the wedge too, which
    --fog-frac-margin covers. `fog_wedge_ncc` must stay high (~0.80): real fog
    wedges score ~0.95, and 0.65 turned forest into fog (test_ss_3 (18,18)).
    See CLAUDE.md, per-tile compositing."""
    r = poly_mask_bbox(poly, wmask.shape[1], wmask.shape[0])
    if r is None:
        return None
    m, (x0, y0, x1, y1) = r
    area = int(m.sum())
    if area == 0:
        return None
    vm = (m > 0) & (wmask[y0:y1, x0:x1] > 0)
    valid_frac = float(vm.sum()) / area
    if valid_frac < min_valid_frac or vm.sum() < 50:
        return {"witness": False, "explored": False, "mean_color": None}
    sub = warped_img[y0:y1, x0:x1]
    ncc = _region_ncc(warped_img, tmpl_gray, wmask, poly) or 0.0
    fog = ncc >= fog_ncc
    if not fog and wedge_poly is not None:
        wedge_ncc = _region_ncc(warped_img, tmpl_gray, wmask, wedge_poly)
        if wedge_ncc is not None and wedge_ncc >= fog_wedge_ncc:
            fog = True
    return {"witness": True, "explored": not fog, "fog_ncc": ncc,
            "mean_color": sub[vm].reshape(-1, 3).astype(np.float32).mean(0)}


# --------------------------------------------------- per-pixel fog evidence ---
def fog_illumination(warped_img, template, sel):
    """Per-channel gain+offset carrying the template's fog colors to this
    shot's, fitted on this shot's fog-locked pixels (FOG_LOCK_NCC), so it
    measures only the screenshot's illumination. Subsampling the fit was tried
    and does not pay: the `template[sel]` gather dominates."""
    A = template[sel].astype(np.float32)
    B = warped_img[sel].astype(np.float32)
    out = np.zeros((3, 2), np.float32)
    for c in range(3):
        M = np.stack([A[:, c], np.ones(len(A), np.float32)], 1)
        out[c] = np.linalg.lstsq(M, B[:, c], rcond=None)[0]
    return out


def fog_pixel_mask(warped_img, template, gain, tol=26.0):
    """Per-pixel 'this pixel is fog', with no spatial window at all.

    An anchored fog pixel equals the template's once illumination is fitted
    out, so a pixelwise compare can see a ~10px fringe, which a windowed
    correlation cannot (it straddles the occluder)."""
    pred = template.astype(np.float32) * gain[:, 0] + gain[:, 1]
    return np.abs(warped_img.astype(np.float32) - pred).max(2) <= tol


def tile_fog_fraction(fogpix, wmask, poly, W, H, min_px=60):
    """Fraction of one tile's *full* rhombus that is fog in this source.

    The full rhombus, not the inset one: a tall city can fill the centre and
    leave fog visible only around the rim."""
    r = poly_mask_bbox(poly, W, H)
    if r is None:
        return None
    m, (x0, y0, x1, y1) = r
    sel = (m > 0) & (wmask[y0:y1, x0:x1] > 0)
    if sel.sum() < min_px:
        return None
    return float(fogpix[y0:y1, x0:x1][sel].mean())


# ------------------------------------------------------- template px scale ---
# Template-space px thresholds were measured at REFERENCE_TILE_PX per tile and
# are multiplied by tile_px_scale() where used, since renders differ (78.5-80.5
# in Overlays/). Only PLATE_BAND/PLATE_HALF_W still need it; prefer sizes in
# tile widths for anything new. **Round a scaled bound compared against an
# integer** (`<= 10` becomes `10.01 <= 10`), which once cost 5 sets a bar each.
# The small "too few px" floors (_region_ncc, sample_tile, tile_fog_fraction)
# are deliberately not scaled.
REFERENCE_TILE_PX = 89.8    # a reference scale, not a file: do not re-base it


def tile_px_scale(u_col):
    """How much larger this template's tile is than the scale the px constants
    above were measured at. 0.87-0.90 across the Overlays/ renders."""
    return float(np.linalg.norm(u_col)) / REFERENCE_TILE_PX


# ------------------------------------------------------ Elyrion ruin vision ---
# Elyrion sees each fogged ruin as a cluster of rainbow flames drawn over the
# fog; the cluster's pooled centroid lands in the ruin's tile. The flame is the
# game's own sprite, composited as C = f*a*S + (1 - f*a)*F with the fog F
# known from the anchored template, so with D = C - F and K = a*(S - F) a match
# is scored by corr = <D,K>/(|D||K|), which is independent of the fade f.
# Score the correlation, never the residual (blank fog fits best), and never
# by HSV statistics (they cost two sets most of their ruins). See CLAUDE.md,
# Elyrion ruin vision.
RUIN_SPRITE = "Rainbowflame.png"
RUIN_SPRITE_DIR = "Assets"

RUIN_FLAME_TILE_FRAC = 0.26   # flame width / tile step; one measured scale fits
                              # every flame on both board sizes

RUIN_MATCH_MIN_CORR = 0.68    # mid-gap between control sets' false peaks and
                              # the weakest real flame (0.815)

RUIN_MATCH_MIN_SUPPORT = 0.35  # fraction of the flame on in-frame pixels.
                               # Inert on the corpus (wmask already drops
                               # off-frame pixels) but kept as the guard
                               # against a 0/0 correlation.

RUIN_MATCH_MIN_FADE = 0.05    # a flame only tints (darkens) the fog

RUIN_NOMINATE_SAT = 100       # search only near saturated pixels (0: all fog);
                              # see ruin_search_boxes. ~1/3 of the phase.

RUIN_PEAK_SEP_FRAC = 0.18     # min spacing between reported flames, in tiles

RUIN_MARK_BGR = (170, 30, 110)  # violet: high contrast on near-white fog and
                                # 92 deg of hue from the red spawn-zone layer.
                                # Do not drift it toward magenta/pink.

_ruin_sprite_cache = {}


def ruin_sprite_path():
    """Assets/Rainbowflame.png, via _asset_path."""
    return _asset_path(os.path.join(RUIN_SPRITE_DIR, RUIN_SPRITE))


def load_ruin_sprite():
    """The flame as (bgr float, alpha 0..1) cropped to its alpha extent, or None.

    None switches detection off and report() prints NO-RUIN-SPRITE. There is
    deliberately no fallback detector."""
    path = ruin_sprite_path()
    if path in _ruin_sprite_cache:
        return _ruin_sprite_cache[path]
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    out = None
    if im is not None and im.ndim == 3 and im.shape[2] == 4:
        a = im[:, :, 3].astype(np.float64) / 255.0
        ys, xs = np.nonzero(a > 0)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            out = (im[y0:y1, x0:x1, :3].astype(np.float64), a[y0:y1, x0:x1])
    _ruin_sprite_cache[path] = out
    return out


def ruin_flame_kernel(sprite, u_col):
    """The sprite resampled to the size a flame is drawn at on this board."""
    spr_bgr, spr_a = sprite
    h, w = spr_a.shape
    tw = max(3, int(round(RUIN_FLAME_TILE_FRAC * float(np.linalg.norm(u_col)))))
    th = max(3, int(round(tw * h / float(w))))
    return (cv2.resize(spr_bgr, (tw, th), interpolation=cv2.INTER_AREA),
            cv2.resize(spr_a, (tw, th), interpolation=cv2.INTER_AREA))


def ruin_match_maps(warped_bgr, template, gain, valid, box, kernel, fog_mean):
    """Match the flame over the fog region: (corr, fade, support, (y0, x0)).

    The maps cover the crop `box`, indexed by kernel-window centre, with
    (y0, x0) its origin; the caller reads nothing outside it, and the fog
    prediction is built on the crop for the same reason. K uses one fog colour
    (fog is uniform tile to tile) so it stays a fixed kernel; D uses the exact
    per-pixel prediction.

    Kept affordable (a naive version took 7.6s) by summing over channels before
    correlating where the kernel allows, matchTemplate (DFT) rather than
    filter2D, and cropping to the fog region."""
    spr_bgr, spr_a = kernel
    kh, kw = spr_a.shape
    y0, x0, y1, x1 = box
    if y1 - y0 < kh or x1 - x0 < kw:
        return None
    # Grow the slice by half a kernel of *real* pixels; zero padding only where
    # the canvas ends. A box edge otherwise truncates the denominator and
    # inflates the score there.
    H, W = warped_bgr.shape[:2]
    py, px = kh // 2, kw // 2                 # kernel half-extents, top/left
    qy, qx = kh - 1 - py, kw - 1 - px         # and bottom/right
    oy0, ox0, oy1, ox1 = y0, x0, y1, x1       # the box we must produce scores for
    y0, x0 = max(0, oy0 - py), max(0, ox0 - px)
    y1, x1 = min(H, oy1 + qy), min(W, ox1 + qx)
    pad_t, pad_l = py - (oy0 - y0), px - (ox0 - x0)   # zeros still needed
    pad_b, pad_r = qy - (y1 - oy1), qx - (x1 - ox1)

    sel = valid[y0:y1, x0:x1]
    v = sel.astype(np.float32)
    # the fog this shot would show here (as in fog_pixel_mask)
    fog = np.clip(template[y0:y1, x0:x1].astype(np.float32) * gain[:, 0]
                  + gain[:, 1], 0, 255)
    obs = warped_bgr[y0:y1, x0:x1].astype(np.float32)
    K = (spr_a[:, :, None] * (spr_bgr - fog_mean)).astype(np.float32)
    D = (obs - fog) * v[:, :, None]

    # matchTemplate scores only windows wholly inside its input, so pad by half
    # a kernel or the crop's rim is never scored (basin_treaties' ruin at
    # (0,9)). v pads to 0 too, so such windows get low support, not a score.
    def _pad(a):
        return cv2.copyMakeBorder(a, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=0)
    ch, cw = oy1 - oy0, ox1 - ox0
    num = np.zeros((ch, cw), np.float32)
    for c in range(3):
        num += cv2.matchTemplate(_pad(np.ascontiguousarray(D[:, :, c])),
                                 np.ascontiguousarray(K[:, :, c]),
                                 cv2.TM_CCORR)
    dsq = cv2.matchTemplate(_pad((D * D).sum(2)),
                            np.ones((kh, kw), np.float32), cv2.TM_CCORR)
    kk = cv2.matchTemplate(_pad(v), np.ascontiguousarray((K * K).sum(2)),
                           cv2.TM_CCORR)

    kk_full = max(float((K * K).sum()), 1e-9)
    # Exclude low-support windows rather than divide by an epsilon, which
    # produced corr values of 1279 and 2.6e7 and put ruins on six boards.
    good = kk > RUIN_MATCH_MIN_SUPPORT * kk_full * 0.5
    kk_safe = np.where(good, kk, 1.0)
    corr = np.where(good, num / np.sqrt(np.maximum(dsq, 1e-9) * kk_safe), 0.0)
    return (corr[:ch, :cw], np.where(good, num / kk_safe, 0.0)[:ch, :cw],
            np.where(good, kk / kk_full, 0.0)[:ch, :cw], (oy0, ox0))


def _ruin_peaks(corr, allowed, radius):
    """Accepted matches, strongest first, no two within `radius`.

    Suppression runs *within the allowed set*: against the raw map a gated-out
    neighbor can shadow an accepted flame (basin_treaties (0,9) was lost that
    way, and no threshold change could fix it)."""
    k = 2 * radius + 1
    masked = np.where(allowed, corr, -np.inf).astype(np.float32)
    peak = (masked >= cv2.dilate(masked, np.ones((k, k), np.uint8))) & allowed
    ys, xs = np.nonzero(peak)
    if not len(ys):
        return []
    out, taken = [], []
    for idx in np.argsort(-corr[ys, xs]):
        y, x = int(ys[idx]), int(xs[idx])
        if any((y - ty) ** 2 + (x - tx) ** 2 < radius * radius
               for ty, tx in taken):
            continue
        taken.append((y, x))
        out.append((y, x))
    return out


def ruin_search_boxes(warped_bgr, template, gain, valid, fog_area, kh, kw):
    """Where to run the matched filter, as a list of boxes, plus the fog color
    every box must share.

    With RUIN_NOMINATE_SAT off, one box: the fog region's bounding box. With it
    on, one box per component of the saturated pixels (12% of the fog region
    against the bounding box's 80%). Saturation only nominates here: faint
    flames are missed, but a ruin's marker is a *cluster*, so every Elyrion
    set keeps its exact count at thresholds 0-120 (140 starts losing ruins).
    Returns (boxes, fog_mean, search); fog_mean is shared by every box so a
    flame's score does not depend on which box holds it, and `search` is where
    a peak may be accepted (None when nomination is off)."""
    ys, xs = np.nonzero(fog_area)
    if not len(ys):
        return [], np.float32([255, 255, 255]), None
    H, W = fog_area.shape
    fy0, fy1 = max(0, int(ys.min()) - kh), min(H, int(ys.max()) + kh + 1)
    fx0, fx1 = max(0, int(xs.min()) - kw), min(W, int(xs.max()) + kw + 1)
    sel = fog_area[fy0:fy1, fx0:fx1] & valid[fy0:fy1, fx0:fx1]
    if sel.any():
        fog_pred = np.clip(template[fy0:fy1, fx0:fx1].astype(np.float32)
                           * gain[:, 0] + gain[:, 1], 0, 255)
        fog_mean = fog_pred[sel].reshape(-1, 3).mean(0)
    else:
        fog_mean = np.float32([255, 255, 255])
    if not RUIN_NOMINATE_SAT:
        return [(fy0, fx0, fy1, fx1)], fog_mean, None

    sat = cv2.cvtColor(warped_bgr[fy0:fy1, fx0:fx1], cv2.COLOR_BGR2HSV)[:, :, 1]
    nom = ((sat >= RUIN_NOMINATE_SAT) & sel).astype(np.uint8)
    if not nom.any():
        return [], fog_mean, None
    # A flame's centre can sit up to half a kernel from its nominating pixels.
    search = np.zeros(fog_area.shape, bool)
    search[fy0:fy1, fx0:fx1] = cv2.dilate(
        nom, np.ones((kh, kw), np.uint8)).astype(bool)
    n, _lab, st, _ = cv2.connectedComponentsWithStats(nom, 8)
    boxes = []
    for c in range(1, n):
        by = st[c, cv2.CC_STAT_TOP] + fy0
        bx = st[c, cv2.CC_STAT_LEFT] + fx0
        bh, bw = st[c, cv2.CC_STAT_HEIGHT], st[c, cv2.CC_STAT_WIDTH]
        # plus a whole kernel of context for matchTemplate
        boxes.append((max(0, by - kh), max(0, bx - kw),
                      min(H, by + bh + kh), min(W, bx + bw + kw)))
    return boxes, fog_mean, search


def detect_ruin_vision(warped_bgr, wmask, fog_area, template, gain, origin,
                       u_col, u_row, sprite):
    """Elyrion ruin-vision flames in one warped source, as a list of
    (i, j, area, footprint_mask).

    Runs in template space, so one kernel fits every zoom, and only on this
    source's own fog. Scores a map rather than labeling blobs, so there is no
    close, island test, area bound or colour band to tune."""
    if sprite is None:
        return []
    valid = (wmask > 0)
    if not (fog_area & valid).any():
        return []
    kernel = ruin_flame_kernel(sprite, u_col)
    kh, kw = kernel[1].shape
    boxes, fog_mean, search = ruin_search_boxes(warped_bgr, template, gain,
                                                valid, fog_area, kh, kw)
    if not boxes:
        return []
    radius = max(2, int(round(RUIN_PEAK_SEP_FRAC * float(np.linalg.norm(u_col)))))

    # Boxes overlap by their halo, so suppression is global, not per box.
    cands = []
    for box in boxes:
        got = ruin_match_maps(warped_bgr, template, gain, valid, box, kernel,
                              fog_mean)
        if got is None:
            continue
        corr, fade, support, (cy, cx) = got
        ch, cw = corr.shape
        ok = (fog_area[cy:cy + ch, cx:cx + cw]
              & valid[cy:cy + ch, cx:cx + cw]
              & (support >= RUIN_MATCH_MIN_SUPPORT)
              & (fade > RUIN_MATCH_MIN_FADE)
              & (corr >= RUIN_MATCH_MIN_CORR))
        if search is not None:
            # Only inside `search` is a window wholly on real data; at a box
            # edge the truncated denominator inflates the score (dropping this
            # put false ruins on six sets).
            ok &= search[cy:cy + ch, cx:cx + cw]
        if not ok.any():
            continue
        for (py, px) in _ruin_peaks(corr, ok, radius):
            cands.append((float(corr[py, px]), py + cy, px + cx))
    if not cands:
        return []
    foot = kernel[1] > 0.15          # the flame's own silhouette, for reporting
    out, taken = [], []
    for (_, fy, fx) in sorted(cands, key=lambda c: -c[0]):
        if any((fy - ty) ** 2 + (fx - tx) ** 2 < radius * radius
               for ty, tx in taken):
            continue
        taken.append((fy, fx))
        i, j = tile_of_point((fx, fy), origin, u_col, u_row)
        y0, x0 = fy - kh // 2, fx - kw // 2
        ys0, xs0 = max(0, y0), max(0, x0)
        ys1 = min(fog_area.shape[0], y0 + kh)
        xs1 = min(fog_area.shape[1], x0 + kw)
        mask = np.zeros(fog_area.shape, bool)
        if ys1 > ys0 and xs1 > xs0:
            mask[ys0:ys1, xs0:xs1] = foot[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
            mask &= fog_area & valid
        out.append((i, j, int(mask.sum()), mask))
    return out


def cluster_ruin_tiles(hits, origin, u_col, u_row):
    """Collapse per-tile ruin detections into one entry per actual ruin.

    Ruins are never adjacent (8-neighborhood; some clusters meet only at a
    corner), so neighboring detections are one cluster straddling a border.
    Each merged cluster goes to the tile holding the centroid of its pixels
    pooled across all sources."""
    tiles = sorted(hits)
    seen, out = set(), {}
    for t in tiles:
        if t in seen:
            continue
        stack, comp = [t], []
        seen.add(t)
        while stack:                      # flood fill over adjacent detections
            c = stack.pop()
            comp.append(c)
            for u in tiles:
                if u not in seen and max(abs(u[0] - c[0]), abs(u[1] - c[1])) <= 1:
                    seen.add(u)
                    stack.append(u)
        merged = [h for c in comp for h in hits[c]]
        sy = sx = tot = 0.0
        for _, _, mask in merged:
            ys, xs = np.nonzero(mask)
            sy += ys.sum(); sx += xs.sum(); tot += ys.size
        key = tile_of_point((sx / tot, sy / tot), origin, u_col, u_row)
        if key not in comp:               # centroid landed off the cluster --
            key = comp[0]                 # shouldn't happen; keep it in-cluster
        out[key] = (merged, sorted(comp))
    return out


# ------------------------------------------------- city population bar ---
# The population bar under a city's name is owner-only, so the shot showing it
# is the owner's; without promotion a sharper non-owner shot wins its tiles and
# the bar vanishes. City UI scales with board zoom, so in template space it is
# a fixed size. See CLAUDE.md, city population bars.
PLATE_BAND = (8.0, 56.0)              # above the south vertex; template px at
PLATE_HALF_W = 95.0                   # REFERENCE_TILE_PX, scaled at use
PLATE_EDGE_MIN = 12.0                 # gray levels across a row -- a step, not
                                      # an edge detector's idea of an edge


def plate_edge_run(gray, vx, vy, s):
    """Longest horizontal edge run in the band a city's name plate occupies,
    for the city whose south vertex is (vx, vy), in REFERENCE_TILE_PX units. A
    run (positive evidence of one long horizontal edge), which isometric
    terrain cannot fake. Used only as a claim tiebreak."""
    lo, hi = PLATE_BAND
    y0, y1 = int(vy - hi * s), int(vy - lo * s)
    x0, x1 = int(vx - PLATE_HALF_W * s), int(vx + PLATE_HALF_W * s)
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(gray.shape[0], y1), min(gray.shape[1], x1)
    if y1 - y0 < 4 or x1 - x0 < 20:
        return 0.0
    g = cv2.GaussianBlur(gray[y0:y1, x0:x1].astype(np.float32), (3, 3), 0)
    d = np.abs(cv2.filter2D(g, cv2.CV_32F, np.float32([[-1], [0], [1]])))
    e = (d > PLATE_EDGE_MIN).astype(np.uint8)
    # Close along x only: a plate's edge is broken by the letters standing on
    # it and by antialiasing, never by anything that makes it two edges.
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    best = 0
    for row in e:
        if not row.any():
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row, [0]))))
        best = max(best, int((edges[1::2] - edges[::2]).max()))
    return best / s


# Detection is anchored, not searched: a bar is centred on its city tile's
# south vertex, so each interior tile is tested for no bar, a short bar or a
# capped bar. All lengths are in tile widths. The top and bottom edges are a
# pair constrained by BAR_HEIGHT rather than two absolutely placed rows, so a
# couple of px of anchor error does not drop a bar. See CLAUDE.md for the
# measurements and the labeled ground truth (50/50, 0 FP).
BAR_TOP_BAND = (-0.130, 0.020)  # rows the bar's top edge can occupy, from the
BAR_BOT_BAND = (0.030, 0.210)   # vertex; measured -0.093..-0.020 and +0.070..
                                # +0.149, each widened by 0.04 for anchor slack
BAR_HEIGHT = (0.125, 0.225)     # the pair's separation. Confirmed static
                                # relative to the tile; measured 0.138..0.213
BAR_HALVES = (0.482, 0.720)     # the only two legal half-widths: a short bar
                                # spans 0.965 tiles and a capped one 1.44, a
                                # clean 3:2 with nothing in between
BAR_EDGE_MIN = 18.0             # gray levels across a row
BAR_END_MARGIN = 0.06           # how far past an end to require absence
BAR_END_INSET = 0.04            # ignore this much at each end when scoring the
                                # edges -- the caps are rounded, so they fade
BAR_WIN = (0.26, 0.26, 0.95)    # up, down, half-width of the search window
BAR_MIN_EDGE_COLS = 0.30        # of the short bar's width, for a row to count
BAR_SCORE_MIN = 0.34            # deliberately low: the color test below is
                                # what discriminates, so the geometry can be
                                # permissive. At 0.55 the corpus loses 12
                                # confirmed-real bars.

# The modal colour inside an already-located box corroborates a bar; colour
# never *finds* one (a fill score called 225 tiles bar-like). Real bars read
# 228 off-white, saturated blue or saturated red; white false positives (ice,
# snow, UI) read 252, and those ~24 levels are the whole margin.
BAR_MODE_V = (198, 240)         # off-white body
BAR_MODE_S = 60                 # ...must be this desaturated
BAR_BLUE_S = 180                # a filled segment is vividly blue. Water reads
                                # S=186 against a real blue bar's 187, so this
                                # cannot separate them -- the polarity rule in
                                # detect_population_bars is what does.
BAR_RED_S = 150                 # scorched_earth's Icalus reads S=255
# The prefilter colour box, preset because it runs before any geometry (the
# second call uses the measured rows). Inside the short bar, so inside either
# width; rows centred on the median bar interior, the widest margin of five
# spans swept.
BAR_BOX_ROWS = (0.000, 0.080)
BAR_BOX_HALVES = (0.40, 0.62)

# No two cities sit within two tiles of each other (a game rule; every labeled
# set's minimum separation is exactly 3). Two detections closer than this
# contradict each other (promote_city_bars).
CITY_MIN_GAP = 2

# How strongly a shot claims a tile of a city's 3x3 block, strongest first.
# "Strong" tiles are the city's own and its S/SW/SE neighbours -- the only ones
# a bar can reach -- and "weak" the rest of the block. A detected bar outranks
# the vision rule at each strength. See promote_city_bars for the tiebreaks.
CLAIM_BAR_STRONG = 3        # detected bar, on a tile its bar can occupy
CLAIM_VISION_STRONG = 2     # sole seer of the block, same tiles
CLAIM_BAR_WEAK = 1          # detected bar, rest of the block
CLAIM_VISION_WEAK = 0       # sole seer of the block, rest of the block


def _bar_edges(bgr):
    """Signed horizontal-edge strength per pixel.

    A bar's top and bottom are opposite horizontal steps; requiring |dy| to
    dominate |dx| rejects isometric terrain, where nothing is horizontal."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                         .astype(np.float32), (3, 3), 0)
    sy = cv2.filter2D(g, cv2.CV_32F, np.float32([[-1], [0], [1]]))
    sx = cv2.filter2D(g, cv2.CV_32F, np.float32([[-1, 0, 1]]))
    sy[np.abs(sy) < 2.0 * np.abs(sx)] = 0.0
    return sy


def _bar_mode(bgr, wmask, origin, u_col, u_row, i, j, half, rows=None):
    """Modal color in a box at tile (i, j)'s south vertex: (bgr, is_bar, is_red).

    `rows` is an absolute (y0, y1) span once the bar is measured, else the
    preset BAR_BOX_ROWS. `wmask` must be `valid`: out-of-frame pixels would
    otherwise win the mode near a shot's edge (and are not exactly black after
    JPEG)."""
    tile = float(np.linalg.norm(u_col))
    vx, vy = origin + (i + 1) * u_col + (j + 1) * u_row
    if rows is None:
        y0 = int(round(vy + BAR_BOX_ROWS[0] * tile))
        y1 = int(round(vy + BAR_BOX_ROWS[1] * tile))
    else:
        y0, y1 = rows
    x0, x1 = int(round(vx - half * tile)), int(round(vx + half * tile))
    if (y1 <= y0 or x1 <= x0 or y0 < 0 or x0 < 0
            or y1 >= bgr.shape[0] or x1 >= bgr.shape[1]):
        return None, False, False
    patch = bgr[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
    patch = patch[wmask[y0:y1 + 1, x0:x1 + 1].reshape(-1) > 0]
    if len(patch) < 20:
        return None, False, False
    # Mode of the packed 1-D key: 8x faster than np.unique(axis=0) or bincount.
    q = (patch // 6).astype(np.int32)
    key = (q[:, 0] << 12) | (q[:, 1] << 6) | q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    top = int(vals[counts.argmax()])
    b, g, r = (top >> 12) * 6, ((top >> 6) & 63) * 6, (top & 63) * 6
    H, S, V = (int(z) for z in cv2.cvtColor(
        np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0])
    red = (H <= 8 or H >= 172) and S >= BAR_RED_S and V >= 150
    ok = ((S <= BAR_MODE_S and BAR_MODE_V[0] <= V <= BAR_MODE_V[1])
          or (100 <= H <= 118 and S >= BAR_BLUE_S and V >= 150)
          or red)
    return (b, g, r), ok, red


def _bar_at(sy, origin, u_col, u_row, i, j, dark):
    """Score the two width hypotheses at tile (i, j)'s south vertex.

    `dark` inverts the edge polarity: a red bar is darker than grass in gray
    (scorched_earth's Icalus)."""
    up, dn, halfw = BAR_WIN
    tile = float(np.linalg.norm(u_col))
    vx, vy = origin + (i + 1) * u_col + (j + 1) * u_row
    ry0, ry1 = int(round(vy - up * tile)), int(round(vy + dn * tile))
    rx0, rx1 = int(round(vx - halfw * tile)), int(round(vx + halfw * tile))
    if (ry0 < 0 or rx0 < 0 or ry1 >= sy.shape[0] or rx1 >= sy.shape[1]
            or ry1 - ry0 < 6):
        return None
    win = sy[ry0:ry1 + 1, rx0:rx1 + 1]
    rows, w = win.shape
    pos, neg = win >= BAR_EDGE_MIN, win <= -BAR_EDGE_MIN
    if dark:
        pos, neg = neg, pos
    npos, nneg = pos.sum(axis=1), neg.sum(axis=1)

    def band(lo, hi):
        return (max(0, int(round(lo * tile + (vy - ry0)))),
                min(rows - 1, int(round(hi * tile + (vy - ry0)))))

    # Edges by row profile (the vertex column often sits on a segment divider).
    # The bands keep the pair off the name plate above; BAR_HEIGHT lets them
    # stay loose.
    t0, t1 = band(*BAR_TOP_BAND)
    b0, b1 = band(*BAR_BOT_BAND)
    hlo = max(2, int(round(BAR_HEIGHT[0] * tile)))
    hhi = int(round(BAR_HEIGHT[1] * tile))
    floor = BAR_MIN_EDGE_COLS * 2 * BAR_HALVES[0] * tile
    best = None
    for yt in range(t0, t1 + 1):
        if npos[yt] < floor:
            continue
        for yb in range(max(b0, yt + hlo), min(b1, yt + hhi) + 1):
            if nneg[yb] >= floor and (best is None
                                      or min(npos[yt], nneg[yb]) > best[0]):
                best = (min(npos[yt], nneg[yb]), yt, yb)
    if best is None:
        return None
    _sc, yt, yb = best
    tol = max(1, int(round(0.02 * tile)))

    def rowband(m, y):
        return m[max(0, y - tol):min(rows, y + tol + 1)].any(axis=0)

    top_col, bot_col = rowband(pos, yt), rowband(neg, yb)
    both = top_col & bot_col
    cx = vx - rx0

    def cols(a, b):
        lo, hi = int(round(cx + a * tile)), int(round(cx + b * tile))
        return slice(max(0, min(lo, w)), max(0, min(hi, w)))

    scored = []
    for h in BAR_HALVES:
        # Score each side alone (a unit may cover one end), with coverage and
        # the end test from the *same* side, or a capped bar with one buried
        # end scores highest as a short one.
        for sgn in (-1, 1):
            a, b = sorted((0.0, sgn * (h - BAR_END_INSET)))
            inner = cols(a, b)
            if inner.stop - inner.start < 4:
                continue
            e0, e1 = sorted((sgn * h, sgn * (h + BAR_END_MARGIN)))
            out = both[cols(e0, e1)]
            end = 1.0 - float(out.mean()) if out.size else 0.0
            scored.append((min(float(top_col[inner].mean()),
                               float(bot_col[inner].mean())) * end, h))
    if not scored:
        return None
    score, half = max(scored)
    return {"score": score, "half": half,
            "px": (int(round(vx - half * tile)), ry0 + yt,
                   int(round(2 * half * tile)), yb - yt)}


def detect_population_bars(warped_bgr, wmask, origin, u_col, u_row, n):
    """Population bars in one warped source, as
    [(city_tile, bbox, width_class, plate_run)].

    Segments are never counted (dividers do not survive every capture);
    `width_class` is 2 for a short bar and 3 for a capped one, and is not a
    segment count. Rim tiles are skipped: cities never sit on the rim."""
    sy = _bar_edges(warped_bgr)
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    tile_scale = tile_px_scale(u_col)
    bars = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            # Colour first: independent of the geometry and far cheaper.
            _mode, ok, red = _bar_mode(warped_bgr, wmask, origin, u_col, u_row,
                                       i, j, BAR_BOX_HALVES[0])
            if not ok:
                continue
            best = None
            for dark in (False, True):
                # Dark polarity for red only: unscoped it admits water and ice,
                # which colour cannot reject (S=186 against a blue bar's 187).
                if dark and not red:
                    continue
                m = _bar_at(sy, origin, u_col, u_row, i, j, dark)
                if m and (best is None or m["score"] > best["score"]):
                    best = m
            if best is None or best["score"] < BAR_SCORE_MIN:
                continue
            # Re-read the colour inside the measured bar: columns from the
            # width class, rows from the edges, inset a fifth of the height. A
            # preset band could slide onto the white name plate (mode 252).
            wide = BAR_BOX_HALVES[BAR_HALVES.index(best["half"])]
            by, bh = best["px"][1], best["px"][3]
            inset = max(1, int(round(0.2 * bh)))
            m2, ok2, _r = _bar_mode(warped_bgr, wmask, origin, u_col, u_row,
                                    i, j, wide,
                                    rows=(by + inset, by + bh - inset))
            if m2 is not None and not ok2:
                continue
            vx, vy = origin + (i + 1) * u_col + (j + 1) * u_row
            bars.append(((i, j), best["px"],
                         2 if best["half"] == BAR_HALVES[0] else 3,
                         plate_edge_run(gray, vx, vy, tile_scale)))
    return bars


# ------------------------------------------------- player identification ---
# The bottom action row is four round buttons (Settings, Game Stats, Tech Tree,
# Exit/End Turn), and Game Stats shows the screenshot's own player's head icon.
# Matching it against the game's head renders (Assets/Heads) says which shots
# share a player, for --overlays vision. Approximate by nature: the catalog
# need not cover every skin, so a drawn outline is an aid, not ground truth.
# See CLAUDE.md, player identification.
HEAD_ICON_DIR = "Assets/Heads"
HEAD_ICON_CANON = 96          # every button crop is normalized to this; swept
                              # 72/96/128/200, margins stop improving at 96

# Fractional x of buttons 0, 2 and 3, over 63 portrait shots. Game Stats (1) is
# never located directly (its colourful icon fails the ring test: 0 of 63) but
# interpolated, the row being evenly spaced and centred.
BUTTON_ROW_ANCHOR_X = {0: 0.216, 2: 0.594, 3: 0.783}
BUTTON_ROW_MATCH_TOL = 0.08    # inside the anchors' spread, short of the ~0.19
                              # gap to a neighbor
BUTTON_ROW_BOTTOM_FRAC = 0.18  # search band, as a fraction of image height
                              # (--bottom-crop's 0.15 just clears the row)
BUTTON_RING_BRIGHTNESS = 205   # the ring is bright white on black; board
                              # content rarely reaches this band
BUTTON_MIN_CIRCULARITY = 0.70  # contour area / enclosing-circle area


def _button_row_blob_candidates(img):
    """Bright, sufficiently circular blobs in the bottom action-button band,
    as (cx, cy, r) fractions of (w, h, w). The primary nominator, enough on
    most shots; the Hough and blue-fill nominators cover the rest, and only
    _fit_button_row decides what is real. Its own shape test implies a real
    ring, so unlike the other two it skips _button_fill_is_plausible.

    No morphological close: one used to bridge rings to nearby bright board
    content and cost badland_test/oum.jpg and star_change/oum2.png anchors."""
    band, y0 = _button_band(img)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bright = (gray > BUTTON_RING_BRIGHTNESS).astype(np.uint8) * 255
    return _button_blob_circles(img, bright, y0, BUTTON_MIN_CIRCULARITY,
                                verify_fill=False)


def _button_band(img):
    """The bottom band the button row sits in, and the row it starts at."""
    h = img.shape[0]
    y0 = h - int(h * BUTTON_ROW_BOTTOM_FRAC)
    return img[y0:, :], y0


def _button_blob_circles(img, mask, y0, min_circularity, verify_fill):
    """Button candidates from the blobs of `mask` (a mask of the bottom band
    starting at row y0), as (cx, cy, r) fractions of (w, h, w).

    Shared shape tests: size, radius range, circularity, and not cut off by
    the image's bottom edge. `verify_fill` adds _button_fill_is_plausible."""
    h, w = img.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < 0.015 * w or r > 0.11 * w:
            continue
        if area / (np.pi * r * r) < min_circularity:
            continue
        cy_abs = cy + y0
        if cy_abs + r > h - 2:
            continue
        if verify_fill and not _button_fill_is_plausible(img, cx, cy_abs, r):
            continue
        out.append((cx / w, cy_abs / h, r / w))
    return out


BUTTON_FILL_LO, BUTTON_FILL_HI = 0.62, 0.85  # sampled annulus, in ring radii:
                                              # inside the stroke, outside the
                                              # icon glyph
BUTTON_FILL_GRAY_MAX = 90     # real fill: gray 0-2, p99 19, max 44 (200 samples)
BUTTON_FILL_STD_MAX = 30      # real fill: std 0-2, p99 19; dark terrain 90-103
BUTTON_FILL_BLUE_HUE = (90, 115)  # the "not your turn" recolor: measured hue
BUTTON_FILL_BLUE_SAT_MIN = 130    # 96-105, saturation 130-220


def _button_fill_is_plausible(img, cx, cy, r):
    """Is the fill just inside this candidate's ring a real button's -- flat
    black, or the "not your turn" blue -- rather than terrain? Flatness is
    what separates them: dark terrain is as dark as a button (basin_treaties,
    perilous_test) but textured. Used by the Hough and blue nominators only."""
    h, w = img.shape[:2]
    y0 = max(0, int(cy - BUTTON_FILL_HI * r))
    y1 = min(h, int(cy + BUTTON_FILL_HI * r) + 1)
    x0 = max(0, int(cx - BUTTON_FILL_HI * r))
    x1 = min(w, int(cx + BUTTON_FILL_HI * r) + 1)
    if y1 <= y0 or x1 <= x0:
        return False
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (d2 >= (BUTTON_FILL_LO * r) ** 2) & (d2 <= (BUTTON_FILL_HI * r) ** 2)
    pixels = img[y0:y1, x0:x1][mask]
    if len(pixels) < 5:
        return False
    if pixels.mean(axis=1).std() > BUTTON_FILL_STD_MAX:
        return False
    if pixels.mean() < BUTTON_FILL_GRAY_MAX:
        return True
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    return BUTTON_FILL_BLUE_HUE[0] <= hsv[0] <= BUTTON_FILL_BLUE_HUE[1] and hsv[1] >= BUTTON_FILL_BLUE_SAT_MIN


def _button_row_hough_candidates(img):
    """Button rings via Hough transform, for a ring the board's jagged edge
    touches and so fuses into one component with the whole board
    (basin_treaties/q.png, test_ss_elyruins/hood.png, u_forest2/ely.png).
    Noisy on textured fog -- missized_test/z1.jpg, with no button row in
    frame, yields a plausible-looking circle pair -- so every circle must pass
    _button_fill_is_plausible. Fallback only; same (cx, cy, r) fractions as
    the blob nominator."""
    h, w = img.shape[:2]
    band, y0 = _button_band(img)
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    min_r = max(1, int(0.015 * w))
    max_r = max(min_r + 1, int(0.11 * w))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5,
                                minDist=int(0.08 * w), param1=80, param2=40,
                                minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return []
    out = []
    for cx, cy, r in circles[0]:
        cy_abs = cy + y0
        if cy_abs + r > h - 2:
            continue
        if not _button_fill_is_plausible(img, cx, cy_abs, r):
            continue
        out.append((cx / w, cy_abs / h, r / w))
    return out


BUTTON_BLUE_MIN_CIRCULARITY = 0.30  # loose: a recolored Game Stats ring
                                     # measures 0.42-0.43; the fill check and
                                     # row fit arbitrate


def _button_row_blue_candidates(img):
    """Saturated-blue blobs in the button band, for the "not your turn"
    state, where Game Stats' ring and Exit's whole fill turn blue and Exit
    has no ring for the other nominators to find. Terrain blue is far less
    saturated (S <= 33 against the button's 130+). Recovers
    perilous_test/xin.png's Exit; Game Stats itself fails the fill check,
    its glyph filling the sampled annulus (scorched_earth/bard.png)."""
    band, y0 = _button_band(img)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= BUTTON_FILL_BLUE_HUE[0]) & (hsv[:, :, 0] <= BUTTON_FILL_BLUE_HUE[1]) &
            (hsv[:, :, 1] >= BUTTON_FILL_BLUE_SAT_MIN) & (hsv[:, :, 2] >= 100)).astype(np.uint8) * 255
    return _button_blob_circles(img, mask, y0, BUTTON_BLUE_MIN_CIRCULARITY,
                                verify_fill=True)


BUTTON_ROW_FIT_MAX_RESID = 0.006  # all 66 corpus shots with 3+ candidates fit
                                   # on slots {0, 2, 3} under 0.003
BUTTON_ROW_RADIUS_RATIO = 1.35    # the buttons are one size


def _fit_button_row(cands):
    """Fit 2-4 button candidates to one evenly spaced, screen-centred row:
    slot i at cx = 0.5 + (i - 1.5) * d, for one shared d.

    Every nominator above only proposes; this decides. Because the intercept
    is fixed at the screen centre, any k >= 2 points over-determine d, so a
    tight fit corroborates whatever this device's own spacing is. That is
    what fixed-table matching could not do: test_screenshots/h.jpg has a
    spacing of 0.084 against the corpus's 0.189. Slot pairs need not be
    symmetric. The best fit wins on most points, then lowest residual, never
    first found.

    Returns (d, {idx: (cx, cy, r)}), or None if nothing fits tightly enough."""
    cands = sorted(cands, key=lambda c: c[0])
    n = len(cands)
    best = None
    for k in (4, 3, 2):
        if n < k:
            continue
        for cand_idxs in itertools.combinations(range(n), k):
            pts = [cands[i] for i in cand_idxs]
            rs = [p[2] for p in pts]
            if max(rs) > BUTTON_ROW_RADIUS_RATIO * min(rs):
                continue
            for slot_idxs in itertools.combinations(range(4), k):
                u = np.array([i - 1.5 for i in slot_idxs])
                xs = np.array([p[0] - 0.5 for p in pts])
                d = float((u * xs).sum() / (u * u).sum())
                resid = float(np.max(np.abs(0.5 + u * d - np.array([p[0] for p in pts]))))
                if resid > BUTTON_ROW_FIT_MAX_RESID:
                    continue
                if best is None or (k, -resid) > (best[0], -best[1]):
                    best = (k, resid, d, dict(zip(slot_idxs, pts)))
    if best is None:
        return None
    _, _, d, slot_map = best
    return d, slot_map


def _merge_button_candidates(base, extra):
    """base + extra, minus any extra within 0.01 (x and y) of a base
    candidate, so one button found twice cannot pass for two points."""
    out = list(base)
    for c in extra:
        if not any(abs(c[0] - b[0]) < 0.01 and abs(c[1] - b[1]) < 0.01 for b in out):
            out.append(c)
    return out


def locate_game_stats_icon(img):
    """Where the Game Stats button's icon sits in this (raw, unwarped)
    screenshot, as (cx, cy, r) in pixels, or None with fewer than two buttons
    to place it from (zero or one anchor produced confident wrong matches).

    The blob nominator is tried alone first; only if its row fit fails do
    the Hough and blue nominators join. If no row fits, falls back to the
    BUTTON_ROW_ANCHOR_X table, keeping one candidate per slot: two buttons
    in one slot's window must not count as two anchors."""
    h, w = img.shape[:2]
    cands = _button_row_blob_candidates(img)
    fit = _fit_button_row(cands)
    if fit is None:
        cands = _merge_button_candidates(cands, _button_row_hough_candidates(img))
        cands = _merge_button_candidates(cands, _button_row_blue_candidates(img))
        fit = _fit_button_row(cands)
    if fit is not None:
        d, slot_map = fit
        cx = 0.5 - 0.5 * d
        cy = float(np.median([p[1] for p in slot_map.values()]))
        r = float(np.median([p[2] for p in slot_map.values()]))
        return int(round(cx * w)), int(round(cy * h)), int(round(r * w))
    by_idx = {}
    for cx, cy, r in cands:
        idx, canon_x = min(BUTTON_ROW_ANCHOR_X.items(), key=lambda kv: abs(cx - kv[1]))
        d = abs(cx - canon_x)
        if d < BUTTON_ROW_MATCH_TOL and (idx not in by_idx or d < by_idx[idx][0]):
            by_idx[idx] = (d, cx, cy, r)
    anchors = [(idx, cx, cy, r) for idx, (d, cx, cy, r) in by_idx.items()]
    if len(anchors) < 2:
        return None
    idxs = np.float32([a[0] for a in anchors])
    xs = np.float32([a[1] for a in anchors])
    A = np.stack([idxs, np.ones_like(idxs)], 1)
    d, a0 = np.linalg.lstsq(A, xs, rcond=None)[0]
    cx = a0 + d
    cy = float(np.median([a[2] for a in anchors]))
    r = float(np.median([a[3] for a in anchors]))
    return int(round(cx * w)), int(round(cy * h)), int(round(r * w))


# The crop handed to the matcher, in ring radii: just wide enough to hold the
# largest head swept, since extra background dilutes the correlation.
HEAD_ICON_REGION = 1.4
# The head's height in ring radii varies by icon (1.235-1.742 over 16 shots),
# so it is swept, never assumed.
HEAD_SCALE_LO, HEAD_SCALE_HI, HEAD_SCALE_STEPS = 1.15, 1.85, 8
HEAD_ICON_MIN_PIXELS = 60       # too little sprite left to say anything
HEAD_SPRITE_MAX = 256           # catalog sprites are stored no larger than
                                # this (assets ship at up to 1024x1024)

# But it is fixed per *icon*: the same icon across screenshots and devices
# varies 0.00-0.06, against 1.22-1.74 between icons. So an entry with a
# confident corpus measurement is swept narrowly around it (tolerance 2.7x the
# largest spread seen, finer steps, half the cost); an uncalibrated entry keeps
# the full sweep.
HEAD_SCALE_ENTRY_TOL = 0.08
HEAD_SCALE_ENTRY_STEPS = 4
HEAD_SCALE_BY_ENTRY = {
    "x2.png": 1.220, "ai2.png": 1.270, "i2.png": 1.270, "c.png": 1.292,
    "x.png": 1.360, "c2.png": 1.390, "y.png": 1.405, "p.png": 1.420,
    "i.png": 1.440, "z.png": 1.480, "l.png": 1.510, "v2.png": 1.540,
    "e.png": 1.550, "o.png": 1.571, "q.png": 1.600, "h.png": 1.620,
    "y2.png": 1.620, "k.png": 1.740, "h2.png": 1.740,
}


def _head_scale_sweep(name):
    """Scale candidates for one catalog entry (see HEAD_SCALE_BY_ENTRY)."""
    center = HEAD_SCALE_BY_ENTRY.get(name)
    if center is None:
        return np.linspace(HEAD_SCALE_LO, HEAD_SCALE_HI, HEAD_SCALE_STEPS)
    lo = max(HEAD_SCALE_LO, center - HEAD_SCALE_ENTRY_TOL)
    hi = min(HEAD_SCALE_HI, center + HEAD_SCALE_ENTRY_TOL)
    return np.linspace(lo, hi, HEAD_SCALE_ENTRY_STEPS)


def head_icon_region(img, cx, cy, r):
    """The Game Stats button's neighbourhood, normalized to a fixed canonical
    size, as float32 -- or None if the location is degenerate.

    Normalizing by the button radius lets one scale sweep serve every device.
    No interior circle is cut: the ring and rank badge are excluded at match
    time by the sprite's own alpha, whereas a fixed circle clipped the head on
    some captures and kept ring on others."""
    R = int(round(r * HEAD_ICON_REGION))
    if R < 8:
        return None
    pad = max(0, R - cy, R - cx, cy + R - img.shape[0], cx + R - img.shape[1])
    if pad > 0:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cx, cy = cx + pad, cy + pad
    box = img[cy - R:cy + R, cx - R:cx + R]
    if box.shape[0] < 8 or box.shape[1] < 8:
        return None
    return cv2.resize(box, (HEAD_ICON_CANON, HEAD_ICON_CANON),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


PLAYER_HEAD_MIN_CORR = 0.55     # floor on the winning entry's own score
PLAYER_HEAD_MIN_MARGIN = 0.10   # lead it must hold over the runner-up
HEAD_CLUSTER_MIN_NCC = 0.90     # two shots' own icons this alike are one player
HEAD_SAME_ICON_NCC = 0.98       # ...and this alike are the same icon, so only
                                # one needs a (costly) catalog match. Far above
                                # the closest different pair's 0.926.

_head_catalog_cache = None


def head_icon_dir():
    """Assets/Heads, via _asset_path."""
    return _asset_path(HEAD_ICON_DIR)


def load_head_catalog():
    """Every Assets/Heads/*.png as a (premultiplied color, alpha) pair at its
    own native aspect ratio, keyed by filename, cached at module scope.

    Composited over black (the button's fill) and cropped to the alpha extent,
    but never squashed to a square: aspects run 0.561-1.169, and squashing
    held correct matches near 0.54 where they now reach 0.94-0.99. Capped at
    HEAD_SPRITE_MAX, which took the phase from 478 to 104 ms a shot for
    identical scores."""
    global _head_catalog_cache
    if _head_catalog_cache is not None:
        return _head_catalog_cache
    cat = {}
    for f in sorted(glob.glob(os.path.join(head_icon_dir(), "*.png"))):
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        alpha = im[:, :, 3]
        ys, xs = np.where(alpha > 8)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        a = (alpha.astype(np.float32) / 255.0)[:, :, None]
        comp = (im[:, :, :3].astype(np.float32) * a)[y0:y1, x0:x1]
        amask = alpha[y0:y1, x0:x1].astype(np.float32) / 255.0
        long_side = max(comp.shape[:2])
        if long_side > HEAD_SPRITE_MAX:
            k = HEAD_SPRITE_MAX / long_side
            wh = (max(int(round(comp.shape[1] * k)), 1),
                  max(int(round(comp.shape[0] * k)), 1))
            comp = cv2.resize(comp, wh, interpolation=cv2.INTER_AREA)
            amask = cv2.resize(amask, wh, interpolation=cv2.INTER_AREA)
        cat[os.path.basename(f)] = (comp, amask)
    _head_catalog_cache = cat
    return cat


def _head_scores(region, catalog):
    """Every catalog entry's best masked correlation against this region, as
    {filename: ncc}, searching scale and position.

    Only pixels under the sprite's own alpha are compared, so the ring, fill
    and rank badge never score. matchTemplate finds each scale's peak cheaply;
    the masked correlation is then taken in a 3x3 window around it, since the
    unmasked peak can sit a pixel or two off."""
    C = HEAD_ICON_CANON
    rr = C / (2.0 * HEAD_ICON_REGION)       # the button's radius, canonically
    scores = {}
    for name, (comp, amask) in catalog.items():
        best = -1.0
        for hr in _head_scale_sweep(name):
            th = int(round(rr * hr))
            if th < 8:
                continue
            tw = int(round(comp.shape[1] * (th / comp.shape[0])))
            if tw < 8 or tw >= C or th >= C:
                continue
            t = cv2.resize(comp, (tw, th), interpolation=cv2.INTER_AREA)
            m = cv2.resize(amask, (tw, th), interpolation=cv2.INTER_AREA) > 0.5
            if int(m.sum()) < HEAD_ICON_MIN_PIXELS:
                continue
            b = t[m].ravel()
            b = b - b.mean()
            nb = float(np.sqrt((b * b).sum()))
            if nb <= 0:
                continue
            res = cv2.matchTemplate(region, t, cv2.TM_CCOEFF_NORMED)
            oy, ox = np.unravel_index(int(res.argmax()), res.shape)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    Y, X = oy + dy, ox + dx
                    if Y < 0 or X < 0 or Y + th > C or X + tw > C:
                        continue
                    a = region[Y:Y + th, X:X + tw][m].ravel()
                    a = a - a.mean()
                    na = float(np.sqrt((a * a).sum()))
                    if na > 0:
                        best = max(best, float((a * b).sum()) / (na * nb))
        if best > -1.0:
            scores[name] = best
    return scores


def match_head_icon(region, catalog):
    """The catalog filename this icon region looks most like, as (key, ncc),
    or None if nothing clears the confidence gate.

    Correlation is over colour (BGR pooled): a head renders in one fixed
    palette for every viewer. Capture pipelines can differ by a gamut
    conversion that mean-centring does not undo, which is immaterial at the
    current margins. The floor rejects an uncatalogued icon (an Oumaji
    Khondor shot scores 0.39); the margin rejects a tie. Correct matches run
    0.83-0.99 at margins of 0.20-0.59."""
    scores = _head_scores(region, catalog)
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_name, top_ncc = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
    if top_ncc < PLAYER_HEAD_MIN_CORR or top_ncc - runner_up < PLAYER_HEAD_MIN_MARGIN:
        return None
    return top_name, top_ncc


def player_icon_region(img):
    """This screenshot's own Game Stats icon, canonically framed, or None when
    locate_game_stats_icon cannot place it."""
    loc = locate_game_stats_icon(img)
    return None if loc is None else head_icon_region(img, *loc)


def icon_similarity(a, b):
    """Plain NCC between two canonically framed icon regions, channels pooled."""
    u = a.ravel() - a.mean()
    v = b.ravel() - b.mean()
    d = float(np.sqrt((u * u).sum() * (v * v).sum()))
    return float((u * v).sum()) / d if d > 0 else 0.0


def group_shots_by_icon(region_of):
    """Group shot names by whose Game Stats icon they carry, as a list of
    lists, without consulting the catalog at all.

    Shots are compared with each other, a far easier question than naming the
    tribe, and one that works for icons the catalog lacks: same-player pairs
    correlate 0.926-1.000, different players 0.017-0.837. The one pair in
    between (vengir_cultist, 0.926: one tribe, two skins) is left to the
    catalog split in identify_players. Two players on the same tribe and skin
    are indistinguishable by any method."""
    names = [n for n in region_of if region_of[n] is not None]
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if icon_similarity(region_of[a], region_of[b]) >= HEAD_CLUSTER_MIN_NCC:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return [sorted(g) for g in groups.values()]


# Each tribe's default colour from the "Color" column of
# https://polytopia.fandom.com/wiki/Tribes (hex RGB in the comment), in the
# wiki's order, plus the two skins that change it. One row per distinct colour.
TRIBE_COLORS_BGR = (
    ("Xin-xi",   (0, 0, 204)),      # cc0000
    ("Imperius", (255, 0, 0)),      # 0000ff
    ("Bardur",   (20, 37, 53)),     # 352514
    ("Oumaji",   (0, 255, 255)),    # ffff00
    ("Kickoo",   (0, 255, 0)),      # 00ff00
    ("Hoodrick", (0, 102, 153)),    # 996600
    ("Luxidoor", (214, 59, 171)),   # ab3bd6
    ("Vengir",   (255, 255, 255)),  # ffffff
    ("Zebasi",   (0, 153, 255)),    # ff9900
    ("Ai-Mo",    (170, 226, 54)),   # 36e2aa
    ("Quetzali", (74, 92, 39)),     # 275c4a
    ("Yadakk",   (28, 35, 125)),    # 7d231c
    ("Aquarion", (129, 131, 243)),  # f38381
    ("Elyrion",  (153, 0, 255)),    # ff0099
    ("Polaris",  (133, 161, 182)),  # b6a185
    ("Cymanti",  (0, 253, 194)),    # c2fd00
    ("Aquarion (Forgotten)", (48, 140, 103)),  # 678c30, confirmed on Aquarion's own wiki page
    ("Cymanti (New Dawn)",   (79, 250, 204)),  # ccfa4f, measured off a screenshot -- not on the wiki
)
_TRIBE_COLOR_OF = dict(TRIBE_COLORS_BGR)

# Assets/Heads filename -> tribe (or colour-changing skin). Base tribes are
# coded by first letter ("ai"/"aq" for the one collision); a "2" suffix is a
# skin of that tribe, which shares its colour except aq2 and c2.
HEAD_TRIBE = {
    "x.png": "Xin-xi", "x2.png": "Xin-xi",       # x2 = Sha-po
    "i.png": "Imperius", "i2.png": "Imperius",   # i2 = Lirepacci
    "b.png": "Bardur", "b2.png": "Bardur",       # b2 = Baergoff
    "o.png": "Oumaji",
    "k.png": "Kickoo", "k2.png": "Kickoo",       # k2 = Ragoo
    "h.png": "Hoodrick", "h2.png": "Hoodrick",   # h2 = Yorthwober
    "l.png": "Luxidoor", "l2.png": "Luxidoor",   # l2 = Aumux
    "v.png": "Vengir", "v2.png": "Vengir",       # v2 = Cultist
    "z.png": "Zebasi",
    "ai.png": "Ai-Mo", "ai2.png": "Ai-Mo",       # ai2 = To-Li
    "q.png": "Quetzali", "q2.png": "Quetzali",   # q2 = Iqaruz
    "y.png": "Yadakk", "y2.png": "Yadakk",       # y2 = Urkaz
    "aq.png": "Aquarion", "aq2.png": "Aquarion (Forgotten)",
    "e.png": "Elyrion", "e2.png": "Elyrion",     # e2 = Midnight
    "p.png": "Polaris",
    "c.png": "Cymanti", "c2.png": "Cymanti (New Dawn)",
}


def tribe_default_color(player_key):
    """This catalog filename's tribe (or named skin)'s own default color
    (BGR), or None if HEAD_TRIBE doesn't know the filename."""
    return _TRIBE_COLOR_OF.get(HEAD_TRIBE.get(player_key))


PLAYER_VISION_PALETTE = (
    (255, 90, 0), (0, 140, 255), (40, 180, 40), (200, 0, 200),
    (255, 220, 0), (30, 90, 200), (0, 200, 200), (140, 100, 255),
)  # BGR for a player whose tribe colour is taken or unknown. Clear of every
   # tribe colour, RUIN_MARK_BGR and the spawn-zone red; cycles past 8.


def _assign_vision_colors(keys):
    """Map each identified player key (already sorted) to a BGR color: that
    tribe's own default when nobody else identified in this merge already
    has it, else the next unused color from PLAYER_VISION_PALETTE.

    The case is a base tribe and its own skin in one merge. First claim wins
    in sorted order, so the skin is bumped -- to the palette, never to another
    tribe's colour, which would read as that tribe."""
    used = set()
    assigned = {}
    unresolved = []
    for key in keys:
        color = tribe_default_color(key)
        if color is not None and color not in used:
            assigned[key] = color
            used.add(color)
        else:
            unresolved.append(key)
    spare = 0
    for key in unresolved:
        color = PLAYER_VISION_PALETTE[spare % len(PLAYER_VISION_PALETTE)]
        spare += 1
        assigned[key] = color
        used.add(color)
    return assigned


# (di, dj, vertex offset, vertex offset): the neighbor at (i+di, j+dj) lies
# across the edge between lattice vertices (i,j)+offset0 and (i,j)+offset1,
# vertex (a,b) being origin + a*u_col + b*u_row.
_VISION_EDGE_DIRS = (
    (-1, 0, (0, 0), (0, 1)),   # west edge, border with tile (i-1, j)
    (1, 0, (1, 0), (1, 1)),    # east edge, border with tile (i+1, j)
    (0, -1, (0, 0), (1, 0)),   # north edge, border with tile (i, j-1)
    (0, 1, (0, 1), (1, 1)),    # south edge, border with tile (i, j+1)
)

VISION_BANDS_PER_EDGE = 6   # colour bands per tile edge that several players'
                            # boundaries share


def _draw_vision_edge(out, p0, p1, colors, thick):
    """One tile-edge segment, solid if only one player's boundary reaches it,
    else banded through every player whose boundary does. Every edge divides
    evenly, so a straight shared frontier reads as one continuous stripe."""
    if len(colors) == 1:
        cv2.line(out, tuple(np.round(p0).astype(int)),
                 tuple(np.round(p1).astype(int)), colors[0], thick, cv2.LINE_AA)
        return
    n = VISION_BANDS_PER_EDGE
    for k in range(n):
        a = p0 + (p1 - p0) * (k / n)
        b = p0 + (p1 - p0) * ((k + 1) / n)
        cv2.line(out, tuple(np.round(a).astype(int)),
                 tuple(np.round(b).astype(int)), colors[k % len(colors)],
                 thick, cv2.LINE_AA)


VISION_FADE_FRAC = 0.25    # how far a frontier's color wash reaches into the
                           # interior, as a fraction of one tile width
VISION_FADE_ALPHA = 0.75   # wash opacity immediately behind the frontier line


def _draw_vision_fade(out, explored, keys_sorted, color_of, origin, u_col, u_row):
    """Blend each player's colour into their own territory, fading out over
    VISION_FADE_FRAC of a tile from the frontier (a distance transform of the
    player's tile set). The mask is first opened with a disk of the fade's
    size so convex corners wrap in an arc rather than a miter -- a look, not a
    truer distance."""
    h, w = out.shape[:2]
    tile_px = (float(np.linalg.norm(u_col)) + float(np.linalg.norm(u_row))) / 2
    fade_px = VISION_FADE_FRAC * tile_px
    pad = int(np.ceil(fade_px)) + 2
    corner_r = max(round(fade_px), 1)
    corner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (2 * corner_r + 1, 2 * corner_r + 1))
    for key in keys_sorted:
        tiles = explored[key]
        ij = np.array(list(tiles))
        i0, j0 = ij[:, 0].min(), ij[:, 1].min()
        i1, j1 = ij[:, 0].max() + 1, ij[:, 1].max() + 1
        corners = np.stack([origin + i0 * u_col + j0 * u_row,
                             origin + i1 * u_col + j0 * u_row,
                             origin + i0 * u_col + j1 * u_row,
                             origin + i1 * u_col + j1 * u_row])
        x0 = max(int(np.floor(corners[:, 0].min())) - pad, 0)
        y0 = max(int(np.floor(corners[:, 1].min())) - pad, 0)
        x1 = min(int(np.ceil(corners[:, 0].max())) + pad, w)
        y1 = min(int(np.ceil(corners[:, 1].max())) + pad, h)
        if x1 <= x0 or y1 <= y0:
            continue
        sub_origin = origin - [x0, y0]
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for (i, j) in tiles:
            poly = tile_poly(sub_origin, u_col, u_row, i, j)
            cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
        rounded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, corner_kernel)
        dist = cv2.distanceTransform(rounded, cv2.DIST_L2, 5)
        # Background also reads distance 0, so force its alpha to 0.
        a = np.where(rounded > 0, np.clip(1 - dist / fade_px, 0, 1), 0.0)
        a = (a * VISION_FADE_ALPHA)[..., None]
        region = out[y0:y1, x0:x1].astype(np.float32)
        col = np.array(color_of[key], np.float32)
        out[y0:y1, x0:x1] = (region * (1 - a) + col * a).astype(np.uint8)


def _player_explored_tiles(samples, shot_names):
    """Tiles any of `shot_names` classified as explored: one player's own
    witnessed territory, from the same samples winner selection used."""
    return {ij for ij, per in samples.items()
            if any(per.get(n, {}).get("explored") for n in shot_names)}


def draw_player_vision(out, samples, by_player, origin, u_col, u_row, thick):
    """Outline, in a distinct color per player, the union of tiles each
    identified player's own shot(s) witnessed as explored, over a fading wash
    (_draw_vision_fade). `by_player` maps a player key to that player's shot
    names. Drawn per tile edge rather than per-player contour, so an edge on
    several players' frontiers is banded in all their colours instead of
    showing whoever was drawn last."""
    explored = {key: _player_explored_tiles(samples, shot_names)
                for key, shot_names in by_player.items()}
    keys_sorted = [k for k in sorted(explored) if explored[k]]
    color_of = _assign_vision_colors(keys_sorted)

    edge_players = {}   # (vertex, vertex) -> [player key, ...], insertion order
    for key in keys_sorted:
        tiles = explored[key]
        for (i, j) in tiles:
            for di, dj, v0off, v1off in _VISION_EDGE_DIRS:
                if (i + di, j + dj) in tiles:
                    continue          # interior to this player's own region
                v0 = (i + v0off[0], j + v0off[1])
                v1 = (i + v1off[0], j + v1off[1])
                edge = (v0, v1) if v0 <= v1 else (v1, v0)
                edge_players.setdefault(edge, []).append(key)

    _draw_vision_fade(out, explored, keys_sorted, color_of, origin, u_col, u_row)
    for (v0, v1), keys in edge_players.items():
        p0 = origin + v0[0] * u_col + v0[1] * u_row
        p1 = origin + v1[0] * u_col + v1[1] * u_row
        _draw_vision_edge(out, p0, p1, [color_of[k] for k in keys], thick)


# `vision-each` marks unseen tiles with a flat white wash rather than fog art,
# whose texture competes with the terrain beneath it at partial opacity.
VISION_EACH_WASH_BGR = (255, 255, 255)
VISION_EACH_WASH_ALPHA = 0.55   # the terrain must stay visible underneath


def _vision_each_unseen_mask(winner_tiles, explored_self, origin, u_col, u_row, W, Hc):
    """uint8 mask of the tiles in the explored union `winner_tiles` that this
    player's own `explored_self` lacks. Tiles nobody explored keep plain fog."""
    mask = np.zeros((Hc, W), np.uint8)
    for (i, j) in winner_tiles:
        if (i, j) in explored_self:
            continue
        poly = tile_poly(origin, u_col, u_row, i, j, 0.0)
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
    return mask


def render_vision_each(out, unseen_mask, color=VISION_EACH_WASH_BGR,
                        alpha=VISION_EACH_WASH_ALPHA):
    """`out` with a translucent `color` wash wherever `unseen_mask` is set."""
    a = (unseen_mask.astype(np.float32) / 255.0 * alpha)[..., None]
    blended = out.astype(np.float32) * (1.0 - a) + np.array(color, np.float32) * a
    return np.clip(blended, 0, 255).astype(np.uint8)


def vision_each_slug(key):
    """Filesystem-safe stem for a vision-each output file, from a by_player
    key: a catalog filename ("i2.png") or "unnamed player N"."""
    if key.endswith(".png"):
        return key[:-4]
    digits = "".join(ch for ch in key if ch.isdigit())
    return "unnamed" + (digits or "0")


# ---------------------------------------------- board renders & templates ---
# Overlay/template file resolution and loading, plus the per-render and
# per-shot geometry caches (template_geometry, ShotCache) that both the map
# size detection below and the main anchoring path draw on.
OVERLAY_DIR = "Overlays"


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _asset_path(rel):
    """`rel` under the working directory if it exists there, else under this
    script's directory (existing or not, so errors name a real path). The
    fallback is what finds assets from the bot, which runs polymerge in a
    temp dir."""
    return rel if os.path.exists(rel) else os.path.join(_SCRIPT_DIR, rel)


def _asset_glob(pattern):
    """The first match of `pattern` under the working directory, else under
    this script's directory, or None. Same search order as _asset_path."""
    for base in (os.curdir, _SCRIPT_DIR):
        hits = sorted(glob.glob(os.path.join(base, pattern)))
        if hits:
            return hits[0]
    return None


def overlay_path(n, layer):
    """Where Overlays/<name>-<layer>.png lives (see _asset_path). None when
    the board size has no such name."""
    name = BOARD_SIZE_NAMES.get(n)
    if name is None:
        return None
    return _asset_path(os.path.join(OVERLAY_DIR, f"{name}-{layer}.png"))


def template_path_for(n):
    """The blank all-fog render for an NxN board, or None at a size with no
    name. Deliberately no fallback: a missing render must fail by name, not
    merge silently against other fog art. --template overrides."""
    return overlay_path(n, "blank")


# The decorative layers, bottom-up in paint order, and their Overlays/ stems.
# Ruin markers are drawn after all of them (paste_composite).
OVERLAY_LAYERS = (("shade", "shaded"),
                  ("grid", "gridded"),
                  ("spawns", "*spawns"),
                  ("push", "push"))

# Layers that fill tiles are clipped to fog, where they help, and kept off
# explored terrain, where they would dull it; the grid and push arrows are
# read against the map and cover the whole board.
OVERLAY_FOG_ONLY = frozenset({"shade", "spawns"})

# Opacity on top of the file's own alpha; the grid lies over explored terrain.
OVERLAY_ALPHA = {"grid": 0.7}

# The CLI default. polybot always passes --overlays (default: none).
OVERLAY_DEFAULT = "shade"


def overlay_layer_path(n, stem):
    """Resolve one layer file for an NxN board, or None if it has none.

    `stem` may hold a `*`: the spawn layer's filename encodes its zone grid
    (2spawns, 3spawns), which is matched rather than assumed."""
    name = BOARD_SIZE_NAMES.get(n)
    if name is None:
        return None
    return _asset_glob(os.path.join(OVERLAY_DIR, f"{name}-{stem}.png"))


def _read_png(path):
    """An image with any alpha kept and 16-bit (normal-*.png) normalized to
    8, or None."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is not None and im.dtype == np.uint16:
        im = (im / 257.0).astype(np.uint8)
    return im


def load_overlay(path, shape):
    """One decorative layer as (bgr, alpha) in 0..1 float, or None.

    Layers share their blank's exact frame, so none is warped; a layer whose
    canvas differs from the template's is the wrong file and is rejected."""
    im = _read_png(path)
    if im is None:
        return None
    if im.ndim != 3 or im.shape[2] != 4 or im.shape[:2] != shape[:2]:
        return None
    return im[:, :, :3].astype(np.float32), \
        (im[:, :, 3].astype(np.float32) / 255.0)[:, :, None]


def paint_overlays(out, wanted, n, fog_mask=None):
    """Alpha-blend the requested layers onto the composite, in OVERLAY_LAYERS
    order. Returns the names that had no file for this board size.

    Straight, not premultiplied, alpha (the push and spawn layers need it).
    Layers in OVERLAY_FOG_ONLY are clipped to `fog_mask`, the unexplored
    tiles."""
    missing = []
    for name, stem in OVERLAY_LAYERS:
        if name not in wanted:
            continue
        path = overlay_layer_path(n, stem)
        layer = load_overlay(path, out.shape) if path else None
        if layer is None:
            missing.append(name)
            continue
        bgr, a = layer
        if name in OVERLAY_FOG_ONLY and fog_mask is not None:
            a = a * fog_mask
        a = a * OVERLAY_ALPHA.get(name, 1.0)
        np.copyto(out, np.clip(out.astype(np.float32) * (1.0 - a) + bgr * a,
                               0, 255).astype(np.uint8))
    return missing


def load_template(path, dark_thresh, erode_px):
    """A template's BGR pixels and its silhouette masks, as
    (bgr, valid_t, edge_t).

    The renders are premultiplied against black, so dropping alpha leaves the
    black-sky image the pipeline expects. The silhouette comes from the same
    brightness test the screenshots get; **never from the alpha channel**,
    which cuts the antialiased fringe and moved the tile step enough to cost a
    set most of its fog lock."""
    if not path:
        return None, None, None      # a board size with no render of its own
    im = _read_png(path)
    if im is None:
        return None, None, None
    if im.ndim == 2:
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        bgr = np.ascontiguousarray(im[:, :, :3])
    edge_t = build_valid_mask(bgr, [], dark_thresh, 0)
    valid_t = build_valid_mask(bgr, [], dark_thresh, erode_px)
    return bgr, valid_t, edge_t


_template_cache = {}


def template_geometry(path, dark_thresh, erode_px):
    """One render's pixels plus everything geometric derived from them, once.

    Shared by the size pre-pass (_probe_basis) and the merge, which would
    otherwise each pay ~300 ms at 20x20. Keyed on every input that can change
    the answer, (path, dark_thresh, erode_px).

    Returns None for a render that is not on disk, exactly as load_template
    does, so the caller still owns the refusal."""
    key = (path, dark_thresh, erode_px)
    if key not in _template_cache:
        bgr, valid_t, edge_t = load_template(path, dark_thresh, erode_px)
        if bgr is None:
            _template_cache[key] = None
        else:
            corners = detect_corners(edge_t)
            _template_cache[key] = {
                "bgr": bgr, "valid": valid_t, "edge": edge_t,
                "corners": corners,
                "edges": edge_lines(board_boundary(edge_t)),
            }
    return _template_cache[key]


def output_crop(origin, u_col, u_row, n, W, Hc, pad=10):
    """The rectangle paste_composite crops the canvas to, shared with --base's
    size check and base_output_size so they cannot drift from it."""
    full = np.float32([origin, origin + n * u_col,
                       origin + n * u_col + n * u_row, origin + n * u_row])
    x0c, y0c = np.floor(full.min(0)).astype(int)
    x1c, y1c = np.ceil(full.max(0)).astype(int)
    x0c, y0c = max(x0c - pad, 0), max(y0c - pad, 0)
    x1c, y1c = min(x1c + pad, W), min(y1c + pad, Hc)
    return x0c, y0c, x1c, y1c


def base_output_size(shape_hw, dark_thresh, erode_px):
    """Which board size's own composite has exactly these pixel dimensions,
    or None if none does.

    Exact match only: an 18x18 composite scaled ~1.1x lands within a pixel of
    20x20's size, so a resized --base is refused, never guessed at. Largest
    first, as in _probe_basis, so a 20x20 base loads one template."""
    h, w = shape_hw
    hit = None
    for n in sorted(MAP_SIZE_CHOICES, reverse=True):
        tgeom = template_geometry(template_path_for(n), dark_thresh, erode_px)
        if tgeom is None:
            continue
        t_top, t_right, _t_bottom, t_left, _res = tgeom["corners"]
        origin, u_col, u_row = build_lattice(t_top, t_right, t_left, n)
        Wt, Ht = tgeom["bgr"].shape[1], tgeom["bgr"].shape[0]
        x0c, y0c, x1c, y1c = output_crop(origin, u_col, u_row, n, Wt, Ht)
        if (x1c - x0c, y1c - y0c) == (w, h):
            if hit is not None:
                return None  # ambiguous -- refuse rather than guess
            hit = n
    return hit


class ShotCache:
    """Per-shot measurements several parts of a run all want: the board
    outline, and the fog's repeat period.

    Both are wanted by detect_map_size and again by anchor_to_template. **The
    key carries every input that can change the answer** -- the mask (via a
    per-shot generation counter), the basis and crop bands for the outline,
    dir_a and tile_px (which sets the sweep's phase) for the period -- so a
    hit is bit-identical and a miss recomputes. On a 20x20 board the pre-pass
    parameters match the anchor's and every key hits; elsewhere nothing does.

    Masks are not hashable, so anything that replaces a shot's masks must call
    `invalidate`."""

    def __init__(self):
        self._gen, self._boundary, self._period = {}, {}, {}

    def invalidate(self, name):
        """Call after replacing a shot's masks; see anchor_to_template."""
        self._gen[name] = self._gen.get(name, 0) + 1

    def boundary(self, name, mask, dirs, top_crop=0.0, bottom_crop=0.0):
        key = (name, self._gen.get(name, 0), dirs[0].tobytes(), dirs[1].tobytes(),
               top_crop, bottom_crop)
        if key not in self._boundary:
            self._boundary[key] = board_boundary(mask, dirs, top_crop=top_crop,
                                                 bottom_crop=bottom_crop)
        return self._boundary[key]

    def period(self, name, img, valid, hsv, dir_a, tile_px):
        """(s_it, period_px, ncc) or None -- fog_period_scale's own answer.

        Takes the image rather than its gray conversion so a hit skips that
        too."""
        key = (name, self._gen.get(name, 0), dir_a.tobytes(), float(tile_px))
        if key not in self._period:
            self._period[key] = fog_period_scale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), valid, hsv, dir_a, tile_px)
        return self._period[key]


class Shot:
    """One screenshot: its image, masks, anchor and warp -- everything the merge
    computes about it that is read more than once.

    - Masks (valid, valid_raw, edge_mask, frame) are built at construction
      and rebuilt in place by build_masks for the sunrise-sky fallback;
      frame_raw and hsv never change. badge_mask/badge_halo come from
      detect_badge (None with no badge).
    - to_template and its warp (warped, wmask, wmask_raw, pmask, pmask_raw,
      scale) are set by anchoring and replaced together by warp_shot; all stay
      None on a dropped shot. `prior` is the unrefined anchor.
    - gain/fogpix come from fit_fog_pixels (gain None when too little fog
      locked); bars/ruins are None until their phase runs; sift_features is
      memoized by sift_hops.
    - sift_mask() and terrain_mask() are lazy methods because they need only
      the shot itself and fixed constants."""
    __slots__ = ("img", "hsv", "valid", "valid_raw", "frame", "frame_raw",
                 "edge_mask", "badge_mask", "badge_halo",
                 "to_template", "zoom_source", "implied_n", "prior",
                 "scale", "warped", "wmask", "wmask_raw", "pmask", "pmask_raw",
                 "gain", "fogpix", "bars", "ruins",
                 "_sift_mask", "_terrain_mask", "sift_features")

    def __init__(self, img, args, rects):
        self.img = img
        # The paste-time counterpart of valid (see build_frame_mask).
        self.frame_raw = build_frame_mask(img, rects, args.top_crop,
                                          args.bottom_crop)
        self.badge_mask = None
        self.badge_halo = None
        self.build_masks(args, rects)
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.gain = None
        self.fogpix = None
        self.bars = None
        self.ruins = None
        self._terrain_mask = None
        self.sift_features = None
        self.to_template = None
        self.zoom_source = None
        self.implied_n = None
        self.prior = None
        self.scale = None
        self.warped = None
        self.wmask = None
        self.wmask_raw = None
        self.pmask = None
        self.pmask_raw = None

    def build_masks(self, args, rects, drop_sky=False):
        """(Re)build the two brightness masks for this shot, in place.

        Called from __init__, and again by sky_rebuild_for with the
        sunrise-sky test."""
        self.valid_raw = build_valid_mask(self.img, rects, args.dark_thresh,
                                          args.erode_px, args.top_crop,
                                          args.bottom_crop, drop_sky=drop_sky)
        # Geometry uses the un-eroded mask: erosion eats a zoom-dependent
        # amount of board and would bias the scale.
        self.edge_mask = build_valid_mask(self.img, rects, args.dark_thresh,
                                          0, args.top_crop, args.bottom_crop,
                                          drop_sky=drop_sky)
        # sift_mask() is derived from the masks just replaced.
        self._sift_mask = None
        self._subtract_badges()

    def _subtract_badges(self):
        """Take the capture badge off valid/frame and its halo off edge_mask,
        after every mask build. Forgetting the halo on a rebuild brings the
        phantom edge back (see badge_halo)."""
        badge = self.badge_mask
        self.valid = self.valid_raw & ~badge if badge is not None else self.valid_raw
        self.frame = self.frame_raw & ~badge if badge is not None else self.frame_raw
        if self.badge_halo is not None:
            self.edge_mask = self.edge_mask & ~self.badge_halo

    def detect_badge(self):
        """Find this shot's capture badge, if any, and exclude it from the
        masks that must not see it.

        Returns the blob list for the caller to log. badge_mask is set even
        when empty, since --debug-dir writes badges_<name>.png for every
        shot."""
        badge, found = detect_capture_badges(self.img, self.valid)
        if found:
            self.badge_halo = badge_halo(badge, found)
        self.badge_mask = badge
        self._subtract_badges()
        return found

    def sift_mask(self):
        """Pixels SIFT may take features from: the board, and nothing else.

        Two shots of one replay have identical UI, which matches perfectly
        (replay_ss2: 330 inliers on an identity transform against the real
        match's 113). Lazy, since only SIFT paths need it."""
        if self._sift_mask is None:
            self._sift_mask = self.valid & board_region(
                self.edge_mask, (BOARD_DIR_A, BOARD_DIR_B))
        return self._sift_mask

    def terrain_mask(self):
        """Pixels saturated enough that they cannot be the fog cube.

        Only counts SIFT inliers on terrain (SIFT_TERRAIN_MIN_INLIERS); it
        never classifies a tile."""
        if self._terrain_mask is None:
            self._terrain_mask = self.hsv[:, :, 1] >= PIXEL_FOG_SAT
        return self._terrain_mask


# ------------------------------------------------------ map size detection ---
def _probe_basis(dark_thresh, erode_px=0):
    """The board's projection directions, tile step and per-direction span,
    from any template.

    Nothing measured with these depends on which template supplied them:
    `tile_px` only centres fog_period_scale's 12x-wide sweep, and `span`
    only normalizes two edge pairs against each other (a and b differ by
    0.17% even on a square board). Returns None when no template is on disk.

    Largest first, so on a 20x20 board ShotCache's keys match the anchor's."""
    for n in sorted(MAP_SIZE_CHOICES, reverse=True):
        g = template_geometry(template_path_for(n), dark_thresh, erode_px)
        if g is None:
            continue
        t_top, t_right, _, t_left, _ = g["corners"]
        dir_a, dir_b = BOARD_DIR_A, BOARD_DIR_B
        _, u_col, _ = build_lattice(t_top, t_right, t_left, n)
        t_off, _ = g["edges"]
        span = [t_off[1] - t_off[0], t_off[3] - t_off[2]]
        return dir_a, dir_b, float(np.linalg.norm(u_col)), span
    return None


def _no_measurement_reason(why):
    """Why no shot could be measured, phrased for the person who has to fix it.

    Counting tiles needs both an opposite edge pair and readable fog, and the
    two failures want different advice. This is channel copy: the first line
    is polybot's headline, carrying the cause and the remedy and nothing about
    the mechanism (CLAUDE.md, the bot's refusal conventions)."""
    tail = " " + RESTATE_SIZE.format(size_list())
    kinds = set(why.values())
    if kinds == {"no-fog"}:
        # No re-photographing remedy: there is no fog to bring back.
        return "these screenshots have no fog left to measure." + tail + "."
    if kinds == {"no-span"}:
        return ("no screenshot spans the whole board." + tail
                + ", or with at least one screenshot that shows two opposite "
                  "sides of the board.")
    # Otherwise name the reason per shot, so the summary cannot contradict the
    # detail lines above it.
    said = {"no-fog": "no fog left to measure against",
            "no-span": "does not span the board",
            "weak-pair": "its only edge pair is too weak to trust",
            "phantom-pair": "its edge pairs disagree, so one is a phantom",
            "no-outline": "no board outline found"}
    detail = "; ".join(f"{n} ({said.get(w, w)})" for n, w in sorted(why.items()))
    return f"no screenshot could be measured -- {detail}.{tail}."


def detect_map_size(names, imgs, edge_mask, valid, hsv, dark_thresh,
                    min_support, min_scale_support, erode_px=0, cache=None,
                    top_crop=0.0, bottom_crop=0.0):
    """Measure the board's size off the screenshots themselves.

    A shot spanning the board in one direction counts its tiles as span / fog
    period, both in its own pixels, so no template is involved. The sizes are
    at least 2 apart and honest readings land within 0.32 tiles. The
    phantom-edge rules are the same as anchor_to_template's, and with no
    measurement this raises rather than guess: a wrong size is the most
    destructive failure there is. See CLAUDE.md, "Detecting the map size
    outright". `cache` shares the measurements with anchoring (ShotCache);
    `top_crop`/`bottom_crop` must match the masks'."""
    cache = cache if cache is not None else ShotCache()
    basis = _probe_basis(dark_thresh, erode_px)
    if basis is None:
        # An install fault: the headline says so, the path goes on line two.
        raise SystemExit("this is not installed correctly, so it cannot work "
                         "the board size out.\n(no Overlays/<name>-blank.png "
                         "on disk to take the board's projection from -- see "
                         "the Dockerfile.)")
    dir_a, dir_b, tile_px, t_span = basis
    tags = ["a-min", "a-max", "b-min", "b-max"]
    implied = {}
    # Why each abstaining shot abstained: the refusal is built from these, and
    # it is all a player sees (these prints go to stdout).
    why = {}
    with PHASES("detect map size"):
        for n in names:
            pts = cache.boundary(n, edge_mask[n], (dir_a, dir_b), top_crop, bottom_crop)
            if len(pts) < 100:
                print(f"  {n}: no board outline -- no size measurement")
                why[n] = "no-outline"
                continue
            off, support = edge_lines(pts)
            have = [c >= min_scale_support for c in support]
            pairs = [(k, lo, hi) for k, (lo, hi) in enumerate([(0, 1), (2, 3)])
                     if have[lo] and have[hi]]
            # anchor_to_template's phantom-edge rules.
            if len(pairs) == 1:
                _, lo, hi = pairs[0]
                if min(support[lo], support[hi]) < min_support:
                    print(f"  {n}: sole edge pair too weak to trust "
                          f"({support[lo]}/{support[hi]}, need {min_support}) "
                          f"-- no size measurement")
                    why[n] = "weak-pair"
                    continue
            elif not pairs:
                seen = " ".join(t for t, h in zip(tags, have) if h) or "none"
                print(f"  {n}: no opposite edge pair [{seen}] -- this shot does "
                      f"not span the board, so it cannot count its tiles")
                why[n] = "no-span"
                continue
            got = cache.period(n, imgs[n], valid[n], hsv[n], dir_a, tile_px)
            if got is None:
                print(f"  {n}: no periodic fog to measure the tile step "
                      f"against -- no size measurement")
                why[n] = "no-fog"
                continue
            period = got[1]
            spans = [off[hi] - off[lo] for _, lo, hi in pairs]
            if len(spans) == 2:
                # Normalized per direction, as anchor_to_template does.
                scales = [s / t_span[k] for s, (k, _, _) in zip(spans, pairs)]
                spread = abs(scales[0] / scales[1] - 1)
                if spread > EDGE_PAIR_MAX_SPREAD:
                    print(f"  {n}: two edge pairs disagree by "
                          f"{100 * spread:.1f}% -- at least one edge is a "
                          f"phantom, no size measurement")
                    why[n] = "phantom-pair"
                    continue
            implied[n] = float(np.mean(spans)) / period - BOARD_SPAN_WALL_TILES
            print(f"  {n}: board spans {implied[n]:.2f} tiles "
                  f"(fog period {period:.1f}px)")

    if not implied:
        raise SystemExit("cannot detect the map size: "
                         + _no_measurement_reason(why))
    # A number near no real size is broken, not dissenting (missized_test).
    implied, discarded = plausible_sizes(implied)
    if discarded:
        print(implausible_note(discarded))
    if not implied:
        detail = "  ".join(f"{n}={v:.2f}" for n, v in sorted(discarded.items()))
        raise SystemExit(
            "cannot detect the map size: no screenshot measured anything "
            "close to a real board size. " + RESTATE_SIZE.format(size_list()) + "."
            + f"\n(measured {detail})")
    vals = sorted(implied.values())
    spread = vals[-1] - vals[0]
    if spread > MAP_SIZE_DETECT_MAX_SPREAD:
        raise SystemExit(
            f"the screenshots disagree about the board size by {spread:.2f} "
            f"tiles -- are these all screenshots of the same board? "
            + RESTATE_SIZE.format(size_list()) + "."
            + f"\n(measured {', '.join(f'{k} {v:.2f}' for k, v in implied.items())})")
    med = float(np.median(vals))
    size = int(round(med))
    if size not in MAP_SIZE_CHOICES:
        raise SystemExit(
            f"this looks like a {size}x{size} board, which is not a size this "
            f"can merge. The sizes it handles are " + size_list()
            + f".\n(measured {med:.2f} tiles across)")
    print(f"detected map size: {size}x{size} (measured {med:.2f} tiles across "
          f"{len(implied)} of {len(names)} shot(s))")
    return size


# -------------------------------------------------------------------- main ---
# A tile is "locked" when it matches the fog art this well: far above
# --fog-ncc, since it asks whether the lattice is right (real fog scores 0.95+).
FOG_LOCK_NCC = 0.7

# Fog-locked pixels a shot needs before its illumination is fitted; below it
# the shot gets no fog-pixel evidence (see fit_fog_pixels).
FOG_GAIN_MIN_PX = 5000


class Board(collections.namedtuple(
        "Board", "template_path template tmpl_gray dir_a dir_b origin u_col "
                 "u_row N W Hc t_corners t_edge_off tile_px base_rect")):
    """The board this run merges onto: its template and the tile lattice
    derived from it. Built once by load_board and never replaced afterwards
    (the fields cannot be reassigned; the arrays they hold are not copied, so
    do not write into them).

    The methods are the module-level lattice helpers with this board's
    geometry filled in. Those helpers keep their own signatures because
    shoreline/ imports them."""
    __slots__ = ()

    @property
    def lattice(self):
        """(origin, u_col, u_row), for helpers that take the three."""
        return self.origin, self.u_col, self.u_row

    def tile_poly(self, i, j, inset=0.0):
        return tile_poly(self.origin, self.u_col, self.u_row, i, j, inset)

    def tile_top_wedge(self, i, j):
        return tile_top_wedge(self.origin, self.u_col, self.u_row, i, j)

    def tile_mask_bbox(self, i, j, inset=0.0):
        return tile_mask_bbox(self.origin, self.u_col, self.u_row, i, j,
                              self.W, self.Hc, inset)

    def tile_of_point(self, xy):
        return tile_of_point(xy, self.origin, self.u_col, self.u_row)


class Run:
    """The state main()'s phases share, one attribute per cross-phase value.

    Each phase reads what it needs off this and stores what later phases
    need; see main() for the order. Attributes start as None (or empty) and
    are filled in by the phase named beside them."""

    def __init__(self, args, overlays):
        self.args = args
        self.overlays = overlays
        # load_inputs
        self.ui = self.shots = self.names = self.badge_found = None
        self.base_bgr = None
        # select_map_shots
        self.all_names = None
        # resolve_map_size
        self.size_was_detected = None
        self.shot_cache = None
        # load_board
        self.board = None
        # sample_tiles onward
        self.samples = {}
        self.fog_lock = None
        self.size_unverified = False
        self.player_of = {}
        self.no_head_catalog = False
        self.winner = self.priority = None
        self.badge_fallback = self.fog_demoted = None
        self.bar_promoted = self.vision_promoted = None
        self.capped_of = {}
        self.ruin_hits = {}
        self.n_raw = 0
        self.no_ruin_sprite = False
        self.ruin_no_fog = []
        self.conflicts = None
        self.comparable = 0
        self.out = None
        self.thick = None
        self.by_player = {}
        self.crop = None


def sky_rebuild_for(run, n):
    """Rebuild one shot's masks with the sunrise-sky test.

    Called by anchor_to_template only when the ordinary masks fail. Writes
    into the Shot as well as returning, so everything downstream sees the
    masks the anchor was fitted on."""
    args, ui, shots = run.args, run.ui, run.shots
    def rebuild():
        shots[n].build_masks(args, ui.get(n, []), drop_sky=True)
        return shots[n].edge_mask, shots[n].valid
    return rebuild


def anchor_all(run):
    """Anchor every shot, in two passes: independently first, then a SIFT
    zoom hint for whatever could not manage it alone.

    Shared by --cross-check and the merge so the two cannot diverge."""
    args, names, shots, shot_cache, board = (
        run.args, run.names, run.shots, run.shot_cache, run.board)
    M_of, src_of, implied_of, scale_of, failed = {}, {}, {}, {}, []
    prior_of = {}

    def _record(n, M, implied):
        """File one shot's anchor into M_of/scale_of/implied_of."""
        M_of[n] = M
        scale_of[n] = float(np.hypot(M[0, 0], M[1, 0]))
        if implied is not None:
            implied_of[n] = implied

    for n in names:
        print(f"anchoring {n}:")
        try:
            s = shots[n]
            M, src_of[n], implied, prior_of[n] = anchor_to_template(
                s.img, s.edge_mask, s.valid, s.hsv, board.tmpl_gray,
                board.t_edge_off, board.dir_a, board.dir_b, *board.lattice,
                args.map_size,
                args.min_edge_support, n, refine=not args.no_refine,
                min_scale_support=args.min_scale_support,
                sky_rebuild=sky_rebuild_for(run, n), cache=shot_cache,
                top_crop=args.top_crop, bottom_crop=args.bottom_crop)
            _record(n, M, implied)
        except SystemExit as e:
            print(f"  no self-anchor: {e}")
            failed.append(n)
    if failed and M_of:
        # Keep the re-anchor outside this: nested PHASES double-count.
        with PHASES("anchor: SIFT zoom fallback"):
            feats = {n: sift_features(shots[n].img, shots[n].sift_mask(), args.nfeatures,
                                      args.contrast)
                     for n in list(M_of) + failed}
            hints = {}
            for n in failed:
                # Ranked and gated on terrain inliers only.
                best = (0, 0, None, None)
                for m in M_of:
                    M_nm, inl, terr = pair_transform(
                        *feats[n], *feats[m], args.ratio, args.reproj,
                        terrain=(shots[n].terrain_mask(), shots[m].terrain_mask()))
                    if M_nm is not None and terr > best[0]:
                        best = (terr, inl, m, M_nm)
                hints[n] = best
        for n in list(failed):
            print(f"re-anchoring {n} from an already-anchored shot:")
            terr, inl, m, M_nm = hints[n]
            if terr < SIFT_TERRAIN_MIN_INLIERS:
                print(f"  dropped -- best SIFT match has only {terr} "
                      f"inliers on terrain (need "
                      f"{SIFT_TERRAIN_MIN_INLIERS}; {inl} counting fog)")
                continue
            k = float(np.hypot(M_nm[0, 0], M_nm[1, 0]))
            print(f"  {inl} SIFT inliers against {m}, {terr} of them on "
                  f"terrain (relative scale {k:.4f})")
            # A complete n -> template transform: its scale is the zoom hint
            # and its translation the pan hint, so both are at one zoom.
            borrowed = (to_h(M_of[m]) @ to_h(M_nm))[:2]
            try:
                s = shots[n]
                M, src_of[n], implied, prior_of[n] = anchor_to_template(
                    s.img, s.edge_mask, s.valid, s.hsv, board.tmpl_gray,
                    board.t_edge_off, board.dir_a, board.dir_b, *board.lattice,
                    args.map_size, args.min_edge_support, n,
                    refine=not args.no_refine,
                    min_scale_support=args.min_scale_support,
                    # M_nm maps n's pixels to m's, so 1 n-px = k m-px,
                    # and m's own anchor converts those to template px
                    zoom_hint=scale_of[m] * k,
                    sky_rebuild=sky_rebuild_for(run, n),
                    pan_hint=borrowed, cache=shot_cache,
                    top_crop=args.top_crop, bottom_crop=args.bottom_crop)
                _record(n, M, implied)
                failed.remove(n)
            except SystemExit as e:
                print(f"  dropped -- {e}")
    return M_of, src_of, implied_of, prior_of


def warp_shot(run, n):
    """Put one shot on the canvas with all its masks; rerun whenever its
    anchor changes."""
    shots, board = run.shots, run.board
    s = shots[n]
    Mn = s.to_template
    size = (board.W, board.Hc)
    s.warped = cv2.warpAffine(s.img, Mn[:2], size, flags=cv2.INTER_LANCZOS4)
    s.wmask = cv2.warpAffine(s.valid, Mn[:2], size, flags=cv2.INTER_NEAREST)
    s.wmask_raw = cv2.warpAffine(s.valid_raw, Mn[:2], size, flags=cv2.INTER_NEAREST)
    s.pmask = cv2.warpAffine(s.frame, Mn[:2], size, flags=cv2.INTER_NEAREST)
    s.pmask_raw = cv2.warpAffine(s.frame_raw, Mn[:2], size, flags=cv2.INTER_NEAREST)
    s.scale = float(np.hypot(Mn[0, 0], Mn[1, 0]))


def sample_shot(run, n):
    """Classify every tile for one shot, replacing whatever it said before."""
    args, shots, samples, board = run.args, run.shots, run.samples, run.board
    shot = shots[n]
    for i in range(board.N):
        for j in range(board.N):
            s = sample_tile(shot.warped, shot.wmask, board.tmpl_gray,
                            board.tile_poly(i, j, args.tile_inset),
                            args.fog_ncc, args.min_valid_frac,
                            wedge_poly=board.tile_top_wedge(i, j),
                            fog_wedge_ncc=args.fog_wedge_ncc)
            per = samples.setdefault((i, j), {})
            per.pop(n, None)
            if s is not None:
                per[n] = s


# How many of n's tiles lock onto the fog art (FOG_LOCK_NCC). With a wrong
# lattice almost nothing locks; see --min-fog-lock in CLAUDE.md.
def locked(run, n):
    samples = run.samples
    return sum(1 for s in samples.values()
               if s.get(n, {}).get("fog_ncc", 0.0) >= FOG_LOCK_NCC)


def sift_hops(run, n, witnesses):
    """Where each anchored shot's SIFT geometry says n belongs.

    One (inliers, m, anchor implied for n, gap from n's own anchor in
    tiles) per witness clearing SIFT_TERRAIN_MIN_INLIERS, most inliers
    first. The gap is --cross-check's measurement. Features are memoized on
    the Shot, separately from anchor_all's (no corpus set reaches this)."""
    args, shots, board = run.args, run.shots, run.board
    with PHASES("SIFT anchor hop"):
        for m in witnesses + [n]:
            s = shots[m]
            if s.sift_features is None:
                s.sift_features = sift_features(s.img, s.sift_mask(),
                                                args.nfeatures, args.contrast)
        out = []
        for m in witnesses:
            M_nm, inl, terr = pair_transform(
                *shots[n].sift_features, *shots[m].sift_features,
                args.ratio, args.reproj,
                terrain=(shots[n].terrain_mask(), shots[m].terrain_mask()))
            if M_nm is None or terr < SIFT_TERRAIN_MIN_INLIERS:
                continue
            A = shots[m].to_template @ to_h(M_nm)   # n's pixels -> template
            via = A @ np.linalg.inv(shots[n].to_template)
            got = cv2.transform(np.float32(board.t_corners).reshape(-1, 1, 2),
                                via[:2]).reshape(-1, 2)
            gap = float(np.max(np.linalg.norm(
                got - np.float32(board.t_corners), axis=1))) / board.tile_px
            out.append((inl, m, A, gap))
        return sorted(out, key=lambda r: -r[0])


def corroborate_anchor(run, n, witnesses):
    """Is n's anchor confirmed by an already-anchored shot's SIFT geometry?

    Needs the terrain-inlier floor (fog matching the wrong repeat) and
    MISANCHOR_CORROBORATE_MAX_TILES (a genuine match that disagrees). Any
    witness may agree, not only the best-matching one, since inlier count
    does not measure anchor quality. Returns a phrase for the log, or None."""
    for inl, m, _A, gap in sift_hops(run, n, witnesses):
        if gap <= MISANCHOR_CORROBORATE_MAX_TILES:
            return (f"sits {gap:.3f} tiles from where {m}'s SIFT geometry "
                    f"puts it, on {inl} inliers")
    return None


def tile_predicate_mask(run, n, keep, inset=0.0):
    """Boolean canvas of every tile whose sample for n satisfies `keep`
    (fog-locked tiles for the illumination fit, fog tiles for ruin search)."""
    samples, board = run.samples, run.board
    canvas = np.zeros((board.Hc, board.W), bool)
    for (i, j), per in samples.items():
        s = per.get(n)
        if s is None or not keep(s):
            continue
        r = board.tile_mask_bbox(i, j, inset)
        if r is None:
            continue
        m, (x0, y0, x1, y1) = r
        canvas[y0:y1, x0:x1] |= (m > 0)
    return canvas


def rank(run, cands, key):
    """Eligible sources, best first: least fog on the tile, then sharpest.

    Sharpest is *ascending* shot.scale (the least upscaled shot). Sources
    with more than --fog-frac-margin more fog than the cleanest are moved to
    the back, not dropped, so they can still fill pixels nobody else has
    (ordinary disagreement: median 0.001, p95 0.019 on test_ss_3). Returns
    (order, number demoted)."""
    args, shots, board = run.args, run.shots, run.board
    poly = board.tile_poly(key[0], key[1], 0.0)
    frac = {n: (tile_fog_fraction(shots[n].fogpix, shots[n].wmask, poly,
                                  board.W, board.Hc) or 0.0)
            for n in cands}
    lo = min(frac.values())
    clean = sorted([n for n in cands if frac[n] <= lo + args.fog_frac_margin],
                   key=lambda n: shots[n].scale)
    foggy = sorted([n for n in cands if frac[n] > lo + args.fog_frac_margin],
                   key=lambda n: shots[n].scale)
    return clean + foggy, len(foggy)


def parse_args():
    """The command line, with the one check that needs argparse's own error."""
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*",
                    help="screenshots to merge. May be omitted entirely when "
                         "--base is given and only its overlays are wanted -- "
                         "see --base.")
    ap.add_argument("-o", "--out", default="merged.png")
    ap.add_argument("--ui-mask", help="JSON of per-file exclusion rectangles, "
                    "for anything --top-crop/--bottom-crop don't cover (a "
                    "mid-screen dialog, say). Not required for the ordinary "
                    "score/turn banner or action-button row.")
    ap.add_argument("--top-crop", type=float, default=0.15,
                    help="fraction of image height to exclude from the top on "
                         "every input, unconditionally, to clear the score/turn "
                         "HUD without per-screenshot setup (see build_frame_mask)")
    ap.add_argument("--bottom-crop", type=float, default=0.15,
                    help="fraction of image height to exclude from the bottom "
                         "on every input, unconditionally, to clear the "
                         "action-button row (see build_frame_mask)")
    ap.add_argument("--map-size", type=int, choices=list(MAP_SIZE_CHOICES),
                    help="board is this many tiles on a side (always square, "
                         "always a full diamond). Omit it to measure it off the "
                         "screenshots instead (see detect_map_size); that needs "
                         "at least one shot spanning the whole board in one "
                         "direction, and fails loudly rather than guessing when "
                         "no shot does. Get it wrong and the fog test "
                         "degenerates -- see --min-fog-lock and the board-size "
                         "check, which catch exactly that.")
    ap.add_argument("--single", action="store_true",
                    help="refuse more than one input image. Every shot is "
                         "anchored independently either way, so this changes "
                         "no geometry -- it is an assertion, for isolating "
                         "whether a problem is in one shot's own anchor "
                         "rather than in how it sits against the others.")
    ap.add_argument("--min-edge-support", type=int, default=150,
                    help="boundary pixels needed before a board edge counts as "
                         "visible enough to fix pan on its own")
    ap.add_argument("--min-scale-support", type=int, default=30,
                    help="boundary pixels needed before a board edge can join "
                         "an *opposite pair* for the zoom prior -- deliberately "
                         "far below --min-edge-support, since a pair "
                         "cross-checks itself (see anchor_to_template)")
    ap.add_argument("--no-refine", action="store_true",
                    help="anchor from board edges alone, skipping the joint "
                         "zoom+pan refinement against the fog art")
    ap.add_argument("--min-fog-lock", type=int, default=8,
                    help="fail unless at least one shot matches this many tiles "
                         f"against the template's fog art at NCC >= {FOG_LOCK_NCC}. "
                         "Guards against a wrong --map-size/--template, which "
                         "otherwise degrades silently into calling the whole "
                         "board explored. A board with no fog left (a replay, a "
                         "finished game) does not need 0: a run that locks "
                         "nothing refuses only when the size was *detected*, and "
                         "warns and merges when it was stated.")
    ap.add_argument("--cross-check", action="store_true",
                    help="anchor every image to the template independently, then "
                         "report how far those anchors disagree with the SIFT "
                         "relative geometry between the same shots. QA only -- "
                         "produces no merged output.")
    ap.add_argument("--template", help="blank all-fog board render to register "
                    "against and use as the fog base layer (default: "
                    "Overlays/<size-name>-blank.png -- see template_path_for)")
    ap.add_argument("--base", help="a composite this program wrote earlier, "
                    "used as the paste canvas's starting pixels instead of "
                    "blank template fog, so a new screenshot can extend it "
                    "without resupplying the shots it was built from. Must "
                    "be this program's own, unmodified output -- its pixel "
                    "dimensions name the board size exactly (see "
                    "base_output_size), and a resized or re-encoded copy is "
                    "refused rather than guessed at. images may be omitted "
                    "entirely with --base, to redraw its --overlays alone.")
    ap.add_argument("--fog-ncc", type=float, default=0.4,
                    help="a tile counts as fog when it correlates at least this "
                         "well with the template's fog art. Measured on a real "
                         "shot, genuine fog scores a median 0.88 and explored "
                         "tiles 0.02, with nothing between 0.16 and 0.54, so "
                         "anything in 0.3-0.5 gives the same answer. Replaces "
                         "the old saturation cutoff, which called mountains, "
                         "snow and ice fog because they are pale.")
    ap.add_argument("--fog-wedge-ncc", type=float, default=0.80,
                    help="a tile also counts as fog if just its occlusion-free "
                         "top wedge (see tile_top_wedge) matches this well, "
                         "catching fog tiles whose whole-tile score is diluted "
                         "by a city or mountain overlapping from the south. "
                         "Keep it high: fog is one fixed render, so a real fog "
                         "wedge scores ~0.95, and anything in 0.75-0.85 behaves "
                         "identically across every set it was measured on.")
    ap.add_argument("--ruin-vision", action="store_true",
                    help="detect Elyrion ruin-vision markers (the rainbow "
                         "diamonds an Elyrion player sees on fogged ruin "
                         "tiles), report their tiles, and outline them on the "
                         "composite's fog so the merge keeps that knowledge. "
                         "A tile another player has explored is reported but "
                         "not outlined -- its real terrain is already there. "
                         "Off by default: only meaningful when an Elyrion "
                         "player contributed a screenshot.")
    ap.add_argument("--city-bars", action="store_true",
                    help="detect the owner-only city population bar and give "
                         "the shot that shows it priority on that city's 3x3 "
                         "block, so the bar survives into the composite "
                         "instead of being overwritten by a sharper shot that "
                         "cannot see it. Off by default.")
    ap.add_argument("--overlays", default=None,
                    help="comma-separated decorative layers to draw on the "
                         "composite: " +
                         ", ".join(name for name, _ in OVERLAY_LAYERS) +
                         ", vision, vision-each, or 'none'. The first four "
                         "come from the same Overlays/ renders as the board "
                         "itself, so they need no registration; not every "
                         "board size has every one -- a missing layer is "
                         "skipped and reported, never an error. `vision` and "
                         "`vision-each` are computed rather than loaded: both "
                         "identify each shot's own player from its Game "
                         "Stats icon (see player_icon_region). `vision` outlines "
                         "what each identified player's shot(s) alone "
                         "witnessed as explored, in a different color per "
                         "player, on the one composite --out writes. "
                         "`vision-each` instead writes one *additional* "
                         "composite per identified player, alongside --out, "
                         "named <out>_vision_<player>.<ext>: the same "
                         "composite, with a flat translucent white wash laid "
                         "back over any tile the union explored that this "
                         "player's own shot(s) did not, so what somebody "
                         "else revealed but this player has not personally "
                         "seen still reads as unexplored to them. Both are "
                         "best-effort -- a shot whose player cannot be "
                         "identified with confidence simply contributes no "
                         "outline/composite. "
                         f"Default: {OVERLAY_DEFAULT}.")
    ap.add_argument("--no-badge-filter", action="store_true",
                    help="skip capture-city/capture-ruin badge detection "
                         "(see detect_capture_badges)")
    ap.add_argument("--fog-frac-margin", type=float, default=0.04,
                    help="on a tile witnessed by several shots, a shot whose "
                         "rhombus is this much more fog (by per-pixel match "
                         "against the template's fog art) than the least-foggy "
                         "one loses the tile, however sharp it is. Catches a "
                         "shot whose fog is hidden behind a tall city -- see "
                         "tile_fog_fraction.")
    ap.add_argument("--tile-inset", type=float, default=0.25,
                    help="shrink each tile toward its center by this fraction "
                         "before sampling/comparing, to dodge neighbor-tile "
                         "and UI-icon bleed at tile borders")
    ap.add_argument("--min-valid-frac", type=float, default=0.5,
                    help="min fraction of a tile's (inset) sample area that must "
                         "be non-UI/non-dark pixels for an image to witness it")
    ap.add_argument("--consistency-thresh", type=float, default=40.0,
                    help="max allowed mean-color distance between two images' "
                         "witnesses of the same explored tile before flagging a conflict")
    ap.add_argument("--dark-thresh", type=int, default=25)
    ap.add_argument("--erode-px", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio")
    ap.add_argument("--reproj", type=float, default=6.0)
    ap.add_argument("--nfeatures", type=int, default=20000)
    ap.add_argument("--contrast", type=float, default=0.02)
    ap.add_argument("--max-shots", type=int,
                    help="refuse the run if more than this many inputs survive "
                         "the menu prefilter. Counted *after* it deliberately: "
                         "the limit exists to bound the cost and the quality of "
                         "a merge, and a menu screenshot contributes to "
                         "neither, so spending the budget on one would refuse "
                         "merges that are well inside it. No limit by default.")
    ap.add_argument("--debug-dir")
    args = ap.parse_args()

    if not args.images and not args.base:
        ap.error("no screenshots given, and no --base to update either -- "
                 "nothing to merge")
    return args


def resolve_overlays(args):
    """The --overlays set, defaulted and validated."""
    if args.overlays is None:
        # With --base, default to no layers, so an update is a quiet
        # continuation of the prior merge.
        args.overlays = "none" if args.base else OVERLAY_DEFAULT

    # Validated up front so a typo fails before the merge work. vision and
    # vision-each have no Overlays/ file and are drawn separately.
    known = {name for name, _ in OVERLAY_LAYERS} | {"vision", "vision-each"}
    overlays = {p.strip().lower() for p in args.overlays.split(",") if p.strip()}
    overlays.discard("none")
    unknown = overlays - known
    if unknown:
        raise SystemExit(
            f"unknown overlay(s): {', '.join(sorted(unknown))}. "
            f"Choose from {', '.join(sorted(known))}, or 'none'.")
    return overlays


def load_inputs(run):
    """Read every screenshot into a Shot, and the --base image."""
    args = run.args
    with PHASES("load images + masks + badges"):
        ui = {}
        if args.ui_mask:
            with open(args.ui_mask) as fh:
                ui = json.load(fh)
        names = [os.path.basename(p) for p in args.images]
        shots = {}
        badge_found = set()
        for path, name in zip(args.images, names):
            im = cv2.imread(path)
            if im is None:
                # basename only: this reaches a Discord channel
                raise SystemExit(f"cannot read {name} -- it appears invalid")
            shots[name] = Shot(im, args, ui.get(name, []))
            if not args.no_badge_filter:
                found = shots[name].detect_badge()
                if found:
                    print(f"{name}: excluding {len(found)} capture-badge "
                          f"blob(s) {found}")
                    # badge_mask is set even when empty, so record hits here.
                    badge_found.add(name)

        # The prior composite is never a shot: it only seeds the paste canvas
        # (paste_composite), its geometry being exactly known.
        base_bgr = cv2.imread(args.base) if args.base else None
        if args.base and base_bgr is None:
            raise SystemExit("cannot read --base image -- it appears invalid")
    run.ui = ui
    run.names = names
    run.shots = shots
    run.badge_found = badge_found
    run.base_bgr = base_bgr


def select_map_shots(run):
    """Drop menu screenshots, then apply --max-shots and --single."""
    args, names, shots = run.args, run.names, run.shots
    # Menus go before anything else sees the shots, detect_map_size included.
    all_names = list(names)
    with PHASES("menu-screenshot prefilter"):
        for name in list(names):
            frac = board_angle_fraction(shots[name].img, (BOARD_DIR_A, BOARD_DIR_B),
                                        args.top_crop, args.bottom_crop)
            if frac < MENU_BOARD_ANGLE_FRAC:
                # Console only; the channel gets the DROPPED summary below.
                print(f"{name}: not a view of the board -- only {frac:.0%} "
                      f"of its detail runs at a board angle (a score screen "
                      f"or other menu drawn over the map?)")
                names.remove(name)
    if all_names and not names:
        # (A --base-only run starts with no names and is not a refusal.)
        raise SystemExit("these look like score screens or menus rather than "
                         "the map itself. Please retry with in-game "
                         "screenshots of the map.")

    # Counted after the menu prefilter, which only this program can run (the
    # bot has undownloaded attachments): menus must not spend the budget.
    if args.max_shots is not None and len(names) > args.max_shots:
        raise SystemExit(f"too many screenshots: {len(names)} of these show the "
                         f"map, and the limit is {args.max_shots}. Please retry "
                         f"with fewer.")
    if args.single and len(names) != 1:
        raise SystemExit(f"--single takes exactly one image, got {len(names)}")
    run.all_names = all_names


def resolve_map_size(run):
    """Settle args.map_size: stated, read off --base, or measured."""
    args, names, shots, base_bgr = run.args, run.names, run.shots, run.base_bgr
    # An explicit --map-size is always obeyed. A --base's pixel size names the
    # board exactly, so it beats measuring, and counts as *stated* (a zero
    # fog lock then warns rather than refuses).
    size_was_detected = args.map_size is None and not args.base
    # One cache for the run, shared with anchoring (ShotCache).
    shot_cache = ShotCache()
    if args.map_size is None and args.base:
        args.map_size = base_output_size(base_bgr.shape[:2], args.dark_thresh,
                                         args.erode_px)
        if args.map_size is None:
            raise SystemExit(
                f"--base doesn't match the pixel size of any board this "
                f"program renders ({size_list()}). It has to be this "
                f"program's own, unmodified merge output.")
        print(f"base map size: {args.map_size}x{args.map_size} (base image "
              f"is {base_bgr.shape[1]}x{base_bgr.shape[0]}px, exactly that "
              f"size's own composite dimensions)")
    elif size_was_detected:
        # Plain dicts, since shoreline/polyshore.py calls this too.
        args.map_size = detect_map_size(
            names, {n: s.img for n, s in shots.items()},
            {n: s.edge_mask for n, s in shots.items()},
            {n: s.valid for n, s in shots.items()},
            {n: s.hsv for n, s in shots.items()},
            args.dark_thresh, args.min_edge_support,
            args.min_scale_support,
            erode_px=args.erode_px,
            cache=shot_cache,
            top_crop=args.top_crop, bottom_crop=args.bottom_crop)
    run.size_was_detected = size_was_detected
    run.shot_cache = shot_cache


def load_board(run):
    """Load the template and derive the board lattice from it."""
    args, base_bgr = run.args, run.base_bgr
    template_path = args.template or template_path_for(args.map_size)
    tgeom = template_geometry(template_path, args.dark_thresh, args.erode_px)
    template = tgeom["bgr"] if tgeom else None
    if template is None:
        # Install fault: the path goes below the headline.
        raise SystemExit(f"this is not installed correctly, so it cannot merge "
                         f"this board size.\n({template_path} is missing or "
                         f"unreadable.)")
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    print(f"template: {template_path}")

    t_top, t_right, t_bottom, t_left, t_residual = tgeom["corners"]
    print(f"template corners: top={tuple(t_top.round(0))} right={tuple(t_right.round(0))} "
          f"bottom={tuple(t_bottom.round(0))} left={tuple(t_left.round(0))} "
          f"(residual {t_residual:.1f}px, should be tiny)")
    dir_a, dir_b = BOARD_DIR_A, BOARD_DIR_B
    origin, u_col, u_row = build_lattice(t_top, t_right, t_left, args.map_size)
    t_corners = (t_top, t_right, t_bottom, t_left)
    t_edge_off, _t_edge_sup = tgeom["edges"]
    tile_px = float(np.linalg.norm(u_col))
    print(f"tile step: {tile_px:.2f}px")

    # Any --base must match this size's composite exactly, however the size
    # was chosen.
    base_rect = None
    if args.base:
        Wt, Ht = template.shape[1], template.shape[0]
        x0c, y0c, x1c, y1c = output_crop(origin, u_col, u_row, args.map_size,
                                         Wt, Ht)
        want = (x1c - x0c, y1c - y0c)
        got = (base_bgr.shape[1], base_bgr.shape[0])
        if got != want:
            raise SystemExit(
                f"--base is {got[0]}x{got[1]}px, but a "
                f"{args.map_size}x{args.map_size} board's own composite is "
                f"{want[0]}x{want[1]}px. It has to be an unmodified merge "
                f"output at this size.")
        base_rect = (x0c, y0c, x1c, y1c)

    # The composite is built in the template's own frame.
    W, Hc = template.shape[1], template.shape[0]
    N = args.map_size
    run.board = Board(
        template_path=template_path, template=template, tmpl_gray=tmpl_gray,
        dir_a=dir_a, dir_b=dir_b, origin=origin, u_col=u_col, u_row=u_row,
        N=N, W=W, Hc=Hc, t_corners=t_corners, t_edge_off=t_edge_off,
        tile_px=tile_px, base_rect=base_rect)


def run_cross_check(run):
    """--cross-check: report anchors against SIFT geometry, and stop."""
    args, names, shots, board = run.args, run.names, run.shots, run.board
    anchors = anchor_all(run)[0]
    feats = {n: sift_features(shots[n].img, shots[n].sift_mask(), args.nfeatures, args.contrast)
             for n in names}
    sift_edges = {(a, b): pair_transform(*feats[a], *feats[b], args.ratio,
                                         args.reproj)
                  for a, b in itertools.combinations(names, 2)}
    print(f"\nindependent anchors vs SIFT relative geometry "
          f"({len(anchors)}/{len(names)} shots anchorable):")
    rows = cross_check(anchors, sift_edges, board.t_corners, board.tile_px)
    if not rows:
        raise SystemExit("no pair has both an anchor and a SIFT transform")
    for a, b, n_inl, err, tiles in rows:
        flag = "" if tiles < CROSS_CHECK_FLAG_TILES else "   <-- ANCHORS DISAGREE"
        print(f"  {a[:30]:30s} vs {b[:30]:30s} inliers={n_inl:5d} "
              f"corner gap={err:7.1f}px = {tiles:.3f} tiles{flag}")
    worst = max(r[4] for r in rows)
    print(f"\nworst disagreement: {worst:.3f} tiles")


def adopt_anchors(run):
    """Anchor every shot and file the results onto the Shots."""
    names, all_names, shots = run.names, run.all_names, run.shots
    # Each shot is anchored to the template independently, never registered
    # to the others, so one bad pairwise match cannot throw off a chain.
    M_of, src_of, implied_of, prior_M_of = anchor_all(run)
    # Stored homogeneous (3x3), since later phases swap and invert them.
    for n, M in M_of.items():
        shots[n].to_template = to_h(M)
    for n, src in src_of.items():
        shots[n].zoom_source = src
    for n, v in implied_of.items():
        shots[n].implied_n = v
    for n, M in prior_M_of.items():
        if M is not None:
            shots[n].prior = to_h(M)
    if all_names and not M_of:
        # (A --base-only run has nothing to anchor.)
        raise SystemExit(
            "no valid images found -- none of them could be placed on the "
            "board. They need to be in-game screenshots showing part of the "
            "map, with two adjoining sides of the board in frame -- two "
            "opposite sides are not enough. Zooming out usually does it.")
    # A partial merge still looks fine, so name what was left out, menus
    # included. polybot parses this line: keep the prefix and n/m shape.
    dropped_names = [n for n in all_names if shots[n].to_template is None]
    if dropped_names:
        print(f"DROPPED {len(dropped_names)}/{len(all_names)}: "
              + ", ".join(dropped_names))
    names = [n for n in names if shots[n].to_template is not None]
    run.names = names


def identify_players(run):
    """--overlays vision/vision-each: who took each shot."""
    overlays, names, shots = run.overlays, run.names, run.shots
    # Runs once `names` is final, so dropped shots are not probed.
    player_of = {}
    no_head_catalog = False
    if overlays & {"vision", "vision-each"}:
        with PHASES("player identification"):
            catalog = load_head_catalog()
            no_head_catalog = not catalog
            if no_head_catalog:
                print(f"\nNO-HEAD-CATALOG: cannot read "
                      f"{os.path.join(head_icon_dir(), '*.png')}, so no shot "
                      f"could be matched to a player -- vision outlines/"
                      f"per-player composites skipped")
            # First which shots are one player (icons compared with each
            # other), then which tribe (the catalog), which only picks colour.
            region_of = {n: player_icon_region(shots[n].img) for n in names}
            groups = group_shots_by_icon(region_of)
            # One (costly) catalog match per distinct icon, not per shot.
            named, matched = {}, []
            for group in groups:
                for n in group:
                    same = next((m for m in matched
                                 if icon_similarity(region_of[n], region_of[m])
                                 >= HEAD_SAME_ICON_NCC), None)
                    if same is not None:
                        named[n] = named[same]
                        continue
                    named[n] = match_head_icon(region_of[n], catalog)
                    matched.append(n)
            anon = 0
            for group in groups:
                # Two confident names in one group are one tribe in two skins
                # (vengir_cultist): split on the catalog's word.
                keys = sorted({named[n][0] for n in group if named.get(n)})
                if len(keys) > 1:
                    parts = [[n for n in group
                              if named.get(n) and named[n][0] == k] for k in keys]
                    # An unnamed shot joins the first part.
                    for n in group:
                        if not named.get(n):
                            parts[0].append(n)
                else:
                    parts = [group]
                for part in parts:
                    # The part takes its best-scoring name; with none it is
                    # still a player, under a non-catalog key (palette colour).
                    best = max((named[n] for n in part if named.get(n)),
                               key=lambda kn: kn[1], default=None)
                    if best is None:
                        anon += 1
                        best = (f"unnamed player {anon}", 0.0)
                    for n in part:
                        player_of[n] = best
            for n in names:
                player_of.setdefault(n, None)
    run.player_of = player_of
    run.no_head_catalog = no_head_catalog


def check_board_size(run):
    """Refuse a stated size the shots' own tile count contradicts."""
    args, shots, size_was_detected = (
        run.args, run.shots, run.size_was_detected)
    # Stronger than --min-fog-lock, which a fog-period anchor passes at a wrong
    # size too. The median of two is their mean, so implausible readings are
    # dropped first (MAP_SIZE_PLAUSIBLE_TOL).
    implied_n_of, implied_n_bad = plausible_sizes(
        {n: s.implied_n for n, s in shots.items() if s.implied_n is not None})
    if implied_n_bad and not size_was_detected:
        # (detect_map_size has already printed it otherwise)
        print(implausible_note(implied_n_bad))
    if implied_n_of:
        med = float(np.median(list(implied_n_of.values())))
        if abs(med - args.map_size) > MAP_SIZE_STATED_TOL:
            # Channel copy: one sentence. The per-shot evidence is on stdout.
            got = int(round(med))
            raise SystemExit(
                f"This looks like a {got}x{got} board, not "
                f"{args.map_size}x{args.map_size}. Please retry with size "
                f"{got} or without a size.")


def sample_tiles(run):
    """Warp every shot onto the canvas and classify every tile."""
    names, samples, board = run.names, run.samples, run.board
    with PHASES("warp to canvas"):
        for n in names:
            warp_shot(run, n)
        print(f"canvas: {board.W} x {board.Hc} (template)")

    with PHASES("tile sampling"):
        for i in range(board.N):
            for j in range(board.N):
                samples[(i, j)] = {}
        for n in names:
            sample_shot(run, n)


def fog_lock_guards(run):
    """Count fog lock, undo fog-blind refinements, and apply --min-fog-lock."""
    args, names, shots, size_was_detected, board = (
        run.args, run.names, run.shots, run.size_was_detected, run.board)
    fog_lock = {n: locked(run, n) for n in names}
    # A refinement that locked no fog was fitted to noise (tests/fogless), so
    # the shot goes back to its unrefined prior -- even on a tie at zero, which
    # is the point. One locked tile keeps the refinement. Runs before the
    # report and the guard, and regardless of --min-fog-lock.
    for n in [n for n in names if not fog_lock[n] and shots[n].prior is not None]:
        shots[n].to_template = shots[n].prior
        warp_shot(run, n)
        sample_shot(run, n)
        fog_lock[n] = locked(run, n)
        print(f"  {n}: refining it locked no fog, so its edge anchor stands "
              f"unrefined (that prior locks {fog_lock[n]})")
    print("\nfog lock (tiles matching the template's fog art at NCC >= "
          f"{FOG_LOCK_NCC}): "
          + "  ".join(f"{n}={fog_lock[n]}" for n in names))
    size_unverified = False
    # (A --base-only run has no names, and max() of nothing would raise.)
    if names and args.min_fog_lock > 0 and max(fog_lock.values()) < args.min_fog_lock:
        # A wrong size and a fogless board look the same here, so split on who
        # claimed the size: a *detected* size came from fog, so none locking
        # is an error; a *stated* one is warned about and merged (a wrong one
        # is still caught by check_board_size or CONFLICT_FRAC_SUSPECT). See
        # --min-fog-lock in CLAUDE.md.
        if size_was_detected:
            raise SystemExit(
                "the board size measured from these screenshots does not fit "
                "them. " + RESTATE_SIZE.format(size_list()) + "."
                + f"\n(no shot locked onto the fog artwork: best "
                f"{max(fog_lock.values())} tiles, need {args.min_fog_lock})")
        size_unverified = True
        print(f"\nWARNING: no shot locked onto the fog artwork (best "
              f"{max(fog_lock.values())} tiles). Merging as "
              f"{board.N}x{board.N} because that is what was asked for, but nothing here "
              f"can confirm it -- fog is what this check compares against, and "
              f"there is none to compare. If the board really has no fog left "
              f"(a replay, or a finished game) that is expected; otherwise "
              f"check the size.")
    run.fog_lock = fog_lock
    run.size_unverified = size_unverified


def borrow_anchors(run):
    """Give a shot with no fog lock a lender's anchor, if its own fog agrees."""
    args, names, shots, fog_lock = (
        run.args, run.names, run.shots, run.fog_lock)
    # A shot with zero fog locked had no say in its own anchor, so it may
    # borrow a whole anchor by SIFT -- only from a lender that clears
    # --min-fog-lock, and only if the borrowed anchor locks more of the
    # borrower's own fog. That test also picks the lender, since inlier count
    # does not (a near-identical view out-matches regardless). No corpus set
    # reaches this: exercise changes deliberately.
    if args.min_fog_lock > 0:
        lenders = [m for m in names if fog_lock[m] >= args.min_fog_lock]
        for n in [n for n in names if fog_lock[n] == 0]:
            keep_M, keep_lock, best = shots[n].to_template, fog_lock[n], None
            for inl, m, A, gap in sift_hops(run, n, lenders):
                shots[n].to_template = A
                warp_shot(run, n)
                sample_shot(run, n)
                lock = locked(run, n)
                print(f"    {n} anchored from {m} ({inl} SIFT inliers, moves "
                      f"{gap:.3f} tiles) locks {lock} fog tiles")
                if best is None or lock > best[0]:
                    best = (lock, m, A, inl, gap)
            if best is None:
                continue
            lock, m, A, inl, gap = best
            shots[n].to_template = A if lock > keep_lock else keep_M
            warp_shot(run, n)
            sample_shot(run, n)
            if lock > keep_lock:
                fog_lock[n] = lock
                print(f"  re-anchored {n} from {m}'s SIFT geometry ({inl} "
                      f"inliers, moved {gap:.3f} tiles): it locked no fog of "
                      f"its own, so its refinement had nothing to correct the "
                      f"edge fit against. Now locks {lock}")


def drop_misanchored(run):
    """Drop a fog-period shot that locks no fog and nothing corroborates."""
    # A fog-period zoom was measured on the shot's own fog, so locking none of
    # it means a wrong anchor that would paste fog over others' terrain. But
    # terrain is periodic too (star_change/oum.png), so SIFT corroboration can
    # save the shot first; it can never drop one that would have survived.
    args, names, all_names, shots, samples, fog_lock = (
        run.args, run.names, run.all_names, run.shots, run.samples, run.fog_lock)
    if args.min_fog_lock > 0:
        suspect = [n for n in names
                   if shots[n].zoom_source == "fog-period" and fog_lock[n] == 0]
        witnesses = [n for n in names if n not in suspect]
        for n in suspect:
            keep = corroborate_anchor(run, n, witnesses)
            if keep is not None:
                print(f"  keeping {n}: zero fog tiles locked, but it has almost "
                      f"no fog in frame and {keep} -- the anchor is corroborated "
                      f"by geometry that does not depend on fog")
                continue
            print(f"  dropping {n}: anchored from its own fog's repeat period, "
                  f"yet zero of its tiles lock onto the template's fog art, and "
                  f"no anchored shot's SIFT geometry corroborates it -- that "
                  f"anchor cannot be right, and keeping it would paste its fog "
                  f"over other shots' terrain")
            names.remove(n)
            for per in samples.values():
                per.pop(n, None)
            del fog_lock[n]
        if all_names and not names:
            # (A --base-only run starts with no names.)
            raise SystemExit(
                "none of these screenshots could be placed on the board. "
                "Check they are all of the same board, and that the size is "
                "right.")


def fit_fog_pixels(run):
    """Per-pixel fog evidence for each shot, which rank() compares across
    sources on one tile: a tall city can hide fog from the tile test in every
    shot, but the occluder cancels between them (test_ss_3 (9,8))."""
    args, names, shots, board = run.args, run.names, run.shots, run.board
    with PHASES("fog pixel masks"):
        for n in names:
            s = shots[n]
            locked_mask = tile_predicate_mask(run, 
                n, lambda s: s.get("fog_ncc", 0.0) >= FOG_LOCK_NCC,
                args.tile_inset)
            sel = locked_mask & (s.wmask > 0)
            # Too little fog to fit: an all-false mask never demotes the shot.
            if int(sel.sum()) >= FOG_GAIN_MIN_PX:
                s.gain = fog_illumination(s.warped, board.template, sel)
                s.fogpix = fog_pixel_mask(s.warped, board.template, s.gain)
            else:
                s.fogpix = np.zeros((board.Hc, board.W), bool)


def select_winners(run):
    """Rank the sources for every explored tile."""
    args, names, shots, badge_found, samples, board = (
        run.args, run.names, run.shots, run.badge_found, run.samples, run.board)
    # A tile with no clean witness falls back to badge-covered ones rather than
    # show fog. priority[key] keeps the whole order, since the paste falls
    # through it wherever the winner's frame ends mid-tile.
    winner, priority, badge_fallback, fog_demoted = {}, {}, [], []
    with PHASES("winner selection"):
        for i in range(board.N):
            for j in range(board.N):
                key = (i, j)
                eligible = [n for n, s in samples[key].items() if s["explored"]]
                if eligible:
                    order, n_foggy = rank(run, eligible, key)
                    if n_foggy:
                        fog_demoted.append(key)
                    winner[key] = order[0]
                    priority[key] = (order, False)  # False: use each shot's pmask
                    continue
                # With no badge anywhere the re-sample would change nothing.
                if not badge_found:
                    continue
                poly = board.tile_poly(i, j, args.tile_inset)
                wedge = board.tile_top_wedge(i, j)
                raw_eligible = []
                for n in names:
                    s = sample_tile(shots[n].warped, shots[n].wmask_raw, board.tmpl_gray, poly,
                                    args.fog_ncc, args.min_valid_frac,
                                    wedge_poly=wedge,
                                    fog_wedge_ncc=args.fog_wedge_ncc)
                    if s is not None and s["explored"]:
                        raw_eligible.append(n)
                if raw_eligible:
                    order, _ = rank(run, raw_eligible, key)
                    winner[key] = order[0]
                    priority[key] = (order, True)  # True: use each shot's pmask_raw
                    badge_fallback.append(key)
    run.winner = winner
    run.priority = priority
    run.badge_fallback = badge_fallback
    run.fog_demoted = fog_demoted


def promote_city_bars(run):
    """--city-bars: give a city's owner its 3x3 block."""
    args, names, shots, samples, winner, priority, capped_of, board = (
        run.args, run.names, run.shots, run.samples, run.winner, run.priority,
        run.capped_of, run.board)
    # The shot showing a bar is its city's owner's, so that source is promoted
    # on the city's 3x3 block: a superset of the tiles the bar can touch, and
    # the block the owner is guaranteed to see. Promotion only reorders
    # sources that witnessed the tile as explored, so a mis-detection costs
    # sharpness, never truth. See CLAUDE.md, city population bars.
    bar_promoted, vision_promoted = [], []
    if args.city_bars:
        # Vision promotes too: a source that alone sees a whole 3x3 block is
        # the only candidate owner of a city there. It cancels wherever every
        # source sees the block, so it only acts at frontiers and frame edges.
        # All claims are collected before any is applied.
        with PHASES("city-bar detection"):
            for n in names:
                shots[n].bars = detect_population_bars(shots[n].warped, shots[n].wmask,
                                                        *board.lattice, board.N)

            # Only a capped bar reaches the city's S/SW/SE neighbors.
            def _capped(width_class):
                return width_class >= 3

            # How well backed a city is, taking its best evidence across shots.
            def _evidence(city):
                best = (0, 0.0, 0)
                for n in names:
                    for c, bbox, width_class, plate in shots[n].bars:
                        if c == city:
                            best = max(best, (1 if _capped(width_class) else 0,
                                              plate, bbox[2]))
                return best

            # Detections within CITY_MIN_GAP contradict each other: drop the
            # weaker (capped width, then plate, then span; the higher tile on
            # a tie) until none remain. The loser keeps its vision claim.
            while True:
                cities = {c for n in names for c, _b, _n, _p in shots[n].bars}
                pair = next(((a, b) for a in sorted(cities)
                             for b in sorted(cities)
                             if a < b and max(abs(a[0] - b[0]),
                                              abs(a[1] - b[1])) <= CITY_MIN_GAP),
                            None)
                if pair is None:
                    break
                loser = min(pair, key=lambda c: (_evidence(c), (-c[0], -c[1])))
                for n in names:
                    shots[n].bars = [x for x in shots[n].bars if x[0] != loser]

            # One city's bar in several shots: the sharpest owns it.
            owner_of, capped_of, plate_of = {}, {}, {}
            for n in names:
                for city, bbox, width_class, plate in shots[n].bars:
                    if city not in owner_of or shots[n].scale < shots[owner_of[city]].scale:
                        owner_of[city] = n
                        plate_of[city] = plate
                        capped_of[city] = _capped(width_class)

        with PHASES("vision-based promotion"):
            seen_of = {}
            for ci in range(board.N):
                for cj in range(board.N):
                    block = [(ci + di, cj + dj) for di in (-1, 0, 1)
                             for dj in (-1, 0, 1)
                             if 0 <= ci + di < board.N and 0 <= cj + dj < board.N]
                    seers = [n for n in names
                             if all(samples.get(k, {}).get(n, {}).get("explored")
                                    for k in block)]
                    if len(seers) == 1:
                        seen_of[(ci, cj)] = seers[0]

        # tile -> (rank, claiming shot, whether a detected bar made the claim).
        # Lowest rank wins; see the CLAIM_* strengths.
        claims = {}
        for source, strong_w, weak_w, from_bar in (
                (owner_of, CLAIM_BAR_STRONG, CLAIM_BAR_WEAK, True),
                (seen_of, CLAIM_VISION_STRONG, CLAIM_VISION_WEAK, False)):
            for (ci, cj), n in source.items():
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        key = (ci + di, cj + dj)
                        w = strong_w if (di >= 0 and dj >= 0) else weak_w
                        # Strength, capped width, plate, proximity (a city's
                        # own tile first, which keeps its label whole), then
                        # sharpness. Vision claims score plate 0, which only
                        # ever meets another vision claim. CLAUDE.md works
                        # through the tiles that pin this order down.
                        d = max(abs(di), abs(dj))
                        full = 0 if capped_of.get((ci, cj), False) else 1
                        plate = plate_of.get((ci, cj), 0.0)
                        rank = (-w, full, -plate, d, shots[n].scale, n)
                        if key not in claims or rank < claims[key][0]:
                            claims[key] = (rank, n, from_bar)
        for key, (_rank, n, from_bar) in claims.items():
            if key not in priority:
                continue
            order, raw = priority[key]
            if n not in order:
                continue
            priority[key] = ([n] + [m for m in order if m != n], raw)
            if winner.get(key) != n:
                (bar_promoted if from_bar else vision_promoted).append(key)
            winner[key] = n
    run.bar_promoted = bar_promoted
    run.vision_promoted = vision_promoted
    run.capped_of = capped_of


def detect_ruins(run):
    """--ruin-vision: find Elyrion ruin markers on each shot's own fog."""
    args, names, shots, board = run.args, run.names, run.shots, run.board
    # Searched only on tiles this source itself witnessed as fog.
    ruin_hits = {}                        # (i, j) -> ([(n, area, mask)], tiles)
    n_raw = 0                             # detections before adjacency merging
    no_ruin_sprite = False                # asset missing: reported, never faked
    ruin_no_fog = []                      # shots with no fog reference to match
    if args.ruin_vision:
        with PHASES("ruin-vision detection"):
            sprite = load_ruin_sprite()
            no_ruin_sprite = sprite is None
            for n in names:
                fog_area = tile_predicate_mask(run, 
                    n, lambda s: s.get("witness") and not s["explored"])
                fog_area &= shots[n].wmask > 0
                # Without a fitted gain there is no fog to subtract: report the
                # shot and skip it rather than match against a guess.
                if sprite is None or shots[n].gain is None:
                    if sprite is not None:
                        ruin_no_fog.append(n)
                    shots[n].ruins = []
                    continue
                # wmask, not pmask: pmask's dark pixels inflate |D| (u_forest
                # loses 2 of its 9 ruins).
                found = detect_ruin_vision(shots[n].warped, shots[n].wmask, fog_area,
                                           board.template, shots[n].gain,
                                           *board.lattice, sprite)
                shots[n].ruins = found
                for i, j, area, mask in found:
                    if 0 <= i < board.N and 0 <= j < board.N:
                        ruin_hits.setdefault((i, j), []).append((n, area, mask))
            n_raw = len(ruin_hits)
            ruin_hits = cluster_ruin_tiles(ruin_hits, *board.lattice)
    run.ruin_hits = ruin_hits
    run.n_raw = n_raw
    run.no_ruin_sprite = no_ruin_sprite
    run.ruin_no_fog = ruin_no_fog


def check_conflicts(run):
    """Tiles whose sources disagree about their content."""
    args, samples = run.args, run.samples
    with PHASES("conflict check"):
        conflicts = []
        comparable = 0          # tiles where two sources can be compared at all
        for key, per_img in samples.items():
            witnesses = [n for n, s in per_img.items()
                        if s["explored"] and s["mean_color"] is not None]
            if len(witnesses) < 2:
                continue
            comparable += 1
            colors = [per_img[n]["mean_color"] for n in witnesses]
            maxd = max(float(np.linalg.norm(a - b))
                      for a, b in itertools.combinations(colors, 2))
            if maxd > args.consistency_thresh:
                conflicts.append((key, maxd, witnesses))
    run.conflicts = conflicts
    run.comparable = comparable


def paste_composite(run):
    """Build the composite and write --out."""
    (args, overlays, shots, base_bgr, samples, player_of, winner, priority,
     ruin_hits, board) = (
        run.args, run.overlays, run.shots, run.base_bgr, run.samples,
        run.player_of, run.winner, run.priority, run.ruin_hits, run.board)
    with PHASES("paste composite"):
        out = board.template.copy()
        # --base seeds the canvas. The paste writes only tiles explored this
        # run, so new content always wins and the rest of the base survives.
        if args.base:
            bx0, by0, bx1, by1 = board.base_rect
            out[by0:by1, bx0:bx1] = base_bgr
        for (i, j), (order, raw) in priority.items():
            r = board.tile_mask_bbox(i, j)
            if r is None:
                continue
            m, (x0, y0, x1, y1) = r
            region_out = out[y0:y1, x0:x1]
            filled = np.zeros(m.shape, bool)
            for n in order:
                pm = shots[n].pmask_raw if raw else shots[n].pmask
                avail = (m > 0) & (pm[y0:y1, x0:x1] > 0) & ~filled
                if not avail.any():
                    continue
                region_out[avail] = shots[n].warped[y0:y1, x0:x1][avail]
                filled |= avail
        # Layers go on after the tiles and before the ruin markers. The fog-only
        # layers are clipped to tiles nobody explored: not in `winner`, and on
        # a --base run not explored in the base either (its pixels, already in
        # template frame, get the ordinary fog test), or shade would tint the
        # base's own territory.
        base_explored = set()
        if args.base and overlays & OVERLAY_FOG_ONLY:
            with PHASES("base fog classification"):
                base_canvas = np.zeros_like(board.template)
                bx0, by0, bx1, by1 = board.base_rect
                base_canvas[by0:by1, bx0:bx1] = base_bgr
                base_valid = np.zeros((board.Hc, board.W), np.uint8)
                base_valid[by0:by1, bx0:bx1] = 255
                for i in range(board.N):
                    for j in range(board.N):
                        if (i, j) in winner:
                            continue
                        s = sample_tile(
                            base_canvas, base_valid, board.tmpl_gray,
                            board.tile_poly(i, j, args.tile_inset),
                            args.fog_ncc, args.min_valid_frac,
                            wedge_poly=board.tile_top_wedge(i, j),
                            fog_wedge_ncc=args.fog_wedge_ncc)
                        if s is not None and s["explored"]:
                            base_explored.add((i, j))
        fog_only = None
        if overlays & OVERLAY_FOG_ONLY:
            fog_only = np.zeros((board.Hc, board.W), np.uint8)
            for i in range(board.N):
                for j in range(board.N):
                    if (i, j) in winner or (i, j) in base_explored:
                        continue
                    poly = board.tile_poly(i, j, 0.0)
                    cv2.fillConvexPoly(fog_only,
                                       np.round(poly).astype(np.int32), 1)
            fog_only = fog_only.astype(np.float32)[:, :, None]
        missing_overlays = paint_overlays(out, overlays, board.N, fog_only)
        if missing_overlays:
            # polybot lifts this line into its caption (same shape as DROPPED).
            print(f"NO-OVERLAY {len(missing_overlays)}: "
                  f"{' '.join(sorted(missing_overlays))} -- not available on a "
                  f"{board.N}x{board.N} board")
        thick = max(3, int(round(np.linalg.norm(board.u_col) * 0.075)))
        # Vision outlines: after the layers, before the ruin markers.
        by_player = {}
        for n, ident in player_of.items():
            if ident is not None:
                by_player.setdefault(ident[0], []).append(n)
        if by_player and "vision" in overlays:
            draw_player_vision(out, samples, by_player, *board.lattice, thick)
        # A fogged ruin tile gets the Elyrion player's own view of it, clipped
        # to the tile's rhombus and carried back through that shot's inverse
        # illumination fit so it matches the surrounding fog, plus a violet
        # outline. Tiles someone explored are skipped. pmask, not wmask: this
        # is a paste.
        for key, (hits, comp) in sorted(ruin_hits.items()):
            if key in winner:
                continue
            # Sharpest witness first (ascending scale, as everywhere else),
            # then the one that saw most of the cluster.
            src = min(hits, key=lambda h: (shots[h[0]].scale, -h[1]))[0]
            r = board.tile_mask_bbox(key[0], key[1])
            if r is not None:
                m, (x0, y0, x1, y1) = r
                sel = (m > 0) & (shots[src].pmask[y0:y1, x0:x1] > 0)
                if sel.any():
                    patch = shots[src].warped[y0:y1, x0:x1].astype(np.float32)
                    g = shots[src].gain
                    if g is not None:
                        patch = (patch - g[:, 1]) / np.where(
                            np.abs(g[:, 0]) < 1e-3, 1.0, g[:, 0])
                    out[y0:y1, x0:x1][sel] = np.clip(
                        patch, 0, 255).astype(np.uint8)[sel]
            poly = board.tile_poly(key[0], key[1], 0.08)
            cv2.polylines(out, [np.round(poly).astype(np.int32)], True,
                          RUIN_MARK_BGR, thick, cv2.LINE_AA)

    # The same rectangle --base was validated against.
    x0c, y0c, x1c, y1c = output_crop(*board.lattice, board.N, board.W, board.Hc)
    with PHASES("encode + write output"):
        cv2.imwrite(args.out, out[y0c:y1c, x0c:x1c])
    run.crop = (x0c, y0c, x1c, y1c)
    run.out = out
    run.thick = thick
    run.by_player = by_player


def write_vision_each(run):
    """--overlays vision-each: one extra composite per player."""
    x0c, y0c, x1c, y1c = run.crop
    args, overlays, samples, winner, out, thick, by_player, board = (
        run.args, run.overlays, run.samples, run.winner, run.out, run.thick,
        run.by_player, run.board)
    # Each: the composite, washed white over tiles the union explored that this
    # player did not, with the player's own outline on top.
    vision_each_paths = []
    if "vision-each" in overlays and by_player:
        with PHASES("vision-each per-player composites"):
            out_stem, out_ext = os.path.splitext(args.out)
            out_ext = out_ext or ".png"
            for key in sorted(by_player):
                explored_self = _player_explored_tiles(samples, by_player[key])
                unseen = _vision_each_unseen_mask(
                    winner, explored_self, *board.lattice, board.W, board.Hc)
                per_out = render_vision_each(out, unseen)
                draw_player_vision(per_out, samples, {key: by_player[key]},
                                   *board.lattice, thick)
                path = f"{out_stem}_vision_{vision_each_slug(key)}{out_ext}"
                cv2.imwrite(path, per_out[y0c:y1c, x0c:x1c])
                vision_each_paths.append(path)
    if vision_each_paths:
        # polybot lifts this line (same shape as DROPPED).
        print(f"VISION-EACH {len(vision_each_paths)}: "
              + " ".join(vision_each_paths))


def report(run):
    """The run's summary on stdout."""
    (args, overlays, names, shots, samples, size_unverified, player_of,
     no_head_catalog, winner, badge_fallback, fog_demoted, bar_promoted,
     vision_promoted, capped_of, ruin_hits, n_raw, no_ruin_sprite, ruin_no_fog,
     conflicts, comparable, board) = (
        run.args, run.overlays, run.names, run.shots, run.samples,
        run.size_unverified, run.player_of, run.no_head_catalog, run.winner,
        run.badge_fallback, run.fog_demoted, run.bar_promoted,
        run.vision_promoted, run.capped_of, run.ruin_hits, run.n_raw,
        run.no_ruin_sprite, run.ruin_no_fog, run.conflicts, run.comparable,
        run.board)
    total = board.N * board.N
    print(f"\nmap: {board.N}x{board.N} = {total} tiles")
    print(f"explored (union): {len(winner)}/{total} ({100 * len(winner) / total:.1f}%)")
    if args.base:
        carried = total - len(winner)
        print(f"BASE {carried}/{total}: tile(s) carried over unchanged from "
              f"the base image -- the union above counts only this run's "
              f"own screenshot(s)")
    if badge_fallback:
        print(f"  of which {len(badge_fallback)} tile(s) had no clean witness and fell "
              f"back to a badge-covered source: {sorted(badge_fallback)}")
    if fog_demoted:
        print(f"  {len(fog_demoted)} tile(s) taken off a sharper source that showed "
              f"more fog there (occluded fog, see --fog-frac-margin): "
              f"{sorted(fog_demoted)}")
    print("\nper-shot tile counts:")
    for n in names:
        seen = sum(1 for s in samples.values() if s.get(n, {}).get("explored"))
        won = sum(1 for w in winner.values() if w == n)
        print(f"  {n:40s} witnessed-explored={seen:4d}  won={won:4d}")
    if args.city_bars:
        n_bars = sum(len(shots[n].bars) for n in names)
        if n_bars:
            print(f"\ncity population bars: {n_bars} found (owner-only, so each "
                  f"marks that shot as the city's owner):")
            for n in names:
                for (ci, cj), bbox, width_class, _pl in sorted(shots[n].bars):
                    kind = "full bar" if width_class >= 3 else "short bar"
                    print(f"  city ({ci},{cj}): {kind}, seen by {n}")
            print(f"  {len(set(bar_promoted))} tile(s) changed hands to keep a "
                  f"bar intact: {sorted(set(bar_promoted))}")
            # Spliced bars: a tile the bar crosses went to a source not showing
            # it. No other number can see this.
            shown_by = {}
            for n in names:
                for city, bbox, width_class, _pl in shots[n].bars:
                    shown_by.setdefault(city, set()).add(n)
            spliced = []
            for city, srcs in sorted(shown_by.items()):
                covered = set()
                for n in srcs:
                    for c2, (bx, by, bw, bh), _n, _p in shots[n].bars:
                        if c2 != city:
                            continue
                        # A grid over the bbox: corners alone miss a middle tile.
                        for px in np.linspace(bx, bx + bw, 12):
                            for py in np.linspace(by, by + bh, 4):
                                covered.add(board.tile_of_point((px, py)))
                lost = sorted(t for t in covered
                              if t in winner and winner[t] not in srcs)
                if lost:
                    wid = max(b[1][2] for n in srcs
                              for b in [x for x in shots[n].bars
                                        if x[0] == city])
                    spliced.append((city, lost, wid, capped_of.get(city, False)))
            if spliced:
                # A spliced capped ("complete") bar is a real defect; a short
                # one is usually a false positive correctly overridden.
                detail = "  ".join(
                    f"{c}[{w}px{',complete' if f else ''}]->"
                    f"{','.join(str(t) for t in l)}"
                    for c, l, w, f in spliced)
                n_full = sum(1 for _, _, _, f in spliced if f)
                print(f"  {len(spliced)} bar(s) spliced -- a tile they cross "
                      f"went to a source not showing them"
                      + (f", {n_full} of them COMPLETE (a real defect)"
                         if n_full else
                         " (all incomplete, so probably false positives being "
                         "correctly overridden)")
                      + f": {detail}")
        else:
            print("\ncity population bars: none found")
        if vision_promoted:
            print(f"  {len(set(vision_promoted))} tile(s) went to the only source "
                  f"seeing their whole 3x3 block (a city's owner always does, so "
                  f"this keeps an undetected bar too)")

    if args.ruin_vision:
        # Machine-readable like NO-OVERLAY: "could not look" must not read as
        # "no ruins".
        if no_ruin_sprite:
            print(f"\nNO-RUIN-SPRITE: cannot read {ruin_sprite_path()}, so "
                  f"Elyrion ruin markers were not searched for")
        elif ruin_no_fog:
            print(f"\nruin vision: {len(ruin_no_fog)} shot(s) locked too little "
                  f"fog to fit a fog reference against, so were not searched: "
                  f"{', '.join(sorted(ruin_no_fog))}")
        if ruin_hits:
            marked = sum(1 for k in ruin_hits if k not in winner)
            merged_note = (f" (from {n_raw} detections; ruins are never "
                           f"adjacent, so neighboring ones are one diamond "
                           f"cluster across a tile border)") if n_raw > len(ruin_hits) else ""
            print(f"\nElyrion ruin vision: {len(ruin_hits)} ruin(s) found"
                  f"{merged_note}, {marked} marked on the composite:")
            for (i, j), (hits, comp) in sorted(ruin_hits.items()):
                srcs = ", ".join(sorted({h[0] for h in hits}))
                extra = [c for c in comp if c != (i, j)]
                note = " -- already explored by another player, not marked" \
                    if (i, j) in winner else ""
                merged = f" [merged with {extra}]" if extra else ""
                print(f"  tile ({i},{j}): seen by {srcs}{merged}{note}")
        elif not no_ruin_sprite:
            print("\nElyrion ruin vision: no markers found on any fogged tile")

    if overlays & {"vision", "vision-each"} and not no_head_catalog:
        identified = {n: ident for n, ident in player_of.items() if ident is not None}
        if identified:
            n_players = len({key for key, _ncc in identified.values()})
            print(f"\nplayer identification: {n_players} player(s) identified "
                  f"from the Game Stats icon:")
            for n in names:
                ident = player_of.get(n)
                if ident is not None:
                    print(f"  {n:40s} -> {ident[0]} (ncc={ident[1]:.2f})")
        unmatched = [n for n in names if player_of.get(n) is None]
        if unmatched:
            print(f"  {len(unmatched)} shot(s) not confidently matched to a "
                  f"player, no outline drawn: {sorted(unmatched)}")

    print(f"\nconflicts: {len(conflicts)} tile(s) with inconsistent content "
          f"across sources (mean color dist > {args.consistency_thresh})")
    for (i, j), d, ws in sorted(conflicts, key=lambda c: -c[1])[:10]:
        print(f"  tile ({i},{j}): dist={d:.1f} sources={ws}")

    # The backstop for a stated size nothing else could check
    # (CONFLICT_FRAC_SUSPECT). It warns rather than refuses: the size was the
    # person's to state.
    if size_unverified and comparable >= CONFLICT_FRAC_MIN_COMPARABLE:
        frac = len(conflicts) / float(comparable)
        if frac > CONFLICT_FRAC_SUSPECT:
            print(f"\nWARNING: {100 * frac:.0f}% of tiles disagree across "
                  f"sources, which is far more than two shots of one board "
                  f"should. {board.N}x{board.N} is probably the wrong size -- at the "
                  f"right one this stays under 17%. The merge was written "
                  f"anyway; check it before trusting it.")
        # Deliberately no "looks fine" branch: a low fraction proves nothing.

    print(f"\nwrote {args.out}")


def write_debug(run):
    """--debug-dir output."""
    args, names, shots, samples, winner, conflicts, out, board = (
        run.args, run.names, run.shots, run.samples, run.winner, run.conflicts,
        run.out, run.board)
    if args.debug_dir:
        with PHASES("debug overlays"):
            os.makedirs(args.debug_dir, exist_ok=True)
            pal = {n: c for n, c in zip(names, itertools.cycle(
                [(60, 60, 255), (60, 220, 60), (255, 180, 60), (255, 60, 220),
                 (60, 220, 220), (200, 200, 200)]))}

            prov = (board.template.astype(np.float32) * 0.35).astype(np.uint8)
            for (i, j), n in winner.items():
                r = board.tile_mask_bbox(i, j)
                if r is None:
                    continue
                m, (x0, y0, x1, y1) = r
                region = prov[y0:y1, x0:x1]
                region[m > 0] = pal[n]
            cv2.imwrite(os.path.join(args.debug_dir, "provenance.png"), prov)

            conf_img = prov.copy()
            for (i, j), d, ws in conflicts:
                poly = board.tile_poly(i, j, 0.0).astype(np.int32)
                cv2.polylines(conf_img, [poly], True, (255, 255, 255), 3)
            cv2.imwrite(os.path.join(args.debug_dir, "conflicts.png"), conf_img)

            grid = out.copy()
            for i in range(board.N + 1):
                p0 = board.origin + i * board.u_col
                p1 = board.origin + i * board.u_col + board.N * board.u_row
                cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 0, 0), 1)
            for j in range(board.N + 1):
                p0 = board.origin + j * board.u_row
                p1 = board.origin + board.N * board.u_col + j * board.u_row
                cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 0, 0), 1)
            cv2.imwrite(os.path.join(args.debug_dir, "grid_overlay.png"), grid)

            # Undecorated warped sources and the lattice, for offline tools
            # (tools/ruinsprite.py); the overlays above paint over the pixels.
            for n in names:
                cv2.imwrite(os.path.join(args.debug_dir, f"warped_{n}.png"),
                            shots[n].warped)
            with open(os.path.join(args.debug_dir, "anchor.json"), "w") as fh:
                json.dump({
                    "map_size": int(board.N),
                    "origin": [float(v) for v in board.origin],
                    "u_col": [float(v) for v in board.u_col],
                    "u_row": [float(v) for v in board.u_row],
                    "template": os.path.basename(board.template_path),
                    "shots": {n: {
                        "scale": float(shots[n].scale),
                        # this shot's own fog tiles, where ruins are searched
                        "fog_tiles": [list(k) for k, per in sorted(samples.items())
                                      if n in per and per[n].get("witness")
                                      and not per[n]["explored"]],
                        "explored_tiles": [list(k) for k, per in sorted(samples.items())
                                           if n in per and per[n].get("witness")
                                           and per[n]["explored"]],
                        # fog_illumination's fit, None if too little fog
                        "fog_gain": (shots[n].gain.tolist()
                                     if shots[n].gain is not None else None),
                    } for n in names},
                }, fh, indent=1)

            for n in names:
                dim = shots[n].warped.copy().astype(np.float32)
                for (i, j), per_img in samples.items():
                    if per_img.get(n, {}).get("explored"):
                        continue
                    r = board.tile_mask_bbox(i, j)
                    if r is None:
                        continue
                    m, (x0, y0, x1, y1) = r
                    region = dim[y0:y1, x0:x1]
                    region[m > 0] *= 0.35
                cv2.imwrite(os.path.join(args.debug_dir, f"explored_{n}.png"),
                           np.clip(dim, 0, 255).astype(np.uint8))

            # The detector overlays below are written even when empty, so a
            # miss or misfire is visible.
            if args.city_bars:
                for n in names:
                    vis = shots[n].warped.copy()
                    for (ci, cj), (bx, by, bw, bh), _wc, _pl in shots[n].bars:
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                poly = board.tile_poly(ci + di, cj + dj, 0.0)
                                cv2.polylines(vis, [np.round(poly).astype(np.int32)],
                                              True, (0, 255, 255), 2)
                        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh),
                                      (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(args.debug_dir, f"bars_{n}.png"), vis)

            if args.ruin_vision:
                for n in names:
                    vis = shots[n].warped.copy()
                    for i, j, area, mask in shots[n].ruins:
                        vis[mask] = (0, 0, 255)
                        poly = board.tile_poly(i, j, 0.0)
                        cv2.polylines(vis, [poly.astype(np.int32)], True,
                                      (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(args.debug_dir, f"ruins_{n}.png"), vis)

            for n in names:
                badge = shots[n].badge_mask
                if badge is None:
                    continue
                vis = shots[n].img.copy()
                vis[badge > 0] = (0, 0, 255)
                cv2.imwrite(os.path.join(args.debug_dir, f"badges_{n}.png"), vis)

            print(f"debug output in {args.debug_dir}/:\n"
                  f"  provenance.png       winner per tile\n"
                  f"  conflicts.png        tiles whose sources disagree, outlined\n"
                  f"  grid_overlay.png     the tile lattice on the composite\n"
                  f"  anchor.json          lattice, per-shot scale, tiles, fog gain\n"
                  f"  warped_<n>.png       each shot on the template canvas\n"
                  f"  explored_<n>.png     what each shot witnessed as explored\n"
                  f"  badges_<n>.png       capture-badge pixels excluded, in red\n"
                  f"                       (written for every shot, empty or not)\n"
                  + ("  bars_<n>.png         population bars found\n"
                     if args.city_bars else "")
                  + ("  ruins_<n>.png        ruin-vision pixels accepted, in red\n"
                     if args.ruin_vision else ""))


def main():
    args = parse_args()
    run = Run(args, resolve_overlays(args))
    load_inputs(run)
    select_map_shots(run)
    resolve_map_size(run)
    load_board(run)
    if args.cross_check:
        run_cross_check(run)
        return
    adopt_anchors(run)
    identify_players(run)
    check_board_size(run)
    sample_tiles(run)
    fog_lock_guards(run)
    borrow_anchors(run)
    drop_misanchored(run)
    fit_fog_pixels(run)
    select_winners(run)
    promote_city_bars(run)
    detect_ruins(run)
    check_conflicts(run)
    paste_composite(run)
    write_vision_each(run)
    report(run)
    write_debug(run)


if __name__ == "__main__":
    _t0 = time.perf_counter()
    try:
        main()
    finally:
        PHASES.report(time.perf_counter() - _t0)
