#!/usr/bin/env python3
"""
polymerge.py -- merge multiple Polytopia screenshots of the same map into a
single composite showing the union of all players' explored (non-fog) area.

What the game guarantees (established facts, not guesses -- the whole pipeline
leans on these, so changing one means changing the code that exploits it):
  * The camera is a fixed orthographic isometric projection and never rotates.
    Zoom is continuous (pinch), within limits. So any two views of one board
    differ by pan + zoom only: a 4-DOF similarity, never a homography.
  * The board is always a full NxN diamond -- never clipped, never missing a
    corner. --map-size states N; omitting it measures N off the screenshots
    instead (detect_map_size), which refuses rather than guessing when no shot
    spans the board.
  * A tile's *footprint* is fixed: the width:depth ratio of the top face is a
    constant of the projection. Its *height* is not -- each tile is a 3D box
    whose height varies by content (a fog cube stands taller than a plains
    tile, which stands taller than water). Three consequences run through this
    whole file: a board silhouette includes a side wall and so overstates the
    board's extent; tall content (mountains, big cities) is drawn extending
    toward the viewer, occluding part of the tile *north* of its own; and the
    four board edges are not equally good evidence (next point).
  * The boxes stand on a common base, so the SE and SW silhouette edges show
    that base, while the NW and NE edges are rim tiles' *top* faces and so sit
    lower in a screenshot than in the all-fog template wherever the rim has
    been explored. In the oblique basis dir_a points southeast and dir_b
    southwest, so a-min = NW, a-max = SE, b-min = NE, b-max = SW. See
    anchor_to_template, and the standing decision in CLAUDE.md about why
    preferring the bottom lips did *not* improve the merge.
  * Fog artwork is completely deterministic -- one identical render per tile,
    no per-tile rotation, jitter or variation. That is what makes correlation
    against a blank all-fog template render (Overlays/<name>-blank.png)
    a reliable fog *detector* and zoom/pan reference.
    Do not substitute a color test: mountains, snow and ice are as pale as
    fog and a saturation cutoff silently drops them. See --fog-ncc.
  * Terrain, cities, roads, ruins and territory borders render identically for
    every player, so two shots of one tile should agree on content.
  * Elyrion sees ruins through fog, each ruin marked with a cluster of
    several small rainbow diamonds drawn on top of the fog. Only that player's
    screenshots show them, and the cluster is neither centered on nor contained
    by its tile -- but its pooled centroid lands inside the right one.
    --ruin-vision merges them; see detect_ruin_vision.
  * Ruins are never adjacent, in either the edge- or corner-sharing sense. So
    two ruin detections on neighboring tiles are one diamond cluster
    straddling a tile border, not two ruins -- see cluster_ruin_tiles.

Usage:
  python3 polymerge.py shot1.jpg shot2.jpg shot3.jpg --map-size 20 -o merged.png \
      --debug-dir debug/

N must match the board or nothing works: the tile lattice period comes straight
from it, and a wrong N puts every tile's fog art out of phase with the template,
which makes the fog test call the whole board explored. That failure used to be
silent. Three checks now catch it, in descending strength: the board-size check
in main (span / fog period, owing nothing to N), --min-fog-lock, and the
post-merge conflict fraction (CONFLICT_FRAC_SUSPECT) for a stated size the other
two cannot reach.

Optionally, --ui-mask takes per-file exclusion rectangles in each image's own
pixel coords, for chrome that --top-crop/--bottom-crop don't cover:
  { "shot1.jpg": [[820,0,1850,225],[0,1700,2732,2048]],
    "shot2.jpg": [[280,90,820,230]] }

Not handled yet (would produce spurious --consistency-thresh conflicts rather
than wrong geometry, so the conflict report is left intact to surface them):
  * Resource icons a player cannot see for lack of the relevant technology, so
    one shot shows a resource where another shows bare terrain.
  * Unit art desaturating once that unit has moved this turn, so the same unit
    differs in color between two players' shots.

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
    """Pixels within the photographed frame and not UI chrome -- deliberately
    *not* excluding dark pixels, unlike build_valid_mask. This is the mask
    pasting layers against, and there "dark" must not mean "not photographed
    here": a winning source can legitimately be dark in patches (roof shadow, a
    tree trunk, a dim rooftop) and those are content, not holes. Conflating the
    two makes the paste fall through to a second, differently-aligned source at
    every dark speck. See the mask taxonomy in CLAUDE.md.

    `top_crop`/`bottom_crop` are fractions of image height, applied to every
    input identically regardless of its own layout. Deliberately unintelligent:
    the HUD (score/turn banner, action-button row) is always docked top and
    bottom in a band of fairly consistent relative height across phones and
    tablets, so a fixed crop needs no per-image judgment and generalizes to a
    screenshot this program has never seen. The cost is that on a shot with
    little margin above the board it can clip the board's own top corner.
    `ui_rects` remains available for whatever the band crop doesn't cover (a
    mid-screen dialog, say), but is not required."""
    h, w = img.shape[:2]
    m = np.full((h, w), 255, np.uint8)
    if top_crop > 0:
        m[:int(round(h * top_crop)), :] = 0
    if bottom_crop > 0:
        m[h - int(round(h * bottom_crop)):, :] = 0
    for x0, y0, x1, y1 in ui_rects:
        m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 0
    return m


# Local standard deviation below which a neighborhood counts as featureless.
# Verified across every test set at 2.0, 4.0 and 8.0 with identical outcomes,
# so 4.0 sits in the middle of a wide indifference band rather than on an
# edge. Being well clear of 2.0 also leaves room for JPEG noise, which raises
# the sky's measured texture on compressed screenshots.
SKY_SMOOTH_STD = 4.0


def sky_mask(img, std_thresh=SKY_SMOOTH_STD, k=9):
    """Pixels belonging to the empty space *behind* the board.

    The obvious test -- "sky is black" -- is what `--dark-thresh` encodes, and
    it is wrong: the game runs a slow sunrise that progressively lightens the
    background the longer it is left open, so the sky's brightness is not a
    constant of the render at all. Measured on tests/pol_archi_test/pol.png it
    runs V=13 at the top of the frame to V=60 at the bottom, sailing past the
    threshold, after which the silhouette fuses with the background and *no*
    board edge can be found -- all four come back as phantoms.

    So identify the sky by what it is rather than how bright it is: **empty**.
    It carries no content, only a smooth gradient and a few stars, whereas
    board content -- faceted terrain, territory borders, sprites -- always has
    local variation. Sky measures a median local std of 0 against the board's
    24-27.

    Smoothness alone is not enough, though, and using it alone actively breaks
    things: where the board is genuinely flat (desert sand, calm water, ice)
    it deletes real silhouette, which cost `badland_test` both of its
    b-direction edges at every threshold tried. The fix is to also require
    **connectivity to the image border**. Sky is by definition outside the
    board, so it always reaches the frame edge, while a flat patch in the
    board's interior cannot get there without crossing textured content. That
    restriction makes interior terrain untouchable however flat it is.

    Brightness never enters, which is the point: however far the sunrise goes,
    empty space stays empty."""
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
    # `drop_sky` is a *fallback*, off by default, and deliberately so. It
    # catches sky the darkness test misses once the game's sunrise has
    # lightened it (see sky_mask), but it is not free: it also nibbles the
    # silhouette wherever the board's own rim is smooth, shifting a
    # weakly-supported board edge. See the sky_rebuild call site (below, in
    # anchor_to_template) for why main only turns this on as a last resort,
    # and CLAUDE.md for the case where applying it unconditionally regressed
    # a set.
    if drop_sky:
        m[sky_mask(img)] = 0
    if erode_px > 0:
        m = cv2.erode(m, np.ones((erode_px, erode_px), np.uint8))
    return m


def _fill_enclosed_holes(mask):
    """A binary mask with every hole that is *fully enclosed* by foreground
    filled in, via the standard flood-from-the-border trick: flood-fill the
    background from a corner, then anything still background afterward was
    unreachable from outside, i.e. an enclosed hole."""
    h, w = mask.shape
    flood = mask.copy()
    ffm = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffm, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def badge_halo(badge, found):
    """The badge mask grown to cover the glow around it, for geometry only.

    detect_capture_badges finds the blue disk. The badge is drawn with a bright
    halo bleeding well past that disk, and while a few glowing pixels do not
    matter when the question is "what color is this tile", they matter a great
    deal to board_boundary: a badge sitting over the board's rim spills brightness
    into the sky, --dark-thresh admits it as board, and the silhouette grows a
    bump where there is no board at all. That is not hypothetical -- it is what
    put star_change/oum2.png a full tile out. Its badge sits on the NW rim, and
    the halo cost that edge its support (58 boundary px, below the bar), leaving
    pan in the a direction resting on a lone phantom a-max line 0.88 tiles away.
    Excluding the halo restores the edge to 607 px and the shot anchors on its
    own to within 0.03 tiles.

    The dilation is the blob's own radius rather than a constant, because the
    halo is drawn in proportion to the badge and the badge is HUD: it is the
    same size in a shot's own pixels regardless of zoom, but not across devices.
    Measured on that badge, the glow reaches ~0.7 of the disk radius past its
    edge; a full radius is the round number that covers it with margin. Eating a
    little real rim alongside it is safe in a way it would not be for the color
    masks -- edge_lines fits a line of *known direction* by trimmed median over
    hundreds of points, so a local notch is an outlier, while a phantom bump is
    a systematic lie.

    Note the badge is a *pin*: it is drawn above the tile it marks, so a capture
    on a rim tile puts it over the sky beyond the board entirely, with nothing
    but its own glow connecting it to the silhouette. That is the worst case for
    the boundary and the reason this keys purely on the badge's appearance --
    never assume it overlaps the board at all."""
    r = max(int(round(float(np.sqrt(a / np.pi)))) for a, *_ in found)
    return cv2.dilate(badge.astype(np.uint8),
                      np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)


def detect_capture_badges(img, valid, h_lo=90, h_hi=112, s_lo=70, s_hi=170,
                          v_lo=190, ring_v=235, ring_s=60,
                          min_area=1500, min_fill=0.45, min_aspect=0.55,
                          min_ring_frac=0.30):
    """Flag pixels belonging to a capture HUD badge: a solid sky-blue disk
    (hue ~90-112, fairly saturated) wrapped in a glowing near-white ring,
    floated over whatever tile a capture is in progress on. It is per-player,
    per-moment UI, not persistent map content, so it should be excluded the
    same way UI-chrome rectangles are -- but it moves with the action, so a
    fixed rectangle can't cover it and it has to be found by appearance
    instead. The game draws two of these, one for a pending city capture and
    one for a pending ruin capture; the thresholds below were calibrated on
    the instances present in this project's test screenshots, so a badge
    variant that never appeared there may still need widening the ranges.

    Color alone is not enough. Both the surrounding fog and a handful of
    small blue decoration icons scattered on fog tiles are close to this same
    hue, so the blue+white color test is only a candidate filter; shape does
    the real discriminating, calibrated against real badges found in this
    project's test screenshots plus their false-positive neighbors:
      - `min_fill`, `min_aspect`: the badge is a near-solid disk. Fog's blue
        facets are angular and scattered, not compact, and fail this on their
        own -- *except* that the icon drawn in the badge's center (crossed
        swords, a sword-and-shield, etc) is white, not blue, so it silhouettes
        as a hole in the raw blue mask and can split one disk into two
        crescents, each of which fails the fill/aspect test alone.
      - Reconnect those crescents *topologically* (`_fill_enclosed_holes`),
        never by a large dilate. Closing the gap by distance is blind to what
        sits on the other side of it: close enough to bridge a badge's own
        icon-notch is also close enough to bridge open space to a same-hued
        neighbor, which is how a cluster of decoration icons once became a
        badge. The icon-shaped gap is fully enclosed by the disk's blue ring,
        while a neighboring icon is a separate component with open background
        between, so hole-filling can never reach it however close it sits.
      - `min_ring_frac`: requires a real near-white band around a *substantial
        majority* of the shape's own perimeter, not just brightness somewhere
        nearby -- fog is bright as a whole, but any one small patch of it does
        not encircle another small patch of it.
      - `min_area`: confirmed badges measured 3500-8300px here; the worst
        color+shape false positive found (that same decoration-icon cluster)
        measured 800-900px -- a >4x gap even before the hole-fill change made
        it unreachable in the first place. May need revisiting against future
        screenshots with a badge at a very different zoom.

    Restricting the color test to `valid` keeps dark/UI-chrome pixels (which
    can have meaningless hue at very low saturation) from ever entering it."""
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


# --------------------------------------------------------------- tile grid ---
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
    """Locate a board's 4 tile-grid vertices from a complete silhouette mask
    (the template's). Only reliable on a complete mask -- a screenshot's own
    coverage can be clipped short of the true vertex, which is what the
    board-edge line fit is for.

    The board is a slab of 3D boxes, not a flat diamond, so the silhouette is
    not the tile grid. At the leftmost and rightmost columns the mask runs from
    the top face's corner all the way down the slab's ~70px side wall, so take
    the *top* of that column, not its mean -- the top is the top face's corner,
    which is what the tile grid is indexed on. Using the mean puts the corner
    half a wall-height low and skews both lattice steps vertically by 3.7%
    while leaving the horizontal component correct, so it does not look like a
    scale error and is easy to miss.

    The near vertex is likewise the bottom of the near wall, so it is derived by
    completing the parallelogram rather than measured; the residual reports how
    far the measured near vertex is off horizontally, where the wall does not
    displace it.

    The side corners are taken from the first column that is a *whole* wall,
    which matters more than it sounds. The very outermost column is sometimes a
    thin antialiasing spur a few px tall, and its vertical position is
    arbitrary -- so taking its top puts the corner wherever the spur happened to
    land. Measured on the Overlays/ renders: small-blank's west extreme is a
    16px spur at y=713 against a real 61px wall at y=670 one column in, and
    tiny-blank has spurs on both sides at y=520 and y=550. detect_corners then
    reports a board whose two lattice steps differ by 2%, which is not a
    plausible shape for a square board and skews every tile position derived
    from it. Requiring the full wall costs the sizes that never had a spur
    nothing at all -- their extreme column already qualifies, so the answer is
    unchanged to the pixel."""
    ys, xs = np.where(mask > 0)
    y0 = ys.min(); top = np.array([xs[ys == y0].mean(), float(y0)])
    left = _side_corner(mask, xs.min(), +1)
    right = _side_corner(mask, xs.max(), -1)
    bottom = right + left - top
    y1 = ys.max()
    residual = float(abs(xs[ys == y1].mean() - bottom[0]))
    return top, right, bottom, left, residual


# The projection is fixed, so the two edge families have one angle at every
# board size and on every device -- a constant, not something to re-estimate
# per render. It is NOT atan(3/5) = 30.9638; the actual slope is 0.5986, and
# four independent routes agree on it (see CLAUDE.md, "Don't feed the exact
# 3:5 angle into edge_lines").
#
# Deriving it per render from three corner *pixels* is what this replaced --
# noise masquerading as a per-render property, for a quantity that cannot vary
# (CLAUDE.md has the scatter that gave it away).
#
# The corners are still measured and still matter -- they remain the source of
# the two step *lengths* (see build_lattice), which are genuinely per-render.
# It is only the two *directions* that are constants, so callers take them from
# here rather than through a function that accepted corners and ignored them.
#
# NOT perpendicular: Polytopia's isometric board is a rotated rhombus, not a
# rotated square (the two families meet at ~118 degrees, not 90), so treating
# this as an orthonormal basis would silently rotate dir_b away from its true
# direction.
BOARD_EDGE_SLOPE = 0.5986            # rise/run of the dir_a family
BOARD_DIR_A = np.array([1.0, BOARD_EDGE_SLOPE]) / np.hypot(1.0, BOARD_EDGE_SLOPE)
BOARD_DIR_B = np.array([-BOARD_DIR_A[0], BOARD_DIR_A[1]])


# Chrome is docked to the screen frame and runs horizontally or vertically; a
# board edge never does. The camera is fixed isometric and never rotates, so a
# silhouette edge is only ever at dir_a or dir_b, identical at every N. The
# discriminator is that game guarantee, not a tuned number.
#
# What makes it *discriminating* is the smoothing window rather than the
# tolerance. BOARD_ANGLE_SMOOTH averages the outline tangent over 2k px, so a
# fragment has to sustain a board angle across ~70px of outline before it
# registers at all: the test screens by edge *length* as much as by angle, which
# a tolerance alone cannot do. It has to, because chrome is not reliably
# axis-aligned -- the replay play button's upper edge sits 1.1 deg off dir_a,
# and a near-equilateral triangle beside a ~30 deg board edge is a coincidence
# that will recur. Do not narrow the window to save time.
#
# 80px sits between all corpus chrome (<= 30) and board halves severed by a
# full-width dialog (>= 173). Three other formulations were measured and are
# worse -- component area as a share, the fraction of the outline at a board
# angle, and a per-component edge_lines support test. Do not retry them;
# CLAUDE.md records what each one does.
#
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
        if len(runs):
            best = max(best, int(runs.max()))
    return best


def _board_component(m, dirs=None):
    """`m` with everything detached from the board's own silhouette removed.

    board_boundary outlines every component of the mask, and an edge line is
    fitted to whatever lies furthest out in that direction -- so a scrap of
    bright chrome sitting *outside* the board captures that edge outright and
    leaves the real one orphaned, supported by nothing. It is not a near miss,
    and three different pieces of chrome have caused it:

      - fogless/s1.png and s2.png carry the replay's turn-timeline strip, which
        lands just below the --top-crop band and is 8.4% of the board's own area.
        It takes a-min on one shot and b-min on the other, leaving each with a
        single edge and no way to pan-anchor, so both are dropped and the merge
        reports "no valid images found".
      - u_forest2/ely.png carries the game's collapsed side-drawer tab, 577px
        past the true NE rim: that rim's 800 outline points then support nothing
        and b-min reports 18 against --min-edge-support 150.
      - pol_archi_test/kick.png is the same defect from score banner glyphs
        landing just below the crop, and is the origin of that shot's
        long-documented "phantom" a-max of 30 points.

    The strip is horizontal, the tab vertical, the glyphs too small to sustain a
    run at any angle -- which is the shape of the fix. See
    BOARD_COMPONENT_MIN_EDGE_RUN for the discriminator and the measured margins.

    Without `dirs` nothing is filtered; the template call sites pass nothing
    because a render is a single clean component."""
    nc, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if nc < 3:                            # background plus at most one region
        return m
    if dirs is None:
        return m
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = [big]
    for k in range(1, nc):
        if k == big or stats[k, cv2.CC_STAT_AREA] < BOARD_COMPONENT_MIN_AREA:
            continue
        x, y, w, h = stats[k, :4]
        # Measured inside the component's own bbox, padded so the outline is
        # never clipped by the array edge. Chrome bboxes are a few hundred px
        # against a whole screenshot, which is what keeps this cheap.
        one = np.zeros((h + 2, w + 2), np.uint8)
        one[1:-1, 1:-1] = (lab[y:y + h, x:x + w] == k) * 255
        if _longest_board_angle_run(one, dirs) >= BOARD_COMPONENT_MIN_EDGE_RUN:
            keep.append(k)
    return np.where(np.isin(lab, keep), m, 0).astype(m.dtype)


# --- menu screenshots (score screen, tech tree, ...) ------------------------
#
# A *score screen* -- the scoreboard drawn full-screen over a dimmed copy of
# the map -- is board-ish enough to anchor, and merging one is silently
# destructive: added to goon_test2 it locked 63 fog tiles and won 50, on a board
# it does not belong to.
#
# Neither existing guard can see it, and one cannot in principle.
# --min-fog-lock fails because the fog test is NCC and NCC is
# illumination-invariant *by design* -- the whole reason it is used -- so a fog
# cube under a scrim locks like any other. A brightness test fails because the
# two populations interleave: a zoomed-in shot of a dark board is just as dim.
# Do not retry either.
#
# What separates them is the projection, a game fact rather than an appearance:
# every edge on the board runs at dir_a or dir_b and *nothing on it is
# horizontal*, while a menu is laid out with the screen. Same test as
# _board_component's above, asked of a whole frame, and reusing its
# BOARD_ANGLE_TOL because it is the same physical question.
#
# **The HUD crop is load-bearing here, not incidental.** At top_crop and
# bottom_crop 0 the margin does not merely shrink, it *inverts*, because a real
# shot's score banner and action-button row are horizontal chrome of exactly the
# kind this keys on. Crop before asking.
#
# 0.35 is mid-gap, 1.48x clear both ways, on an evidence base of three score
# screens. A menu with *no map behind it* is a different case and the anchor
# path rejects it on its own -- do not spend margin tuning this bar to catch
# one. CLAUDE.md has the measurements.
MENU_BOARD_ANGLE_FRAC = 0.35   # below this, the frame is not a view of the board
MENU_PROBE_WIDTH = 256         # long edge the probe downsamples to
MENU_EDGE_PCT = 92.0           # gradient magnitude percentile counted as an edge


def board_angle_fraction(img, dirs, top_crop=0.0, bottom_crop=0.0,
                         width=MENU_PROBE_WIDTH):
    """Of this frame's strong-edge energy running either at a board angle or
    with the screen, the share running at a board angle.

    ~0.6-0.9 for a screenshot of the board, ~0.2 for a menu drawn over one. See
    MENU_BOARD_ANGLE_FRAC for the measured populations and the bar.

    The screen-aligned energy is the denominator rather than all of it, which
    makes the answer a *ratio between two families* instead of a share of
    whatever else the frame happens to contain -- so it does not move with how
    textured the shot is. Costs ~7ms per shot at the default width."""
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

    Same cleanup and same component filter board_boundary applies, exposed so
    SIFT can be told to look at the board rather than at whatever else survived
    the crop. See the note at the sift_features call sites for why that matters:
    the replay UI is byte-identical between two shots of one replay, so features
    on it match perfectly and RANSAC returns the identity transform."""
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return _board_component(m, dirs)


def board_boundary(mask, dirs=None, frame_margin=10, open_px=15, close_px=15):
    """The board's silhouette outline within one image, as an (N,2) point array.

    Interior holes (UI cut-outs, dark terrain) are filled so they cannot emit
    boundary points, detached chrome is dropped (see _board_component), and
    stretches that merely follow the image frame are dropped too -- those are
    where the photo ran out, not where the board ends.

    The cleanup is an open *and* a close, and it must be that pair rather than
    an erosion: both operations leave a straight half-plane exactly where it
    was, so the board edge does not move, while an erosion would pull it inward
    by a few of that image's own pixels -- which is a different distance in
    board units for every zoom level, and so becomes a scale error in the fit.

    Note the pad goes on *after* the morphology: closing a padded mask can
    bridge the 1px ring, and then the flood fill leaks and returns nothing."""
    # board_region is exactly this outline's own prologue -- the same open,
    # close and component filter -- so the two cannot drift apart. The component
    # filter runs after the morphology, so a scrap the open already dissolved is
    # never tested, and before the pad, for the same reason the pad goes last.
    m = board_region(mask, dirs, open_px, close_px)
    pad = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    outside = pad.copy()
    cv2.floodFill(outside, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8),
                  (0, 0), 255)
    filled = (pad | cv2.bitwise_not(outside))[1:-1, 1:-1]
    bnd = cv2.subtract(filled, cv2.erode(filled, np.ones((3, 3), np.uint8)))
    h, w = mask.shape
    ys, xs = np.where(bnd > 0)
    keep = ((xs > frame_margin) & (ys > frame_margin) &
            (xs < w - 1 - frame_margin) & (ys < h - 1 - frame_margin))
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float64)


def _line_mode(vals, bw=2.0, smooth=3):
    """Offset of the sharpest straight line hiding in `vals`: the peak of a
    2px histogram, not the middle. A board edge shows the slab's 3D side wall,
    so the boundary points near it form a ~30px-thick band; a median would sit
    in the middle of that band, whereas the template's own corners come from
    the silhouette's outer extreme. The mode locks onto the same crisp outer
    line in both."""
    n = int(np.ceil((vals.max() - vals.min()) / bw)) + 1
    hist, edges = np.histogram(vals, bins=n, range=(vals.min(), vals.min() + n * bw))
    hist = np.convolve(hist, np.ones(smooth) / smooth, mode="same")
    k = int(hist.argmax())
    sel = (vals >= edges[k] - bw) & (vals <= edges[k + 1] + bw)
    return float(np.median(vals[sel]))


def edge_lines(pts, dir_a, dir_b, windows=(40.0, 15.0, 8.0), tol=3.0):
    """Fit the four board edges as lines of known direction. Returns their
    offsets [a_min, a_max, b_min, b_max] in the oblique dir_a/dir_b basis, plus
    how many boundary points support each.

    The direction of every edge is already known (the camera never rotates), so
    only the offset is unknown, and it is estimated from the points that
    actually lie on that edge. That is the point of doing this rather than
    taking a percentile over the silhouette's *area*, which trims a fixed
    fraction of the board's extent and so shrinks the board by a couple of
    percent no matter how clean the input.

    The support counts are the real output as much as the offsets are: an edge
    that isn't in frame gets a phantom line at the extreme of whatever else was
    there, and only its support count gives it away."""
    basis_inv = np.linalg.inv(np.stack([dir_a, dir_b], axis=1))
    ab = pts @ basis_inv.T
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

    Deliberately a nominator and never a classifier -- the same distinction that
    makes RUIN_NOMINATE_SAT acceptable. Mountains, snow, ice and pale sand are
    exactly as desaturated as the fog cube, so this admits 83.5% of a Polaris
    player's valid pixels (xizauh/pol.jpg) and 49.2% of a desert board's
    (badland_test/oum.jpg). It is safe only where a false positive costs work
    rather than an answer."""
    return (valid > 0) & (hsv[:, :, 1] < PIXEL_FOG_SAT) & (gray > 140)


# joint_register scores a shot's fog against the template's, and sums only the
# top JOINT_TOP_K tiles -- so a tile that cannot be fog by colour is paid for at
# every candidate and then discarded by the sort. Dropping those from the
# sample grid is the single largest saving in the program (see CLAUDE.md, the
# fog-colour tile prefilter). Three things about it are load-bearing:
#
# - **The keep-set is fixed once**, at full resolution from the incoming edge
#   prior, and reused at every pyramid level -- never decided per candidate
#   (which would gather at exactly the coordinates meant to be skipped) or
#   recomputed per level (the prior moves too little across the whole search
#   for it to matter, and per-level/once-only measure identically).
# - **The floor is not optional.** Some shots keep fewer tiles by colour alone
#   than JOINT_TOP_K itself, so without a floor a fog-poor shot's objective
#   collapses to a sum over almost nothing. Topping up costs fog-heavy shots
#   nothing, since the threshold already keeps more than the floor there.
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


# How far a fog autocorrelation peak must stand above the valley before it to
# count as a real period rather than the tail of a smooth decay. Genuine fog
# peaks measured 0.35+ of prominence (ely1: 0.968 at the peak against 0.62 a
# few px away); the desert false positive had none at all, decaying
# monotonically from the shortest shift.
PERIOD_MIN_PROMINENCE = 0.10


def _peak_parabola(ts, ss, k):
    if 0 < k < len(ss) - 1 and (ss[k - 1] - 2 * ss[k] + ss[k + 1]) < 0:
        return ts[k] + (ts[k] - ts[k - 1]) * (ss[k - 1] - ss[k + 1]) / (
            2 * (ss[k - 1] - 2 * ss[k] + ss[k + 1]))
    return float(ts[k])


def fog_period_scale(gray, valid, hsv, dir_a, tile_px,
                     lo=0.30, hi=3.75, min_ncc=0.50):
    """The image's zoom, read off the fog art's own repeat period.

    Fog is one deterministic render per tile, so a screenshot's fog region is
    exactly periodic along the board directions, with period = one tile step in
    that image's own pixels. Autocorrelating the fog with itself along dir_a
    (known exactly -- the camera never rotates) therefore peaks at the tile
    step, and zoom = template step / measured step. No template involved.

    **Do not measure zoom by matching fog patches against the template
    instead.** Fog is periodic, so a patch at a *wrong* scale correlates
    strongly against a different repeat of the pattern, and a zoom outside the
    swept range then yields a confident wrong answer rather than no answer. A
    period cannot alias that way: a shifted repeat *is* the quantity being
    measured, and multiples of the fundamental sit a factor of 2 apart rather
    than a few percent. Where a multiple does land in range (a zoomed-in shot's
    2x harmonic), taking the smallest strong peak keeps the fundamental.

    That last defense holds only while the fundamental is itself inside the
    sweep, which is what `lo` and `hi` are for. Both are set from the zoom
    extremes in tests/, with roughly half again in hand past each:
    missized_test/z2.png is a whole 18x18 board at s_it=2.23, period 40.4px, and
    basin_treaties/q.png is the most zoomed-in shot at s_it=0.398, period 202px
    and no opposite edge pair, so it depends on this fallback entirely.
    Narrowing either drops such a shot onto a harmonic or off the sweep
    altogether. Note the prominence test below needs a few samples of runway
    *before* a peak, so `lo` must clear the lowest real period by more than one
    coarse step rather than merely reach it. Widening `hi` cannot pull an
    existing shot onto a worse (larger) peak, since peaks are scanned
    smallest-first.

    Coarse sweep at quarter resolution, then a fine full-resolution pass with
    sub-pixel (parabolic) refinement around the winner: ~0.2% accuracy on the
    shots measured, against the +-3% window joint_register explores.

    Only needed by a shot with no opposite pair of board edges; an edge pair is
    a cheaper prior of similar quality. Returns (s_it, period_px, ncc) or
    None if no periodic fog is found."""
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
    # Require a genuine *peak*, not merely a high score. A periodic texture's
    # autocorrelation dips and comes back up, so its peak stands proud of the
    # valley before it; a smooth non-periodic region just decays monotonically
    # from the shortest shift and its "best" score is that decay, not a
    # period. Ignoring this is what let a desert board (tests/badland_test,
    # 27% of the frame pale sand and white cities that the fog color test
    # wrongly admits) report a confident 35px period where the truth was
    # ~118px -- an anchor 3x off that the fog-lock guard then had to catch.
    peaks = [i for i in range(1, len(ss) - 1)
             if ss[i] >= ss[i - 1] and ss[i] > ss[i + 1] and ss[i] >= min_ncc
             and ss[i] - float(ss[:i].min()) >= PERIOD_MIN_PROMINENCE]
    if not peaks:
        return None
    k = peaks[0]              # smallest period: the fundamental, not a harmonic
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
        # The coarse score at the peak actually chosen -- not ss.max(), which is
        # a different peak whenever the fundamental is not the strongest one
        # (missized_test/z2.png: fundamental 0.68 against its 2x harmonic 0.83).
        period, score = float(t0), float(ss[k])
    if score < min_ncc:
        return None
    return tile_px / period, period, score


# ------------------------------------------------------- joint registration ---
def _tile_sample_grid(origin, u_col, u_row, n, inset, div, max_px=320):
    """Integer sample coordinates (X, Y arrays of shape (n*n, px)) covering
    every tile's inset interior, at 1/div resolution.

    Every tile is the same shape, so one rasterized mask is built and reused for
    all of them with only the per-tile origin changing. That is what makes a
    search over hundreds of (zoom, dx, dy) candidates affordable at all --
    rasterizing a polygon per tile per candidate would be tens of thousands of
    fillConvexPoly calls per image.

    `max_px` caps each tile's sample set, because the fog art's features are
    ~10px across, so a few hundred evenly-spread samples measure the same
    correlation as all ~4000 px of a full-resolution tile. Evenly spread, not
    strided -- see the note at the cap below for why that distinction cost real
    accuracy.

    It is divided by `div` so that a coarse pyramid level actually costs less
    per candidate, which is the whole point of having a pyramid. **Do not make
    it a flat constant**: a tile holds only ~276 px at div=4, so a flat cap
    never binds there, every level gathers about the same ~300 samples per tile,
    and since the coarse levels evaluate *more* candidates (343 vs 175) they
    cost more in total than the full-resolution level rather than less -- a
    pyramid that shrinks the images but not the work. Dividing by `div` cut the
    phase by a third.

    Note the floor below binds before the division does at both coarse levels:
    the cap works out at 160 / 160 / 320 for div = 4 / 2 / 1, so div=2 samples
    exactly as many pixels per tile as div=4 and is dearer only because the beam
    visits it more often.

    The 160 floor is empirical and worth keeping. Thinning the coarse levels
    all the way down (a floor of 24) is faster still, but it costs accuracy on
    exactly the shots that can least afford it -- one with little fog in frame
    has few tiles constraining the fit, so a noisier per-tile correlation moves
    which tiles land in `_fog_alignment_score`'s top_k. That pushed
    `test_ss_2` (whose `cym1.jpg` locks just 15 fog tiles) from 0.021 to 0.059
    cross-check tiles, past this project's 0.05 healthy bar. At 160 no set
    exceeds 0.056, which is where the worst set already sat before any of
    this. The merges themselves were identical either way -- same explored
    union, same or fewer conflicts -- so this is tuning a leading indicator,
    not a fix."""
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
        # Take exactly `cap` evenly spread pixels, rather than every step'th.
        # Striding by an integer makes the sample count depend on how many
        # pixels the template happens to put in a tile, which is a property of
        # the render rather than of the board: at div=4 the original templates
        # gave 276 px/tile -> step 2 -> 138 samples, while the Overlays renders
        # give 222 -> step 2 -> 111, a fifth fewer probes at the level that
        # chooses which zoom branch to follow. That thinning is exactly what
        # this function's own note warns about, and it is the systematic half
        # of why a smaller template anchored worse.
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

    Only the most fog-like tiles are counted: explored terrain correlates with
    fog at about 0.0, so on a mostly-explored shot including every tile would
    bury the signal. Fully vectorized -- a couple of gathers plus row-wise
    reductions -- because this is the inner loop of the whole program.

    `tmpl_vals` is the template already gathered at (Y, X), not the template
    itself. That gather depends on neither the pan offset nor the zoom, so it
    is identical for every call within a pyramid level -- several hundred of
    them -- and hoisting it out drops a third of this function's memory
    traffic. See joint_register, which computes it once per level."""
    # Gather through a *flat* take rather than a 2-D fancy index. Same values,
    # same order, bit-identical result -- but numpy's 2-D fancy indexing builds
    # the coordinate pair per element, and a 1-D take does not: measured 0.630ms
    # against 0.086ms for this function's two gathers, which is 7.3x on them and
    # 1.76x on the whole call. It is what makes the per-pan cost small enough
    # that the sweep is dominated by the arithmetic rather than by addressing.
    #
    # `base` is the flat index at zero pan and is invariant for a whole pyramid
    # level; a pan of (dx, dy) is the *scalar* `off` = dy*width + dx, so the only
    # per-call address work is one integer add over the sample grid.
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

    The masked correlation has to centre *both* sides over whichever samples are
    valid, and which samples those are depends on the pan -- a pan slides the
    sample grid, so a tile at the edge of what this shot photographed reads real
    pixels at one offset and the warp's zero border at another. That is why the
    template side cannot simply be hoisted out of the loop the way `tmpl_vals`
    is: `b - mean(b over valid)` is a different vector at every pan.

    For a tile whose samples are *all* valid the mean is over all of them, so it
    is pan-independent and `bc_pre = b - mean(b)` and `bden = |bc_pre|` come from
    the caller, computed once per pyramid level. Then sum(ac*bc) collapses to
    sum(a*bc_pre) because sum(bc_pre) = 0, and sum(ac^2) to
    sum(a^2) - sum(a)^2/n. Three reductions over the shot's samples instead of
    the ~ten passes the masked form needs, and no validity gather at all --
    measured 0.079ms against 0.534ms on a div=2 call.

    It reaches the same quantity by a different route, so it agrees with the
    masked form to ~1e-5 relative rather than bit-for-bit. See joint_register for
    where it is used and why not at div=1."""
    a = img_flat.take(base + off).astype(np.float32).reshape(shape)
    sa = a.sum(1)
    den = np.sqrt(np.maximum((a * a).sum(1) - sa * sa / shape[1], 0.0)) * bden
    ncc = np.where(den > 1e-6, (a * bc_pre).sum(1) / np.maximum(den, 1e-6), 0.0)
    k = min(top_k, ncc.size)
    return float(np.sort(ncc)[-k:].sum())


# (div, zoom half-span, zoom step, pan radius in div-px, pan step in div-px,
#  how many zoom candidates this level hands to the next).
# The coarse level explores the whole plausible range around the edge-derived
# prior (+-3%, and +-24 full-res px of pan); each finer level only has to cover
# the previous level's step size, which is why the spans shrink so fast.
#
# The last field narrows as the levels sharpen, which is the whole shape of the
# fix: the blurry level cannot tell its candidates apart and so must not choose,
# while the full-resolution level is the one that actually discriminates and
# needs no successor. See the note below on why several are carried at all.
JOINT_LEVELS = ((4, 0.030, 0.010, 6, 2, 3),
                (2, 0.010, 0.003, 3, 1, 2),
                (1, 0.003, 0.001, 2, 1, 1))


# The last JOINT_LEVELS field is how many zoom candidates a level hands to the
# next. It is 3, 2, 1 and not 1, 1, 1: a coarse level's score surface is noisy
# near its optimum, so its best-scoring zoom is often in the wrong basin, and
# the finer levels have to be given more than one hypothesis to arbitrate. That
# is the mechanism behind the "refinement moves a correct prior to a worse
# position" problem, and the tell is that the score function was right all along
# -- only the search was wrong.
#
# The taper is a cost decision: scoring is 90-99% of this phase, so every extra
# candidate costs a full pan sweep. Do not narrow it -- not at the coarsest
# level, and least of all before div=1, which is where the discrimination
# actually happens. CLAUDE.md has the measurement for each width.


def _prune_beam(cand, width, min_sep):
    """The `width` best candidates, no two within `min_sep` in zoom.

    The separation is the point. Plain top-N returns N neighbors of the same
    optimum, which explores nothing; requiring a gap of one search step makes
    the survivors genuinely different hypotheses."""
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

    A tile is kept if at least JOINT_TILE_FOG_FRAC of its samples could be fog
    by colour under the edge-derived prior, with JOINT_TILE_FLOOR tiles kept
    regardless (topped up from the ranking) so a fog-poor shot still has an
    objective. See the constants for why each half is there.

    Measured at the prior across all 74 corpus shots, this excludes no tile that
    goes on to lock fog at the final anchor on 71 of them; the three exceptions
    (goon_test2/imp 11 of 58, control_c 5 of 76, control_d 1 of 88) lose
    accuracy rather than truth, since the tile is still merged and only its vote
    on the anchor is dropped. The floor recovers most of that in practice."""
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

    Jointly, not one after the other, because they are not independent: a zoom
    error shows up as a pan error and vice versa, so optimizing one while the
    other is wrong walks toward the wrong answer. The image pyramid explores
    where pixels are cheap and lets the full-resolution level only polish.
    (This replaced a slower two-stage zoom-then-pan design; see CLAUDE.md.)

    Returns the image -> template affine."""
    Ht, Wt = tmpl_gray.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keep, n_fogish = _fogish_tiles(img, valid, gray, origin, u_col, u_row, n,
                                   Wt, Ht, scale0, trans0)
    # Sum the best `top_k` per-tile correlations, but never more of them than
    # there are tiles that could be fog at all. The score's job is to add up the
    # evidence, and a tile that is not fog still contributes a correlation
    # against fog art that moves with the candidate for reasons unrelated to
    # alignment -- so on a shot with 11 fog-ish tiles a fixed k of 60 sums 11
    # signals and 49 noise terms, and the noise wins. The colour count is a
    # generous upper bound on the fog (it admits 83.5% of a Polaris shot's
    # pixels), which is exactly what makes it safe to cap by: it errs toward
    # leaving k alone. 14 of the corpus's 77 shots see a smaller k.
    top_k = min(top_k, max(n_fogish, 1))
    # The board's centre in template space. The zoom sweep below pivots about
    # it; see the note at the sweep for why that is not the same search.
    board_c = origin + (n / 2.0) * u_col + (n / 2.0) * u_row
    beam = [(-2.0, float(scale0), float(trans0[0]), float(trans0[1]))]
    for div, s_span, s_step, p_rad, p_step, emit in JOINT_LEVELS:
      # Timed per level rather than as one total, because the shape of the
      # taper is the thing worth watching: measured on a 4-shot merge it runs
      # 0.35s / 1.20s / 1.90s at div=4 / 2 / 1, about 45% of the whole merge.
      # div=2 is not a cheaper level than div=4 -- it samples the same 160 px
      # per tile (see _tile_sample_grid's floor) and is simply visited more --
      # so a single total would hide which level a change actually moved.
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
          # Most scored tiles have no invalid sample anywhere in this level's
          # search, and those are far cheaper to score (_fog_full_score). Only
          # the coarse levels take it. Not because they are the best fit -- they
          # are the worst, since pan radius is in each level's own pixels so
          # div=4 slides +-24 template px against div=1's +-2, and erodes four
          # times as much boundary (mean fully-valid share 0.63 / 0.74 / 0.80
          # over 21 shots). They are simply where the time is: div=4 and div=2
          # are ~69% of the scoring cost, and div=1 stays on the exact masked
          # score so the final answer is still chosen at full fidelity.
          fast = div > 1
          if fast:
              bc_pre = tvals - tvals.mean(1)[:, None]
              bden = np.sqrt((bc_pre * bc_pre).sum(1))
              pan_se = np.zeros((2 * p_rad + 1, 2 * p_rad + 1), np.uint8)
              pan_se[::p_step, ::p_step] = 1
          zooms = []
          seen = set()
          for _, s_cur, tx0, ty0 in beam:
              # Where this beam entry thinks the board's centre sits in the
              # shot's own pixels. Holding that point still is what makes the
              # zoom sweep and the pan sweep independent -- see below.
              piv = (board_c - np.array([tx0, ty0])) / s_cur
              # Snap the sweep to a grid shared by every beam candidate rather
              # than one centered on each. Beam entries sit only s_step apart
              # (that is _prune_beam's separation rule), so their +-s_span
              # sweeps overlap by about half -- but sweeps centered on different
              # points land on interleaved grids and never coincide, so the
              # duplicate zooms all get scored twice. On a shared grid they
              # collide exactly and the second one is free.
              lo = int(np.floor((s_cur - s_span) / s_step))
              hi = int(np.ceil((s_cur + s_span) / s_step))
              for q in range(lo, hi + 1):
                  s = q * s_step
                  # p_template = s * p_image + t scales about the image ORIGIN,
                  # so holding t fixed and changing s does not rescale the board
                  # in place -- it swings it across the frame by (s_cur - s)
                  # times the board's distance from that origin, which is most
                  # of a screenshot. At every pyramid level the sweep's ends
                  # displaced the board centre further than that level's pan
                  # reach could pull it back (see CLAUDE.md for the table), so
                  # the extreme zooms were scored while guaranteed
                  # misregistered, for a reason unrelated to whether the zoom
                  # is right, and the score surface was biased toward the prior.
                  #
                  # Carrying the translation that holds the board centre fixed
                  # removes the coupling: each zoom is scored at its own best
                  # centring, and dx/dy below then search genuine pan.
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

          # The fast path's tile set is settled ONCE for the whole level, as the
          # intersection over every zoom candidate in it. Per candidate looks
          # equivalent and is not (see CLAUDE.md for the measurement that caught
          # it): a different zoom warps the shot differently, so its coverage
          # boundary moves and the boundary tiles flip, and the score is a sum
          # over the top JOINT_TOP_K per-tile correlations -- so a candidate
          # whose set happened to be larger would draw its top-k from more
          # tiles and win for a reason unrelated to alignment. Fixing the set
          # makes every candidate in a level answer the same question.
          #
          # A tile qualifies only if it is valid at every *pan* too, which is
          # what the erosion by the level's pan grid answers in one pass.
          # borderValue 0 so anything the erosion reads off the padded canvas
          # counts as invalid, rather than as no constraint, which is what
          # cv2.erode assumes by default.
          sel = None
          if fast:
              for s, tx, ty in zooms:
                  full = (cv2.erode(_warp_valid(s, tx, ty), pan_se,
                                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
                          .ravel().take(base).reshape(shape) > 0).all(1)
                  sel = full if sel is None else (sel & full)
                  # The intersection only shrinks, so a level that is already
                  # short of top_k cannot recover -- stop rather than warping
                  # the rest for an answer that is settled. This is what keeps
                  # the pre-pass from costing the fog-poor shots anything.
                  if int(sel.sum()) < JOINT_TOP_K:
                      sel = None
                      break
          if sel is not None:
              # A fog-poor shot keeps almost nothing here (star_change/oum holds
              # 25 of 120 tiles at div=4), and below this bar the objective would
              # be a sum over fewer terms than it is meant to select from. Such a
              # level falls back to the exact masked score in full -- never a
              # mix, since the two forms agree only to ~1e-5 and the beam
              # compares candidates within a level against each other.
              #
              # The bar is the CONSTANT, not the per-shot `top_k` above. Those
              # were the same number until the fog-ish cap landed, and keeping
              # them tied would put every fog-poor shot onto the fast path as a
              # side effect of shrinking its k -- two changes wearing one
              # constant's name. Read it as a floor on how much evidence a level
              # must hold before approximating is worth it, which is a different
              # question from how much of that evidence the score sums.
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
              # Only the exact path reads validity, and it re-warps rather than
              # keeping every candidate's mask alive from the pre-pass: this
              # branch is the rare one, and holding ~20 padded masks costs more
              # than the warp it saves.
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


# A board silhouette overstates the board by the slab's side wall, so counting
# tiles across it (span / fog period) lands high by a fixed amount. Measured
# over the 30 shots with an opposite edge pair across all ten sets, span/period
# sits in [N+0.61, N+0.90] -- mean 0.78, and consistent between the 18x18 and
# 20x20 boards, as it should be for an absolute wall height expressed in tiles.
# Subtracting it recovers N to within +-0.13, i.e. a ~0.4-tile margin either
# way before the rounded answer would change.
BOARD_SPAN_WALL_TILES = 0.78

# Polytopia's board sizes, by the name the game gives each. The renders in
# Overlays/ are named this way, and the mapping is measured rather than assumed:
# feeding each <name>-blank.png through this file's own span/fog-period
# measurement returns 11.01, 14.00, 16.01, 17.86, 20.03 and 30.05 tiles.
BOARD_SIZE_NAMES = {11: "tiny", 14: "small", 16: "normal",
                    18: "large", 20: "huge", 30: "massive"}

# Board sizes with a template on disk, and so the only sizes --map-size accepts
# or detect_map_size can answer with.
#
# 30x30 is deliberately held back even though its render is present and
# correct: its 2x fog-period harmonic sits close enough to both 14 and 16 that
# a misread period could be silently detected as one of them and merged
# wrong -- the single worst failure this program has. Every other size's
# halving lands nowhere near a real board and is discarded harmlessly; 30
# would not be. See CLAUDE.md, "30x30 is deferred", for what restoring it needs.
MAP_SIZE_DEFERRED = (30,)
MAP_SIZE_CHOICES = tuple(n for n in sorted(BOARD_SIZE_NAMES)
                         if n not in MAP_SIZE_DEFERRED)


def size_list():
    """The board sizes, phrased for a player rather than for a shell.

    Every message this appears in is one polybot passes straight to a Discord
    channel, and the reader there has never seen this program's command line:
    naming --map-size sends them hunting for a flag they have no way to type,
    and the bot's own equivalent is a bare word (`!merge 20`). So these
    messages say what to *do* -- retry with the size included -- and leave the
    flag to --help, where the only person who can use it is already looking.

    Same comma-then-"or" shape as polybot's unrecognized-size reply, because
    four "or"s read badly at five sizes."""
    return (", ".join(str(n) for n in MAP_SIZE_CHOICES[:-1])
            + f" or {MAP_SIZE_CHOICES[-1]}")


# The remedy clause every "we could not work the size out" refusal ends with.
# One constant so the four of them cannot drift apart -- they are the same
# sentence to the player whichever internal check produced them.
#
# The unsupported-size refusal deliberately does *not* use it and is not drift:
# there the size is already known and simply is not one this can merge, so
# "please retry including the map size" would be advice that cannot help. It
# names size_list() on its own instead.
# No trailing full stop: one caller continues the sentence with a second
# remedy ("..., or with at least one screenshot that ...").
RESTATE_SIZE = "Please retry including the map size ({})"

# How far two shots' independently measured board sizes may differ before
# neither is believed. Across the corpus's edge-pair shots every measurement
# lands within 0.32 tiles of the truth, and the sizes being told apart are 2
# tiles from each other, so 0.5 still catches the case this exists for --
# screenshots of two *different* boards handed to one merge, which nothing else
# here detects. Watch this one: the worst intra-set spread is now 0.36
# (badland_test2), so the headroom is 1.4x rather than the 3x it once was. See
# the deferred item in CLAUDE.md on the period measurement's phase sensitivity,
# which is where the headroom went.
MAP_SIZE_DETECT_MAX_SPREAD = 0.5

# How far a single measurement may sit from *any* real board size before it is
# discarded as not a measurement of a board at all, rather than weighed against
# the others.
#
# **The count and the anchor are separate measurements from separate evidence**,
# which is what makes dropping the number the right response rather than
# dropping the shot: a shot with an opposite edge pair takes its zoom from the
# edges and never consults the fog period for it, so a shot whose count is
# wrecked by a harmonic can still be anchored perfectly. One such number used to
# fail an entire merge.
#
# 0.9 separates the populations by an order of magnitude both ways -- the worst
# honest measurement in the corpus is 0.32 out, the harmonic was 9.4 -- and it
# cannot swallow a real disagreement, because half of any supported size is not
# a size. Staying under 1.0 also keeps a genuine reading of a *neighboring* size
# plausible, so two different boards still reach the spread check above.
MAP_SIZE_PLAUSIBLE_TOL = 0.9


def plausible_sizes(implied):
    """Split {name: tiles-across} into the measurements that could describe a
    board and those that could not, as (kept, discarded).

    Discarded ones are not evidence about anything -- see
    MAP_SIZE_PLAUSIBLE_TOL. Callers report them and carry on with `kept`."""
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

# Inliers required before a SIFT match is trusted to supply a shot's zoom, a
# pan offset or a whole anchor -- counted only over inliers landing on *terrain*
# at both ends, and the only inlier bar in the file over pair_transform's own
# 12-match floor.
#
# **Counting raw inliers instead does not work, in either direction.** The
# hazard is fog matching the wrong repeat of itself, and that hazard is made
# entirely of fog: the worst spurious pair in the corpus reaches 115 raw inliers
# while a genuine match runs as low as 108. Restricted to terrain those same two
# read 0 and 100. So do not raise a raw bar to cover fog -- do not count the
# fog-borne inliers at all. CLAUDE.md carries both populations in full.
SIFT_TERRAIN_MIN_INLIERS = 60

# How far a shot may sit from where an anchored shot's SIFT geometry puts it and
# still count as corroborated, when the per-image fog-lock guard would otherwise
# drop it (see the drop site in main). In tiles, the same quantity --cross-check
# prints. Every set at its correct map size reports well under this, and the
# failures it must still catch are far coarser -- a shot anchored a few percent
# off is displaced by a few percent of the whole board -- so 0.20 sits clear.
#
# **Nothing in the corpus reaches this path any more** (see CLAUDE.md's
# star_change history for why), so it is not regression-tested by
# tools/baseline.py -- exercise a change here deliberately.
MISANCHOR_CORROBORATE_MAX_TILES = 0.20

# How far two opposite edge pairs may disagree on the zoom before neither is
# believed. Both measure the same board, so they should agree closely -- over
# the whole corpus the observed spreads run 0.00% to 1.24%, and the one shot
# that exceeds this reports 25.99%. There is no middle ground in the data, so
# the exact value anywhere in ~2-10% behaves identically.
EDGE_PAIR_MAX_SPREAD = 0.03


# A lone edge pair is believed on a weak side only if its span agrees with how
# wide the board already *looks* the other way. The board is square, so the two
# spans match to 0.17% in the template; and the silhouette's observed extent
# along the other axis is a lower bound on that direction's span -- equal when
# both its edges are in frame, short when one is cut off. So a real pair lands at
# ratio ~1 or a little above, and a pair resting on a phantom lands well off it.
#
# [0.95, 1.15] clears the nearest genuine pair in the corpus by 3.5% and the
# nearest phantom by 8%. **The evidence is four genuine samples**, which is
# thinner calibration than most numbers here -- CLAUDE.md lists both
# populations.
#
# It fails safe twice over: it is consulted only when the pair is already too
# weak for min_support, so nothing that passes today can start failing, and a
# pair that fails it falls back to exactly the old behaviour of looking for
# another zoom source.
LONE_PAIR_EXTENT_LO = 0.95
LONE_PAIR_EXTENT_HI = 1.15


def anchor_to_template(img, mask, valid, hsv, tmpl_gray, t_off, dir_a, dir_b,
                       origin, u_col, u_row, n_tiles, min_support, label,
                       refine=True, min_scale_support=30, zoom_hint=None,
                       sky_rebuild=None, pan_hint=None, cache=None):
    """Map one image onto the template.

    The board edges supply the starting estimate for both zoom and pan, and
    joint_register then refines both together against the fog artwork.

    Edges alone are not enough for zoom: a screenshot's silhouette includes the
    slab's 3D side walls, and a span always pairs a height-dependent top edge
    with a bottom lip, so the estimate carries a small systematic bias (mean
    +0.28%, at most 0.92% over the sets that existed when it was measured). But
    they are an excellent
    *prior*, and starting from them is what lets the refinement search a +-3%
    window rather than sweeping the whole plausible zoom range. A shot with no
    opposite pair of edges at all has no edge-derived zoom and falls back to
    the fog art's own repeat period (fog_period_scale; see that docstring for
    why period measurement replaced patch matching). An edge *pair* is still
    preferred when one exists, because it cross-checks itself: both sides
    should imply the same scale even when each is individually weak. So
    `min_scale_support` is deliberately much lower than `min_support`: a
    median line fit only needs enough boundary points to be robust to
    outliers, not enough to trust one edge's absolute position with no
    cross-check, which is the higher bar `min_support` sets for pan below.
    (Measured once on a real shot with two weak-but-present edge pairs: the
    pairs landed within 0.04 tiles of ground truth after refinement, agreeing
    with each other to 0.14%.)

    Pan is where the two halves of the basis matter. Each board edge is a line
    of constant `a` or constant `b`, so seeing one edge pins exactly one of the
    two offsets -- which is why one edge per direction is the requirement, and
    why an *opposite* pair is not enough however well supported (it pins the
    same offset twice). Fog cannot supply the other one: it is periodic, so
    correlation says where a point sits within a tile, never which tile.

    `pan_hint` is the way out for a shot that shows only one direction's edge.
    It is an image -> template affine borrowed from an already-anchored shot by
    SIFT, and only the direction with no edge of its own is taken from it; the
    direction that has an edge keeps that edge. Terrain is not periodic, so the
    hint has none of fog's ambiguity. The caller must have cleared
    SIFT_TERRAIN_MIN_INLIERS first -- that floor is the whole gate here, because
    a borrowed offset cannot then be checked against the shot's own fog the way a
    borrowed zoom can (the fog test cannot distinguish one lattice repeat from
    the next, which is the reason the offset had to be borrowed at all).

    Returns (affine, zoom_source, implied_n, prior). `prior` is the
    unrefined edge/fog-period transform, or None when the returned affine *is*
    that prior because refinement was skipped. main keeps it so a refinement
    that cannot prove itself on the shot's own fog can be undone -- see the
    zero-lock block there. zoom_source is "edges" or
    "fog-period", so the caller can hold fog-period-anchored shots to the extra
    fog-lock check that their anchoring path warrants. implied_n is this shot's
    own independent estimate of the board size, or None -- see
    BOARD_SPAN_WALL_TILES and the map-size check in main."""
    # t_off is the *template's* four edge offsets, fitted by the same
    # board_boundary/edge_lines pair used below. It deliberately is not derived
    # from the tile-grid corners: a screenshot's silhouette edge includes the
    # slab's 3D side wall, so matching it against a top-face-based reference
    # would offset every shot by the wall. Fitting both sides identically makes
    # the wall cancel -- but only to the extent that the two boards have the
    # same content on the rim, since the wall's height varies with the tile.
    # The template is all fog, so a screenshot whose rim is explored does not
    # cancel exactly (see the edge bullet in the module docstring).
    span = [t_off[1] - t_off[0], t_off[3] - t_off[2]]

    # A private cache still dedups this call's own repeats (the sky rebuild
    # re-fits, and anchor_all can re-anchor a shot in its second pass); a shared
    # one additionally reuses whatever the size pre-pass already measured. See
    # ShotCache for why sharing is only ever a hit when the parameters match.
    cache = cache if cache is not None else ShotCache()

    tags = ["a-min", "a-max", "b-min", "b-max"]

    def fit_edges(m):
        with PHASES("anchor: board outline + edge fit"):
            pts = cache.boundary(label, m, (dir_a, dir_b))
            if len(pts) < 100:
                return pts, None, None, False
            off, support = edge_lines(pts, dir_a, dir_b)
        have = [c >= min_support for c in support]
        print(f"  board edges ({len(pts)} boundary px): " + "  ".join(
            f"{t}={'ok' if h else 'MISSING'}({c})"
            for t, h, c in zip(tags, have, support)))
        return pts, off, support, (have[0] or have[1]) and (have[2] or have[3])

    pts, off, support, pan_ok = fit_edges(mask)
    # Sunrise fallback, and deliberately *only* reached on failure. The game
    # slowly lightens the sky the longer it is left open, and once it passes
    # --dark-thresh the silhouette fuses with the background and every edge
    # comes back a phantom. sky_rebuild swaps in masks that identify the sky
    # by being *featureless* instead (see sky_mask), which fixes that outright
    # -- CLAUDE.md has the measured supports before and after.
    #
    # It costs a black-sky shot nothing, which is the point of hanging it here
    # rather than pre-computing: those shots take the first branch and never
    # build the second mask. It is also not free to apply blindly -- it
    # nibbles silhouette wherever the board's own rim is smooth, which cost a
    # real set accuracy when tried on every shot (see CLAUDE.md). So: only
    # when the ordinary masks have already failed, where there is nothing left
    # to lose.
    if not pan_ok and sky_rebuild is not None:
        print("  no usable board edge from brightness alone -- retrying with "
              "the sunrise-sky test")
        mask, valid = sky_rebuild()
        # The masks this shot is measured from have just been replaced, so
        # anything already measured from the old ones is stale. Invalidating
        # here rather than inside sky_rebuild keeps that obligation next to the
        # reassignment it belongs to, and covers a caller that passed no cache.
        cache.invalidate(label)
        pts, off, support, pan_ok = fit_edges(mask)
    if off is None:
        raise SystemExit(f"{label}: no board outline found (only {len(pts)} "
                         f"boundary px) -- is any board edge actually in frame?")
    have = [c >= min_support for c in support]
    # One entry per *direction*, which is the unit pan is actually decided in:
    # a-min/a-max both fix the offset along dir_a, b-min/b-max the one along
    # dir_b. Either edge of a pair will do; what cannot be missing is a whole
    # direction.
    have_dir = [have[0] or have[1], have[2] or have[3]]
    # A shot showing only one direction's edge is not lost if the caller has a
    # SIFT hint: that direction keeps its own edge and the other is borrowed
    # (see pan_hint above). Deliberately still requires one real edge -- a shot
    # with none at all would be taking its whole position from another shot,
    # which is the register-the-group design this pipeline is built to avoid,
    # and it has no independent evidence left to be judged on.
    can_borrow = pan_hint is not None and any(have_dir)
    if not pan_ok and not can_borrow:
        seen = " ".join(t for t, h in zip(tags, have) if h) or "none"
        raise SystemExit(f"{label}: pan needs one board edge from each direction, "
                         f"only found [{seen}] -- merge it with a shot that shows "
                         f"more of the board.")
    # True only when an offset really is taken from the hint. A hint is passed
    # to *every* shot re-anchored in anchor_all's second pass, including one
    # that has both directions in frame and merely needed a zoom (that is what
    # the second pass was originally for), and such a shot must go on using its
    # own edges and its own refinement. Conflating the two cost
    # pol_archi_test/kick.png its whole fog lock (17 -> 0) and took that set
    # from 0.035 to 0.191 cross-check tiles.
    #
    # That measurement is historical: kick.png finds all four of its edges now
    # (_board_component dropped the chrome that was capturing a-max), so it
    # never reaches the second pass and the corpus no longer reproduces it. The
    # distinction still has to be made -- any shot re-anchored there is offered
    # a hint, and most of them should ignore it.
    borrow_pan = can_borrow and not pan_ok

    have_scale = [c >= min_scale_support for c in support]
    edge_scales = [(off[hi] - off[lo]) / span[k]
                   for k, (lo, hi) in enumerate([(0, 1), (2, 3)])
                   if have_scale[lo] and have_scale[hi]]
    # A *lone* pair has nothing to cross-check against, so the low
    # min_scale_support bar does not apply to it: with two pairs a weak side is
    # fine because the printed spread would expose it, but with one an
    # edge barely above the phantom floor is simply believed and can produce a
    # confident, wrong scale (see CLAUDE.md, "A lone edge pair needs both sides
    # properly supported", for the case this guards and the fix's history).
    # Enforced below by the extent check rather than by the support count,
    # because the support count also rejected a genuine weak pair.
    if len(edge_scales) == 1:
        lo, hi = (0, 1) if (have_scale[0] and have_scale[1]) else (2, 3)
        weak = min(support[lo], support[hi]) < min_support
        ratio = None
        if weak:
            k = 0 if lo == 0 else 1
            reg = board_region(mask, (dir_a, dir_b))
            ys, xs = np.where(reg > 0)
            if len(xs) >= 100:
                b_inv = np.linalg.inv(np.stack([dir_a, dir_b], axis=1))
                ab = np.stack([xs, ys], axis=1) @ b_inv.T
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
        # Two pairs are trusted *because* they cross-check each other, so
        # actually act on the check rather than only printing it. Healthy
        # spreads across the whole corpus run 0.00-1.24%; anything remotely
        # near EDGE_PAIR_MAX_SPREAD means at least one "edge" is a phantom
        # line fitted to whatever else was in frame, and averaging the two
        # just splits the difference between a real measurement and a wrong
        # one. test_ss_elyruins/hood.png reports 25.99% and its average is
        # ~94% too large, which anchors it into nonsense (2 fog tiles locked
        # against 95 when its zoom comes from the fog period instead).
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
        # This shot spans the whole board in at least one direction, so it can
        # also *count* the tiles across it: measure the fog's repeat period in
        # its own pixels and divide. Costs one extra fog_period_scale (~0.2s)
        # and is what catches a wrong --map-size (see main).
        with PHASES("anchor: board-size check"):
            got = cache.period(label, img, valid, hsv, dir_a,
                               np.linalg.norm(u_col))
        if got is not None:
            spans = [off[hi] - off[lo] for k, (lo, hi) in enumerate([(0, 1), (2, 3)])
                     if have_scale[lo] and have_scale[hi]]
            implied_n = float(np.mean(spans)) / got[1] - BOARD_SPAN_WALL_TILES
            print(f"  board size implied by span/fog-period: {implied_n:.2f} tiles")
    elif zoom_hint is not None:
        # Supplied by the caller from a SIFT match against an already-anchored
        # shot. Preferred over the fog period when available, because SIFT
        # relative geometry is trustworthy to ~0.2px (see cross_check) whereas
        # the period needs a real expanse of fog to measure.
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

    # s is the image's zoom *relative to* the template (image px per template
    # px), so the image -> template transform scales by its reciprocal.
    # Per axis: offset_template = s_it * offset_image + coord(translation).
    s_it = 1.0 / s
    # The hint's own translation, split into the same two per-direction offsets
    # the edges produce, so one can be substituted for the other. Its scale is
    # already s_it: a caller passing pan_hint passes the same transform's scale
    # as zoom_hint, and the two must agree or the borrowed offset would be
    # measured against a different zoom than it is used at.
    if borrow_pan:
        basis_inv = np.linalg.inv(np.stack([dir_a, dir_b], axis=1))
        hint_shift = basis_inv @ np.array([pan_hint[0, 2], pan_hint[1, 2]])
    # Blend the two edges of a pair by support -- deliberately, even though the
    # bottom lip (a-max = SE, b-max = SW) is the individually more accurate edge
    # and the top one carries a height-dependent bias. Preferring the bottom lip
    # gives a better prior and a *worse* merge, because joint_register already
    # absorbs pan error of this size while --cross-check rewards priors that are
    # consistent across shots over priors that are individually accurate. This
    # was measured and reverted; don't re-land it as an accuracy fix. See the
    # standing decision "a better prior is not a better answer" in CLAUDE.md.
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
        # Free corroboration: the hint covers *both* directions, so the one
        # that kept its own edge can be compared against it. They are
        # independent measurements -- a silhouette line against matched
        # terrain -- so a large disagreement means one of them is wrong and
        # says which direction to look in.
        for d in range(2):
            if have_dir[d]:
                print(f"  {'ab'[d]}-direction edge sits "
                      f"{shift[d] - hint_shift[d]:+.1f}px from where the SIFT "
                      f"hint puts it")
    trans = shift[0] * dir_a + shift[1] * dir_b
    if borrow_pan or not refine:
        # A borrowed pan is deliberately *not* handed to joint_register, and
        # this is the difference between the feature working and not working.
        # The refinement scores candidates by fog alignment, and a shot that
        # had to borrow an offset is one the board edges could not place --
        # which in practice is a zoomed-in shot with barely any fog in frame,
        # so its score surface is noise and refining walks a good prior off
        # into the argmax of that noise (CLAUDE.md has the measured case).
        # Nothing is given up by stopping here: both halves of this anchor come
        # from SIFT relative geometry, good to ~0.2px (see cross_check), where
        # fog has no absolute position to offer at all.
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
# Of the tiles two sources can actually be compared on, the fraction that
# disagree -- above which a size nothing could verify is probably just wrong.
#
# This measures the *harm* a wrong --map-size does rather than guessing at its
# cause: an out-of-phase lattice puts each source's fog onto another source's
# terrain, and fog against terrain is an enormous colour distance, so
# --consistency-thresh sees it directly. It is the only signal here that catches
# a wrong size without having to find fog first.
#
# **The denominator is comparable tiles, not the board**, and that is
# load-bearing: conflicts need two witnesses, so dividing by the board dilutes
# the number by however little the shots overlap, and two shots of disjoint
# islands land under any usable bar while the few tiles they share disagree
# wholesale.
#
# **It only ever rules a size out.** A low number is evidence of nothing -- a
# wrong size can read 0.000 when barely any tiles are comparable, and a fogless
# board reads the same at every size because there is no fog for a bad lattice
# to smear -- so it gets no message at all. MIN_COMPARABLE stops a handful of
# shared tiles producing a confident-looking percentage. CLAUDE.md has the
# populations 0.18 sits between.
CONFLICT_FRAC_SUSPECT = 0.18
CONFLICT_FRAC_MIN_COMPARABLE = 20


def cross_check(anchors, sift_edges, t_corners, tile_px):
    """How far independently-anchored shots disagree with each other, in tiles.

    The 4-corner fit residual printed by the group path cannot fail: fitting a
    4-DOF similarity to 4 corners of a rhombus is near-exactly determined, so it
    reads ~0.7px whether the anchor is right or 5% off. This can fail. Pairwise
    SIFT is trustworthy at the 0.2px level (verified: a homography buys nothing
    over a similarity, so the camera really is pan+zoom), which makes the
    relative geometry between two shots known independently of any anchor. So
    anchor A and B separately, hop A->B->template via SIFT, and see whether you
    land where anchoring A directly said you would. Any gap is anchor error."""
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


def build_lattice(top, right, left, n):
    """The two tile step vectors, and the board's north corner as the origin.

    Direction comes from the fixed basis and only the *length* from the corners.
    The lattice and the silhouette are at the same angle -- confirmed with the
    project owner, and measured to 0.005 degrees on huge (see the note above
    BOARD_EDGE_SLOPE) -- so taking the direction from three corner pixels here
    would put the tile grid at a slightly different angle from the edge fit for
    no reason, which is the one thing that is certainly wrong."""
    u_col = BOARD_DIR_A * (np.linalg.norm(right - top) / n)
    u_row = BOARD_DIR_B * (np.linalg.norm(left - top) / n)
    return top, u_col, u_row


def tile_of_point(xy, origin, u_col, u_row):
    """The (i, j) of the tile containing a point on the canvas.

    The inverse of tile_poly's own origin + i*u_col + j*u_row, floored. Three
    callers wrote this out longhand -- the ruin snap, the ruin cluster snap and
    the spliced-bar check -- each inverting the basis for itself."""
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

    Tall sprites (cities, mountains) are drawn extending *upward* on screen
    from their own tile, so any occlusion of a tile's fog art always bleeds in
    from the tile to its south -- i.e. from this wedge's far edge, never from
    its near one. A tile that is fog but partly hidden behind its southern
    neighbor therefore still has a clean, unoccluded corner here, even when the
    tile as a whole scores too low on the ordinary whole-tile test."""
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
    """tile_poly + poly_mask_bbox in one call -- the mask and slice bounds for
    one tile's rhombus on the canvas, or None if it falls entirely outside it.

    Four call sites in main built this by hand at inset 0.0 (the composite
    paste, the provenance/explored debug overlays, and the per-tile fog-area
    scan), always with the same origin/u_col/u_row/W/H already in scope."""
    return poly_mask_bbox(tile_poly(origin, u_col, u_row, i, j, inset), W, H)


def _region_ncc(warped_img, tmpl_gray, wmask, poly, min_px=40):
    """Normalized correlation between one warped region and the same-shaped
    patch of template, used for both the whole-tile and top-wedge fog tests.

    The shot is already warped into template space, so one bbox indexes both
    sides. There used to be an `img_offset` shifting the shot's side of that,
    from when the composite canvas was a computed union of the shots rather
    than the template's own frame; no caller ever passed a non-zero one."""
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

    Saturation cannot do this job. It was the original test, and it is wrong for
    every desaturated *terrain*: mountains, snow and ice are all pale gray-white
    and get called fog, so the composite quietly drops them and lets the
    template show through. Correlation separates cleanly -- on a real shot the
    tiles that really are fog score a median 0.88 while explored tiles score
    0.02, with nothing at all between 0.16 and 0.54 -- because a mountain is not
    merely 'less desaturated' than fog, it is a completely different picture.

    Being mean-and-variance-normalized, it also survives the template and the
    screenshot being rendered at different overall illumination.

    A tile that is fog but partly occluded by a city/mountain to its south (see
    tile_top_wedge) dilutes that whole-tile score into an ambiguous middle band
    instead of a clean high one. wedge_poly, when given, is a second and much
    narrower test: if the occlusion-free top wedge alone matches fog strongly,
    the tile counts as fog even though its whole-tile score didn't clear
    fog_ncc. It is a partial remedy, not a complete one: a city's sprite grows
    with its level, so a big enough city reaches into the wedge as well and
    both tests miss. --fog-frac-margin is what covers that case.

    fog_wedge_ncc has to be set high, and the reason is the same determinism
    that makes the whole-tile test work: genuine fog is one fixed render, so a
    fog wedge correlates around 0.95, not marginally. Measured over 4011
    (tile, source) observations across the then-four test sets, tiles that look
    certainly explored (whole-tile score <= 0.20) have wedge scores with a 95th
    percentile of 0.334, and raising the threshold from 0.65 to 0.80 cuts the
    tiles this test flips from 15 to 5. Anything in 0.75-0.85 behaves
    identically. The old 0.65 default sat inside the false-positive band and
    cost real tiles -- it is what turned the forest at test_ss_3 tile (18,18)
    into fog, on wedge scores of 0.662 and 0.668 against whole-tile scores of
    0.12. (One survivor at 0.80 looks like a false positive and is not:
    goon_test2 q.jpg tile (14,13) scores 0.986 on the wedge, and its neighbors
    score 0.93-0.95 whole-tile fog in that same shot, so it is genuinely fog
    behind an occluder -- exactly what this test is for.)"""
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
    shot's. Fitted on pixels known to be fog in this shot (see FOG_LOCK_NCC),
    so it measures nothing but the screenshot's overall illumination, which is
    the only thing that differs between two renders of the same fog art.

    Fitting on every selected pixel looks wasteful -- two free parameters per
    channel against up to 1.6M samples -- and subsampling was tried and is
    *not* worth it: the cost is dominated by the `template[sel]` fancy-index
    extraction, which happens before any thinning could apply, so striding
    afterwards saved 181ms->124ms at 1.6M px and actually cost 20ms->38ms at
    200k. It also perturbs the fitted gain slightly. Don't re-land it."""
    A = template[sel].astype(np.float32)
    B = warped_img[sel].astype(np.float32)
    out = np.zeros((3, 2), np.float32)
    for c in range(3):
        M = np.stack([A[:, c], np.ones(len(A), np.float32)], 1)
        out[c] = np.linalg.lstsq(M, B[:, c], rcond=None)[0]
    return out


def fog_pixel_mask(warped_img, template, gain, tol=26.0):
    """Per-pixel 'this pixel is fog', with no spatial window at all.

    Fog is one deterministic render, so an anchored shot's fog pixel simply
    *equals* the template's pixel at that same position once illumination is
    accounted for. Comparing colors pixelwise is what makes this able to see a
    ~10px fringe -- the obvious alternative, a windowed local correlation,
    cannot: an 11x11 window centered in a 10px fringe still straddles whatever
    is occluding it, and the correlation dies. Measured on the tile north of
    Ichphy that windowed version scored the fogged shots 0.034 against the
    clear ones' 0.023 (no separation at all), where this scores 0.089 vs
    0.000."""
    pred = template.astype(np.float32) * gain[:, 0] + gain[:, 1]
    return np.abs(warped_img.astype(np.float32) - pred).max(2) <= tol


def tile_fog_fraction(fogpix, wmask, poly, W, H, min_px=60):
    """Fraction of one tile's *full* rhombus that is fog in this source.

    Deliberately the full rhombus, not the --tile-inset one the fog test uses:
    a city tall enough to fill a tile's inset center leaves its fog visible
    only around the rim, so the inset view is exactly the view that cannot see
    it. On the tile north of Ichphy the inset scores actually run the wrong
    way (0.144 for the fogged shot vs 0.173 for the clear one)."""
    r = poly_mask_bbox(poly, W, H)
    if r is None:
        return None
    m, (x0, y0, x1, y1) = r
    sel = (m > 0) & (wmask[y0:y1, x0:x1] > 0)
    if sel.sum() < min_px:
        return None
    return float(fogpix[y0:y1, x0:x1][sel].mean())


# ------------------------------------------------------- template px scale ---
# A sprite detector works in *template* space so its size thresholds can be
# constants: after warping, a name plate is the same height whatever the shot's
# zoom. That assumes one more thing than it looks -- that every template renders
# a tile at the same number of pixels -- and the Overlays/ renders do not (78.5
# to 80.5 px per tile against the 89.8 these were measured at, so ~12% out).
#
# So such a threshold is written as the pixel count it took at the scale it was
# measured at, and multiplied by tile_px_scale() where it is used. Only
# PLATE_BAND/PLATE_HALF_W still need that; the bar geometry is in tile widths
# and the ruin kernel sizes itself from |u_col|, which are the same idea taken
# further and the better pattern for anything new.
#
# **A scaled bound compared against an integer must be rounded.** PLATE_* get
# away with a raw float because they only index slices. A bound tested against
# an integer pixel count is carried across the very integer it was chosen to
# include -- at a ~0.1% scale `<= 10` becomes `10.01 <= 10` -- and that cost 5
# sets a city bar each when the bar detector was built on px constants, with
# nothing but tools/baseline.py able to see it. (_bar_at and _bar_mode round for
# a different reason: they scale by the tile step and never touch
# tile_px_scale.)
#
# Deliberately *not* scaled: the small px floors in _region_ncc, sample_tile and
# tile_fog_fraction. Those are "too few pixels to say anything" guards rather
# than measurements, and --min-valid-frac already gates the same thing far more
# strictly, so none of them can bind.
REFERENCE_TILE_PX = 89.8    # the scale the px constants above were measured at.
                            # It is a *reference*, not a file, and must not be
                            # re-based onto Overlays/: doing so means
                            # re-deriving PLATE_* in the same commit, straight
                            # into the rounding trap noted above. The whole
                            # point of tile_px_scale is that this stays fixed
                            # while the loaded template varies.


def tile_px_scale(u_col):
    """How much larger this template's tile is than the scale the px constants
    above were measured at. 0.87-0.90 across the Overlays/ renders."""
    return float(np.linalg.norm(u_col)) / REFERENCE_TILE_PX


# ------------------------------------------------------ Elyrion ruin vision ---
# Elyrion's tribe ability marks each ruin hidden under fog with a cluster of
# several small rainbow flames drawn *on top of* the fog. Only an Elyrion
# player's screenshot shows these, but the cluster carries real map information
# worth merging -- the fogged tile has a ruin on it. It is not centered on its
# tile and can spill over the border, but its pooled centroid lands inside the
# right one (confirmed with the project owner), which is what tile assignment
# below relies on.
#
# The flame is a *known asset* (Assets/Rainbowflame.png, the game's own
# sprite), so this predicts what a marker would look like here and asks how
# well that matches, rather than inferring an appearance from screenshots.
# **Do not score it by HSV statistics instead** -- CLAUDE.md has the case
# against that, and it is not a style preference: it silently cost two sets
# most of their ruins.
#
# Alpha compositing is exact and invertible:
#
#     C = f*a*S + (1 - f*a)*F
#
# S and a are the sprite's color and alpha, known per pixel. F is the fog
# behind it, known per pixel because fog is one deterministic render and the
# shot is anchored to it -- fog_illumination already fits this shot's lighting
# onto the template. So the only unknown at a position is f, this frame's
# fade, and with D = C - F and K = a * (S - F) it is a one-parameter
# projection: f = <D,K>/<K,K>.
#
# **Score the correlation, not the residual -- this is the whole design.** A
# residual is minimized by there being nothing there, since fitting noise costs
# less error than fitting a real departure from fog. corr = <D,K>/(|D||K|) asks
# what fraction of whatever departs from fog here is flame-shaped, and f drops
# out of it entirely -- independent of how the marker was captured.
#
# Three game facts do the rest, and none of them is a threshold: the flame is
# always drawn at the same size relative to the tile, so there is one kernel
# and no scale search; the flames move and fade, which is why position is
# searched and f is fitted rather than assumed; and ruins are never adjacent,
# which cluster_ruin_tiles uses to merge a cluster straddling a tile border.
RUIN_SPRITE = "Rainbowflame.png"
RUIN_SPRITE_DIR = "Assets"

RUIN_FLAME_TILE_FRAC = 0.26   # a flame's width as a fraction of the tile step.
                              # Measured, not assumed -- swept over genuine
                              # flames in three sets and two board sizes; see
                              # CLAUDE.md for the sweep. One scale fits every
                              # flame, confirming the owner's fixed-size fact.

RUIN_MATCH_MIN_CORR = 0.68    # the bar a match must clear. Set mid-gap between
                              # the strongest false peak across the no-Elyrion
                              # control sets and the weakest genuine flame; see
                              # CLAUDE.md for both populations. Do not move it
                              # toward the false-peak side for recall -- there
                              # is nothing to gain and the cliff below is steep.

RUIN_MATCH_MIN_SUPPORT = 0.35  # how much of the flame must land on in-frame
                               # pixels before the match counts as measured at
                               # all. Out-of-frame pixels are *excluded* from
                               # the correlation rather than treated as zero,
                               # because a warped shot is black beyond its own
                               # frame and counting that black as evidence drags
                               # every edge flame down.
                               #
                               # **Currently inert, deliberately kept.** wmask
                               # has already removed the out-of-frame pixels, so
                               # the sliver case this guards against cannot
                               # arise against it (it does against a rawer
                               # mask -- see CLAUDE.md). What it still buys is
                               # the numerical guard: as support goes to zero so
                               # does the correlation's denominator, and 0/0 is
                               # not a score. Keep it low and keep it.

RUIN_MATCH_MIN_FADE = 0.05    # a flame *tints* the fog, so it can only darken
                              # it. A fitted fade at or below zero means
                              # whatever is there is brighter than the fog
                              # rather than a tint on it, which no flame ever
                              # is. Nearly free, and it is a statement about the
                              # sprite rather than a tuned number.

RUIN_NOMINATE_SAT = 100       # only search near saturated pixels; 0 searches
                              # the whole fog region. This is the old HSV
                              # detector's idea kept for the one job it is good
                              # at -- see ruin_search_boxes for why it is safe
                              # here and was not safe as a classifier, and for
                              # the sweep that sets it. Worth ~33% of the phase.

RUIN_PEAK_SEP_FRAC = 0.18     # minimum spacing between two reported flames, as
                              # a fraction of the tile step. A cluster's flames
                              # sit a few px apart, so this is what stops one
                              # flame being reported several times without
                              # merging two real ones.

RUIN_MARK_BGR = (170, 30, 110)  # deep violet outline on a fogged ruin tile.
                                # Chosen for *contrast against fog*, which is
                                # the whole job: fog is near-white, and this
                                # violet's low perceived brightness gives far
                                # more contrast than a bright marker (amber, the
                                # former choice) can. It also has to be
                                # unmistakably not the spawn-zone layer, another
                                # diamond outline -- that layer is a saturated
                                # *red* and this is violet, the largest hue
                                # separation of any candidate tried. Do not
                                # drift it toward magenta/pink; see CLAUDE.md
                                # for both margins measured.

_ruin_sprite_cache = {}


def ruin_sprite_path():
    """Assets/Rainbowflame.png: the working directory, else next to this script.

    Same order as overlay_path and for the same reason -- the Discord bot runs
    polymerge with cwd set to a per-merge temp dir, so the script-relative
    fallback is what makes the asset findable there.
    """
    rel = os.path.join(RUIN_SPRITE_DIR, RUIN_SPRITE)
    return _first_existing(
        rel, os.path.join(os.path.dirname(os.path.abspath(__file__)), rel))


def load_ruin_sprite():
    """The flame as (bgr float, alpha 0..1) cropped to its alpha extent, or None.

    None is not an error to swallow: it means ruin detection cannot run at all,
    and main reports NO-RUIN-SPRITE rather than quietly detecting nothing. There
    is deliberately no fallback detector to degrade into: a second, unmeasured
    detector that runs only once something has already gone wrong degrades
    silently, which is worse than not running at all.
    """
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

    The three maps cover the *crop* around `fog_area`, not the canvas, and
    (y0, x0) is where that crop sits. They are indexed by the center of the
    kernel window. Returning the crop rather than pasting it back into a
    canvas-sized array is not bookkeeping for its own sake -- the caller only
    ever reads these inside the same crop, and three full-canvas float maps per
    shot cost more to allocate and scan than the correlation does.

    The fog prediction is built here, on the crop, for the same reason. Built
    over the whole canvas in float64 it is a 121MB array of which about a third
    gets read, and it measured 0.146s per shot against the correlation's 0.282s
    -- 30% of the phase spent predicting fog nobody looks at.

    The kernel uses one fog color rather than the per-pixel field, or it stops
    being a fixed kernel and none of this stays a convolution. That costs almost
    nothing in accuracy, because these renders' fog is uniform tile to tile
    (per-tile mean spans 228.6-230.2 at 20x20, std 0.39) -- K varies far less
    across the board than D does. D itself still uses the exact per-pixel fog
    prediction.

    Three things keep this affordable, and it is worth saying what the naive
    version costs, because it is not a small factor: nine full-canvas
    filter2D passes with a ~21x33 kernel took **7.6s** on a fog-heavy board,
    against a whole-phase budget of 0.4-0.7s.

    * Sum over channels *before* convolving wherever the kernel does not depend
      on the channel. sum_c conv(v, K_c^2) is conv(v, sum_c K_c^2), and
      sum_c conv(D_c^2, ones) is one box filter of the summed square. Nine
      passes become three correlations plus two cheap ones, with identical
      output.
    * matchTemplate rather than filter2D. It is the same correlation, but it
      picks a DFT when the kernel is this large instead of evaluating ~700
      multiplies per pixel.
    * Crop to the fog region first. A marker only exists on fog, so everything
      outside it plus a kernel of margin is work whose answer is discarded --
      and on a developed board that is most of the canvas.
    """
    spr_bgr, spr_a = kernel
    kh, kw = spr_a.shape
    y0, x0, y1, x1 = box
    if y1 - y0 < kh or x1 - x0 < kw:
        return None
    # Grow the slice by half a kernel of *real* pixels, so a window centered at
    # the box's own edge is still measured on the board rather than on padding.
    # Only where the canvas runs out does zero padding stand in, which is the
    # same treatment a window hanging off the photo has always had. Without
    # this, every box boundary is a fictional edge that truncates the
    # correlation's denominator and inflates the score there -- harmless when
    # there was one box round the whole fog region, and not harmless at all
    # once there is one box per nominated cluster.
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
    # the fog this shot would show here, from the template under its own
    # illumination -- the same prediction fog_pixel_mask trusts to +-26 levels
    fog = np.clip(template[y0:y1, x0:x1].astype(np.float32) * gain[:, 0]
                  + gain[:, 1], 0, 255)
    obs = warped_bgr[y0:y1, x0:x1].astype(np.float32)
    K = (spr_a[:, :, None] * (spr_bgr - fog_mean)).astype(np.float32)
    D = (obs - fog) * v[:, :, None]

    # Pad by half a kernel with zeros before correlating. matchTemplate only
    # evaluates windows that fit wholly inside its input, so without this the
    # outermost half-kernel of the crop is never scored -- and a flame there is
    # not a hypothetical: basin_treaties' ruin at (0,9) sits on the board's
    # north rim and was silently lost. Zero is the correct pad rather than a
    # convenience, because v pads to 0 too, so a window hanging off the edge is
    # measured on its in-frame part and reported with the low support that
    # earns, which is exactly how a window hanging off the *photo* is treated.
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
    # Zero support means the window sits entirely off the photo, and there is no
    # score to be had there. It must be *excluded*, not divided through with a
    # small epsilon: corr is a ratio whose denominator vanishes with the
    # support, so an epsilon turns "nothing here" into an enormous number.
    # Measured, that produced corr values of 1279 and 2.6e7 -- and once boxes
    # multiplied the number of edges, some of that garbage landed above the
    # acceptance bar and put ruins on six boards that cannot contain one.
    good = kk > RUIN_MATCH_MIN_SUPPORT * kk_full * 0.5
    kk_safe = np.where(good, kk, 1.0)
    corr = np.where(good, num / np.sqrt(np.maximum(dsq, 1e-9) * kk_safe), 0.0)
    return (corr[:ch, :cw], np.where(good, num / kk_safe, 0.0)[:ch, :cw],
            np.where(good, kk / kk_full, 0.0)[:ch, :cw], (oy0, ox0))


def _ruin_peaks(corr, allowed, radius):
    """Accepted matches, strongest first, no two within `radius`.

    A dilation marks every pixel that is the best in its neighborhood, which
    on a plateau is the whole top of it; taking the strongest first and
    suppressing the rest is what turns that into one flame per flame.
    """
    k = 2 * radius + 1
    # Suppress *within the allowed set*, not against the raw map. Dilating the
    # raw corr lets a position that failed a gate still shadow an allowed
    # neighbor, and that is not a corner case: a flame at the edge of a photo
    # sits right beside windows that hang further off it, which score higher on
    # a sliver of pixels and are rejected for exactly that reason. Measured on
    # basin_treaties (0,9) -- corr 0.815, support 0.37 -- the ruin was found,
    # gated in, and then silently suppressed by a neighbor that had been gated
    # out. Lowering either threshold could never have fixed it.
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

    With RUIN_NOMINATE_SAT off this is one box: the fog region's bounding box.

    With it on, the search is narrowed to saturated places first. That is the
    old HSV detector's idea reused for the one job it is actually good at.
    Saturation failed as a *classifier* -- mean S over already-thresholded
    pixels measured how crisply a marker was captured, which is what cost two
    sets most of their ruins -- but "is there anything colorful here" is a
    different and much easier question, and fog answers it cleanly: bare fog
    runs a median S of 43 and a 99th percentile of 90.

    The threshold is safe because of a game fact rather than a margin in the
    pixels. Per-*flame* recall really does suffer -- the weakest genuine flame
    in the corpus peaks at S=71, below bare fog's own 99th percentile, so no
    threshold separates flames from fog pixel by pixel. But a ruin's marker is a
    *cluster*, and a cluster is not made only of faint flames: dropping its
    weakest one or two changes nothing once cluster_ruin_tiles has run. Swept
    against the real pipeline, every Elyrion set reports its exact ruin count at
    every threshold from 0 to 120, and only at 140 does the corpus start losing
    ruins (37 -> 31). 100 sits comfortably inside that.

    Boxes come from the *components* of the nomination, not its bounding box,
    and the difference is the whole point: measured over 16 Elyrion shots the
    nomination's bounding box is still 80% of the fog region -- markers are
    scattered, so one box round them all saves nothing -- while the sum of the
    component boxes is 12%. Correlation cost is linear in area (8.7 ms/Mpx,
    flat over a 16x range), so that is close to an 8x saving, against a
    per-box overhead measured at +14% for 4 boxes and +40% for 64.

    fog_mean is computed once, over the whole fog region, and handed to every
    box. Letting each box fit its own would make the kernel differ from box to
    box, so a flame's score would depend on which box happened to contain it.
    """
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
    # The searchable set: a flame's center can sit up to half a kernel from the
    # saturated pixels that nominated it. Accepting *only* inside this is what
    # keeps a box's edge from mattering -- see the note in detect_ruin_vision.
    search = np.zeros(fog_area.shape, bool)
    search[fy0:fy1, fx0:fx1] = cv2.dilate(
        nom, np.ones((kh, kw), np.uint8)).astype(bool)
    n, _lab, st, _ = cv2.connectedComponentsWithStats(nom, 8)
    boxes = []
    for c in range(1, n):
        by = st[c, cv2.CC_STAT_TOP] + fy0
        bx = st[c, cv2.CC_STAT_LEFT] + fx0
        bh, bw = st[c, cv2.CC_STAT_HEIGHT], st[c, cv2.CC_STAT_WIDTH]
        # a flame center can sit up to half a kernel outside the saturated
        # pixels that nominated it, and matchTemplate needs a whole kernel of
        # context beyond that
        boxes.append((max(0, by - kh), max(0, bx - kw),
                      min(H, by + bh + kh), min(W, bx + bw + kw)))
    return boxes, fog_mean, search


def detect_ruin_vision(warped_bgr, wmask, fog_area, template, gain, origin,
                       u_col, u_row, sprite):
    """Elyrion ruin-vision flames in one warped source, as a list of
    (i, j, area, footprint_mask).

    Runs in template space, so the one kernel is the right size by construction
    whatever the shot's zoom, and only inside tiles this source itself witnessed
    as fog -- a marker is drawn on fog, and that restriction is a game fact
    rather than a filter that could be traded away.

    Deliberately no morphological close, no component labeling, no island
    test, no area bounds and no hue or saturation bands. Scoring a map rather
    than labeling blobs is what avoids them: a close would bridge a marker on
    the fog frontier into the explored terrain beside it and swallow it whole,
    and shape already rejects the terrain bleed an island test was for. Measured
    on the match alone, sets that cannot contain a ruin yield zero false peaks.
    """
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

    # Score each box in its own coordinates and collect canvas-space peaks.
    # Suppression has to be global rather than per box, because the boxes carry
    # a kernel of halo each and so overlap: the same flame can be a local
    # maximum in two of them.
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
            # Accept only where the nomination reaches. This is not a second
            # filter, it is what makes a box's edge irrelevant: matchTemplate
            # pads a box with zeros, so a window straddling the boundary has
            # part of its |D| denominator truncated and scores *higher* than
            # the truth. Every position inside `search` has its whole kernel
            # inside the box on real data, by how the boxes were built.
            # Measured, dropping this gate put false ruins on six sets that
            # cannot contain one -- restricting the search made the detector
            # find *more*, which is the tell that a boundary is being scored.
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


# ------------------------------------------------- city population bar ---
# The bar drawn under a city's name label is the one genuinely owner-only
# element of the city UI: only that city's owner's screenshot renders it. (The
# "* N" beside the name is *not* an ownership test -- it is stars-per-turn and
# also shows on a foreign city carrying an embassy. See CLAUDE.md.)
#
# Without this the merge discards it: winner selection ranks by sharpness, so a
# non-owner's shot can win the tiles the bar sits on and the population simply
# vanishes.
#
# Detection runs in *template* space because the city UI scales with board
# zoom: measured across the game's two zoom extremes, 2.857x apart, bar heights
# differ by 2.70x. So after warping, a piece of city UI is a fixed size whatever
# the zoom or the device, which is what lets PLATE_* below be constants at all
# (carried to the loaded template's scale by tile_px_scale).
PLATE_BAND = (8.0, 56.0)              # above the south vertex; template px at
PLATE_HALF_W = 95.0                   # REFERENCE_TILE_PX, scaled at use
PLATE_EDGE_MIN = 12.0                 # gray levels across a row -- a step, not
                                      # an edge detector's idea of an edge


def plate_edge_run(gray, vx, vy, s):
    """Longest horizontal edge run in the band a city's name plate occupies,
    for the city whose south vertex is (vx, vy). In REFERENCE_TILE_PX units, so
    it is comparable across templates and zooms like every other length here.

    Deliberately a *run*, not a fraction or a count: a plate is one long
    unbroken horizontal edge, and asking for positive evidence of that is what
    makes terrain unable to fake it. The same reasoning as
    BOARD_COMPONENT_MIN_EDGE_RUN, which throws screen-aligned chrome off the
    board silhouette."""
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



# **Detection is anchored, not searched.** A bar is always centered on its city
# tile's south vertex, and the merge already knows every south vertex exactly --
# so there is nothing to look for. Go to the vertex, examine the fixed region a
# bar would have to occupy, and ask whether it is one: a hypothesis test with
# three outcomes per tile (no bar, short bar, capped bar), with no component
# labels, segment window, aspect/solidity test or run grouping -- searching
# bottom-up needs all of that and gets the object backwards, since the game
# draws a bar and then subdivides it.
#
# See CLAUDE.md for the measurements behind each constant, the city-bar ground
# truth these were scored against, and the several approaches that failed.
#
# Everything is in **tile widths**, not template px, so none of it passes
# through REFERENCE_TILE_PX -- the geometry is a property of the board. The two
# edges are found as a *pair whose separation is constrained*, not as two
# independently placed rows: placing each row absolutely made the detector only
# as accurate as the anchor (an anchor shift of a couple of px could drop a real
# bar with nothing else in the run changing), and the bar's *height* is the
# tight, reliable quantity -- constraining the pair on it buys real slack for a
# constraint that is stronger, not weaker.
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

# **The bar's color, as a corroborator at a known place -- never as a way to
# find anything.** The distinction is the whole reason this is allowed here at
# all: a color *fill* score fails badly, because "bright and desaturated" is
# fog, snow and pale sand, and it called 225 tiles bar-like across three sets.
# Asking "is the modal color inside this already-known rectangle the color a
# bar is painted" is a different and much easier question -- the same way
# RUIN_NOMINATE_SAT is safe where a saturation classifier is not.
#
# Measured over 55 labeled tiles: a real bar's modal color is 228 off-white, a
# saturated blue, or a saturated red, while every white false positive -- ice,
# snow, UI panels -- reads 252. The game does not paint the bar pure white, and
# those ~24 gray levels are the entire margin.
BAR_MODE_V = (198, 240)         # off-white body
BAR_MODE_S = 60                 # ...must be this desaturated
BAR_BLUE_S = 180                # a filled segment is vividly blue. Water reads
                                # S=186 against a real blue bar's 187, so this
                                # cannot separate them -- the polarity rule in
                                # detect_population_bars is what does.
BAR_RED_S = 150                 # scorched_earth's Icalus reads S=255
# The *prefilter* color box -- the only one still preset, because it runs before
# any geometry exists. That is what lets the color test reject a tile before the
# row search happens; the second, post-geometry call takes its rows from the bar
# it just measured (see detect_population_bars). The short bar's footprint is a
# subset of the capped one, so the small box is inside the bar whichever length
# this one turns out to be.
#
# The rows are centred on the median bar interior (-0.047..0.127), which is what
# keeps the box on the bar under a couple of px of anchor error: it leaves 0.020
# tile widths of margin against the lowest top edge observed and 0.034 against
# the second-lowest bottom edge. Swept over five spans, this is the widest
# minimum margin at both ends.
BAR_BOX_ROWS = (0.000, 0.080)
BAR_BOX_HALVES = (0.40, 0.62)

# **No two cities sit within two tiles of each other** -- the placement rule is
# stronger than "never adjacent" (confirmed with the project owner). Verified
# against the 50 confirmed cities in the ground-truth table: the minimum
# Chebyshev separation is exactly 3 on every labeled set, never 1 or 2.
#
# Used as a contradiction when two detections land too close (see main). Using
# it the other way -- to skip scanning near a confident detection, since the
# rule says nothing can be there -- was tried and is **not** worth it: measured
# 0.44s against 0.46s on a 5-shot merge, which is noise. There is little to save
# because the color test already rejects most tiles before any of the geometry
# runs, so the tiles a skip would remove are the cheap ones.
CITY_MIN_GAP = 2


def _bar_edges(bgr):
    """Signed horizontal-edge strength per pixel.

    A bar is an oblong of near-uniform color, so going down the image its top
    and bottom are two opposite steps. The board is isometric -- every board
    edge, tile border, territory dash and terrain facet runs at dir_a (30.7 deg)
    or dir_b (149 deg) -- so requiring |dy| to dominate |dx| is what makes
    terrain unable to fake it. This is _board_component's chrome test applied to
    a much smaller object.
    """
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                         .astype(np.float32), (3, 3), 0)
    sy = cv2.filter2D(g, cv2.CV_32F, np.float32([[-1], [0], [1]]))
    sx = cv2.filter2D(g, cv2.CV_32F, np.float32([[-1, 0, 1]]))
    sy[np.abs(sy) < 2.0 * np.abs(sx)] = 0.0
    return sy


def _bar_mode(bgr, wmask, origin, u_col, u_row, i, j, half, rows=None):
    """Modal color in a box at tile (i, j)'s south vertex: (bgr, is_bar, is_red).

    `rows` is an absolute (y0, y1) span when the caller has already measured the
    bar; otherwise the preset band is used. Only the first, prefiltering call
    has no geometry to work from -- see detect_population_bars.

    `wmask` is this shot's `valid` mask, which is exactly the right filter --
    it drops UI chrome, out-of-frame pixels, and pixels too dark to judge color
    by. Out-of-frame pixels in particular otherwise win the mode outright on any
    tile near a shot's edge, and they are not even black after JPEG: at
    badland_test3 (9,15) two pixels of 520 are true zero against 43 under a
    channel max of 12.
    """
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
    # This runs once per candidate tile per shot, so it is worth getting right.
    # Pack the quantized triple into one integer and take the mode of a *1-D*
    # array: measured 0.032 ms/call against 0.263 for np.unique(axis=0), which
    # has to sort a structured view, and 1.09 for np.bincount, which allocates a
    # 262k-element array every call. Same answer, 8x faster than the next best.
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

    `dark` inverts which way the two edges step. A bar is not always brighter
    than what is behind it: pure red converts to a gray of about 76 while grass
    sits near 130, so a red bar is a *darker* oblong on brighter ground and both
    its edges run the other way. scorched_earth's Icalus is six red segments
    with no white, and a bright-on-dark test finds no row pair there at all.
    """
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

    # Find the two edges by *row profile*, not per column. The vertex column
    # frequently sits on a divider between segments, where the strong vertical
    # gradient suppresses the horizontal one; and the name plate directly above
    # is also a bright horizontal rectangle, so an unconstrained search returns
    # the plate's edges instead -- which reads as a bottom edge *above* the
    # vertex, something no bar can have. The bands are what prevent that; the
    # height constraint is what lets them be loose enough to survive a couple
    # of px of anchor error (see BAR_HEIGHT).
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
        # Score each side alone and keep the better. A bar is centered, so the
        # two sides measure the same object and one clean side settles it --
        # which is what makes a unit standing on an end harmless. Coverage and
        # the end test must come from the *same* side, or the two halves of the
        # evidence contradict each other and a capped bar with one buried end
        # scores highest as a short one.
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

    Vocabulary, which is the game's: a *bar* is the oblong meter under a city's
    name plate; it is subdivided into *segments*; a segment may carry a *dot*.
    Nothing here looks at segments or dots -- dividers do not survive every
    capture (replay_ss2's Bergo has none at all), so they cannot be required.

    `width_class` is 2 for a short bar and 3 for a capped one, named for the
    segment counts those lengths correspond to. It is *not* a segment count and
    must not be reported as one.

    Rim tiles are skipped outright: cities never sit on row or column 0 or n-1,
    so only (n-2)^2 of n^2 tiles are examined -- about 80% at either board size.
    """
    sy = _bar_edges(warped_bgr)
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    tile_scale = tile_px_scale(u_col)
    bars = []
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            # Color first: it is independent of the geometry, far cheaper --
            # one patch and a mode against a row-pair search over the whole
            # band -- and it sits inside the bar whichever length this one is.
            _mode, ok, red = _bar_mode(warped_bgr, wmask, origin, u_col, u_row,
                                       i, j, BAR_BOX_HALVES[0])
            if not ok:
                continue
            best = None
            for dark in (False, True):
                # The dark polarity exists only because red converts to a low
                # gray. White and blue bars are bright and the ordinary
                # polarity finds them; left unscoped it admits four false
                # positives corpus-wide, every one water or ice, which color
                # cannot reject because water reads S=186 against a blue bar's
                # 187.
                if dark and not red:
                    continue
                m = _bar_at(sy, origin, u_col, u_row, i, j, dark)
                if m and (best is None or m["score"] > best["score"]):
                    best = m
            if best is None or best["score"] < BAR_SCORE_MIN:
                continue
            # Re-read the color from the bar the geometry just measured, wider
            # than the prefilter box so the mode gets several times the pixels.
            # Both halves of that box now come from the detection: the width
            # class picks the columns, and the measured edges pick the rows.
            #
            # The rows are the half that mattered. A preset band's bottom sat a
            # median 0.012 tile widths inside the bar's bottom edge and *below*
            # it on four of the 49 labeled bars, so a couple of px of anchor
            # error walked it onto the white name plate and the mode came back
            # 252 -- the exact signature this file records for a false
            # positive. Inset a fifth of the height at each end to stay off the
            # transition rows.
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


def cluster_ruin_tiles(hits, origin, u_col, u_row):
    """Collapse per-tile ruin detections into one entry per actual ruin.

    A ruin's marker is a cluster of several diamonds, and ruins are never
    adjacent in Polytopia -- so two detections on neighboring tiles cannot be
    two ruins. They are one cluster straddling a tile border, reported twice
    because each diamond's own centroid fell on a different side. The rule is
    the game's, which makes this a correction rather than a heuristic: on
    test_ss_elyruins it turns 16 detections into 11 ruins, and any remaining
    adjacency would mean something is wrong.

    Adjacency is the 8-neighborhood, not the 4: three of the five clusters
    measured there meet only at a tile *corner*, which is exactly what a
    cluster sitting near a lattice vertex produces.

    The surviving tile is the one containing the centroid of the cluster's
    combined pixels, pooled across every source that saw it -- the same
    centroid rule used for one diamond, applied to the whole cluster instead
    of an arbitrary piece of it."""
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


# ------------------------------------------------------ map size detection ---
OVERLAY_DIR = "Overlays"


def _first_existing(*paths):
    """The first of these paths that exists, else the last one.

    Returning the last rather than None keeps the caller's error message
    pointing at a real filename instead of "None"."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[-1]


def overlay_path(n, layer):
    """Where Overlays/<name>-<layer>.png lives: the working directory, else
    next to this script. None when the board size has no such name.

    The cwd is checked first so a set can keep its own renders alongside its
    screenshots, but the script-relative fallback is what makes an omitted
    --map-size work from the Discord bot, which runs polymerge with cwd set to
    a per-merge temp dir."""
    name = BOARD_SIZE_NAMES.get(n)
    if name is None:
        return None
    rel = os.path.join(OVERLAY_DIR, f"{name}-{layer}.png")
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
    return _first_existing(rel, here)


def template_path_for(n):
    """The blank all-fog render for an NxN board, or None at a size with no name.

    Overlays/<name>-blank.png is the only source of fog art there is, and
    deliberately the only one: a fallback here could engage only when Overlays/
    was missing, and would then merge *silently* against different fog art
    rather than saying so. A missing render must fail with the filename, which
    is what load_template's caller does.

    --template still overrides."""
    return overlay_path(n, "blank")


# The optional decorative layers, in the order they are painted onto the
# composite, and the Overlays/ filename stem each comes from.
#
# Order matters and is bottom-up: shading is a wash and must go under the grid
# rather than dull it; spawn zones and push arrows are content and sit on top.
# Ruin markers are drawn after all of these (see main), so a marker is never
# dimmed by the shading it happens to land on.
OVERLAY_LAYERS = (("shade", "shaded"),
                  ("grid", "gridded"),
                  ("spawns", "*spawns"),
                  ("push", "push"))

# Layers painted only where the composite is still fog, never over explored
# terrain (confirmed with the project owner).
#
# The split is between layers that *fill* and layers that *reference*. `shade`
# and `spawns` wash whole tiles with color, so over real terrain they dull the
# map art the merge exists to show, while over fog -- a flat expanse of one
# repeated render -- they are what makes it readable as tiles at all.
#
# `grid` and `push` cover the whole board instead, because they are read
# *against* the map rather than laid over it: a lattice is for counting
# coordinates and a push arrow states a fixed property of each tile, and both
# are wanted most where the units and cities are, which is the explored half.
# Thin strokes on a sprite cost far less legibility than a color wash does.
OVERLAY_FOG_ONLY = frozenset({"shade", "spawns"})

# Per-layer opacity multiplier, applied on top of the file's own alpha.
# `grid` covers explored terrain (see OVERLAY_FOG_ONLY above) and full-strength
# lines compete with the map art underneath, so it's cut to let terrain show
# through. Layers not listed here paint at their file's own alpha.
OVERLAY_ALPHA = {"grid": 0.7}

# Shading alone: it is what makes a flat expanse of fog readable as tiles,
# where the grid and the spawn zones are clutter on a map being read for
# territory. Note polybot overrides this with an empty default and always passes
# --overlays explicitly -- a merge nobody asked a question of should hand back
# the map as the game draws it. This default serves the CLI only.
OVERLAY_DEFAULT = "shade"


def overlay_layer_path(n, stem):
    """Resolve one layer file for an NxN board, or None if it has none.

    `stem` may contain a `*` because the spawn layer encodes its zone grid in
    the filename -- <name>-2spawns.png is a 2x2 grid of 4 zones and
    <name>-3spawns.png a 3x3 of 9. That digit is a property of the board size,
    so it is matched rather than assumed. Not every size has every layer:
    massive has no layers at all and tiny has neither push nor spawns."""
    name = BOARD_SIZE_NAMES.get(n)
    if name is None:
        return None
    for base in (os.curdir, os.path.dirname(os.path.abspath(__file__))):
        hits = sorted(glob.glob(os.path.join(base, OVERLAY_DIR,
                                             f"{name}-{stem}.png")))
        if hits:
            return hits[0]
    return None


def load_overlay(path, shape):
    """One decorative layer as (bgr, alpha) in 0..1 float, or None.

    Rejects a layer whose canvas is not the template's. The layers are rendered
    in the same frame as their blank -- verified: per board size every variant
    shares the blank's canvas and the grid lines land exactly on the fog tile
    boundaries -- so no registration or warping is needed, and a mismatch means
    the wrong file rather than something to resample."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype == np.uint16:
        im = (im / 257.0).astype(np.uint8)
    if im.ndim != 3 or im.shape[2] != 4 or im.shape[:2] != shape[:2]:
        return None
    return im[:, :, :3].astype(np.float32), \
        (im[:, :, 3].astype(np.float32) / 255.0)[:, :, None]


def paint_overlays(out, wanted, n, fog_mask=None):
    """Alpha-blend the requested layers onto the composite, in OVERLAY_LAYERS
    order. Returns the names that had no file for this board size.

    Straight (not premultiplied) alpha: the push and spawn layers carry pixels
    whose color exceeds their alpha, which only makes sense unpremultiplied,
    and the dark layers composite identically either way.

    `fog_mask` marks the tiles nobody explored. Layers in OVERLAY_FOG_ONLY are
    clipped to it, so they never tint another player's terrain."""
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

    16-bit files are normalized on the way in (normal-*.png is uint16) and the
    alpha channel is dropped, which is all the conversion these renders need:
    they are premultiplied against black -- RGB is 0 wherever alpha is, verified
    across every file in Overlays/ -- so the remaining BGR *is* the black-sky
    image the rest of the pipeline already expects.

    The silhouette then comes from the same brightness test the screenshots
    get, deliberately. **Do not derive it from the alpha channel**, obvious as
    that looks on a render that ships one: alpha cuts the antialiased fringe
    --dark-thresh keeps, so the template's board is defined differently from
    every screenshot's and the edge fit stops comparing like with like. It moves
    the tile step by only ~0.045%, which is enough to cost a set most of its fog
    lock and triple its cross-check disagreement. A lattice skew is a real
    reason to want this and belongs to detect_corners, which handles it."""
    if not path:
        return None, None, None      # a board size with no render of its own
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, None, None
    if im.dtype == np.uint16:
        im = (im / 257.0).astype(np.uint8)
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

    Every consumer of a board render wants the same six things off it, and a
    run has two consumers: the size pre-pass (_probe_basis) and the merge
    itself. They ask at different times and used to each pay in full -- a load,
    a corner fit, an outline and an edge fit, which is 294ms at 20x20 (164 load,
    42 corners, 72 outline, 17 edge fit).

    Keyed on (path, dark_thresh, erode_px), so a hit is by construction the same
    computation on the same file and returns the same object. `erode_px` reaches
    only `valid_t`; `edge_t` is always built un-eroded, which is why the pre-pass
    can pass the caller's erode_px rather than 0 and share this entry without
    changing anything it reads.

    Returns None for a render that is not on disk, exactly as load_template
    does, so the caller still owns the refusal."""
    key = (path, dark_thresh, erode_px)
    if key not in _template_cache:
        bgr, valid_t, edge_t = load_template(path, dark_thresh, erode_px)
        if bgr is None:
            _template_cache[key] = None
        else:
            corners = detect_corners(edge_t)
            dirs = (BOARD_DIR_A, BOARD_DIR_B)
            _template_cache[key] = {
                "bgr": bgr, "valid": valid_t, "edge": edge_t,
                "corners": corners, "dirs": dirs,
                "edges": edge_lines(board_boundary(edge_t), *dirs),
            }
    return _template_cache[key]


class ShotCache:
    """Per-shot measurements several parts of a run all want: the board
    outline, and the fog's repeat period.

    Both are computed twice today -- once by `detect_map_size` before the board
    size is known, and again by `anchor_to_template` afterwards -- at ~85-250ms
    a shot (outline 18-68ms, period 66-186ms). They are the same measurement
    only when they are made under the same *parameters*, and that is the whole
    design of this class: **the key carries every input that can change the
    answer**, so a hit returns the same bits and a miss recomputes exactly what
    the old code did.

    - the outline depends on the mask and on the projection basis (which reaches
      `_board_component`'s angle test);
    - the period depends on the mask, on `dir_a` (the shift direction) and on
      `tile_px` (which sets the phase of the coarse sweep grid -- see the
      deferred item on that sensitivity, where a different sweep window moves
      the answer by up to 0.9%).

    The pre-pass has to pick a template before it knows the size, so it takes
    those from whichever render `_probe_basis` finds first. That is why the
    probe order is 20, 18, 16, 14, 11 rather than ascending: on a 20x20 board --
    the commonest size, and the one this program is most often asked for -- the
    probe's basis and tile step *are* the ones the anchor will use, every key
    hits, and the pre-pass becomes free. On any other size nothing hits and the
    behavior is bit-identical to not having this class at all. It buys the
    common case and cannot cost the rest.

    A mask is not hashable and identity is not enough (`sky_rebuild` writes new
    masks into main's dicts in place), so a generation counter per shot stands
    in for it and `invalidate` bumps it. Getting that wrong would be the one way
    this could return a stale answer, so it is the caller's single obligation:
    anything that replaces a shot's masks must invalidate."""

    def __init__(self):
        self._gen, self._boundary, self._period = {}, {}, {}

    def invalidate(self, name):
        """Call after replacing a shot's masks; see anchor_to_template."""
        self._gen[name] = self._gen.get(name, 0) + 1

    def boundary(self, name, mask, dirs):
        key = (name, self._gen.get(name, 0), dirs[0].tobytes(), dirs[1].tobytes())
        if key not in self._boundary:
            self._boundary[key] = board_boundary(mask, dirs)
        return self._boundary[key]

    def period(self, name, img, valid, hsv, dir_a, tile_px):
        """(s_it, period_px, ncc) or None -- fog_period_scale's own answer.

        Takes the image rather than its gray conversion so that a hit skips
        that too -- an argument would be evaluated before the call could return
        the cached value. The conversion is deliberately not *retained*: one
        gray frame per shot is ~5.6MB on a large capture and would buy only a
        ~5ms recompute on the at most two misses a shot can have."""
        key = (name, self._gen.get(name, 0), dir_a.tobytes(), float(tile_px))
        if key not in self._period:
            self._period[key] = fog_period_scale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), valid, hsv, dir_a, tile_px)
        return self._period[key]


def _probe_basis(dark_thresh, erode_px=0):
    """The board's projection directions, tile step and per-direction span,
    from any template.

    None of it depends on which template supplies it, which is what makes
    detecting the map size possible at all. `dir_a`/`dir_b` are the fixed
    isometric projection angles -- the camera never rotates and a diamond is a
    diamond at every N. `tile_px` is used only to center fog_period_scale's
    0.30-3.75x sweep, a 12x-wide window that comfortably contains the truth
    whichever template set it. `span` is the template's own width in each
    direction, needed only to compare two edge pairs like for like: the a and b
    spans differ by 0.17% even on a square board (1858.93 vs 1862.05 at 20x20),
    which is small against EDGE_PAIR_MAX_SPREAD's 3% but not against the
    0.00-1.24% that healthy pairs actually report. Returns None when no
    standard template is on disk."""
    # Largest first, and this ordering is load-bearing rather than tidy: the
    # basis and tile step taken here are what ShotCache keys on, so probing the
    # size a board is most likely to be means the pre-pass's measurements are
    # the ones the anchor wants and are reused instead of repeated. Descending
    # from MAP_SIZE_CHOICES rather than a literal list, so a new size cannot
    # fall out of step with it.
    for n in sorted(MAP_SIZE_CHOICES, reverse=True):
        g = template_geometry(template_path_for(n), dark_thresh, erode_px)
        if g is None:
            continue
        t_top, t_right, _, t_left, _ = g["corners"]
        dir_a, dir_b = g["dirs"]
        _, u_col, _ = build_lattice(t_top, t_right, t_left, n)
        t_off, _ = g["edges"]
        span = [t_off[1] - t_off[0], t_off[3] - t_off[2]]
        return dir_a, dir_b, float(np.linalg.norm(u_col)), span
    return None


def _no_measurement_reason(why):
    """Why no shot could be measured, phrased for the person who has to fix it.

    Counting the tiles across a board is span / fog-period, so it needs *both*
    an opposite edge pair and a readable fog repeat, and the two failures want
    opposite advice -- zoom out, versus state the size because no amount of
    zooming will help. The old text named only the first, which on a replay or a
    finished game is simply untrue: fogless/s1.png and s2.png each have an
    opposite edge pair agreeing to 0.03%.

    The first line stands alone as polybot's channel headline, so it carries the
    cause and stays free of internal vocabulary.

    Note "counted" is the honest verb throughout. Nothing here guesses a size;
    it measures one, and refuses when it cannot."""
    tail = " " + RESTATE_SIZE.format(size_list())
    kinds = set(why.values())
    # Both branches state the cause and the remedy and stop there. Why the fog
    # is needed, and what it is measured against, is this file's business and
    # not the player's -- they cannot act on it, and it is the difference
    # between a message read and a message skimmed.
    if kinds == {"no-fog"}:
        # No second remedy, deliberately: the board has no fog left, so no
        # amount of re-photographing it brings the ruler back, and inviting the
        # player to try sends them round a loop that cannot succeed.
        return "these screenshots have no fog left to measure." + tail + "."
    if kinds == {"no-span"}:
        # Here there is a second remedy and it is the better one, so it is
        # offered alongside rather than instead. "Two opposite sides" rather
        # than the internal a-min/a-max vocabulary: what a shot must contain to
        # have its tiles counted is both ends of one direction.
        return ("no screenshot spans the whole board." + tail
                + ", or with at least one screenshot that shows two opposite "
                  "sides of the board.")
    # Mixed, or a cause with no tailored line of its own. Name them per shot, so
    # the summary can never contradict the detail lines printed above it.
    said = {"no-fog": "no fog left to measure against",
            "no-span": "does not span the board",
            "weak-pair": "its only edge pair is too weak to trust",
            "phantom-pair": "its edge pairs disagree, so one is a phantom",
            "no-outline": "no board outline found"}
    detail = "; ".join(f"{n} ({said.get(w, w)})" for n, w in sorted(why.items()))
    return f"no screenshot could be measured -- {detail}.{tail}."


def detect_map_size(names, imgs, edge_mask, valid, hsv, dark_thresh,
                    min_support, min_scale_support, erode_px=0, cache=None):
    """Measure the board's size off the screenshots themselves.

    This is the same quantity the board-size check already computes against a
    supplied --map-size (see anchor_to_template): a shot spanning the whole
    board in one direction can *count* the tiles across it, by dividing that
    span by its own fog repeat period. Both are measured in the shot's own
    pixels, so no template is involved and there is no circularity in using it
    to pick one -- verified directly, goon_test2 reads 17.93/17.87 with the 18
    template loaded and 17.96/17.99 with the 20, the difference being only
    fog_period_scale's slightly different sweep window.

    It is safe here for a reason that does not hold for the general guard: this
    only has to separate 18 from 20, which are 2 tiles apart, and across the
    corpus's 30 edge-pair shots every measurement lands within 0.20 tiles of
    the truth. That is a 5x margin on the rounding, 10x in practice. (Note the
    18-boards all read slightly low -- 17.80 to 17.93 -- so BOARD_SPAN_WALL_TILES
    is a touch large there; nowhere near enough to matter, but that is the
    direction to look if a size is ever misread.)

    The phantom-edge rejections are replicated from anchor_to_template rather
    than skipped, and they are what makes the failure case honest: a sole edge
    pair must clear min_support on *both* sides, and two pairs must agree.
    pol_archi_test is the set with no measurement at all -- kick.png's only
    pair is the phantom-sided one -- and the right answer there is to ask,
    which is why this raises rather than falling back to a default. A guessed
    size is the most destructive mistake available in this program.

    Costs one edge fit and one fog_period_scale per shot (~0.22s). That used to
    be duplicated work -- anchor_to_template measures both again -- and `cache`
    is what reclaims it, on the sizes where the two agree about the parameters
    they measure under. See ShotCache. Only ever runs when --map-size is
    omitted."""
    cache = cache if cache is not None else ShotCache()
    basis = _probe_basis(dark_thresh, erode_px)
    if basis is None:
        # An install fault, not anything the player did, so the headline says
        # so plainly and the path that identifies it goes on the second line
        # for whoever runs the deployment. Same split as the refusals below:
        # polybot promotes only the first line, and puts the rest in a code
        # block underneath.
        raise SystemExit("this is not installed correctly, so it cannot work "
                         "the board size out.\n(no Overlays/<name>-blank.png "
                         "on disk to take the board's projection from -- see "
                         "the Dockerfile.)")
    dir_a, dir_b, tile_px, t_span = basis
    tags = ["a-min", "a-max", "b-min", "b-max"]
    implied = {}
    # Why each abstaining shot abstained. The summary below is built from these
    # rather than asserting one hardcoded cause, because the causes want
    # opposite advice -- zoom out, versus state the size because no amount of
    # zooming will help -- and polybot shows the channel only the summary (it
    # reads stderr, and these prints go to stdout). A hardcoded cause is
    # therefore the *only* thing a player would see, right or wrong.
    why = {}
    with PHASES("detect map size"):
        for n in names:
            pts = cache.boundary(n, edge_mask[n], (dir_a, dir_b))
            if len(pts) < 100:
                print(f"  {n}: no board outline -- no size measurement")
                why[n] = "no-outline"
                continue
            off, support = edge_lines(pts, dir_a, dir_b)
            have = [c >= min_scale_support for c in support]
            pairs = [(k, lo, hi) for k, (lo, hi) in enumerate([(0, 1), (2, 3)])
                     if have[lo] and have[hi]]
            # A lone pair is believed only if both its sides are properly
            # supported, and two pairs only if they agree: exactly the rules
            # anchor_to_template applies, for exactly the same reason. An edge
            # barely above the phantom floor yields a confident wrong span, and
            # here that would be a confident wrong *board size*.
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
                # Normalized by the template's own span in each direction, the
                # same way anchor_to_template forms edge_scales -- comparing raw
                # spans would fold that 0.17% asymmetry into the test.
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
    # Weigh only the measurements that could describe a board. A number nowhere
    # near a real size is a broken measurement rather than a dissenting
    # opinion, and letting it into the spread check below fails the whole run
    # over one shot's misread fog period -- which is what missized_test did.
    implied, discarded = plausible_sizes(implied)
    if discarded:
        print(implausible_note(discarded))
    if not implied:
        detail = "  ".join(f"{n}={v:.2f}" for n, v in sorted(discarded.items()))
        # Headline first and remedy with it; the per-shot numbers are debugging
        # detail and go below, where polybot renders them in a code block the
        # player can ignore.
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
        # Sizes without the "x", the way a player writes one to the bot: the
        # supported list is a list of things to *type*, not of board shapes.
        raise SystemExit(
            f"this looks like a {size}x{size} board, which is not a size this "
            f"can merge. The sizes it handles are " + size_list()
            + f".\n(measured {med:.2f} tiles across)")
    print(f"detected map size: {size}x{size} (measured {med:.2f} tiles across "
          f"{len(implied)} of {len(names)} shot(s))")
    return size


# -------------------------------------------------------------------- main ---
# A tile counts as "locked" when it matches the template's fog art this well.
# Deliberately far above --fog-ncc: that threshold only has to separate fog
# from terrain, whereas this one asks the stronger question of whether the
# lattice itself is right, and a correct anchor puts genuine fog tiles at
# 0.95+ rather than merely above 0.4.
FOG_LOCK_NCC = 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
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
                         ", or 'none'. These come from the same Overlays/ "
                         "renders as the board itself, so they need no "
                         "registration. Not every board size has every layer "
                         "-- a missing one is skipped and reported, never an "
                         f"error. Default: {OVERLAY_DEFAULT}.")
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

    if args.overlays is None:
        args.overlays = OVERLAY_DEFAULT

    # Validated here rather than where it is used, so a typo fails immediately
    # instead of after the ~20s of work that produced the composite.
    known = {name for name, _ in OVERLAY_LAYERS}
    overlays = {p.strip().lower() for p in args.overlays.split(",") if p.strip()}
    overlays.discard("none")
    unknown = overlays - known
    if unknown:
        raise SystemExit(
            f"unknown overlay(s): {', '.join(sorted(unknown))}. "
            f"Choose from {', '.join(sorted(known))}, or 'none'.")

    with PHASES("load images + masks + badges"):
        ui = json.load(open(args.ui_mask)) if args.ui_mask else {}
        names = [os.path.basename(p) for p in args.images]
        imgs, valid, valid_raw, frame, frame_raw = {}, {}, {}, {}, {}
        edge_mask, hsv, badge_mask_of, badge_halo_of = {}, {}, {}, {}
        badge_found = set()

        def build_masks(name, drop_sky=False):
            """(Re)build the two brightness masks for one shot.

            Called at most twice per shot: once at load, and again from
            sky_rebuild_for when the ordinary masks yielded no usable board
            edge, that time with the sunrise-sky test. Those two were written
            out separately for a long time, which is a poor shape for a
            sequence with a step that is easy to leave out -- see
            subtract_badges."""
            rects = ui.get(name, [])
            im = imgs[name]
            valid_raw[name] = build_valid_mask(im, rects, args.dark_thresh,
                                               args.erode_px, args.top_crop,
                                               args.bottom_crop,
                                               drop_sky=drop_sky)
            # Geometry comes off the *un-eroded* mask. --erode-px exists to keep
            # SIFT features and tile samples away from the mask's fringe, but it
            # eats erode_px of the board edge in each image's own pixels -- i.e.
            # a different amount of board in each, since the shots differ in
            # zoom by up to 1.4x. Anchoring off that would bake a scale error
            # into the fit.
            edge_mask[name] = build_valid_mask(im, rects, args.dark_thresh, 0,
                                               args.top_crop, args.bottom_crop,
                                               drop_sky=drop_sky)
            subtract_badges(name)

        def subtract_badges(name):
            """Take this shot's capture badge back off the masks that must not
            see it, after either build above.

            Separate from build_masks for two reasons: the badge is detected
            *from* valid, so at load time it is not known until after the first
            build; and a rebuild replaces valid_raw and edge_mask from scratch,
            so it has to redo exactly this. The halo half is the one that gets
            forgotten, and forgetting it reintroduces the phantom board edge on
            precisely the shots a rebuild is for -- see badge_halo."""
            badge = badge_mask_of.get(name)
            valid[name] = (valid_raw[name] & ~badge if badge is not None
                           else valid_raw[name])
            frame[name] = (frame_raw[name] & ~badge if badge is not None
                           else frame_raw[name])
            if name in badge_halo_of:
                edge_mask[name] = edge_mask[name] & ~badge_halo_of[name]
        for path, name in zip(args.images, names):
            im = cv2.imread(path)
            if im is None:
                # basename, not the full path: this message is passed straight
                # to a Discord channel, and the path is a server-side temp dir
                raise SystemExit(f"cannot read {name} -- it appears invalid")
            imgs[name] = im
            # frame/frame_raw are the paste-time counterparts of valid/valid_raw:
            # same badge handling, but built from build_frame_mask so darkness
            # never disqualifies a pixel from being pasted (see that docstring).
            # Built once: the sky test does not touch it, because it decides
            # what was *photographed*, not what is bright enough to judge.
            frame_raw[name] = build_frame_mask(im, ui.get(name, []), args.top_crop,
                                               args.bottom_crop)
            build_masks(name)
            hsv[name] = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            if not args.no_badge_filter:
                # Detected against the badge-free masks just built, which is
                # why the subtraction is a separate step rather than part of
                # build_masks.
                badge, found = detect_capture_badges(im, valid[name])
                if found:
                    print(f"{name}: excluding {len(found)} capture-badge "
                          f"blob(s) {found}")
                    badge_halo_of[name] = badge_halo(badge, found)
                    badge_found.add(name)
                # Every shot gets an entry, badge or not -- an all-zero mask
                # when nothing was found. --debug-dir relies on that to write
                # badges_<name>.png for every shot, which is deliberate: an
                # empty overlay is how you see the detector did not misfire.
                # So "did any shot have a badge?" cannot be asked of this dict;
                # badge_found answers it instead.
                badge_mask_of[name] = badge
                subtract_badges(name)

    # A score screen is a menu drawn over a dimmed copy of the map, and it
    # anchors well enough to poison a merge (see board_angle_fraction). Dropped
    # here, before anything else looks at these shots -- in particular before
    # detect_map_size, which one could otherwise contribute a board-size
    # measurement to.
    #
    # The projection angles come straight from BOARD_DIR_A/BOARD_DIR_B, so this
    # needs no template and no probe: the camera is fixed orthographic
    # isometric, and those are a constant of it rather than of any one render.
    all_names = list(names)
    with PHASES("menu-screenshot prefilter"):
        for name in list(names):
            frac = board_angle_fraction(imgs[name], (BOARD_DIR_A, BOARD_DIR_B),
                                        args.top_crop, args.bottom_crop)
            if frac < MENU_BOARD_ANGLE_FRAC:
                # Console only; the channel gets the DROPPED summary below.
                print(f"{name}: not a view of the board -- only {frac:.0%} "
                      f"of its detail runs at a board angle (a score screen "
                      f"or other menu drawn over the map?)")
                names.remove(name)
    if not names:
        # Cause and remedy on the one line, as every refusal here does: polybot
        # promotes only the first line to the channel. No mention of edge angles
        # -- a player cannot act on that, and infers the shape of it from the
        # remedy anyway.
        raise SystemExit("these look like score screens or menus rather than "
                         "the map itself. Please retry with in-game "
                         "screenshots of the map.")

    # Counted here rather than by the caller, and that is the whole point of the
    # option: the menu prefilter is the only thing that knows which inputs are
    # map screenshots, it needs the pixels to know it, and polybot has them only
    # as undownloaded attachments at the point where it would otherwise check.
    # Charging a score screen against the budget refuses merges that are well
    # inside it -- a 3v3 where everyone posts a map and a score screen is 12
    # images and 6 shots.
    if args.max_shots is not None and len(names) > args.max_shots:
        raise SystemExit(f"too many screenshots: {len(names)} of these show the "
                         f"map, and the limit is {args.max_shots}. Please retry "
                         f"with fewer.")

    # Only when the caller omitted it: an explicit --map-size is always obeyed,
    # so this can never override a size someone actually meant.
    size_was_detected = args.map_size is None
    # One cache for the whole run, so whatever the size pre-pass measures below
    # is available to the anchor rather than measured again. See ShotCache.
    shot_cache = ShotCache()
    if size_was_detected:
        args.map_size = detect_map_size(names, imgs, edge_mask, valid, hsv,
                                        args.dark_thresh, args.min_edge_support,
                                        args.min_scale_support,
                                        erode_px=args.erode_px,
                                        cache=shot_cache)

    template_path = args.template or template_path_for(args.map_size)
    tgeom = template_geometry(template_path, args.dark_thresh, args.erode_px)
    template = tgeom["bgr"] if tgeom else None
    if template is None:
        # Install fault, like the missing-Overlays refusal in detect_map_size:
        # nothing the player did, so the path goes below the headline.
        raise SystemExit(f"this is not installed correctly, so it cannot merge "
                         f"this board size.\n({template_path} is missing or "
                         f"unreadable.)")
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    print(f"template: {template_path}")

    t_top, t_right, t_bottom, t_left, t_residual = tgeom["corners"]
    print(f"template corners: top={tuple(t_top.round(0))} right={tuple(t_right.round(0))} "
          f"bottom={tuple(t_bottom.round(0))} left={tuple(t_left.round(0))} "
          f"(residual {t_residual:.1f}px, should be tiny)")
    dir_a, dir_b = tgeom["dirs"]
    origin, u_col, u_row = build_lattice(t_top, t_right, t_left, args.map_size)
    t_corners = (t_top, t_right, t_bottom, t_left)
    t_edge_off, _t_edge_sup = tgeom["edges"]
    tile_px = float(np.linalg.norm(u_col))
    print(f"tile step: {tile_px:.2f}px")

    _sift_mask = {}

    def sift_mask_for(n):
        """Pixels SIFT may take features from: the board, and nothing else.

        Chrome that survives the crop is a hazard here in a way it is not for
        the edge fit, because two screenshots of the same *replay* carry
        pixel-identical UI -- the turn timeline, the transport buttons -- and
        identical pixels match perfectly. On tests/replay_ss2 that gives 330
        inliers on a flat identity transform, beating the genuine board match's
        113 and reporting that two different views of the board are the same
        image. Ordinary gameplay shots hide this because their HUD differs
        between captures (score, turn, whose go it is).

        Built lazily and cached: it costs one morphology pass per shot, and only
        on the paths that use SIFT at all."""
        if n not in _sift_mask:
            _sift_mask[n] = valid[n] & board_region(edge_mask[n], (dir_a, dir_b))
        return _sift_mask[n]

    _terrain = {}

    def terrain_mask_for(n):
        """Pixels saturated enough that they cannot be the fog cube.

        Used only to count how many SIFT inliers rest on terrain rather than fog
        (see SIFT_TERRAIN_MIN_INLIERS). This is emphatically *not* fog
        classification -- it never decides what a tile is, only whether a
        correspondence is worth counting -- which is the same distinction that
        makes RUIN_NOMINATE_SAT acceptable while a color-based fog test is
        not."""
        if n not in _terrain:
            _terrain[n] = hsv[n][:, :, 1] >= PIXEL_FOG_SAT
        return _terrain[n]

    def sky_rebuild_for(n):
        """Rebuild one shot's masks with the sunrise-sky test.

        Handed to anchor_to_template and called *lazily* -- only when that
        shot's ordinary masks yield no usable board edge. A black-sky
        screenshot therefore never builds these at all, which is why the
        fallback costs the common case nothing.

        It writes back into main's dicts as well as returning, because
        everything downstream -- the warp, tile sampling, SIFT -- has to see
        the same masks the anchor was fitted on."""
        def rebuild():
            build_masks(n, drop_sky=True)
            return edge_mask[n], valid[n]
        return rebuild

    def anchor_all():
        """Anchor every shot, in two passes: independently first, then a SIFT
        zoom hint for whatever could not manage it alone.

        Shared by --cross-check and the merge so the two cannot diverge --
        they did once, and the symptom was cross-check reporting a set
        unanchorable that the merge handled fine (pol_archi_test)."""
        M_of, src_of, implied_of, scale_of, failed = {}, {}, {}, {}, []
        prior_of = {}

        def _record(n, M, implied):
            """File one shot's anchor into M_of/scale_of/implied_of. Run once
            per shot in each of the two passes below."""
            M_of[n] = M
            scale_of[n] = float(np.hypot(M[0, 0], M[1, 0]))
            if implied is not None:
                implied_of[n] = implied

        for n in names:
            print(f"anchoring {n}:")
            try:
                M, src_of[n], implied, prior_of[n] = anchor_to_template(
                    imgs[n], edge_mask[n], valid[n], hsv[n], tmpl_gray, t_edge_off,
                    dir_a, dir_b, origin, u_col, u_row, args.map_size,
                    args.min_edge_support, n, refine=not args.no_refine,
                    min_scale_support=args.min_scale_support,
                    sky_rebuild=sky_rebuild_for(n), cache=shot_cache)
                _record(n, M, implied)
            except SystemExit as e:
                print(f"  no self-anchor: {e}")
                failed.append(n)
        if failed and M_of:
            # Timed around the SIFT work only. The re-anchor below must stay
            # *outside* this context: anchor_to_template opens phases of its
            # own, and nesting them double-counts, which drives the report's
            # "(unattributed)" line negative and makes the whole timing block
            # untrustworthy.
            with PHASES("anchor: SIFT zoom fallback"):
                feats = {n: sift_features(imgs[n], sift_mask_for(n), args.nfeatures,
                                          args.contrast)
                         for n in list(M_of) + failed}
                hints = {}
                for n in failed:
                    # Ranked by terrain inliers, and gated on them: a match
                    # made of fog is the failure this guards against, so the
                    # fog-borne inliers should neither elect a lender nor count
                    # toward the bar. See SIFT_TERRAIN_MIN_INLIERS.
                    best = (0, 0, None, None)
                    for m in M_of:
                        M_nm, inl, terr = pair_transform(
                            *feats[n], *feats[m], args.ratio, args.reproj,
                            terrain=(terrain_mask_for(n), terrain_mask_for(m)))
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
                # m's anchor composed with the hop onto m is a complete
                # image-n -> template transform. Its scale is what zoom_hint
                # has always been; its *translation* is what anchor_to_template
                # falls back on for a direction with no board edge in frame.
                # Pass both, so the borrowed offset is measured at the same zoom
                # it is used at -- see pan_hint in anchor_to_template.
                borrowed = (to_h(M_of[m]) @ to_h(M_nm))[:2]
                try:
                    M, src_of[n], implied, prior_of[n] = anchor_to_template(
                        imgs[n], edge_mask[n], valid[n], hsv[n], tmpl_gray,
                        t_edge_off, dir_a, dir_b, origin, u_col, u_row,
                        args.map_size, args.min_edge_support, n,
                        refine=not args.no_refine,
                        min_scale_support=args.min_scale_support,
                        # M_nm maps n's pixels to m's, so 1 n-px = k m-px,
                        # and m's own anchor converts those to template px
                        zoom_hint=scale_of[m] * k,
                        sky_rebuild=sky_rebuild_for(n),
                        pan_hint=borrowed, cache=shot_cache)
                    _record(n, M, implied)
                    failed.remove(n)
                except SystemExit as e:
                    print(f"  dropped -- {e}")
        return M_of, src_of, implied_of, prior_of

    if args.cross_check:
        anchors = anchor_all()[0]
        feats = {n: sift_features(imgs[n], sift_mask_for(n), args.nfeatures, args.contrast)
                 for n in names}
        sift_edges = {(a, b): pair_transform(*feats[a], *feats[b], args.ratio,
                                             args.reproj)
                      for a, b in itertools.combinations(names, 2)}
        print(f"\nindependent anchors vs SIFT relative geometry "
              f"({len(anchors)}/{len(names)} shots anchorable):")
        rows = cross_check(anchors, sift_edges, t_corners, tile_px)
        if not rows:
            raise SystemExit("no pair has both an anchor and a SIFT transform")
        for a, b, n_inl, err, tiles in rows:
            flag = "" if tiles < 0.10 else "   <-- ANCHORS DISAGREE"
            print(f"  {a[:30]:30s} vs {b[:30]:30s} inliers={n_inl:5d} "
                  f"corner gap={err:7.1f}px = {tiles:.3f} tiles{flag}")
        worst = max(r[4] for r in rows)
        print(f"\nworst disagreement: {worst:.3f} tiles")
        return

    if args.single and len(names) != 1:
        raise SystemExit(f"--single takes exactly one image, got {len(names)}")

    # Every image is anchored to the template independently -- zoom from its own
    # fog artwork, pan from its own board edges (anchor_to_template) -- and never
    # against each other. Registering the shots to one another first and
    # anchoring the group is the tempting alternative and is worse: one bad
    # pairwise match throws off every shot chained through it, and a group fit
    # has no precision check that can fail (see cross_check's docstring).
    # Per-image anchoring gives each shot its own independent failure mode,
    # which --cross-check verifies shot-by-shot against SIFT's geometry.
    (to_template_of, zoom_source_of, implied_n_of, prior_of) = anchor_all()
    to_template_of = {n: to_h(M) for n, M in to_template_of.items()}
    # Same representation as the anchors themselves -- these are swapped in for
    # one another below, and sift_hops inverts whatever is in to_template_of.
    prior_of = {n: to_h(M) for n, M in prior_of.items() if M is not None}
    if not to_template_of:
        raise SystemExit(
            "no valid images found -- none of them could be placed on the "
            "board. They need to be in-game screenshots showing part of the "
            "map, with two adjoining sides of the board in frame -- two "
            "opposite sides are not enough. Zooming out usually does it.")
    # One summary line naming what got left out, on stdout and in a fixed
    # format, because a *partial* merge is the failure a player is least
    # likely to notice: the composite still looks fine, it is just missing
    # someone's territory. A real user reported exactly this ("failed to
    # attach the oum ss") and only caught it by eye. polybot parses this line
    # to say so in the channel -- keep the prefix stable if you edit it.
    # Over all_names, not names: a shot the menu prefilter removed above is
    # just as absent from the composite as one that failed to anchor, and the
    # player is owed the same line about it.
    dropped_names = [n for n in all_names if n not in to_template_of]
    if dropped_names:
        print(f"DROPPED {len(dropped_names)}/{len(all_names)}: "
              + ", ".join(dropped_names))
    names = [n for n in names if n in to_template_of]

    # Any shot spanning the whole board counts the tiles across it directly
    # (span / fog repeat period), which is an estimate of the board size owing
    # nothing to --map-size. This is the guard that --min-fog-lock alone cannot
    # be: fog_period_scale sizes a shot by matching period against period, so a
    # shot anchored that way locks happily onto the *wrong* board size too, and
    # the fog-lock check waves it through (CLAUDE.md has the measured case).
    # Checked against the median so one odd shot cannot fail an otherwise good
    # run -- but with only two measurements the median *is* their mean, so one
    # broken number drags it half way (CLAUDE.md has that failure too).
    # Measurements that could not describe a board at all are therefore
    # discarded first rather than averaged in; see MAP_SIZE_PLAUSIBLE_TOL.
    implied_n_of, implied_n_bad = plausible_sizes(implied_n_of)
    if implied_n_bad and not size_was_detected:
        # detect_map_size already said this about the same shots when it ran,
        # and the message is three lines.
        print(implausible_note(implied_n_bad))
    if implied_n_of:
        med = float(np.median(list(implied_n_of.values())))
        if abs(med - args.map_size) > 0.4:
            # One sentence, no method and no per-shot table: this goes straight
            # to a Discord channel, where the reader is a player who wants the
            # answer and the fix. The evidence is already on stdout -- every
            # shot printed its own "board size implied by span/fog-period"
            # line during anchoring -- so nothing is lost by leaving it there.
            #
            # "or without a size" is the better of the two remedies and is why
            # it is offered: omitting it makes this program measure the board
            # rather than take anyone's word for it, which is exactly what went
            # wrong here.
            got = int(round(med))
            raise SystemExit(
                f"This looks like a {got}x{got} board, not "
                f"{args.map_size}x{args.map_size}. Please retry with size "
                f"{got} or without a size.")

    # The composite is built in the template's own frame, so the canvas *is*
    # the template. These were two pairs of names for one size back when the
    # canvas was a computed union of wherever the shots landed.
    W, Hc = template.shape[1], template.shape[0]
    warped, wmask, wmask_raw, pmask, pmask_raw, scale = {}, {}, {}, {}, {}, {}
    N = args.map_size
    samples = {}

    def warp_shot(n):
        """Put one shot on the canvas, in all four mask flavors. Factored out
        so a shot whose anchor is revised later (the SIFT pan borrow below) can
        be redone on its own rather than re-running the whole phase."""
        Mn = to_template_of[n]
        warped[n] = cv2.warpAffine(imgs[n], Mn[:2], (W, Hc), flags=cv2.INTER_LANCZOS4)
        wmask[n] = cv2.warpAffine(valid[n], Mn[:2], (W, Hc), flags=cv2.INTER_NEAREST)
        wmask_raw[n] = cv2.warpAffine(valid_raw[n], Mn[:2], (W, Hc), flags=cv2.INTER_NEAREST)
        pmask[n] = cv2.warpAffine(frame[n], Mn[:2], (W, Hc), flags=cv2.INTER_NEAREST)
        pmask_raw[n] = cv2.warpAffine(frame_raw[n], Mn[:2], (W, Hc), flags=cv2.INTER_NEAREST)
        scale[n] = float(np.hypot(Mn[0, 0], Mn[1, 0]))

    def sample_shot(n):
        """Classify every tile for one shot, replacing whatever it said before."""
        for i in range(N):
            for j in range(N):
                s = sample_tile(warped[n], wmask[n], tmpl_gray,
                                tile_poly(origin, u_col, u_row, i, j,
                                          args.tile_inset),
                                args.fog_ncc, args.min_valid_frac,
                                wedge_poly=tile_top_wedge(origin, u_col, u_row,
                                                          i, j),
                                fog_wedge_ncc=args.fog_wedge_ncc)
                per = samples.setdefault((i, j), {})
                per.pop(n, None)
                if s is not None:
                    per[n] = s

    with PHASES("warp to canvas"):
        for n in names:
            warp_shot(n)
        print(f"canvas: {W} x {Hc} (template)")

    with PHASES("tile sampling"):
        for i in range(N):
            for j in range(N):
                samples[(i, j)] = {}
        for n in names:
            sample_shot(n)

    # Fog art is a fixed render, so a correctly anchored shot lands a large
    # number of its fog tiles almost exactly on the template's -- NCC 0.95+,
    # with the explored ones down near 0.0 and very little in between. When the
    # tile lattice is wrong the whole distribution collapses into a blob around
    # 0.0 instead, nothing locks, and every tile gets called explored: the
    # composite then looks plausible but is really each source's fog pasted
    # over every other source's terrain. Measured across the then-four test
    # sets, the best-locking shot in a run scores 105-211 locked tiles when
    # --map-size is right and 2-4 when it is wrong, so this separates them with
    # room to spare. It is a whole-run check because a single shot can honestly
    # have almost no fog in frame (a zoomed-in view of explored territory:
    # test_ss_2/cym1.jpg locks only 15 tiles and is anchored correctly).
    def locked(n):
        return sum(1 for s in samples.values()
                   if s.get(n, {}).get("fog_ncc", 0.0) >= FOG_LOCK_NCC)

    fog_lock = {n: locked(n) for n in names}
    # A refinement that locked nothing had nothing to refine against.
    # joint_register scores candidates by fog alignment, so with no fog in frame
    # it is walking to the argmax of noise rather than refining -- the same
    # reasoning that stops a borrowed pan being refined (see borrow_pan), except
    # a fogless board cannot be spotted up front the way that can, so this check
    # runs after the fact. tests/fogless is the case: see CLAUDE.md for the
    # measurements.
    #
    # **The prior wins even on a tie at zero, and that tie is the whole point.**
    # Fog lock cannot separate two anchors when neither has any fog; what
    # decides is that one was fitted to noise and the other was not. A shot
    # locking even one tile keeps its refinement untouched, so this cannot move
    # an ordinary merge.
    #
    # Deliberately before the report and the guard below, so the printed numbers
    # are the ones the merge actually uses and a recovered prior can satisfy
    # --min-fog-lock. Not gated on --min-fog-lock > 0, since a fogless board has
    # to pass 0 to reach here at all.
    for n in [n for n in names if not fog_lock[n] and prior_of.get(n) is not None]:
        to_template_of[n] = prior_of[n]
        warp_shot(n)
        sample_shot(n)
        fog_lock[n] = locked(n)
        print(f"  {n}: refining it locked no fog, so its edge anchor stands "
              f"unrefined (that prior locks {fog_lock[n]})")
    print("\nfog lock (tiles matching the template's fog art at NCC >= "
          f"{FOG_LOCK_NCC}): "
          + "  ".join(f"{n}={fog_lock[n]}" for n in names))
    size_unverified = False
    if args.min_fog_lock > 0 and max(fog_lock.values()) < args.min_fog_lock:
        # Two things produce a run where nothing locks and they want opposite
        # answers: --map-size is wrong, or the board has no fog left at all (a
        # replay, or a finished game). Nothing available here separates them.
        #
        # So split on *who claimed the size*, which is answerable. **Detected**
        # means the claim is this program's own and has to be self-consistent --
        # detect_map_size reads span over fog period, so it found fog, and a run
        # that then locks none of it is wrong about something. Refuse.
        # **Stated** is the person's assertion, and a wrong one is theirs to
        # make: warn loudly and merge. Refusing every replay to guard against a
        # mistyped size served nobody.
        #
        # A wrong *stated* size is still guarded twice: the board-size check
        # above is independent and stronger wherever it can measure at all, and
        # the conflict fraction below is the backstop where it abstains --
        # which, unlike anything here, measures the *harm* rather than guessing
        # at the cause. Two tests that look like the missing discriminator and
        # are not (fog_period_scale returning None, and a fog-ish colour
        # fraction) are written up in CLAUDE.md; do not reach for either.
        if size_was_detected:
            # The reasoning that makes this a refusal rather than a warning --
            # the size came from the fog, so fog is present and should have
            # locked -- is in the comment above and in the numbers below. The
            # player gets the outcome and what to do about it.
            raise SystemExit(
                "the board size measured from these screenshots does not fit "
                "them. " + RESTATE_SIZE.format(size_list()) + "."
                + f"\n(no shot locked onto the fog artwork: best "
                f"{max(fog_lock.values())} tiles, need {args.min_fog_lock})")
        size_unverified = True
        print(f"\nWARNING: no shot locked onto the fog artwork (best "
              f"{max(fog_lock.values())} tiles). Merging as "
              f"{N}x{N} because that is what was asked for, but nothing here "
              f"can confirm it -- fog is what this check compares against, and "
              f"there is none to compare. If the board really has no fog left "
              f"(a replay, or a finished game) that is expected; otherwise "
              f"check the size.")

    # Per-image counterpart of the run-level guard, applicable only to shots
    # whose zoom came from the fog-period fallback. That fallback measures zoom
    # on a large expanse of the shot's own fog, so if none of its tiles then
    # lock onto the template's fog art, the anchor is wrong -- and a
    # misanchored shot calls its own fog "explored" and pastes it over every
    # other source's terrain. Dropping it keeps the rest of the merge alive.
    #
    # That premise is not airtight: the whole board is periodic at the tile
    # step, not just the fog, so a shot with almost no fog in frame can measure
    # a correct period off crop fields and tile borders and still lock nothing
    # (star_change/oum.png -- see CLAUDE.md). Such a shot is perfectly
    # mergeable, so before dropping it, ask for a second opinion that owes
    # nothing to fog: SIFT against a shot that anchored on its own. This can
    # only ever *save* a shot, never drop one that would have survived -- a
    # spurious SIFT match (fog matching fog) fails the inlier floor or the
    # agreement bar and leaves the drop exactly as it was.
    feat_cache = {}

    def sift_hops(n, witnesses):
        """Where each anchored shot's SIFT geometry says n belongs.

        One (inliers, m, anchor implied for n, gap from n's own anchor in
        tiles) per witness that matches well enough, best-matching first. The
        gap is exactly --cross-check's measurement: hop template -> n -> m ->
        template and see how far you land from where you started.

        Features are cached because both callers below can want the same shot,
        and neither runs on an ordinary merge."""
        with PHASES("SIFT anchor hop"):
            for m in witnesses + [n]:
                if m not in feat_cache:
                    feat_cache[m] = sift_features(imgs[m], sift_mask_for(m),
                                                  args.nfeatures, args.contrast)
            out = []
            for m in witnesses:
                M_nm, inl, terr = pair_transform(
                    *feat_cache[n], *feat_cache[m], args.ratio, args.reproj,
                    terrain=(terrain_mask_for(n), terrain_mask_for(m)))
                if M_nm is None or terr < SIFT_TERRAIN_MIN_INLIERS:
                    continue
                A = to_template_of[m] @ to_h(M_nm)      # n's pixels -> template
                via = A @ np.linalg.inv(to_template_of[n])
                got = cv2.transform(np.float32(t_corners).reshape(-1, 1, 2),
                                    via[:2]).reshape(-1, 2)
                gap = float(np.max(np.linalg.norm(
                    got - np.float32(t_corners), axis=1))) / tile_px
                out.append((inl, m, A, gap))
            return sorted(out, key=lambda r: -r[0])

    # A shot with *no* fog locked has had no say in its own refinement:
    # joint_register scores candidates by fog alignment, so with nothing to
    # align it keeps the edge-derived prior, bias and all. For such a shot
    # another shot's SIFT geometry is better evidence than its own edges, so
    # borrow the whole transform rather than only the zoom (which is what
    # anchor_all's fallback lends a shot that cannot anchor at all).
    #
    # Three things keep this from becoming the old
    # register-the-group-then-anchor design it superficially resembles: only a
    # shot with zero fog evidence is eligible, a lender must have real fog
    # evidence of its own, and the borrowed anchor has to *prove itself* on the
    # borrower's own fog.
    #
    # **That last one also decides which lender, and it has to, because inlier
    # count does not.** The same player's near-identical second view out-matches
    # every other shot whether or not it is itself anchored well -- measured at
    # 4494 inliers against a correct lender's 829, on a shot that was a full
    # tile out. So try every eligible lender and let the borrower's own fog
    # pick. A shot reaching here already has its own unrefined anchor back (the
    # zero-lock block above), so the borrow competes against its best
    # own-evidence anchor rather than against a refinement fitted to noise.
    #
    # **No corpus set reaches this**, so tools/baseline.py cannot verify a
    # change here; exercise it deliberately. CLAUDE.md has the worked case.
    if args.min_fog_lock > 0:
        lenders = [m for m in names if fog_lock[m] >= args.min_fog_lock]
        for n in [n for n in names if fog_lock[n] == 0 and n not in lenders]:
            keep_M, keep_lock, best = to_template_of[n], fog_lock[n], None
            for inl, m, A, gap in sift_hops(n, lenders):
                to_template_of[n] = A
                warp_shot(n)
                sample_shot(n)
                lock = sum(1 for s in samples.values()
                           if s.get(n, {}).get("fog_ncc", 0.0) >= FOG_LOCK_NCC)
                print(f"    {n} anchored from {m} ({inl} SIFT inliers, moves "
                      f"{gap:.3f} tiles) locks {lock} fog tiles")
                if best is None or lock > best[0]:
                    best = (lock, m, A, inl, gap)
            if best is None:
                continue
            lock, m, A, inl, gap = best
            to_template_of[n] = A if lock > keep_lock else keep_M
            warp_shot(n)
            sample_shot(n)
            if lock > keep_lock:
                fog_lock[n] = lock
                print(f"  re-anchored {n} from {m}'s SIFT geometry ({inl} "
                      f"inliers, moved {gap:.3f} tiles): it locked no fog of "
                      f"its own, so its refinement had nothing to correct the "
                      f"edge fit against. Now locks {lock}")

    def corroborate_anchor(n, witnesses):
        """Is n's anchor confirmed by an already-anchored shot's SIFT geometry?

        Both bars have to be cleared, and they guard different failures: the
        terrain-inlier floor (SIFT_TERRAIN_MIN_INLIERS, applied inside
        sift_hops) rejects fog matching the wrong repeat of itself -- the
        confident, high-scoring, badly wrong match -- and the gap bar rejects a
        genuine match that simply disagrees. Returns a phrase describing the
        evidence, or None.

        *Some* anchored shot has to agree, not the best-matching one: sift_hops
        ranks by inlier count, and inlier count is not what decides here (the
        same reasoning as the anchor borrow above -- a near-identical view of
        the same player's own board out-matches every other shot whether or not
        it is anchored well). So every hop that cleared the inlier floor gets
        to corroborate, and the first that also agrees is enough."""
        for inl, m, _A, gap in sift_hops(n, witnesses):
            if gap <= MISANCHOR_CORROBORATE_MAX_TILES:
                return (f"sits {gap:.3f} tiles from where {m}'s SIFT geometry "
                        f"puts it, on {inl} inliers")
        return None

    if args.min_fog_lock > 0:
        suspect = [n for n in names
                   if zoom_source_of[n] == "fog-period" and fog_lock[n] == 0]
        witnesses = [n for n in names if n not in suspect]
        for n in suspect:
            keep = corroborate_anchor(n, witnesses)
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
        if not names:
            # Same audience as the refusals above: "misanchored" names an
            # internal state, and the player needs the consequence and the
            # remedy instead. The per-shot lines just printed carry the cause
            # for the console.
            raise SystemExit(
                "none of these screenshots could be placed on the board. "
                "Check they are all of the same board, and that the size is "
                "right.")

    # Per-pixel fog evidence, used to rank sources against each other on the
    # *same* tile. A tall city can fill a tile's inset center in every shot, so
    # the fog test sees only towers and calls the tile explored even in a shot
    # where it is really fog -- which then wins on sharpness and pastes its own
    # fog fringe. Both shots are looking at the same towers, so comparing their
    # fog fractions cancels the occluder and leaves only the disagreement that
    # matters. Confirmed on the tile north of Ichphy (test_ss_3 tile (9,8)):
    # 0.089/0.084 for the two yad shots, where it is genuinely fog, against
    # 0.000/0.000 for the two cym shots, where it is genuinely grass.
    def tile_predicate_mask(n, keep, inset=0.0):
        """Boolean canvas of every tile whose sample for n satisfies `keep`.

        Two sites in main build this by hand -- the fog-lock mask that picks a
        shot's illumination-fitting tiles below, and the fog-area mask that
        picks a shot's own-witnessed-fog tiles for ruin detection -- differing
        only in `keep` and the inset."""
        canvas = np.zeros((Hc, W), bool)
        for (i, j), per in samples.items():
            s = per.get(n)
            if s is None or not keep(s):
                continue
            r = tile_mask_bbox(origin, u_col, u_row, i, j, W, Hc, inset)
            if r is None:
                continue
            m, (x0, y0, x1, y1) = r
            canvas[y0:y1, x0:x1] |= (m > 0)
        return canvas

    with PHASES("fog pixel masks"):
        fogpix, gain_of = {}, {}
        for n in names:
            locked_mask = tile_predicate_mask(
                n, lambda s: s.get("fog_ncc", 0.0) >= FOG_LOCK_NCC,
                args.tile_inset)
            sel = locked_mask & (wmask[n] > 0)
            # Without enough known-fog pixels there is nothing to fit the shot's
            # illumination on. A shot with no fog in frame also cannot be the one
            # smuggling a fog fringe in, so an all-false mask is the right answer:
            # it simply never disqualifies that source.
            if int(sel.sum()) >= 5000:
                gain_of[n] = fog_illumination(warped[n], template, sel)
                fogpix[n] = fog_pixel_mask(warped[n], template, gain_of[n])
            else:
                fogpix[n] = np.zeros((Hc, W), bool)

    def rank(cands, key):
        """Eligible sources, best first: least fog on the tile, then sharpest.

        Sharpness is *ascending* scale[n]. Mind the direction: scale[n] is the
        factor that blows a shot up to template size, so the smallest value is
        the shot that already had the most of its own pixels on the tile. The
        zoomed-out shots are the ones being upscaled, and they lose twice over
        -- fewer source pixels per tile, and constant-screen-size UI (city
        labels, health bars) covering more board area per tile.

        Fog evidence outranks sharpness, because a sharp shot that cannot
        actually see the tile is worse than a blurry one that can. Sources
        carrying clearly more fog than the best available are pushed to the
        back rather than dropped, so they can still fill pixels no cleaner
        source photographed -- the same reasoning as the badge fallback below.
        The margin is well clear of ordinary disagreement: across test_ss_3's
        186 multi-source tiles the fog-fraction spread between co-eligible
        sources is a median 0.001 and a 95th percentile of 0.019."""
        poly = tile_poly(origin, u_col, u_row, key[0], key[1], 0.0)
        frac = {n: (tile_fog_fraction(fogpix[n], wmask[n], poly, W, Hc) or 0.0)
                for n in cands}
        lo = min(frac.values())
        clean = sorted([n for n in cands if frac[n] <= lo + args.fog_frac_margin],
                       key=lambda n: scale[n])
        foggy = sorted([n for n in cands if frac[n] > lo + args.fog_frac_margin],
                       key=lambda n: scale[n])
        return clean + foggy, len(foggy)

    # A tile with no clean (badge-excluded) witness falls back to raw witnessing
    # -- i.e. a source may win using content that includes a capture badge --
    # rather than showing template fog. Excluding badge pixels is meant to
    # prefer a *cleaner* source when one exists, never to blank out the only
    # available view of a tile.
    #
    # `priority[key]` keeps the *whole* eligible order, not just the winner,
    # because the paste layers through it: a tile whose best source only partly
    # covers it (that shot's own photo frame ends mid-tile) falls through to the
    # next-best source for the leftover pixels, instead of copying the winner's
    # out-of-frame black into the composite.
    winner, priority, badge_fallback, fog_demoted = {}, {}, [], []
    with PHASES("winner selection"):
        for i in range(N):
            for j in range(N):
                key = (i, j)
                eligible = [n for n, s in samples[key].items() if s["explored"]]
                if eligible:
                    order, n_foggy = rank(eligible, key)
                    if n_foggy:
                        fog_demoted.append(key)
                    winner[key] = order[0]
                    priority[key] = (order, pmask)
                    continue
                # Nothing to fall back *to* unless some shot actually carries
                # a badge: with none, wmask_raw is wmask and the re-sample below
                # is guaranteed to reproduce the empty `eligible` it just got.
                # This used to test badge_mask_of for a None, which is never
                # None (see its assignment), so the re-sample ran on every
                # unexplored tile of every merge.
                if not badge_found:
                    continue
                poly = tile_poly(origin, u_col, u_row, i, j, args.tile_inset)
                wedge = tile_top_wedge(origin, u_col, u_row, i, j)
                raw_eligible = []
                for n in names:
                    s = sample_tile(warped[n], wmask_raw[n], tmpl_gray, poly,
                                    args.fog_ncc, args.min_valid_frac,
                                    wedge_poly=wedge,
                                    fog_wedge_ncc=args.fog_wedge_ncc)
                    if s is not None and s["explored"]:
                        raw_eligible.append(n)
                if raw_eligible:
                    order, _ = rank(raw_eligible, key)
                    winner[key] = order[0]
                    priority[key] = (order, pmask_raw)
                    badge_fallback.append(key)

    # The population bar is owner-only, so the shot showing one *is* that
    # city's owner's shot -- no ownership has to be inferred, which is what
    # makes this immune to "* N" also appearing on embassy'd foreign cities.
    # Preserving it is a priority question, not a compositing one: promote
    # that source on the city's 3x3 block and the ordinary paste keeps the bar
    # intact. The 3x3 is deliberately a *superset* of the tiles the bar can
    # touch (the city plus its SW/S/SE), because promoting only some of them
    # would cut the bar in half at a tile border -- worse than not preserving
    # it at all. It is also exactly the block a city's owner is guaranteed to
    # have vision on, so promotion here can never paste fog over another
    # player's terrain.
    #
    # The `n in order` test is the belt-and-braces version of that guarantee:
    # `order` holds only sources that witnessed the tile as explored, so a
    # source is never promoted onto a tile it sees as fog even if the bar was
    # mis-located. A mis-detection can therefore cost sharpness, never truth.
    bars_of, bar_promoted, vision_promoted = {}, [], []
    if args.city_bars:
        # Same promotion, triggered by *vision* rather than by a detected bar.
        # A source that alone sees every tile of some 3x3 block is the only
        # candidate owner of a city there, since the game guarantees an owner
        # sight of all 8 neighbors -- and this needs no detector, so it reaches
        # bars a fused or occluded sprite hides from detect_population_bars.
        # It is self-limiting: inside anyone's own territory every source
        # qualifies and the claim cancels, so it only discriminates where
        # sight genuinely differs (frontiers, frame edges), which is where
        # bars actually get lost. See CLAUDE.md for the measured hit rate and
        # the worked strong/weak-claim case.
        #
        # Blocks running off the board count only their in-range tiles; a rim
        # city has fewer than 8 neighbors and would otherwise never qualify.
        # Everything is collected before anything is applied, so overlapping
        # blocks cannot be resolved by iteration order.
        with PHASES("city-bar detection"):
            for n in names:
                bars_of[n] = detect_population_bars(warped[n], wmask[n],
                                                    origin, u_col, u_row, N)

            # Does this bar physically reach its S/SW/SE neighbors? Only the
            # capped width does; the short one stops on its own tile. Asked
            # here to rank two contradictory detections against each other, and
            # again below for the claim ranking.
            #
            # This was `_complete`, and it measured a detected bbox against a
            # pixel width to tell a whole bar from a fragment. The anchor-first
            # detector only ever emits one of two legal widths, so there are no
            # fragments and nothing to measure -- the bbox argument outlived
            # its use by some margin.
            def _capped(width_class):
                return width_class >= 3

            # How well backed a city is, taking its best evidence across shots.
            def _evidence(city):
                best = (0, 0.0, 0)
                for bars in bars_of.values():
                    for c, bbox, width_class, plate in bars:
                        if c == city:
                            best = max(best, (1 if _capped(width_class) else 0,
                                              plate, bbox[2]))
                return best

            # **No two cities sit within CITY_MIN_GAP tiles of each other**
            # (confirmed with the project owner), so two detections that close
            # cannot both be real -- at least one is an impostor. That is a hard
            # contradiction rather than a heuristic, and it is the one signal
            # here needing no new measurement: the corpus carried 14 such pairs.
            #
            # Resolve on the evidence already in hand -- a complete bar beats a
            # fragment, then plate evidence, then span -- and drop the loser's
            # detections outright, so the report and the splice check agree with
            # the claim ranking. Dropping is safe in the direction that matters:
            # the loser keeps its *vision* claim, which is a separate mechanism,
            # so a real city demoted here can still win its tiles by sight.
            #
            # On an exact tie neither is better and dropping either is a guess,
            # so the higher tile loses purely to keep this deterministic. Repeat
            # until no pair survives, since adjacency can chain.
            while True:
                cities = {c for bars in bars_of.values()
                          for c, _b, _n, _p in bars}
                pair = next(((a, b) for a in sorted(cities)
                             for b in sorted(cities)
                             if a < b and max(abs(a[0] - b[0]),
                                              abs(a[1] - b[1])) <= CITY_MIN_GAP),
                            None)
                if pair is None:
                    break
                loser = min(pair, key=lambda c: (_evidence(c), (-c[0], -c[1])))
                for n in names:
                    bars_of[n] = [x for x in bars_of[n] if x[0] != loser]

            # Several shots can show one city's bar -- two shots by the same
            # player, most obviously. Arbitrate by the ordinary sharpness rule
            # rather than letting dict order decide, so this is deterministic.
            owner_of, capped_of, plate_of = {}, {}, {}
            for n, bars in bars_of.items():
                for city, bbox, width_class, plate in bars:
                    if city not in owner_of or scale[n] < scale[owner_of[city]]:
                        owner_of[city] = n
                        plate_of[city] = plate
                        # Direct evidence that the bar physically covers
                        # its S/SW/SE neighbors, which only the capped width
                        # does. Every detection is one of two legal widths now,
                        # so this is a property of the bar rather than a guess
                        # about how much of one was seen.
                        capped_of[city] = _capped(width_class)

        with PHASES("vision-based promotion"):
            seen_of = {}
            for ci in range(N):
                for cj in range(N):
                    block = [(ci + di, cj + dj) for di in (-1, 0, 1)
                             for dj in (-1, 0, 1)
                             if 0 <= ci + di < N and 0 <= cj + dj < N]
                    seers = [n for n in names
                             if all(samples.get(k, {}).get(n, {}).get("explored")
                                    for k in block)]
                    if len(seers) == 1:
                        seen_of[(ci, cj)] = seers[0]

        # claim strength: 3 = detected bar, on tiles its bar can occupy;
        # 2 = vision, same tiles; 1 = detected bar, rest of the block;
        # 0 = vision, rest of the block. Strongest wins, sharpest breaks ties.
        claims = {}
        for source, strong_w, weak_w in ((owner_of, 3, 1), (seen_of, 2, 0)):
            for (ci, cj), n in source.items():
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        key = (ci + di, cj + dj)
                        w = strong_w if (di >= 0 and dj >= 0) else weak_w
                        # Rank, in full: strength, then whether a capped-width
                        # bar backs the claim, then plate evidence, then
                        # proximity, then sharpness. Every term earns its place
                        # and the order is not arbitrary -- CLAUDE.md works
                        # through the contested tiles that pin it down.
                        #
                        # Two of them in brief. **Proximity alone is not
                        # enough**, because the two cases disagree about it: a
                        # false detection can sit on a real bar's SW tile (the
                        # nearer claim is wrong) or a real city can hold only a
                        # short bar beside a longer impostor (the nearer claim
                        # is right). Width class separates them. **A city's own
                        # tile beats another city's claim on it as a neighbor**
                        # at equal strength -- a bar certainly covers the tile
                        # its city stands on, a neighbor's claim there is
                        # speculative -- and without that the tie falls to
                        # sharpness and splices a city's label across two shots
                        # that render it differently.
                        #
                        # Vision claims carry no plate measurement and score 0.
                        # That is neutral rather than a penalty: strength
                        # already separates vision from detected-bar claims, so
                        # a plate value is only ever compared against another
                        # value of the same kind.
                        d = max(abs(di), abs(dj))
                        full = 0 if capped_of.get((ci, cj), False) else 1
                        plate = plate_of.get((ci, cj), 0.0)
                        best = claims.get(key)
                        cand = (-w, full, -plate, d, scale[n], n)
                        if best is None or cand < (-best[0], best[1], -best[2],
                                                   best[3], scale[best[4]],
                                                   best[4]):
                            claims[key] = (w, full, plate, d, n)
        for key, (w, _f, _p, _d, n) in claims.items():
            if key not in priority:
                continue
            order, md = priority[key]
            if n not in order:
                continue
            priority[key] = ([n] + [m for m in order if m != n], md)
            if winner.get(key) != n:
                (bar_promoted if w >= 3 or (w == 1) else vision_promoted).append(key)
            winner[key] = n

    # Ruin-vision detection runs on the same per-source fog classification the
    # merge already produced: a sprite is only searched for inside tiles this
    # source itself witnessed as fog, which is what keeps saturated explored
    # content (fruit, borders, units) out of the candidate set entirely.
    ruin_of = {}                          # n -> [(i, j, area, mask)]
    ruin_hits = {}                        # (i, j) -> ([(n, area, mask)], tiles)
    n_raw = 0                             # detections before adjacency merging
    no_ruin_sprite = False                # asset missing: reported, never faked
    ruin_no_fog = []                      # shots with no fog reference to match
    if args.ruin_vision:
        with PHASES("ruin-vision detection"):
            sprite = load_ruin_sprite()
            no_ruin_sprite = sprite is None
            for n in names:
                fog_area = tile_predicate_mask(
                    n, lambda s: s.get("witness") and not s["explored"])
                fog_area &= wmask[n] > 0
                # The fog this shot would show if nothing were drawn on it. The
                # gain comes from the fog-pixel-mask phase above, which fits it
                # on tiles this shot locked onto the template's fog art. Without
                # one there is no reference to subtract, and a matched filter
                # against a guessed background is worse than no answer: say so
                # and skip the shot.
                if sprite is None or n not in gain_of:
                    if sprite is not None:
                        ruin_no_fog.append(n)
                    ruin_of[n] = []
                    continue
                # The template and this shot's gain go in, not a prediction
                # built from them: the prediction is only ever read inside the
                # fog crop, so building it whole here spent 30% of the phase on
                # fog nobody looks at.
                #
                # wmask, not pmask, and the mask taxonomy in CLAUDE.md is
                # worth re-reading before changing it. pmask is the *pasting*
                # mask -- it keeps pixels too dark to judge color by, because a
                # winning source may legitimately have dark content. Feeding
                # those here adds their darkness to |D| in the correlation's
                # denominator without adding anything a flame kernel explains,
                # which drags genuine matches down: measured, u_forest drops
                # from 9 ruins to 7 on that change alone.
                found = detect_ruin_vision(warped[n], wmask[n], fog_area,
                                           template, gain_of[n], origin,
                                           u_col, u_row, sprite)
                ruin_of[n] = found
                for i, j, area, mask in found:
                    if 0 <= i < N and 0 <= j < N:
                        ruin_hits.setdefault((i, j), []).append((n, area, mask))
            n_raw = len(ruin_hits)
            ruin_hits = cluster_ruin_tiles(ruin_hits, origin, u_col, u_row)

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

    with PHASES("paste composite"):
        out = template.copy()
        for (i, j), (order, mask_dict) in priority.items():
            r = tile_mask_bbox(origin, u_col, u_row, i, j, W, Hc)
            if r is None:
                continue
            m, (x0, y0, x1, y1) = r
            region_out = out[y0:y1, x0:x1]
            filled = np.zeros(m.shape, bool)
            for n in order:
                avail = (m > 0) & (mask_dict[n][y0:y1, x0:x1] > 0) & ~filled
                if not avail.any():
                    continue
                region_out[avail] = warped[n][y0:y1, x0:x1][avail]
                filled |= avail
        # Decorative layers go on *after* the tiles, so the grid reads over
        # real terrain and not only over the fog it was rendered against. They
        # are drawn before the ruin markers below, which therefore stay the
        # topmost thing on the composite.
        #
        # `winner` holds exactly the tiles somebody explored, so its complement
        # is the fog still showing through from the template -- which is what
        # the shading layer is clipped to. Built from the same tile_poly the
        # paste loop uses, so the two agree tile for tile by construction.
        fog_only = None
        if overlays & OVERLAY_FOG_ONLY:
            fog_only = np.zeros((Hc, W), np.uint8)
            for i in range(N):
                for j in range(N):
                    if (i, j) in winner:
                        continue
                    poly = tile_poly(origin, u_col, u_row, i, j, 0.0)
                    cv2.fillConvexPoly(fog_only,
                                       np.round(poly).astype(np.int32), 1)
            fog_only = fog_only.astype(np.float32)[:, :, None]
        missing_overlays = paint_overlays(out, overlays, N, fog_only)
        if missing_overlays:
            # Named on stdout in the same shape as DROPPED so polybot can lift
            # it into the merge caption -- a player who asked for the grid and
            # silently did not get it would reasonably assume the merge failed.
            print(f"NO-OVERLAY {len(missing_overlays)}: "
                  f"{' '.join(sorted(missing_overlays))} -- not available on a "
                  f"{N}x{N} board")
        # A fogged tile carrying a ruin gets two things: the Elyrion player's
        # own view of that tile, and a violet outline around it.
        #
        # How much gets copied is constrained, and deliberately. The flames are
        # that player's private UI rather than map content, nothing else can
        # corroborate them, and the cluster is
        # drawn at an offset from the tile it refers to and spills across the
        # border. So the copy is clipped to the ruin tile's own rhombus: the
        # player sees the real cluster instead of taking the outline's word for
        # it, and no sprite lands on a tile that did not earn one. A cluster
        # straddling a border therefore shows only its share, which is the
        # honest rendering of a marker that does not belong to one tile.
        #
        # Tiles another player has actually explored are skipped entirely --
        # their real terrain is already in the composite and is strictly better
        # information than either a marker or a copy.
        #
        # Two details make the copy sit right:
        #  * It runs *after* the decorative overlays, so `shade` cannot dull the
        #    flames -- the same reason the outline is drawn last.
        #  * The source is carried back through its own illumination fit
        #    (the inverse of fog_illumination's gain) into template space, so
        #    the pasted fog matches the fog of the tiles around it instead of
        #    showing a rectangle of that shot's exposure.
        #  * pmask, not wmask: this is a *pasting* operation, and the mask
        #    taxonomy reserves wmask for judging color. A flame's own dark
        #    pixels are content here, not untrustworthy data.
        thick = max(3, int(round(np.linalg.norm(u_col) * 0.075)))
        for key, (hits, comp) in sorted(ruin_hits.items()):
            if key in winner:
                continue
            # Sharpest witness first (ascending scale, as everywhere else),
            # then the one that saw most of the cluster.
            src = min(hits, key=lambda h: (scale[h[0]], -h[1]))[0]
            r = poly_mask_bbox(tile_poly(origin, u_col, u_row, key[0], key[1],
                                         0.0), W, Hc)
            if r is not None:
                m, (x0, y0, x1, y1) = r
                sel = (m > 0) & (pmask[src][y0:y1, x0:x1] > 0)
                if sel.any():
                    patch = warped[src][y0:y1, x0:x1].astype(np.float32)
                    g = gain_of.get(src)
                    if g is not None:
                        patch = (patch - g[:, 1]) / np.where(
                            np.abs(g[:, 0]) < 1e-3, 1.0, g[:, 0])
                    out[y0:y1, x0:x1][sel] = np.clip(
                        patch, 0, 255).astype(np.uint8)[sel]
            poly = tile_poly(origin, u_col, u_row, key[0], key[1], 0.08)
            cv2.polylines(out, [np.round(poly).astype(np.int32)], True,
                          RUIN_MARK_BGR, thick, cv2.LINE_AA)

    full = np.float32([origin, origin + N * u_col,
                       origin + N * u_col + N * u_row, origin + N * u_row])
    x0c, y0c = np.floor(full.min(0)).astype(int)
    x1c, y1c = np.ceil(full.max(0)).astype(int)
    pad = 10
    x0c, y0c = max(x0c - pad, 0), max(y0c - pad, 0)
    x1c, y1c = min(x1c + pad, W), min(y1c + pad, Hc)
    with PHASES("encode + write output"):
        cv2.imwrite(args.out, out[y0c:y1c, x0c:x1c])

    total = N * N
    print(f"\nmap: {N}x{N} = {total} tiles")
    print(f"explored (union): {len(winner)}/{total} ({100 * len(winner) / total:.1f}%)")
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
        total = sum(len(b) for b in bars_of.values())
        if total:
            print(f"\ncity population bars: {total} found (owner-only, so each "
                  f"marks that shot as the city's owner):")
            for n in names:
                for (ci, cj), bbox, width_class, _pl in sorted(bars_of.get(n, [])):
                    kind = "full bar" if width_class >= 3 else "short bar"
                    print(f"  city ({ci},{cj}): {kind}, seen by {n}")
            print(f"  {len(set(bar_promoted))} tile(s) changed hands to keep a "
                  f"bar intact: {sorted(set(bar_promoted))}")
            # Did each bar actually survive whole? Promotion changing hands is
            # not the same question: a bar is only preserved if *every* tile it
            # physically crosses went to a source that shows it, and one tile
            # lost splices it -- which reads worse than not preserving it at
            # all. Worth reporting because none of the other numbers can see
            # this. The explored union is invariant under promotion by
            # construction, and the count above counts detections, so a spliced
            # bar leaves both completely unchanged. It is nearly free here,
            # since `winner` and the bar bboxes are already in hand.
            shown_by = {}
            for n, bars in bars_of.items():
                for city, bbox, width_class, _pl in bars:
                    shown_by.setdefault(city, set()).add(n)
            spliced = []
            for city, srcs in sorted(shown_by.items()):
                covered = set()
                for n in srcs:
                    for c2, (bx, by, bw, bh), _n, _p in bars_of.get(n, []):
                        if c2 != city:
                            continue
                        # Sample the whole bbox, not just its corners: the tile
                        # lattice is a rhombus, so a bar only ~1.5 tiles wide
                        # can cross three of them and a few probe points miss
                        # the one in the middle.
                        for px in np.linspace(bx, bx + bw, 12):
                            for py in np.linspace(by, by + bh, 4):
                                covered.add(tile_of_point(
                                    (px, py), origin, u_col, u_row))
                lost = sorted(t for t in covered
                              if t in winner and winner[t] not in srcs)
                if lost:
                    wid = max(b[1][2] for n in srcs
                              for b in [x for x in bars_of.get(n, [])
                                        if x[0] == city])
                    spliced.append((city, lost, wid, capped_of.get(city, False)))
            if spliced:
                # Report the width class, because the two cases this catches
                # want opposite reactions and only that separates them. A
                # *capped* bar losing a tile is the real defect -- the composite
                # shows half a bar, which reads worse than showing none. A short
                # bar losing one is usually the system working: cities sit at
                # least CITY_MIN_GAP apart, so a short bar detected beside a
                # capped one is a false positive being correctly overridden, and
                # cutting it is the point.
                #
                # "complete" here means the capped width, not the old
                # complete-versus-fragment distinction -- the anchor-first
                # detector emits only legal widths, so there are no fragments.
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
        # NO-RUIN-SPRITE mirrors NO-OVERLAY: a machine-readable line the bot
        # lifts into its caption. Saying nothing would let a player read "no
        # ruins found" as "this board has no ruins", which is a different and
        # much stronger claim than "this merge could not look".
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

    print(f"\nconflicts: {len(conflicts)} tile(s) with inconsistent content "
          f"across sources (mean color dist > {args.consistency_thresh})")
    for (i, j), d, ws in sorted(conflicts, key=lambda c: -c[1])[:10]:
        print(f"  tile ({i},{j}): dist={d:.1f} sources={ws}")

    # The backstop for a stated size that nothing else could check. See
    # CONFLICT_FRAC_SUSPECT: this is the only signal here that measures what a
    # wrong size actually does, rather than inferring it from whether fog was
    # found. It reports rather than refuses, because the size was asserted by
    # the person merging and overriding them here would put us back to refusing
    # replays.
    if size_unverified and comparable >= CONFLICT_FRAC_MIN_COMPARABLE:
        # Over the tiles two sources actually *can* be compared on, not over the
        # board. The board is the wrong denominator whenever the shots do not
        # overlap much: basin_treaties' two shots show disjoint islands, so at a
        # wrong size it reads 0.020-0.078 of the board -- under any usable bar --
        # while the handful of tiles they do share disagree wholesale.
        frac = len(conflicts) / float(comparable)
        if frac > CONFLICT_FRAC_SUSPECT:
            print(f"\nWARNING: {100 * frac:.0f}% of tiles disagree across "
                  f"sources, which is far more than two shots of one board "
                  f"should. {N}x{N} is probably the wrong size -- at the "
                  f"right one this stays under 17%. The merge was written "
                  f"anyway; check it before trusting it.")
        # No "looks fine" branch, deliberately: a low number here is not
        # evidence the size is right. Measured by forcing every set to every
        # size, badland_test at 11 and 14 and u_forest at 11 all produce *zero*
        # conflicts at a flatly wrong size, and fogless reads the same at 16 as
        # at 20 because with no fog there is nothing for a bad lattice to smear.
        # So this rules a size out and is otherwise silent -- "nothing
        # contradicts NxN" would be reassurance the number cannot support.

    print(f"\nwrote {args.out}")

    if args.debug_dir:
        with PHASES("debug overlays"):
            os.makedirs(args.debug_dir, exist_ok=True)
            pal = {n: c for n, c in zip(names, itertools.cycle(
                [(60, 60, 255), (60, 220, 60), (255, 180, 60), (255, 60, 220),
                 (60, 220, 220), (200, 200, 200)]))}

            prov = (template.astype(np.float32) * 0.35).astype(np.uint8)
            for (i, j), n in winner.items():
                r = tile_mask_bbox(origin, u_col, u_row, i, j, W, Hc)
                if r is None:
                    continue
                m, (x0, y0, x1, y1) = r
                region = prov[y0:y1, x0:x1]
                region[m > 0] = pal[n]
            cv2.imwrite(os.path.join(args.debug_dir, "provenance.png"), prov)

            conf_img = prov.copy()
            for (i, j), d, ws in conflicts:
                poly = tile_poly(origin, u_col, u_row, i, j, 0.0).astype(np.int32)
                cv2.polylines(conf_img, [poly], True, (255, 255, 255), 3)
            cv2.imwrite(os.path.join(args.debug_dir, "conflicts.png"), conf_img)

            grid = out.copy()
            for i in range(N + 1):
                p0 = origin + i * u_col
                p1 = origin + i * u_col + N * u_row
                cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 0, 0), 1)
            for j in range(N + 1):
                p0 = origin + j * u_row
                p1 = origin + N * u_col + j * u_row
                cv2.line(grid, tuple(p0.astype(int)), tuple(p1.astype(int)), (0, 0, 0), 1)
            cv2.imwrite(os.path.join(args.debug_dir, "grid_overlay.png"), grid)

            # The warped sources and the lattice they were warped onto, as
            # plain data. Everything else in this directory is an *overlay* --
            # a decision already drawn onto the pixels in red or yellow -- which
            # makes it unreadable as input: ruins_<name>.png paints its
            # accepted pixels over the very colors anything downstream would
            # want to measure. Offline tooling that needs to re-measure what a
            # detector saw (tools/ruinsprite.py) needs the pixels themselves and
            # the basis to locate a tile in them, so write both. Pure
            # diagnostics: nothing in the pipeline reads these back.
            for n in names:
                cv2.imwrite(os.path.join(args.debug_dir, f"warped_{n}.png"),
                            warped[n])
            with open(os.path.join(args.debug_dir, "anchor.json"), "w") as fh:
                json.dump({
                    "map_size": int(N),
                    "origin": [float(v) for v in origin],
                    "u_col": [float(v) for v in u_col],
                    "u_row": [float(v) for v in u_row],
                    "template": os.path.basename(template_path),
                    "shots": {n: {
                        "scale": float(scale[n]),
                        # what this shot itself witnessed, which is the cut the
                        # ruin detector runs inside -- a marker is only ever
                        # searched for on a tile this source called fog
                        "fog_tiles": [list(k) for k, per in sorted(samples.items())
                                      if n in per and per[n].get("witness")
                                      and not per[n]["explored"]],
                        "explored_tiles": [list(k) for k, per in sorted(samples.items())
                                           if n in per and per[n].get("witness")
                                           and per[n]["explored"]],
                        # per-channel gain+offset carrying template fog to this
                        # shot's colors; absent when the shot locked too little
                        # fog to fit one (see the fog pixel masks phase)
                        "fog_gain": (gain_of[n].tolist()
                                     if n in gain_of else None),
                    } for n in names},
                }, fh, indent=1)

            for n in names:
                dim = warped[n].copy().astype(np.float32)
                for (i, j), per_img in samples.items():
                    if per_img.get(n, {}).get("explored"):
                        continue
                    r = tile_mask_bbox(origin, u_col, u_row, i, j, W, Hc)
                    if r is None:
                        continue
                    m, (x0, y0, x1, y1) = r
                    region = dim[y0:y1, x0:x1]
                    region[m > 0] *= 0.35
                cv2.imwrite(os.path.join(args.debug_dir, f"explored_{n}.png"),
                           np.clip(dim, 0, 255).astype(np.uint8))

            # Same mandatory-QA reasoning as the badge overlay: show exactly
            # which pixels the bar detector accepted and which 3x3 it promoted,
            # per source, even when it found nothing.
            if args.city_bars:
                for n in names:
                    vis = warped[n].copy()
                    for (ci, cj), (bx, by, bw, bh), _wc, _pl in bars_of.get(n, []):
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                poly = tile_poly(origin, u_col, u_row,
                                                 ci + di, cj + dj, 0.0)
                                cv2.polylines(vis, [np.round(poly).astype(np.int32)],
                                              True, (0, 255, 255), 2)
                        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh),
                                      (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(args.debug_dir, f"bars_{n}.png"), vis)

            # Same mandatory-QA reasoning as the badge overlay below: the ruin
            # matched filter accepts or rejects on a score nobody can eyeball,
            # so when it ran, show exactly which pixels it accepted, per source,
            # even when it found nothing.
            if args.ruin_vision:
                for n in names:
                    vis = warped[n].copy()
                    for i, j, area, mask in ruin_of.get(n, []):
                        vis[mask] = (0, 0, 255)
                        poly = tile_poly(origin, u_col, u_row, i, j, 0.0)
                        cv2.polylines(vis, [poly.astype(np.int32)], True,
                                      (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(args.debug_dir, f"ruins_{n}.png"), vis)

            # Always render what detect_capture_badges flagged, whether or not it
            # found anything -- this is the mandatory visual QA step for a
            # heuristic with a real, demonstrated false-positive mode (see that
            # function's docstring), so a silent miss or misfire is never silent.
            for n in names:
                badge = badge_mask_of.get(n)
                if badge is None:
                    continue
                vis = imgs[n].copy()
                vis[badge > 0] = (0, 0, 255)
                cv2.imwrite(os.path.join(args.debug_dir, f"badges_{n}.png"), vis)

            print(f"debug output in {args.debug_dir}/:\n"
                  f"  provenance.png       winner per tile\n"
                  f"  conflicts.png        tiles whose sources disagree, outlined\n"
                  f"  grid_overlay.png     the tile lattice on the composite\n"
                  f"  anchor.json          per-shot transforms and tile counts\n"
                  f"  warped_<n>.png       each shot on the template canvas\n"
                  f"  explored_<n>.png     what each shot witnessed as explored\n"
                  f"  badges_<n>.png       capture-badge pixels excluded, in red\n"
                  f"                       (written for every shot, empty or not)\n"
                  + ("  bars_<n>.png         population bars found\n"
                     if args.city_bars else "")
                  + ("  ruins_<n>.png        ruin-vision pixels accepted, in red\n"
                     if args.ruin_vision else ""))


if __name__ == "__main__":
    _t0 = time.perf_counter()
    try:
        main()
    finally:
        PHASES.report(time.perf_counter() - _t0)
