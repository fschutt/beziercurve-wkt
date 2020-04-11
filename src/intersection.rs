//! Module for calculating curve-curve

use crate::Point;

pub type CubicCurve = (Point, Point, Point, Point);
pub type Line = (Point, Point);

pub struct CubicCubicIntersection {
    t1: f32,
    curve1: CubicCurve,
    t2: f32,
    curve2: CubicCurve,
    intersection_point: Point,
}

pub struct CubicLineIntersection {
    t1: f32,
    curve1: CubicCurve,
    t2: f32,
    line1: Line,
}

pub enum IntersectionResult {
    Intersection(QuadQuadIntersection),
    InfiniteIntersectionsQuadratic(CubicCurve),
}

// Intersect a quadratic with another quadratic curve
fn cubic_cubic_intersect(a: CubicCurve, b: CubicCurve) -> IntersectionResult {
    if a == b {
        return InfiniteIntersectionsQuadratic(a);
    }

    // TODO
}

// Intersect a quadratic curve with a line
fn cubic_line_intersect(
    (a1, a2, a3): (Point, Point, Point),
    (b1, b2): (Point, Point),
) -> IntersectionResult {

}

// Convert a quadratic bezier into a cubic bezier
fn quad_into_cubic_bezier(c: (Point, Point, Point)) -> CubicCurve {

}

/// Bezier curve intersection algorithm and utilities
/// Directly extracted from PaperJS's implementation bezier curve fat-line clipping
/// The original source code is available under the MIT license at
///
/// https://github.com/paperjs/paper.js/

const TOLERANCE:f32 = 1e-5;
const EPSILON: f32 = 1e-10;

#[inline]
fn is_zero(val: f32) -> bool {
  val.abs() <= EPSILON
}

/// Computes the signed distance of (x, y) between (px, py) and (vx, vy)
#[inline]
fn signed_distance(px: f32, py: f32, vx: f32, vy: f32, x: f32, y: f32) {
    vx -= px;
    vy -= py;
    if is_zero(vx) {
        if vy.is_sign_positive() { px - x } else { x - px }
    } else if (isZero(vy)) {
        if vx.is_sign_positive() { y - py } else { py - y }
    } else {
        (vx * (y - py) - vy * (x - px)) / (vx * vx + vy * vy).sqrt()
    }
}

/// Calculate the convex hull for the non-parametric bezier curve D(ti, di(t)).
///
/// The ti is equally spaced across [0..1] — [0, 1/3, 2/3, 1] for
/// di(t), [dq0, dq1, dq2, dq3] respectively. In other words our CVs for the
/// curve are already sorted in the X axis in the increasing order.
/// Calculating convex-hull is much easier than a set of arbitrary points.
///
/// The convex-hull is returned as two parts [TOP, BOTTOM]. Both are in a
/// coordinate space where y increases upwards with origin at bottom-left
///
/// - TOP: The part that lies above the 'median' (line connecting end points of the curve)
/// - BOTTOM: The part that lies below the median.
fn convex_hull(dq0: f32, dq1: f32, dq2: f32, dq3: f32) -> [Vec<f32>;2] {

    let p0 = [0.0, dq0];
    let p1 = [1.0 / 3.0, dq1];
    let p2 = [2.0 / 3.0, dq2];
    let p3 = [1.0, dq3];

    // Find signed distance of p1 and p2 from line [ p0, p3 ]
    let dist1 = signed_distance(0.0, dq0, 1.0, dq3, 1.0 / 3, dq1);
    let dist2 = signed_distance(0.0, dq0, 1.0, dq3, 2.0 / 3, dq2);

    let flip = false;
    let mut hull;

    // Check if p1 and p2 are on the same side of the line [ p0, p3 ]
    if dist1 * dist2 < 0.0 {
        // p1 and p2 lie on different sides of [ p0, p3 ]. The hull is a
        // quadrilateral and line [ p0, p3 ] is NOT part of the hull so we
        // are pretty much done here.
        // The top part includes p1,
        // we will reverse it later if that is not the case
        hull = [vec![p0, p1, p3], vec![p0, p2, p3]];
        flip = dist1 < 0.0;
    } else {
        // p1 and p2 lie on the same sides of [ p0, p3 ]. The hull can be
        // a triangle or a quadrilateral and line [ p0, p3 ] is part of the
        // hull. Check if the hull is a triangle or a quadrilateral.
        // Also, if at least one of the distances for p1 or p2, from line
        // [p0, p3] is zero then hull must at most have 3 vertices.
        let pmax;
        let cross = 0;
        let distZero = dist1 == 0 || dist2 == 0;
        if (Math.abs(dist1) > Math.abs(dist2)) {
            pmax = p1;
            // apex is dq3 and the other apex point is dq0 vector dqapex ->
            // dqapex2 or base vector which is already part of the hull.
            cross = (dq3 - dq2 - (dq3 - dq0) / 3.0) * (2.0 * (dq3 - dq2) - dq3 + dq1) / 3.0;
        } else {
            pmax = p2;
            // apex is dq0 in this case, and the other apex point is dq3
            // vector dqapex -> dqapex2 or base vector which is already part
            // of the hull.
            cross = (dq1 - dq0 + (dq0 - dq3) / 3.0) * (-2.0 * (dq0 - dq1) + dq0 - dq2) / 3.0;
        }

        // Compare cross products of these vectors to determine if the point
        // is in the triangle [ p3, pmax, p0 ], or if it is a quadrilateral.
        hull = if cross < 0.0 || distZero {
            [vec![p0, pmax, p3], vec![p0, p3]]
        } else {
            [vec![p0, p1, p2, p3], vec![p0, p3]]
        };

        flip = if dist1 { dist1 < 0.0 } else { dist2 < 0 };
    }

    if flip {
      hull.reverse();
    }

    hull
}

/// Clips the convex-hull and returns [tMin, tMax] for the curve contained.
fn clip_convex_hull(hullTop: [Vec<f32>;2], hullBottom: [Vec<f32>;2], dMin: f32, dMax: f32) -> f32 {
    if hullTop[0][1] < dMin {
        // Left of hull is below dMin, walk through the hull until it
        // enters the region between dMin and dMax
        clip_convex_hull_part(hullTop, true, dMin)
    } else if (hullBottom[0][1] > dMax) {
        // Left of hull is above dMax, walk through the hull until it
        // enters the region between dMin and dMax
        clip_convex_hull_part(hullBottom, false, dMax)
    } else {
        // Left of hull is between dMin and dMax, no clipping possible
        hullTop[0][0]
    }
}

fn clip_convex_hull_part(part:[Vec<f32>;2], top: Option<f32>, threshold: f32) -> Option<f32> {
    let [px, py] = part[0];

    for [qx, qy] in &part {
        let a = if top.is_some() { qy >= threshold } else { qy <= threshold };
        if a {
            return Some(px + (threshold - py) * (qx - px) / (qy - py));
        }
        [px, py] = [qx, qy];
    }

    // All points of hull are above / below the threshold
    None
}

/// Calculates the fat line of a curve and returns the maximum and minimum offset widths
/// for the fatline of a curve
fn get_fatline((p0, p1, p2, p3): CubicCurve) -> (f32, f32) {

    // Calculate the fat-line L, for Q is the baseline l and two
    // offsets which completely encloses the curve P.
    let d1 = signed_distance(p0.x, p0.y, p3.x, p3.y, p1.x, p1.y);
    let d2 = signed_distance(p0.x, p0.y, p3.x, p3.y, p2.x, p2.y);
    let factor = if (d1 * d2).is_sign_positive() { 3.0 / 4.0 } else { 4.0 / 9.0 }; // Get a tighter fit
    let dMin = factor * 0.0.min(d1).min(d2);
    let dMax = factor * 0.0.max(d1).max(d2);

    // The width of the 'fatline' is |dMin| + |dMax|
    (dMin, dMax)
}

fn subdivide((p1, c1, c2, p2): CubicCurve, t: f32) -> (CubicCurve, CubicCurve) {

    // Triangle computation, with loops unrolled.
    let u = 1.0 - t;

    // Interpolate from 4 to 3 points
    let p3x = u * p1.x + t * c1.x;
    let p3y = u * p1.y + t * c1.y;
    let p4x = u * c1.x + t * c2.x;
    let p4y = u * c1.y + t * c2.y;
    let p5x = u * c2.x + t * p2.x;
    let p5y = u * c2.y + t * p2.y;

    // Interpolate from 3 to 2 points
    let p6x = u * p3x + t * p4x;
    let p6y = u * p3y + t * p4y;
    let p7x = u * p4x + t * p5x;
    let p7y = u * p4y + t * p5y;

    // Interpolate from 2 points to 1 point
    let p8x = u * p6x + t * p7x;
    let p8y = u * p6y + t * p7y;

    // We now have all the values we need to build the sub-curves [left, right]:
    (
      (Point::new(p1x, p1y), Point::new(p3x, p3y), Point::new(p6x, p6y), Point::new(p8x, p8y),
      (Point::new(p8x, p8y), Point::new(p7x, p7y), Point::new(p5x, p5y), Point::new(p2x, p2y))
    )
}

/// Returns the part of a curve between t1 and t2
fn get_part(mut v: CubicCurve, t1: f32, t2: f32) -> CubicCurve {

    if t1.is_sign_positive() {
        v = subdivide(v, t1).1; // right
    }

    // Interpolate the parameter at 't2' in the new curve and cut there.
    if (t2 < 1) {
        v = subdivide(v, (t2 - t1) / (1.0 - t1)).0; // left
    }

    v
}

// ----------------------------------------------------------------------------------------------------------- JS


/// Calculates the coordinates of the point on a bezier curve at a given t
fn evaluate((p1, c1, c2, p2): CubicCurve, t: f32, type: usize /* this is some kind of enum? */) {
  let [p1x, p1y, c1x, c1y, c2x, c2y, p2x, p2y] = v;
  let x, y;

  // Handle special case at beginning / end of curve
  if (type === 0 && (t < TOLERANCE || t > 1 - TOLERANCE)) {
        let isZero = t < TOLERANCE;
        x = isZero ? p1.x : p2.x;
        y = isZero ? p1.y : p2.y;
  } else {
        // Calculate the polynomial coefficients.
        let cx = 3.0 * (c1x - p1x);
        let bx = 3.0 * (c2x - c1x) - cx;
        let ax = p2x - p1x - cx - bx;

        let cy = 3.0 * (c1y - p1y);
        let by = 3.0 * (c2y - c1y) - cy;
        let ay = p2y - p1y - cy - by;

        if (type === 0) {
            // Calculate the curve point at parameter value t
            x = ((ax * t + bx) * t + cx) * t + p1x;
            y = ((ay * t + by) * t + cy) * t + p1y;
        } else {
            // 1: tangent, 1st derivative
            // 2: normal, 1st derivative
            // 3: curvature, 1st derivative & 2nd derivative
            // Prevent tangents and normals of length 0:
            // http:#stackoverflow.com/questions/10506868/
            if (t < TOLERANCE && c1x == p1x && c1y == p1y || t > 1 - TOLERANCE && c2x == p2x && c2y == p2y) {
              x = c2x - c1x;
              y = c2y - c1y;
            } else if (t < TOLERANCE) {
              x = cx;
              y = cy;
            } else if (t > 1 - TOLERANCE) {
              x = 3.0 * (p2x - c2x);
              y = 3.0 * (p2y - c2y);
            } else {
              // Simply use the derivation of the bezier function for both
              // the x and y coordinates:
              x = (3.0 * ax * t + 2 * bx) * t + cx;
              y = (3.0 * ay * t + 2 * by) * t + cy;
            }
            if (type === 3) {
              // Calculate 2nd derivative, and curvature from there:
              // http://cagd.cs.byu.edu/~557/text/ch2.pdf page#31
              // k = |dx * d2y - dy * d2x| / (( dx^2 + dy^2 )^(3/2))
              let x2 = 6.0 * ax * t + 2.0 * bx;
              let y2 = 6.0 * ay * t + 2.0 * by;
              return (x * y2 - y * x2) / Math.pow(x * x + y * y, 3.0 / 2);
            }
        }
    }

    // The normal is simply the rotated tangent:
    if type == 2 { [y, -x] } else { [x, y] }
}

/// Computes the intersections of two bezier curves
fn curve_intersections(
    v1: CubicCurve,
    v2: CubicCurve,
    tMin: f32 = 0.0,
    tMax: f32 = 1.0,
    uMin: f32 = 0.0,
    uMax: f32 = 1.0,
    oldTDiff: f32 = 1.0,
    reverse: bool = false,
    recursion: usize = 0,
    recursionLimit: usize = 32,
    tLimit: f32 = 0.8
) -> Vec<Intersection> {

  // Avoid deeper recursion.
  // NOTE: @iconexperience determined that more than 20 recursions are
  // needed sometimes, depending on the tDiff threshold values further
  // below when determining which curve converges the least. He also
  // recommended a threshold of 0.5 instead of the initial 0.8
  // See: https:#github.com/paperjs/paper.js/issues/565
  if (recursion > recursionLimit) {
    return Vec::new();
  }

  // Let P be the first curve and Q be the second
  let q0x = v2[0]
  let q0y = v2[1]
  let q3x = v2[6]
  let q3y = v2[7]

  // Calculate the fat-line L for Q is the baseline l and two
  // offsets which completely encloses the curve P.
  let [dMin, dMax] = get_fatline(v2);

  // Calculate non-parametric bezier curve D(ti, di(t)) - di(t) is the
  // distance of P from the baseline l of the fat-line, ti is equally
  // spaced in [0, 1]
  let dp0 = signed_distance(q0x, q0y, q3x, q3y, v1[0], v1[1]);
  let dp1 = signed_distance(q0x, q0y, q3x, q3y, v1[2], v1[3]);
  let dp2 = signed_distance(q0x, q0y, q3x, q3y, v1[4], v1[5]);
  let dp3 = signed_distance(q0x, q0y, q3x, q3y, v1[6], v1[7]);
  let tMinNew = 0.0;
  let tMaxNew = 0.0;
  let tDiff = 0.0;

  // NOTE: the recursion threshold of 4 is needed to prevent issue #571
  // from occurring: https://github.com/paperjs/paper.js/issues/571
  if (q0x == q3x && uMax - uMin <= EPSILON && recursion > 4) {
    // The fatline of Q has converged to a point, the clipping is not
    // reliable. Return the value we have even though we will miss the
    // precision.
    tMaxNew = tMinNew = (tMax + tMin) / 2.0;
    tDiff = 0;
  } else {
    // Get the top and bottom parts of the convex-hull
    let hull = convexHull(dp0, dp1, dp2, dp3);
    let top = hull[0];
    let bottom = hull[1];
    let tMinClip;
    let tMaxClip;
    // Clip the convex-hull with dMin and dMax
    tMinClip = clipConvexHull(top, bottom, dMin, dMax);
    top.reverse();
    bottom.reverse();
    tMaxClip = clipConvexHull(top, bottom, dMin, dMax);
    // No intersections if one of the tvalues are null or 'undefined'
    if (tMinClip === null || tMaxClip === null) {
      return Vec::new();
    }
    // Clip P with the fatline for Q
    let v1d = v1;
    v1 = getPart(v1, tMinClip, tMaxClip);
    tDiff = tMaxClip - tMinClip;
    // tMin and tMax are within the range (0, 1). We need to project it
    // to the original parameter range for v2.
    tMinNew = tMax * tMinClip + tMin * (1 - tMinClip);
    tMaxNew = tMax * tMaxClip + tMin * (1 - tMaxClip);
  }

  let intersections;

  // Check if we need to subdivide the curves
  if (oldTDiff > tLimit && tDiff > tLimit) {
    // Subdivide the curve which has converged the least.
    if (tMaxNew - tMinNew > uMax - uMin) {
      let parts = subdivide(v1, 0.5);
      let t = tMinNew + (tMaxNew - tMinNew) / 2.0;
      intersections = [].concat(
        curveIntersections(v2, parts[0], uMin, uMax, tMinNew, t, tDiff, !reverse, recursion + 1, recursionLimit, tLimit),
        curveIntersections(v2, parts[1], uMin, uMax, t, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit)
      );
    } else {
      let parts = subdivide(v2, 0.5);
      let t = uMin + (uMax - uMin) / 2.0;
      intersections = [].concat(
        curveIntersections(parts[0], v1, uMin, t, tMinNew, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit),
        curveIntersections(parts[1], v1, t, uMax, tMinNew, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit)
      );
    }
  } else if (Math.max(uMax - uMin, tMaxNew - tMinNew) < TOLERANCE) {
    // We have isolated the intersection with sufficient precision
    let t1 = tMinNew + (tMaxNew - tMinNew) / 2.0;
    let t2 = uMin + (uMax - uMin) / 2.0;
    if (reverse) {
      intersections = [[t2, evaluate(v2, t2, 0), t1, evaluate(v1, t1, 0)]];
    } else {
      intersections = [[t1, evaluate(v1, t1, 0), t2, evaluate(v2, t2, 0)]];
    }
  } else {
    intersections = curveIntersections(v2, v1, uMin, uMax, tMinNew, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit);
  }

  return intersections;
}