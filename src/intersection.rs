//! Module for calculating curve-curve

use crate::Point;

pub type Line           = (Point, Point);
pub type QuadraticCurve = (Point, Point, Point);
pub type CubicCurve     = (Point, Point, Point, Point);

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct CubicCubicIntersection {
    pub t1: f32,
    pub curve1: CubicCurve,
    pub t2: f32,
    pub curve2: CubicCurve,
}

impl CubicCubicIntersection {
    pub fn get_intersection_point_1(&self) -> Point {
        evaluate(self.curve1, self.t1)
    }

    pub fn get_intersection_point_2(&self) -> Point {
        evaluate(self.curve2, self.t2)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct CubicLineIntersection {
    pub t1: f32,
    pub curve1: CubicCurve,
    pub t2: f32,
    pub line2: Line,
}

impl CubicLineIntersection {
    pub fn get_intersection_point_1(&self) -> Point {
        evaluate(self.curve1, self.t1)
    }

    pub fn get_intersection_point_2(&self) -> Point {
        // lerp(line.0, line.1, t2)
        let new_x = (1.0 - self.t2) * self.line2.0.x + self.t2 * self.line2.1.x;
        let new_y = (1.0 - self.t2) * self.line2.0.y + self.t2 * self.line2.1.y;
        Point::new(new_x, new_y)
    }
}

pub enum IntersectionResult {
    NoIntersection,
    CubicCubicIntersection(Vec<CubicCubicIntersection>),
    InfiniteIntersectionsCubic(CubicCurve, CubicCurve),
    InfiniteIntersectionsCubicLine(CubicCurve, Line),
}

// Intersect a quadratic with another quadratic curve
pub fn cubic_cubic_intersect(a: CubicCurve, b: CubicCurve) -> IntersectionResult {
    use self::IntersectionResult::*;

    if a == b {
        return InfiniteIntersectionsCubic(a, b);
    }

    let intersections = curve_intersections_inner(a, b, 0.0, 1.0, 0.0, 1.0, 1.0, false, 0, 32, 0.8);

    if intersections.is_empty() {
        NoIntersection
    } else {
        CubicCubicIntersection(intersections)
    }
}

// Intersect a quadratic curve with a line
#[allow(unused_variables)]
pub fn cubic_line_intersect(
    (a1, a2, a3, a4): CubicCurve,
    (b1, b2): Line,
) -> IntersectionResult {
    return IntersectionResult::NoIntersection; // TODO!

    /*
    // based on http://mysite.verizon.net/res148h4j/javascript/script_exact_cubic.html#the%20source%20code
    function cubicRoots(P)
    {
        var a=P[0];
        var b=P[1];
        var c=P[2];
        var d=P[3];

        var A=b/a;
        var B=c/a;
        var C=d/a;

        var Q, R, D, S, T, Im;

        var Q = (3*B - Math.pow(A, 2))/9;
        var R = (9*A*B - 27*C - 2*Math.pow(A, 3))/54;
        var D = Math.pow(Q, 3) + Math.pow(R, 2);    // polynomial discriminant

        var t=Array();

        if (D >= 0)                                 // complex or duplicate roots
        {
            var S = sgn(R + Math.sqrt(D))*Math.pow(Math.abs(R + Math.sqrt(D)),(1/3));
            var T = sgn(R - Math.sqrt(D))*Math.pow(Math.abs(R - Math.sqrt(D)),(1/3));

            t[0] = -A/3 + (S + T);                    // real root
            t[1] = -A/3 - (S + T)/2;                  // real part of complex root
            t[2] = -A/3 - (S + T)/2;                  // real part of complex root
            Im = Math.abs(Math.sqrt(3)*(S - T)/2);    // complex part of root pair

            /*discard complex roots*/
            if (Im!=0)
            {
                t[1]=-1;
                t[2]=-1;
            }

        }
        else                                          // distinct real roots
        {
            var th = Math.acos(R/Math.sqrt(-Math.pow(Q, 3)));

            t[0] = 2*Math.sqrt(-Q)*Math.cos(th/3) - A/3;
            t[1] = 2*Math.sqrt(-Q)*Math.cos((th + 2*Math.PI)/3) - A/3;
            t[2] = 2*Math.sqrt(-Q)*Math.cos((th + 4*Math.PI)/3) - A/3;
            Im = 0.0;
        }

        /*discard out of spec roots*/
        for (var i=0;i<3;i++)
            if (t[i]<0 || t[i]>1.0) t[i]=-1;

        /*sort but place -1 at the end*/
        t=sortSpecial(t);

        console.log(t[0]+" "+t[1]+" "+t[2]);
        return t;
    }
    */

    /*
    //computes intersection between a cubic spline and a line segment
    function computeIntersections(px,py,lx,ly)
    {
        var X=Array();

        var A=ly[1]-ly[0];      //A=y2-y1
        var B=lx[0]-lx[1];      //B=x1-x2
        var C=lx[0]*(ly[0]-ly[1]) +
              ly[0]*(lx[1]-lx[0]);  //C=x1*(y1-y2)+y1*(x2-x1)

        var bx = bezierCoeffs(px[0],px[1],px[2],px[3]);
        var by = bezierCoeffs(py[0],py[1],py[2],py[3]);

        var P = Array();
        P[0] = A*bx[0]+B*by[0];     /*t^3*/
        P[1] = A*bx[1]+B*by[1];     /*t^2*/
        P[2] = A*bx[2]+B*by[2];     /*t*/
        P[3] = A*bx[3]+B*by[3] + C; /*1*/

        var r=cubicRoots(P);

        /*verify the roots are in bounds of the linear segment*/
        for (var i=0;i<3;i++)
        {
            t=r[i];

            X[0]=bx[0]*t*t*t+bx[1]*t*t+bx[2]*t+bx[3];
            X[1]=by[0]*t*t*t+by[1]*t*t+by[2]*t+by[3];

            /*above is intersection point assuming infinitely long line segment,
              make sure we are also in bounds of the line*/
            var s;
            if ((lx[1]-lx[0])!=0)           /*if not vertical line*/
                s=(X[0]-lx[0])/(lx[1]-lx[0]);
            else
                s=(X[1]-ly[0])/(ly[1]-ly[0]);

            /*in bounds?*/
            if (t<0 || t>1.0 || s<0 || s>1.0)
            {
                X[0]=-100;  /*move off screen*/
                X[1]=-100;
            }

            /*move intersection point*/
            I[i].setAttributeNS(null,"cx",X[0]);
            I[i].setAttributeNS(null,"cy",X[1]);
        }
    }
    */
}

//  Determines the intersection point of the line defined by points A and B with the
//  line defined by points C and D.
//
//  Returns YES if the intersection point was found, and stores that point in X,Y.
//  Returns NO if there is no determinable intersection point, in which case X,Y will
//  be unmodified.
#[inline]
pub fn line_line_intersect(
) -> IntersectionResult {

    return IntersectionResult::NoIntersection; // TODO
/*
       //  Fail if either line is undefined.
      if (Ax==Bx && Ay==By || Cx==Dx && Cy==Dy) return IntersectionResult::NoIntersection;


      double  distAB, theCos, theSin, newX, ABpos ;

      //  (1) Translate the system so that point A is on the origin.
      Bx-=Ax; By-=Ay;
      Cx-=Ax; Cy-=Ay;
      Dx-=Ax; Dy-=Ay;

      //  Discover the length of segment A-B.
      distAB=sqrt(Bx*Bx+By*By);

      //  (2) Rotate the system so that point B is on the positive X axis.
      theCos=Bx/distAB;
      theSin=By/distAB;
      newX=Cx*theCos+Cy*theSin;
      Cy  =Cy*theCos-Cx*theSin; Cx=newX;
      newX=Dx*theCos+Dy*theSin;
      Dy  =Dy*theCos-Dx*theSin; Dx=newX;

      //  Fail if the lines are parallel.
      if Cy == Dy return None;

      //  (3) Discover the position of the intersection point along line A-B.
      ABpos=Dx+(Cx-Dx)*Dy/(Dy-Cy);

      //  (4) Apply the discovered position to line A-B in the original coordinate system.
      let x =Ax+ABpos*theCos;
      let y =Ay+ABpos*theSin;

      Some(IntersectionResult:: Point::new(x, y))
*/
}

// Convert a quadratic bezier into a cubic bezier
#[inline]
fn quad_into_cubic_bezier(c: QuadraticCurve) -> CubicCurve {
    const TWO_THIRDS f32 = 2.0 / 3.0;

    let c1_x = c.0.x + TWO_THIRDS * (c.1.x - c.0.x);
    let c1_y = c.0.y + TWO_THIRDS * (c.1.y - c.0.y);

    let c2_x = c.2.x + TWO_THIRDS * (c.1.x - c.2.x);
    let c2_y = c.2.y + TWO_THIRDS * (c.1.y - c.2.y);

    (c.0, Point::new(c1_x, c1_y), Point::new(c2_x, c2_y), c.3)
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
fn signed_distance(px: f32, py: f32, mut vx: f32, mut vy: f32, x: f32, y: f32) -> f32 {
    vx -= px;
    vy -= py;
    if is_zero(vx) {
        if vy.is_sign_positive() { px - x } else { x - px }
    } else if is_zero(vy) {
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
#[inline]
fn convex_hull(dq0: f32, dq1: f32, dq2: f32, dq3: f32) -> [Vec<[f32;2]>;2] {

    let p0 = [0.0, dq0];
    let p1 = [1.0 / 3.0, dq1];
    let p2 = [2.0 / 3.0, dq2];
    let p3 = [1.0, dq3];

    // Find signed distance of p1 and p2 from line [ p0, p3 ]
    let dist1 = signed_distance(0.0, dq0, 1.0, dq3, 1.0 / 3.0, dq1);
    let dist2 = signed_distance(0.0, dq0, 1.0, dq3, 2.0 / 3.0, dq2);

    // Check if p1 and p2 are on the same side of the line [ p0, p3 ]
    let (mut hull, flip) = if dist1 * dist2 < 0.0 {
        // p1 and p2 lie on different sides of [ p0, p3 ]. The hull is a
        // quadrilateral and line [ p0, p3 ] is NOT part of the hull so we
        // are pretty much done here.
        // The top part includes p1,
        // we will reverse it later if that is not the case
        let hull = [vec![p0, p1, p3], vec![p0, p2, p3]];
        let flip = dist1 < 0.0;
        (hull, flip)
    } else {
        // p1 and p2 lie on the same sides of [ p0, p3 ]. The hull can be
        // a triangle or a quadrilateral and line [ p0, p3 ] is part of the
        // hull. Check if the hull is a triangle or a quadrilateral.
        // Also, if at least one of the distances for p1 or p2, from line
        // [p0, p3] is zero then hull must at most have 3 vertices.
        let (cross, pmax) = if dist1.abs() > dist2.abs() {
            // apex is dq3 and the other apex point is dq0 vector dqapex ->
            // dqapex2 or base vector which is already part of the hull.
            let cross = (dq3 - dq2 - (dq3 - dq0) / 3.0) * (2.0 * (dq3 - dq2) - dq3 + dq1) / 3.0;
            (cross, p1)
        } else {
            // apex is dq0 in this case, and the other apex point is dq3
            // vector dqapex -> dqapex2 or base vector which is already part
            // of the hull.
            let cross = (dq1 - dq0 + (dq0 - dq3) / 3.0) * (-2.0 * (dq0 - dq1) + dq0 - dq2) / 3.0;
            (cross, p2)
        };

        let distZero = is_zero(dist1) || is_zero(dist2);

        // Compare cross products of these vectors to determine if the point
        // is in the triangle [ p3, pmax, p0 ], or if it is a quadrilateral.
        let hull = if cross < 0.0 || distZero {
            [vec![p0, pmax, p3], vec![p0, p3]]
        } else {
            [vec![p0, p1, p2, p3], vec![p0, p3]]
        };

        let flip = if is_zero(dist1) { !dist2.is_sign_positive() } else {  !dist1.is_sign_positive() };

        (hull, flip)
    };

    if flip {
      hull.reverse();
    }

    hull
}

/// Clips the convex-hull and returns [tMin, tMax] for the curve contained.
#[inline]
fn clip_convex_hull(hullTop: &[[f32;2]], hullBottom: &[[f32;2]], dMin: f32, dMax: f32) -> Option<f32> {
    if hullTop[0][1] < dMin {
        // Left of hull is below dMin, walk through the hull until it
        // enters the region between dMin and dMax
        clip_convex_hull_part(hullTop, true, dMin)
    } else if hullBottom[0][1] > dMax {
        // Left of hull is above dMax, walk through the hull until it
        // enters the region between dMin and dMax
        clip_convex_hull_part(hullBottom, false, dMax)
    } else {
        // Left of hull is between dMin and dMax, no clipping possible
        Some(hullTop[0][0])
    }
}

#[inline]
fn clip_convex_hull_part(part: &[[f32;2]], top: bool, threshold: f32) -> Option<f32> {
    let mut pxpy = part[0];

    for [qx, qy] in part.iter().copied() {
        let [px, py] = pxpy;
        let a = if top { qy >= threshold } else { qy <= threshold };
        if a {
            return Some(px + (threshold - py) * (qx - px) / (qy - py));
        }
        pxpy = [qx, qy];
    }

    // All points of hull are above / below the threshold
    None
}

/// Calculates the fat line of a curve and returns the maximum and minimum offset widths
/// for the fatline of a curve
#[inline]
fn get_fatline((p0, p1, p2, p3): CubicCurve) -> (f32, f32) {

    // Calculate the fat-line L, for Q is the baseline l and two
    // offsets which completely encloses the curve P.
    let d1 = signed_distance(p0.x, p0.y, p3.x, p3.y, p1.x, p1.y);
    let d2 = signed_distance(p0.x, p0.y, p3.x, p3.y, p2.x, p2.y);
    let factor = if (d1 * d2).is_sign_positive() { 3.0 / 4.0 } else { 4.0 / 9.0 }; // Get a tighter fit
    let dMin = factor * 0.0_f32.min(d1).min(d2);
    let dMax = factor * 0.0_f32.max(d1).max(d2);

    // The width of the 'fatline' is |dMin| + |dMax|
    (dMin, dMax)
}

#[inline]
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
      (p1, Point::new(p3x, p3y), Point::new(p6x, p6y), Point::new(p8x, p8y)),
      (Point::new(p8x, p8y), Point::new(p7x, p7y), Point::new(p5x, p5y), p2)
    )
}

/// Returns the part of a curve between t1 and t2
#[inline]
fn get_part(mut v: CubicCurve, t1: f32, t2: f32) -> CubicCurve {

    if t1.is_sign_positive() {
        v = subdivide(v, t1).1; // right
    }

    // Interpolate the parameter at 't2' in the new curve and cut there.
    if t2 < 1.0 {
        v = subdivide(v, (t2 - t1) / (1.0 - t1)).0; // left
    }

    v
}

/// Calculates the coordinates of the point on a bezier curve at a given t
#[inline]
fn evaluate((p1, c1, c2, p2): CubicCurve, t: f32) -> Point {

  // Handle special case at beginning / end of curve
  if t < TOLERANCE || t > (1.0 - TOLERANCE) {
        let is_zero = t < TOLERANCE;
        let x = if is_zero { p1.x } else { p2.x };
        let y = if is_zero { p1.y } else { p2.y };
        Point::new(x, y)
  } else {
        // Calculate the polynomial coefficients.
        let cx = 3.0 * (c1.x - p1.x);
        let bx = 3.0 * (c2.x - c1.x) - cx;
        let ax = p2.x - p1.x - cx - bx;

        let cy = 3.0 * (c1.y - p1.y);
        let by = 3.0 * (c2.y - c1.y) - cy;
        let ay = p2.y - p1.y - cy - by;

        // Calculate the curve point at parameter value t
        let x = ((ax * t + bx) * t + cx) * t + p1.x;
        let y = ((ay * t + by) * t + cy) * t + p1.y;

        Point::new(x, y)
    }
}

/// Computes the intersections of two bezier curves
#[inline]
fn curve_intersections_inner(
    mut v1: CubicCurve,
    v2: CubicCurve,
    tMin: f32,
    tMax: f32,
    uMin: f32,
    uMax: f32,
    oldTDiff: f32,
    reverse: bool,
    recursion: usize,
    recursionLimit: usize,
    tLimit: f32
) -> Vec<CubicCubicIntersection> {

    // Avoid deeper recursion.
    // NOTE: @iconexperience determined that more than 20 recursions are
    // needed sometimes, depending on the tDiff threshold values further
    // below when determining which curve converges the least. He also
    // recommended a threshold of 0.5 instead of the initial 0.8
    // See: https:#github.com/paperjs/paper.js/issues/565
    if recursion > recursionLimit {
      return Vec::new();
    }

    // Let P be the first curve and Q be the second

    // Calculate the fat-line L for Q is the baseline l and two
    // offsets which completely encloses the curve P.
    let (dMin, dMax) = get_fatline(v2);

    // Calculate non-parametric bezier curve D(ti, di(t)) - di(t) is the
    // distance of P from the baseline l of the fat-line, ti is equally
    // spaced in [0, 1]
    let dp0 = signed_distance(v2.0.x, v2.0.y, v2.3.x, v2.3.y, v1.0.x, v1.0.y);
    let dp1 = signed_distance(v2.0.x, v2.0.y, v2.3.x, v2.3.y, v1.1.x, v1.1.y);
    let dp2 = signed_distance(v2.0.x, v2.0.y, v2.3.x, v2.3.y, v1.2.x, v1.2.y);
    let dp3 = signed_distance(v2.0.x, v2.0.y, v2.3.x, v2.3.y, v1.3.x, v1.3.y);

    // NOTE: the recursion threshold of 4 is needed to prevent issue #571
    // from occurring: https://github.com/paperjs/paper.js/issues/571
    let (tMinNew, tMaxNew, tDiff) = if v2.0.x == v2.3.x && uMax - uMin <= EPSILON && recursion > 4 {
        // The fatline of Q has converged to a point, the clipping is not
        // reliable. Return the value we have even though we will miss the
        // precision.
        let tNew = (tMax + tMin) / 2.0;
        (tNew, tNew, 0.0)
    } else {
        // Get the top and bottom parts of the convex-hull
        let [mut top, mut bottom] = convex_hull(dp0, dp1, dp2, dp3);

        // Clip the convex-hull with dMin and dMax
        let tMinClip = clip_convex_hull(&top, &bottom, dMin, dMax);
        top.reverse();
        bottom.reverse();
        let tMaxClip = clip_convex_hull(&top, &bottom, dMin, dMax);

        // No intersections if one of the tvalues are null or 'undefined'
        let (tMinClip, tMaxClip) = match (tMinClip, tMaxClip) {
            (Some(min), Some(max)) => (min, max),
            _ => return Vec::new(),
        };

        // Clip P with the fatline for Q
        v1 = get_part(v1, tMinClip, tMaxClip);

        // tMin and tMax are within the range (0, 1). We need to project it
        // to the original parameter range for v2.
        let tDiff = tMaxClip - tMinClip;
        let tMinNew = tMax * tMinClip + tMin * (1.0 - tMinClip);
        let tMaxNew = tMax * tMaxClip + tMin * (1.0 - tMaxClip);

        (tMinNew, tMaxNew, tDiff)
    };

    // Check if we need to subdivide the curves
    if oldTDiff > tLimit && tDiff > tLimit {
        // Subdivide the curve which has converged the least.
        if tMaxNew - tMinNew > uMax - uMin {
            let parts = subdivide(v1, 0.5);
            let t = tMinNew + (tMaxNew - tMinNew) / 2.0;
            let mut intersections = Vec::new();
            intersections.append(&mut curve_intersections_inner(v2, parts.0, uMin, uMax, tMinNew, t, tDiff, !reverse, recursion + 1, recursionLimit, tLimit));
            intersections.append(&mut curve_intersections_inner(v2, parts.1, uMin, uMax, t, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit));
            intersections
        } else {
            let parts = subdivide(v2, 0.5);
            let t = uMin + (uMax - uMin) / 2.0;
            let mut intersections = Vec::new();
            intersections.append(&mut curve_intersections_inner(parts.0, v1, uMin, t, tMinNew, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit));
            intersections.append(&mut curve_intersections_inner(parts.1, v1, t, uMax, tMinNew, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit));
            intersections
        }
    } else if (uMax - uMin).max(tMaxNew - tMinNew) < TOLERANCE {
        // We have isolated the intersection with sufficient precision
        let t1 = tMinNew + (tMaxNew - tMinNew) / 2.0;
        let t2 = uMin + (uMax - uMin) / 2.0;
        if reverse {
            vec![CubicCubicIntersection {
                t1: t2,
                curve1: v2,
                t2: t1,
                curve2: v1,
             }]
        } else {
            vec![CubicCubicIntersection {
                t1,
                curve1: v1,
                t2,
                curve2: v2,
            }]
        }
    } else {
        // Recurse
        curve_intersections_inner(v2, v1, uMin, uMax, tMinNew, tMaxNew, tDiff, !reverse, recursion + 1, recursionLimit, tLimit)
    }
}
