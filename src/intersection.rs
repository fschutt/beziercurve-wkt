#![allow(non_snake_case)]

//! Module for calculating curve-curve

use crate::{Point, Line, QuadraticCurve, CubicCurve};

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

// A line-curve intersection can intersect in up to 3 points
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum CubicLineIntersection {
    Intersect1 {
        curve: CubicCurve,
        line: Line,
        t_curve_1: f32,
        t_line_1: f32,
    },
    Intersect2 {
        curve: CubicCurve,
        line: Line,
        t_curve_1: f32,
        t_line_1: f32,
        t_curve_2: f32,
        t_line_2: f32,
    },
    Intersect3 {
        curve: CubicCurve,
        line: Line,
        t_curve_1: f32,
        t_line_1: f32,
        t_curve_2: f32,
        t_line_2: f32,
        t_curve_3: f32,
        t_line_3: f32,
    }
}

impl CubicLineIntersection {

    #[inline]
    pub fn get_intersection_point_1(&self) -> Point {
        use self::CubicLineIntersection::*;
        match self {
            Intersect1 { line, t_line_1, .. } => lerp(line.0, line.1, *t_line_1),
            Intersect2 { line, t_line_1, .. } => lerp(line.0, line.1, *t_line_1),
            Intersect3 { line, t_line_1, .. } => lerp(line.0, line.1, *t_line_1),
        }
    }

    #[inline]
    pub fn get_intersection_point_2(&self) -> Option<Point> {
        use self::CubicLineIntersection::*;
        match self {
            Intersect1 { .. } => None,
            Intersect2 { line, t_line_2, .. } => Some(lerp(line.0, line.1, *t_line_2)),
            Intersect3 { line, t_line_2, .. } => Some(lerp(line.0, line.1, *t_line_2)),
        }
    }

    #[inline]
    pub fn get_intersection_point_3(&self) -> Option<Point> {
        use self::CubicLineIntersection::*;
        match self {
            Intersect1 { .. } => None,
            Intersect2 { .. } => None,
            Intersect3 { line, t_line_3, .. } => Some(lerp(line.0, line.1, *t_line_3)),
        }
    }
}

#[inline]
fn lerp(p1: Point, p2: Point, t: f32) -> Point {
    let new_x = (1.0 - t) * p1.x + t * p2.x;
    let new_y = (1.0 - t) * p1.y + t * p2.y;
    Point::new(new_x, new_y)
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct BezierNormalVector { pub x: f32, pub y: f32 }

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct LineLineIntersection {
    pub t1: f32,
    pub line1: Line,
    pub t2: f32,
    pub line2: Line,
}

impl LineLineIntersection {

    #[inline]
    pub fn get_intersection_point_1(&self) -> Point {
        // lerp(line.0, line.1, t1)
        let new_x = (1.0 - self.t1) * self.line1.0.x + self.t1 * self.line1.1.x;
        let new_y = (1.0 - self.t1) * self.line1.0.y + self.t1 * self.line1.1.y;
        Point::new(new_x, new_y)
    }

    #[inline]
    pub fn get_intersection_point_2(&self) -> Point {
        // lerp(line.0, line.1, t2)
        let new_x = (1.0 - self.t2) * self.line2.0.x + self.t2 * self.line2.1.x;
        let new_y = (1.0 - self.t2) * self.line2.0.y + self.t2 * self.line2.1.y;
        Point::new(new_x, new_y)
    }
}

pub enum IntersectionResult {
    NoIntersection,
    LineLine(LineLineIntersection),
    CubicLine(CubicLineIntersection),
    CubicCubic(Vec<CubicCubicIntersection>),
    InfiniteIntersectionsCubicCubic(CubicCurve, CubicCurve),
    InfiniteIntersectionsCubicLine(CubicCurve, Line),
    InfiniteIntersectionsLineLine(Line, Line),
}

// Intersect a quadratic with another quadratic curve
pub fn curve_curve_intersect(a: CubicCurve, b: CubicCurve) -> IntersectionResult {

    if a == b {
        return IntersectionResult::InfiniteIntersectionsCubicCubic(a, b);
    }

    let intersections = curve_intersections_inner(a, b, 0.0, 1.0, 0.0, 1.0, 1.0, false, 0, 32, 0.8);

    if intersections.is_empty() {
        IntersectionResult::NoIntersection
    } else {
        IntersectionResult::CubicCubic(intersections)
    }
}

/// Calculates the normal vector at a certain point (perpendicular to the curve)
pub fn cubic_bezier_normal(curve: CubicCurve, t: f32) -> BezierNormalVector {

    // 1. Calculate the derivative of the bezier curve
    //
    // This means, we go from 4 control points to 3 control points and redistribute
    // the weights of the control points according to the formula:
    //
    // w'0 = 3(w1-w0)
    // w'1 = 3(w2-w1)
    // w'2 = 3(w3-w2)

    let weight_1_x = 3.0 * (curve.1.x - curve.0.x);
    let weight_1_y = 3.0 * (curve.1.y - curve.0.y);

    let weight_2_x = 3.0 * (curve.2.x - curve.1.x);
    let weight_2_y = 3.0 * (curve.2.y - curve.1.y);

    let weight_3_x = 3.0 * (curve.3.x - curve.2.x);
    let weight_3_y = 3.0 * (curve.3.y - curve.2.y);

    // The first derivative of a cubic bezier curve is a quadratic bezier curve
    // Luckily, the first derivative is also the tangent vector. So all we need to do
    // is to get the quadratic bezier
    let mut tangent = quadratic_interpolate_bezier((
        Point { x: weight_1_x, y: weight_1_y },
        Point { x: weight_2_x, y: weight_2_y },
        Point { x: weight_3_x, y: weight_3_y },
    ), t);

    // We normalize the tangent to have a lenght of 1
    let tangent_length = (tangent.x.powi(2) + tangent.y.powi(2)).sqrt();
    tangent.x /= tangent_length;
    tangent.y /= tangent_length;

    // The tangent is the vector that runs "along" the curve at a specific point.
    // To get the normal (to calcuate the rotation of the characters), we need to
    // rotate the tangent vector by 90 degrees.
    //
    // Rotating by 90 degrees is very simple, as we only need to flip the x and y axis

    BezierNormalVector {
        x: -tangent.y,
        y: tangent.x,
    }
}

#[inline]
fn quadratic_interpolate_bezier(curve: QuadraticCurve, t: f32) -> Point {
    let one_minus = 1.0 - t;
    let one_minus_square = one_minus.powi(2);

    let t_pow2 = t.powi(2);

    let x =         one_minus_square *             curve.0.x
            + 2.0 * one_minus        * t         * curve.1.x
            + 3.0                    * t_pow2    * curve.2.x;

    let y =         one_minus_square *             curve.0.y
            + 2.0 * one_minus        * t         * curve.1.y
            + 3.0                    * t_pow2    * curve.2.y;

    Point { x, y }
}

/// Intersect a cubic curve with a line.
///
/// Based on http://www.particleincell.com/blog/2013/cubic-line-intersection/
pub fn curve_line_intersect(
    (a1, a2, a3, a4): CubicCurve,
    (b1, b2): Line,
) -> IntersectionResult {

    if b1 == b2 {
        return IntersectionResult::NoIntersection;
    }

    let A = b2.y - b1.y; // A = y2 - y1
    let B = b1.x - b2.x; // B = x1 - x2
    let C = b1.x * (b1.y - b2.y) + b1.y * (b2.x - b1.x); // C = x1*(y1-y2)+y1*(x2-x1)

    let bx = bezier_coeffs(a1.x, a2.x, a3.x, a4.x);
    let by = bezier_coeffs(a1.y, a2.y, a3.y, a4.y);

    let p_0 = A * bx.0 + B * by.0;     // t^3
    let p_1 = A * bx.1 + B * by.1;     // t^2
    let p_2 = A * bx.2 + B * by.2;     // t
    let p_3 = A * bx.3 + B * by.3 + C; // 1

    let r = cubic_roots(p_0, p_1, p_2, p_3);

    let mut intersections = (None, None, None);

    // for root in r
    macro_rules! unroll_loop {($index:tt) => ({
        if let Some(t) = r.$index {

            let final_x = bx.0* t * t * t + bx.1 * t * t + bx.2 * t + bx.3;
            let final_y = by.0* t * t * t + by.1 * t * t + by.2 * t + by.3;

            // (final_x, final_y) is intersection point assuming infinitely long line segment,
            // make sure we are also in bounds of the line

            let x_dist = b2.x - b1.x;
            let y_dist = b2.y - b1.y;

            let t_line = if x_dist != 0.0 {
                // if not vertical line
                (final_x - b1.x) / x_dist
            } else {
                (final_y - b1.y) / y_dist
            };

            intersections.$index = if !t.is_sign_positive() || t > 1.0 || !t_line.is_sign_positive() || t_line > 1.0 {
                None
            } else {
                Some((t_line, t))
            }
        }
    })}

    unroll_loop!(0);
    unroll_loop!(1);
    unroll_loop!(2);

    use self::CubicLineIntersection::*;

    match intersections {
        (Some((t_line_1, t_curve_1)), None, None) =>
            IntersectionResult::CubicLine(Intersect1 {
                curve: (a1, a2, a3, a4),
                line: (b1, b2),
                t_curve_1,
                t_line_1,
            }),
        (Some((t_line_1, t_curve_1)), Some((t_line_2, t_curve_2)), None) =>
            IntersectionResult::CubicLine(Intersect2 {
                curve: (a1, a2, a3, a4),
                line: (b1, b2),
                t_curve_1,
                t_line_1,
                t_curve_2,
                t_line_2,
            }),
        (Some((t_line_1, t_curve_1)), Some((t_line_2, t_curve_2)), Some((t_line_3, t_curve_3))) =>
            IntersectionResult::CubicLine(Intersect3 {
                curve: (a1, a2, a3, a4),
                line: (b1, b2),
                t_curve_1,
                t_line_1,
                t_curve_2,
                t_line_2,
                t_curve_3,
                t_line_3,
            }),
        _ => IntersectionResult::NoIntersection,
    }
}

// based on http://mysite.verizon.net/res148h4j/javascript/script_exact_cubic.html#the%20source%20code
#[inline(always)]
fn cubic_roots(a: f32, b: f32, c: f32, d: f32) -> (Option<f32>, Option<f32>, Option<f32>) {

    use std::f32::consts::PI;

    // special case for linear and quadratic case
    if is_zero(a) {
        if is_zero(b) {
           // linear formula

           let p = -1.0 * (d / c);

           let ret = (
               if !p.is_sign_positive() || p > 1.0 { None } else { Some(p) },
               None,
               None
           );

           let ret = sort_special(ret);

           return ret;
        } else {
           // quadratic discriminant
           let d_q = c.powi(2) + 4.0 * b * d;

            if d_q.is_sign_positive() {
                let d_q = d_q.sqrt();

                let m = -1.0 * (d_q + c) / (2.0 * b);
                let n = (d_q - c) / (2.0 * b);

                let ret = (
                    if !m.is_sign_positive() || m > 1.0 { None } else { Some(m) },
                    if !n.is_sign_positive() || n > 1.0 { None } else { Some(n) },
                    None,
                );

                let ret = sort_special(ret);

                return ret;
            }
        }
    }

    let A = b / a;
    let B = c / a;
    let C = d / a;

    let Q = (3.0 * B - A.powi(2)) / 9.0;
    let R = (9.0 * A * B - 27.0 * C - 2.0 * A.powi(3)) / 54.0;
    let D = Q.powi(3) + R.powi(2); // polynomial discriminant

    let ret = if D.is_sign_positive() {

        // complex or duplicate roots
        const ONE_THIRD: f32 = 1.0 / 3.0;

        let D_sqrt = D.sqrt();
        let S = sign(R + D_sqrt) * (R + D_sqrt).abs().powf(ONE_THIRD);
        let T = sign(R - D_sqrt) * (R - D_sqrt).abs().powf(ONE_THIRD);

        let m = -A / 3.0 + (S + T);         // real root
        let n = -A / 3.0 - (S + T) / 2.0;   // real part of complex root
        let p = -A / 3.0 - (S + T) / 2.0;   // real part of complex root

        let mut ret = (
            if !m.is_sign_positive() || m > 1.0 { None } else { Some(m) },
            if !n.is_sign_positive() || n > 1.0 { None } else { Some(n) },
            if !p.is_sign_positive() || p > 1.0 { None } else { Some(p) },
        );

        let imaginary = (3.0_f32.sqrt() * (S - T) / 2.0).abs(); // complex part of root pair

        // discard complex roots
        if !is_zero(imaginary) {
            ret.1 = None;
            ret.2 = None;
        }

        ret
    } else {

        let th = (R / (-1.0 * Q.powi(3)).sqrt()).acos();
        let minus_q_sqrt = (-1.0 * Q).sqrt();

        let m = 2.0 * minus_q_sqrt * (th / 3.0).cos() - A / 3.0;
        let n = 2.0 * minus_q_sqrt * ((th + 2.0 * PI) / 3.0).cos() - A / 3.0;
        let p = 2.0 * minus_q_sqrt * ((th + 4.0 * PI) / 3.0).cos() - A / 3.0;

        // discard out of spec roots
        (
            if !m.is_sign_positive() || m > 1.0 { None } else { Some(m) },
            if !n.is_sign_positive() || n > 1.0 { None } else { Some(n) },
            if !p.is_sign_positive() || p > 1.0 { None } else { Some(p) },
        )
    };

    // sort but place None at the end
    let ret = sort_special(ret);

    ret
}

#[inline]
fn sign(a: f32) -> f32 {
    if a.is_sign_positive() { 1.0 } else { -1.0 }
}

#[inline]
fn bezier_coeffs(a: f32, b: f32, c: f32, d: f32) -> (f32, f32, f32, f32) {
    (
        -a + 3.0*b + -3.0*c + d,
        3.0*a - 6.0*b + 3.0*c,
        -3.0*a + 3.0*b,
        a
    )
}

type OptionTuple = (Option<f32>, Option<f32>, Option<f32>);

// Sort so that the None values are at the end
#[inline]
fn sort_special(a: OptionTuple) -> OptionTuple {
    match a {
        (None, None, None) => (None, None, None),
        (Some(a), None, None) |
        (None, Some(a), None) |
        (None, None, Some(a)) => (Some(a), None, None),
        (Some(a), Some(b), None) |
        (None, Some(a), Some(b)) |
        (Some(b), None, Some(a)) => (Some(a.min(b)), Some(a.max(b)), None),
        (Some(a), Some(b), Some(c)) => {
            let new_a = a.min(b).min(c);
            let new_b = if a < b && b < c { b } else if b < c && c < a { c } else { a };
            let new_c = a.max(b).max(c);
            (Some(new_a), Some(new_b), Some(new_c))
        }
    }
}


//  Determines the intersection point of the line defined by points A and B with the
//  line defined by points C and D.
//
//  Returns YES if the intersection point was found, and stores that point in X,Y.
//  Returns NO if there is no determinable intersection point, in which case X,Y will
//  be unmodified.
#[inline]
pub fn line_line_intersect(
    (a, b): Line,
    (c, d): Line,
) -> IntersectionResult {

    if (a, b) == (c, d) {
        return IntersectionResult::InfiniteIntersectionsLineLine((a, b), (c, d));
    }

    // Check if both points of the line are the same
    if a == b || c == d {
        return IntersectionResult::NoIntersection;
    }

    let (original_a, original_b) = (a, b);
    let (original_c, original_d) = (c, d);

    //  (1) Translate the system so that point A is on the origin.
    let b = Point::new(b.x - a.x, b.y - a.y);
    let mut c = Point::new(c.x - a.x, c.y - a.y);
    let mut d = Point::new(d.x - a.x, d.y - a.y);

    // Get the length from a to b
    let dist_ab = (b.x*b.x + b.y*b.y).sqrt();

    // Rotate the system so that point B is on the positive X axis.
    let cos_b = b.x / dist_ab;
    let sin_b = b.y / dist_ab;

    // Rotate c and d around b
    let new_x = c.x * cos_b + c.y * sin_b;
    c.y = c.y * cos_b - c.x * sin_b;
    c.x = new_x;

    let new_x = d.x * cos_b + d.y * sin_b;
    d.y = d.y * cos_b - d.x * sin_b;
    d.x = new_x;

    // Fail if the lines are parallel
    if c.y == d.y {
        return IntersectionResult::NoIntersection;
    }

    // Calculate the position of the intersection point along line A-B.
    let t = d.x + (c.x - d.x) * d.y / (d.y - c.y);

    let new_x = a.x + t * cos_b;
    let new_y = a.y + t * cos_b;

    // The t projected onto the line a - b
    let t1 = ((b.x - a.x) / (new_x - a.x) + (b.y - a.y) / (new_y - a.y)) / 2.0;
    // The t projected onto the line b - c
    let t2 = ((d.x - c.x) / (new_x - c.x) + (d.y - c.y) / (new_y - c.y)) / 2.0;

    IntersectionResult::LineLine(LineLineIntersection {
        t1,
        line1: (original_a, original_b),
        t2,
        line2: (original_c, original_d),
    })
}

// Convert a quadratic bezier into a cubic bezier
#[inline]
pub fn quadratic_to_cubic_curve(c: QuadraticCurve) -> CubicCurve {
    const TWO_THIRDS: f32 = 2.0 / 3.0;

    let c1_x = c.0.x + TWO_THIRDS * (c.1.x - c.0.x);
    let c1_y = c.0.y + TWO_THIRDS * (c.1.y - c.0.y);

    let c2_x = c.2.x + TWO_THIRDS * (c.1.x - c.2.x);
    let c2_y = c.2.y + TWO_THIRDS * (c.1.y - c.2.y);

    (c.0, Point::new(c1_x, c1_y), Point::new(c2_x, c2_y), c.2)
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
