//! # beziercurve-wkt
//!
//! ## About
//!
//! This libary exists because PostGis (current version 12.0)
//! does not support Bezier curves. However, keeping data in tables
//! is still very useful. It's also sometimes necessary to have a
//! human-readable format of Bezier curves. So instead of keeping
//! the data in a binary format, the data is kept in string format
//! and this libary provides a serializer / deserializer for it.
//!
//! The string format for Bezier curves looks like this:
//!
//! ```no_run,ignore
//! BEZIERCURVE((0.0 1.0, 2.0 1.0), (2.0 1.0, 46.0 20.0, 0.0 0.0), (0.0 0.0, 40.0, 47.0, 50.0 30.0, 2.0 1.0))
//! ```
//!
//! The parser looks at the points and determines the type of point
//! by its length:
//!
//! ```no_run,ignore
//! (x1 y1, x2 y2) -> Line from p1 to p2
//! (x1 y1, x2 y2, x3 y3) -> Quadratic bezier curve from p1 to p3 with control point p2
//! (x1 y1, x2 y2, x3 y3, x4 y4) -> Cubic bezier curve from p1 to p4 with control points p2 and p3
//! ```
//!
//! The reason for duplicating the point on each "item" / section
//! is so that the BezierCurve can be constructed in parallel, if necessary.
//!
//! Additional to serialization / deserialization, this library features tools to:
//!
//! - calculate the bounding box of a curve (necessary for calculating intersection of curves using a quadtree)
//!
//! # Work in progress
//!
//! - calculate intersection(s) between curve-curve and curve-line
//! - calculate the angles of intersections (necessary for ex. to put texts on curves)
//! - cutting curves
//!
//! # License
//!
//! MIT

extern crate quadtree_f32;
#[cfg(feature = "parallel")]
extern crate rayon;

use std::{
    fmt,
    num::ParseFloatError,
    str::FromStr,
    ops::{Index, IndexMut},
};
use quadtree_f32::{Rect, QuadTree, ItemId, Item};
#[cfg(feature = "parallel")]
use rayon::iter::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};

mod intersection;

pub use intersection::{IntersectionResult, Intersection, InfiniteIntersections};
pub use intersection::BezierNormalVector;

pub type Line           = (Point, Point);
pub type QuadraticCurve = (Point, Point, Point);
pub type CubicCurve     = (Point, Point, Point, Point);

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Point { pub x: f64, pub y: f64 }

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.x, self.y)
    }
}

impl Point {

    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Parses a `Point` from a str
    ///
    /// ```rust
    /// # use beziercurve_wkt::Point;
    /// let s = "5.0 7.0";
    /// let parsed = Point::from_str(s).unwrap();
    /// assert_eq!(parsed, Point { x: 5.0, y: 7.0 });
    /// ```
    pub fn from_str(s: &str) -> Result<Self, ParseError> {
        use std::f64;

        let s = s.trim();
        let mut number_iterator = s.split_whitespace();
        let x = number_iterator.next();
        let y = number_iterator.next();

        match (x, y) {
            (Some(x), Some(y)) => Ok(Point::new(f64::from_str(x)?, f64::from_str(y)?)),
            _ => Err(ParseError::FailedToParsePoint(s.to_string())),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Bbox {
    pub max_x: f64,
    pub max_y: f64,
    pub min_x: f64,
    pub min_y: f64,
}

impl fmt::Display for Bbox {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[x: {} - {}, y: {} - {}]", self.max_x, self.min_x, self.max_y, self.min_y)
    }
}

impl Bbox {
    /// Returns the height of the bbox
    #[inline]
    pub fn get_width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Returns the height of the bbox
    #[inline]
    pub fn get_height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn overlaps(&self, other: Bbox) -> bool {
        translate_bbox(*self).overlaps_rect(&translate_bbox(other))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum BezierCurveItem {
    Line(Line),
    QuadraticCurve(QuadraticCurve),
    CubicCurve(CubicCurve),
}

impl fmt::Display for BezierCurveItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::BezierCurveItem::*;
        match self {
            Line((p_start, p_end)) => write!(f, "({}, {})", p_start, p_end),
            QuadraticCurve((p_start, control_1, p_end)) => write!(f, "({}, {}, {})", p_start, control_1, p_end),
            CubicCurve((p_start, control_1, control_2, p_end)) => write!(f, "({}, {}, {}, {})", p_start, control_1, control_2, p_end),
        }
    }
}

macro_rules! get_max_fn {
    ($fn_name:ident, $field:ident, $max:ident, $field_str:expr, $max_str:expr) => {
        /// Returns the
        #[doc = $field_str]
        ///
        #[doc = $max_str]
        /// value of the `BezierCurveItem` - useful for calculating bounding boxes
        #[inline]
        pub fn $fn_name(&self) -> f64 {
            use self::BezierCurveItem::*;
            match self {
                Line((p_start, p_end)) => {
                    p_start.$field.$max(p_end.$field)
                },
                QuadraticCurve((p_start, control_1, p_end)) => {
                    p_start.$field.$max(p_end.$field).$max(control_1.$field)
                },
                CubicCurve((p_start, control_1, control_2, p_end)) => {
                    p_start.$field.$max(p_end.$field).$max(control_1.$field).$max(control_2.$field)
                },
            }
        }
    };
}

impl BezierCurveItem {

    /// Returns the start point of the curve
    pub fn get_first_point(&self) -> Point {
        use self::BezierCurveItem::*;
        match self {
            Line((p_start, _)) => *p_start,
            QuadraticCurve((p_start, _, _)) => *p_start,
            CubicCurve((p_start, _, _, _)) => *p_start,
        }
    }

    /// Returns the end point of the curve
    pub fn get_last_point(&self) -> Point {
        use self::BezierCurveItem::*;
        match self {
            Line((_, p_end)) => *p_end,
            QuadraticCurve((_, _, p_end)) => *p_end,
            CubicCurve((_, _, _, p_end)) => *p_end,
        }
    }

    /// Parses the `BezierCurveItem` from a string
    ///
    /// ```rust
    /// # use beziercurve_wkt::{BezierCurveItem, BezierCurveItem::*, Point};
    /// let parsed1 = BezierCurveItem::from_str("(0.0 1.0, 2.0 1.0)").unwrap();
    /// assert_eq!(parsed1, Line((Point { x: 0.0, y: 1.0 }, Point { x: 2.0, y: 1.0 })));
    ///
    /// let parsed2 = BezierCurveItem::from_str("(0.0 1.0, 2.0 1.0, 3.0 4.0)").unwrap();
    /// assert_eq!(parsed2, QuadraticCurve((Point { x: 0.0, y: 1.0 }, Point { x: 2.0, y: 1.0 }, Point { x: 3.0, y: 4.0 })));
    /// ```
    pub fn from_str(s: &str) -> Result<Self, ParseError> {
        use self::BezierCurveItem::*;

        let s = s.trim();
        if !(s.starts_with("(") && s.ends_with(")")) {
            return Err(ParseError::NoEnclosingBraces);
        }

        let s = &s[1..s.len() - 2]; // remove the two outer braces
        if s.chars().any(|c| c == '(' || c == ')') {
            return Err(ParseError::InternalBracesDetected);
        }

        let mut point_iterator = s.split(",");
        let a = point_iterator.next();
        let b = point_iterator.next();
        let c = point_iterator.next();
        let d = point_iterator.next();

        if point_iterator.next().is_some() {
            return Err(ParseError::TooManyPoints);
        }

        match (a,b,c,d) {
            (Some(a), Some(b), None, None)          => Ok(Line((Point::from_str(a)?, Point::from_str(b)?))),
            (Some(a), Some(b), Some(c), None)       => Ok(QuadraticCurve((Point::from_str(a)?, Point::from_str(b)?, Point::from_str(c)?))),
            (Some(a), Some(b), Some(c), Some(d))    => Ok(CubicCurve((Point::from_str(a)?, Point::from_str(b)?, Point::from_str(c)?, Point::from_str(d)?))),
            _                                       => Err(ParseError::TooFewPoints),
        }
    }

    get_max_fn!(get_max_x, x, max, "x", "max");
    get_max_fn!(get_min_x, x, min, "x", "min");
    get_max_fn!(get_max_y, y, max, "y", "max");
    get_max_fn!(get_min_y, y, min, "y", "min");

    /// Returns the bounding box of this item
    pub fn get_bbox(&self) -> Bbox {
        Bbox {
           max_x: self.get_max_x(),
           max_y: self.get_max_y(),
           min_x: self.get_min_x(),
           min_y: self.get_min_y(),
        }
    }

    /// Returns the intersection of two items (line-curve, curve-curve or line-line intersection).
    ///
    /// Warning: calling this function is expensive, it's recommended to cull
    /// items that don't intersect first by intersecting their bounding boxes.
    pub fn intersect(&self, curve: &Self) -> IntersectionResult {
        use self::BezierCurveItem::*;
        use crate::intersection::*;

        // Preliminary test if the bounding boxes overlap
        let self_bbox = translate_bbox(self.get_bbox());
        let curve_bbox = translate_bbox(curve.get_bbox());
        if !self_bbox.overlaps_rect(&curve_bbox) {
            return IntersectionResult::NoIntersection;
        }

        match (self, curve) {
            (Line(l1), Line(l2))                     => line_line_intersect(*l1, *l2),
            (QuadraticCurve(q1), QuadraticCurve(q2)) => curve_curve_intersect(quadratic_to_cubic_curve(*q1), quadratic_to_cubic_curve(*q2)),
            (CubicCurve(c1), CubicCurve(c2))         => curve_curve_intersect(*c1, *c2),

            (Line(l1), QuadraticCurve(q1))           => curve_line_intersect(quadratic_to_cubic_curve(*q1), *l1),
            (Line(l1), CubicCurve(c1))               => curve_line_intersect(*c1, *l1),

            (QuadraticCurve(q1), Line(l1))           => curve_line_intersect(quadratic_to_cubic_curve(*q1), *l1),
            (QuadraticCurve(q1), CubicCurve(c1))     => curve_curve_intersect(quadratic_to_cubic_curve(*q1), *c1),

            (CubicCurve(c1), Line(l1))               => curve_line_intersect(*c1, *l1),
            (CubicCurve(c1), QuadraticCurve(q1))     => curve_curve_intersect(*c1, quadratic_to_cubic_curve(*q1)),
        }
    }

    /// Returns the normal of the curve / line at t
    pub fn normal(&self, t: f64) -> BezierNormalVector {
        use self::BezierCurveItem::*;
        use crate::intersection::*;
        match self {
            Line(l) => {
                // calculate the rise / run, then simply
                // inverse the axis to rotate by 90 degrees
                let diff_x = l.1.x - l.0.x;
                let diff_y = l.1.y - l.0.y;
                let line_length = (diff_x.powi(2) + diff_y.powi(2)).sqrt();
                BezierNormalVector {
                    x: -diff_y / line_length,
                    y: diff_x / line_length,
                }
            },
            QuadraticCurve(q) => cubic_bezier_normal(quadratic_to_cubic_curve(*q), t),
            CubicCurve(c) => cubic_bezier_normal(*c, t),
        }
    }
}

const fn translate_bbox(bbox: Bbox) -> Rect {
    Rect {
       max_x: bbox.max_x as f32,
       max_y: bbox.max_y as f32,
       min_x: bbox.min_x as f32,
       min_y: bbox.min_y as f32,
    }
}

const fn translate_rect(rect: Rect) -> Bbox {
    Bbox {
       max_x: rect.max_x as f64,
       max_y: rect.max_y as f64,
       min_x: rect.min_x as f64,
       min_y: rect.min_y as f64,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseErrorWithContext {
    /// For each subsequent items `a` and `b` in the curve, `a.get_last_point()` has to match `b.get_first_point()`
    BrokenBezierCurve(usize),
    /// String is not enclosed by the mandatory `BEZIERCURVE()` braces
    NoEnclosingBezierCurve,
    /// Failed to parse error [i]
    FailedToParseItem(ParseError, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    TooManyPoints,
    TooFewPoints,
    NoEnclosingBraces,
    InternalBracesDetected,
    FailedToParsePoint(String),
    F32ParseError(ParseFloatError),
}

impl From<ParseFloatError> for ParseError {
    fn from(e: ParseFloatError) -> Self {
        ParseError::F32ParseError(e)
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BezierCurve {
    pub items: Vec<BezierCurveItem>,
}

impl fmt::Display for BezierCurve {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BEZIERCURVE(")?;
        for i in &self.items {
            i.fmt(f)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl BezierCurve {

    /// Parses a `BezierCurve` from a str
    ///
    /// ```rust
    /// # use beziercurve_wkt::{BezierCurve, BezierCurveItem, Point};
    /// let s = "BEZIERCURVE((0.0 1.0, 2.0 1.0), (2.0 1.0, 3.0 4.0, 0.0 0.0), (0.0 0.0, 0.0 1.0))";
    /// let parsed_curve = BezierCurve::from_str(s).unwrap();
    ///
    /// assert_eq!(parsed_curve, BezierCurve { items: vec![
    ///      BezierCurveItem::Line((Point { x: 0.0, y: 1.0 }, Point { x: 2.0, y: 1.0 })),
    ///      BezierCurveItem::QuadraticCurve((Point { x: 2.0, y: 1.0 }, Point { x: 3.0, y: 4.0 }, Point { x: 0.0, y: 0.0 })),
    ///      BezierCurveItem::Line((Point { x: 0.0, y: 0.0 }, Point { x: 0.0, y: 1.0 })),
    /// ]});
    ///
    /// assert!(parsed_curve.is_closed());
    /// ```
    pub fn from_str(s: &str) -> Result<Self, ParseErrorWithContext> {
        use self::ParseErrorWithContext::*;

        let s = s.trim();
        if !(s.starts_with("BEZIERCURVE(") && s.ends_with(")")) {
            return Err(NoEnclosingBezierCurve);
        }

        let mut s = &s[12..s.len() - 1];
        let mut items = Vec::new();
        let mut last_point = None;

        while let Some((characters_to_skip, character_was_found)) = skip_next_braces(&s, ',') {
            let next_item = if character_was_found { &s[..characters_to_skip] } else { &s[..] };
            let bezier_curve_item =
                BezierCurveItem::from_str(next_item)
                .map_err(|e| FailedToParseItem(e, items.len()))?;

            if let Some(last) = last_point {
                if bezier_curve_item.get_first_point() != last {
                    println!("last point: {:?}, current point {:?}", last, bezier_curve_item.get_first_point());
                    return Err(BrokenBezierCurve(items.len()));
                }
            }
            last_point = Some(bezier_curve_item.get_last_point());
            items.push(bezier_curve_item);
            s = &s[(characters_to_skip + 1)..];
            if !character_was_found {
                break;
            }
        }

        Ok(Self { items })
    }

    pub fn get_bbox(&self) -> Bbox {

        let mut max_x = 0.0_f64;
        let mut min_x = 0.0_f64;
        let mut max_y = 0.0_f64;
        let mut min_y = 0.0_f64;

        for i in &self.items {
            max_x = max_x.max(i.get_max_x());
            min_x = min_x.min(i.get_min_x());
            max_y = max_y.max(i.get_max_y());
            min_y = min_y.min(i.get_min_y());
        }

        Bbox { max_x, min_x, min_y, max_y }
    }

    /// Returns whether the bezier curve is closed
    pub fn is_closed(&self) -> bool {
        match self.items.len() {
            0 => false,
            1 => self.items[0].get_first_point() == self.items[0].get_last_point(),
            n => self.items[0].get_first_point() == self.items[n - 1].get_last_point(),
        }
    }

    /// Builds a quadtree-based cache from the current bezier curve
    pub fn cache(self) -> BezierCurveCache {
        BezierCurveCache::new(self)
    }
}

// Index of the curve item in the curve.items
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CurveIndex(usize);

impl Index<CurveIndex> for BezierCurve {
    type Output = BezierCurveItem;
    fn index(&self, i: CurveIndex) -> &BezierCurveItem {
        &self.items[i.0]
    }
}

impl IndexMut<CurveIndex> for BezierCurve {
    fn index_mut(&mut self, i: CurveIndex) -> &mut BezierCurveItem {
        &mut self.items[i.0]
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct BezierCurveCache {
    curve: BezierCurve,
    quad_tree: QuadTree,
}

const DEFAULT_VEC: Vec<(CurveIndex, Intersection)> = Vec::new();

impl BezierCurveCache {

    /// Creates a new quadtree + bezier cache from a bezier curve
    #[cfg(feature = "parallel")]
    #[inline]
    pub fn new(curve: BezierCurve) -> Self {

        let items = curve.items.par_iter().enumerate().map(|(id, i)| {
            (ItemId(id), Item::Rect(translate_bbox(i.get_bbox())))
        }).collect::<Vec<_>>();

        let quad_tree = QuadTree::new(items.into_iter());

        Self { curve, quad_tree }
    }

    /// Creates a new quadtree + bezier cache from a bezier curve
    #[cfg(not(feature = "parallel"))]
    #[inline]
    pub fn new(curve: BezierCurve) -> Self {

        let quad_tree = QuadTree::new(curve.items.iter().enumerate().map(|(id, i)| {
            (ItemId(id), Item::Rect(translate_bbox(i.get_bbox())))
        }));

        Self { curve, quad_tree }
    }

    #[inline]
    pub fn get_bbox(&self) -> Bbox {
        translate_rect(self.quad_tree.bbox())
    }

    #[inline]
    pub fn get_curve(&self) -> &BezierCurve {
        &self.curve
    }

    #[inline]
    pub fn get_curve_part(&self, i: CurveIndex) -> Option<&BezierCurveItem> {
        self.curve.items.get(i.0)
    }

    #[inline]
    pub fn get_curve_part_mut(&mut self, i: CurveIndex) -> Option<&mut BezierCurveItem> {
        self.curve.items.get_mut(i.0)
    }

    #[cfg(not(feature = "parallel"))]
    #[inline]
    pub fn get_intersections(&self, curve: &Self) -> Vec<(CurveIndex, Intersection)> {
        if curve.curve.items.is_empty() || !self.get_bbox().overlaps(curve.get_bbox()) {
            return DEFAULT_VEC;
        }

        curve.curve.items
        .iter()
        .flat_map(|item| self.get_intersections_with_item(item).into_iter())
        .collect()
    }

    #[cfg(feature = "parallel")]
    #[inline]
    pub fn get_intersections(&self, curve: &Self) -> Vec<(CurveIndex, Intersection)> {
        if curve.curve.items.is_empty() || !self.get_bbox().overlaps(curve.get_bbox()) {
            return DEFAULT_VEC;
        }

        let result = curve.curve.items
        .par_iter()
        .map(|item| self.get_intersections_with_item(item))
        .collect::<Vec<Vec<_>>>();

        result.into_iter().flat_map(|i| i).collect()
    }

    /// Returns the intersection with a single bezier curve item
    #[cfg(feature = "parallel")]
    #[inline]
    pub fn get_intersections_with_item(&self, curve: &BezierCurveItem) -> Vec<(CurveIndex, Intersection)> {
        use intersection::IntersectionResult::*;

        let curve_bbox = translate_bbox(curve.get_bbox());

        if !curve_bbox.overlaps_rect(&self.quad_tree.bbox()) {
            return DEFAULT_VEC;
        }

        let curve_part_ids = self.quad_tree.get_ids_that_overlap(&curve_bbox);

        curve_part_ids
        .par_iter()
        .filter_map(|id| {
            self.curve.items.get(id.0)
            .map(|part| (CurveIndex(id.0), part))
        })
        .filter_map(|(id, item)| match item.intersect(curve) {
            // ignore InfiniteIntersections and NoIntersection
            NoIntersection | Infinite(_) => None,
            FoundIntersection(i) => Some((id, i))
        })
        .collect()
    }

    /// Returns the intersection with a single bezier curve item
    #[cfg(not(feature = "parallel"))]
    #[inline]
    pub fn get_intersections_with_item(&self, curve: &BezierCurveItem) -> Vec<(CurveIndex, Intersection)> {
        use intersection::IntersectionResult::*;

        let curve_bbox = translate_bbox(curve.get_bbox());

        if !curve_bbox.overlaps_rect(&self.quad_tree.bbox()) {
            return DEFAULT_VEC;
        }

        let curve_part_ids = self.quad_tree.get_ids_that_overlap(&curve_bbox);

        curve_part_ids
        .iter()
        .filter_map(|id| {
            self.curve.items.get(id.0)
            .map(|part| (CurveIndex(id.0), part))
        })
        .filter_map(|(id, item)| match item.intersect(curve) {
            // ignore InfiniteIntersections and NoIntersection
            NoIntersection | Infinite(_) => None,
            FoundIntersection(i) => Some((id, i))
        })
        .collect()
    }
}

/// Given a string, returns how many characters need to be skipped
fn skip_next_braces(input: &str, target_char: char) -> Option<(usize, bool)> {

    let mut depth = 0;
    let mut last_character = 0;
    let mut character_was_found = false;

    if input.is_empty() {
        return None;
    }

    for (idx, ch) in input.char_indices() {
        last_character = idx;
        match ch {
            '(' => { depth += 1; },
            ')' => { depth -= 1; },
            c => {
                if c == target_char && depth == 0 {
                    character_was_found = true;
                    break;
                }
            },
        }
    }

    if last_character == 0 {
        // No more split by `,`
        None
    } else {
        Some((last_character, character_was_found))
    }
}

