use std::fmt;
use std::num::ParseFloatError;
use std::str::FromStr;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Point { pub x: f32, pub y: f32 }

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.x, self.y)
    }
}

impl Point {
    /// Parses a `Point` from a str
    ///
    /// ```rust
    /// let s = "5.0 7.0";
    /// let parsed = Point::from_str(s).unwrap();
    /// assert_eq!(parsed, Point { x: 5.0, y: 7.0 });
    /// ```
    pub fn from_str(s: &str) -> Result<Self, ParseError> {
        use std::f32;

        let s = s.trim();
        let mut number_iterator = s.split_whitespace();
        let x = number_iterator.next();
        let y = number_iterator.next();

        match (x, y) {
            (Some(x), Some(y)) => Ok(Point { x: f32::from_str(x)?, y: f32::from_str(y)? }),
            _ => Err(ParseError::FailedToParsePoint(s.to_string())),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Bbox {
    pub max_x: f32,
    pub max_y: f32,
    pub min_x: f32,
    pub min_y: f32,
}

impl fmt::Display for Bbox {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[x: {} - {}, y: {} - {}]", self.max_x, self.min_x, self.max_y, self.min_y)
    }
}

impl Bbox {
    /// Returns the height of the bbox
    #[inline]
    pub fn get_width(&self) -> f32 {
        self.max_x - self.min_x
    }

    /// Returns the height of the bbox
    #[inline]
    pub fn get_height(&self) -> f32 {
        self.max_y - self.min_y
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum BezierCurveItem {
    Line(Point, Point),
    QuadraticCurve(Point,Point, Point),
    CubicCurve(Point, Point, Point, Point),
}

impl fmt::Display for BezierCurveItem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::BezierCurveItem::*;
        match self {
            Line(p_start, p_end) => write!(f, "({}, {})", p_start, p_end),
            QuadraticCurve(p_start, control_1, p_end) => write!(f, "({}, {}, {})", p_start, control_1, p_end),
            CubicCurve(p_start, control_1, control_2, p_end) => write!(f, "({}, {}, {}, {})", p_start, control_1, control_2, p_end),
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
        pub fn $fn_name(&self) -> f32 {
            use self::BezierCurveItem::*;
            match self {
                Line(p_start, p_end) => {
                    p_start.$field.$max(p_end.$field)
                },
                QuadraticCurve(p_start, control_1, p_end) => {
                    p_start.$field.$max(p_end.$field).$max(control_1.$field)
                },
                CubicCurve(p_start, control_1, control_2, p_end) => {
                    p_start.$field.$max(p_end.$field).$max(control_1.$field).$max(control_2.$field)
                },
            }
        }
    };
}

impl BezierCurveItem {

    /// Returns to
    pub fn get_first_point(&self) -> Point {
        use self::BezierCurveItem::*;
        match self {
            Line(p_start, _) => *p_start,
            QuadraticCurve(p_start, _, _) => *p_start,
            CubicCurve(p_start, _, _, _) => *p_start,
        }
    }

    /// Returns the end point of the curve
    pub fn get_last_point(&self) -> Point {
        use self::BezierCurveItem::*;
        match self {
            Line(p_start, _) => *p_start,
            QuadraticCurve(p_start, _, _) => *p_start,
            CubicCurve(p_start, _, _, _) => *p_start,
        }
    }

    /// Parses the `BezierCurveItem` from a string
    ///
    /// ```rust
    /// let s1 = "(0.0 1.0, 2.0 1.0)";
    /// let parsed = BezierCurveItem::from_str(s1).unwrap();
    ///
    /// assert_eq!(parsed, BezierCurveItem::Line(Point { x: 0.0, y: 1.0 }, Point { x: 2.0, y: 1.0 }));
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
            (Some(a), Some(b), None, None) => Ok(Line(Point::from_str(a)?, Point::from_str(b)?)),
            (Some(a), Some(b), Some(c), None) => Ok(QuadraticCurve(Point::from_str(a)?, Point::from_str(b)?, Point::from_str(c)?)),
            (Some(a), Some(b), Some(c), Some(d)) => Ok(CubicCurve(Point::from_str(a)?, Point::from_str(b)?, Point::from_str(c)?, Point::from_str(d)?)),
            _ => Err(ParseError::TooFewPoints)
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
    /// let s = "BEZIERCURVE((0.0 1.0, 2.0 1.0), (2.0 1.0, 0.0 0.0), (0.0 0.0, 2.0 1.0))";
    /// let parsed_curve = BezierCurve::from_str(s).unwrap();
    ///
    /// assert_eq!(parsed_curve, BezierCurve { items: vec![
    ///      BezierCurveItem::Line(Point { x: 0.0, y: 1.0 }, Point { x: 2.0, y: 1.0 }),
    ///      BezierCurveItem::Line(Point { x: 2.0, y: 1.0 }, Point { x: 0.0, y: 0.0 }),
    ///      BezierCurveItem::Line(Point { x: 0.0, y: 0.0 }, Point { x: 2.0, y: 1.0 }),
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

        let mut s = &s[12..s.len() - 2];
        let mut items = Vec::new();
        let mut last_point = None;

        while let Some((characters_to_skip, character_was_found)) = skip_next_braces(&s, ',') {
            let next_item = if character_was_found { &s[..characters_to_skip] } else { &s[..] };
            let bezier_curve_item = BezierCurveItem::from_str(next_item).map_err(|e| FailedToParseItem(e, items.len()))?;
            if let Some(last) = last_point {
                if bezier_curve_item.get_first_point() != last {
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

        let mut max_x = 0.0_f32;
        let mut min_x = 0.0_f32;
        let mut max_y = 0.0_f32;
        let mut min_y = 0.0_f32;

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

fn intersect_bezier_curve() {

}

fn construct_bezier_curve_quadtree() {

}
