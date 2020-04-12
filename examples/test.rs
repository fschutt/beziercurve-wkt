extern crate beziercurve_wkt;

use beziercurve_wkt::BezierCurve;

fn main() {
    let curve1 = BezierCurve::from_str("BEZIERCURVE()").unwrap();
    let curve2 = BezierCurve::from_str("BEZIERCURVE()").unwrap();
    let intersection = curve1.intersect(&curve2);
}