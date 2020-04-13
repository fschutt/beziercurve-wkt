extern crate beziercurve_wkt;

use beziercurve_wkt::BezierCurve;

fn main() {
    let curve1 = BezierCurve::from_str("BEZIERCURVE()").unwrap().cache();
    let curve2 = BezierCurve::from_str("BEZIERCURVE()").unwrap().cache();
    let intersections = curve1.get_intersections(&curve2);
    println!("intersections: {:?}", intersections);
}