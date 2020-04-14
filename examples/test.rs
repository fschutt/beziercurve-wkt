extern crate beziercurve_wkt;

use beziercurve_wkt::BezierCurve;

fn main() {
    let curve1 = BezierCurve::from_str("BEZIERCURVE((0.0 1.0, 0.1 0.2, 1.0 0.0))").unwrap();
    let curve2 = BezierCurve::from_str("BEZIERCURVE((0.0 0.0, 1.0 1.0))").unwrap();
    let intersections = curve1.cache().get_intersections(&curve2.cache());

    println!("intersections: [{}]", intersections.len());

    for (_, i) in intersections {
        println!("i: {:#?}", i);
    }
}