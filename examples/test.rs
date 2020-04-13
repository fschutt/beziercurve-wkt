extern crate beziercurve_wkt;

use beziercurve_wkt::BezierCurve;

fn main() {
    let curve1 = BezierCurve::from_str("BEZIERCURVE((0.0 1.0, 0.2 0.1, 1.0 0.0))").unwrap().cache();
    let curve2 = BezierCurve::from_str("BEZIERCURVE((0.0 0.0, 1.0 1.0))").unwrap().cache();
    println!("curve1: {:#?}", curve1.get_curve());
    println!("curve2: {:#?}", curve2.get_curve());
    let intersections = curve1.get_intersections(&curve2);

    println!("intersections: [{}]", intersections.len());
    for (_, i) in intersections {
        println!("i: {:#?}", i);
    }
}