extern crate beziercurve_wkt;

use beziercurve_wkt::BezierCurve;

fn main() {
    let curve1 = BezierCurve::from_str("BEZIERCURVE((60.0 40.0, 150.0 80.0, 500.0 400.0, 700.0 200.0))").unwrap().cache();
    let curve2 = BezierCurve::from_str("BEZIERCURVE((300.0 50.0, 400.0 450.0))").unwrap().cache();
    let intersections = curve1.get_intersections(&curve2);

    println!("intersections: [{}]", intersections.len());
    for (_, i) in intersections {
        println!("i: {:#?}", i);
    }
}