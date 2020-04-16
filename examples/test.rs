extern crate beziercurve_wkt;

use beziercurve_wkt::BezierCurve;
use beziercurve_wkt::Intersection::*;

const CURVE: &str = "BEZIERCURVE((0.0 1.0, 0.5 0.2, 1.0 0.0))";
const LINE: &str = "BEZIERCURVE((0.0 0.0, 2.0 1.0))";

fn main() {

    let curve1 = BezierCurve::from_str(CURVE).unwrap().cache();
    let curve2 = BezierCurve::from_str(LINE).unwrap().cache();
    let intersections = curve1.get_intersections(&curve2);

    for (_, i) in intersections {
        match i {
            LineLine(l)   => println!("line-line: ({:?})", l.get_intersection_point_1()),
            LineQuad(l)   => println!("line-quad: ({:?})", l.get_intersection_point_1()),
            LineCubic(l)  => println!("line-cubic: ({:?})", l.get_intersection_point_1()),
            QuadLine(l)   => println!("quad-line: ({:?})", l.get_intersection_point_1()),
            QuadQuad(l)   => println!("quad-quad: ({:?})", l.iter().map(|i| i.get_intersection_point_1()).collect::<Vec<_>>()),
            QuadCubic(l)  => println!("quad-cubic: ({:?})", l.iter().map(|i| i.get_intersection_point_1()).collect::<Vec<_>>()),
            CubicLine(c)  => println!("cubic-line: ({:?})", c.get_intersection_point_1()),
            CubicQuad(c)  => println!("cubic-quad: ({:?})", c.iter().map(|i| i.get_intersection_point_1()).collect::<Vec<_>>()),
            CubicCubic(c) => println!("cubic-cubic: {:?}", c.iter().map(|i| i.get_intersection_point_1()).collect::<Vec<_>>()),
        }
    }

    let clipped = curve1.clip(&curve2);
    for new_curve in clipped {
        println!("{}", new_curve);
    }
}