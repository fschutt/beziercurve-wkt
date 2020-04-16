# beziercurve-wkt

## beziercurve-wkt

### About

This libary exists because PostGis (current version 12.0)
does not support Bezier curves. However, keeping data in tables
is still very useful. It's also sometimes necessary to have a
human-readable format of Bezier curves. So instead of keeping
the data in a binary format, the data is kept in string format
and this libary provides a serializer / deserializer for it.

The string format for Bezier curves looks like this:

```no_run,ignore
BEZIERCURVE((0.0 1.0, 2.0 1.0), (2.0 1.0, 46.0 20.0, 0.0 0.0), (0.0 0.0, 40.0, 47.0, 50.0 30.0, 2.0 1.0))
```

The parser looks at the points and determines the type of point
by its length:

```no_run,ignore
(x1 y1, x2 y2) -> Line from p1 to p2
(x1 y1, x2 y2, x3 y3) -> Quadratic bezier curve from p1 to p3 with control point p2
(x1 y1, x2 y2, x3 y3, x4 y4) -> Cubic bezier curve from p1 to p4 with control points p2 and p3
```

The reason for duplicating the point on each "item" / section
is so that the BezierCurve can be constructed in parallel, if necessary.

Additional to serialization / deserialization, this library features tools to:

- calculate the bounding box of a curve (necessary for calculating intersection of curves using a quadtree)
- calculate intersection(s) between curve-curve and curve-line
- calculate the angles of intersections (necessary for ex. to put texts on curves)
- cutting curves

## License

MIT

License: MIT
