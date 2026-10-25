// Interpolation functions: Catmull-Rom and quadratic and cubic Bezier.
//
// Part of the curves, paths and shapes; `../curves.js` gathers them.

export function CatmullRom( t, p0, p1, p2, p3 ) {

	const v0 = ( p2 - p0 ) * 0.5, v1 = ( p3 - p1 ) * 0.5, t2 = t * t, t3 = t * t2;
	return ( 2 * p1 - 2 * p2 + v0 + v1 ) * t3 + ( - 3 * p1 + 3 * p2 - 2 * v0 - v1 ) * t2 + v0 * t + p1;

}

export function QuadraticBezier( t, p0, p1, p2 ) {

	const k = 1 - t;
	return k * k * p0 + 2 * k * t * p1 + t * t * p2;

}

export function CubicBezier( t, p0, p1, p2, p3 ) {

	const k = 1 - t;
	return k * k * k * p0 + 3 * k * k * t * p1 + 3 * k * t * t * p2 + t * t * t * p3;

}
