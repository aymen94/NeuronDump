// Planes, rays, frustums, segments, triangles, spherical and cylindrical coordinates over engine values, and box helpers.
//
// Part of the math classes; `../math.js` gathers them.

import * as W from '../../index.js';
import { WebGLCoordinateSystem, WebGPUCoordinateSystem } from '../constants.js';
import { Vector3 } from '../values.js';
import { extend, free, toEngine } from './boundary.js';

export function boxNumbers( box ) {

	const min = box.min, max = box.max;
	return [ min.x, min.y, min.z, max.x, max.y, max.z ];

}

// Grows `bounds` (min xyz, max xyz) by the box `b` transformed by the column-major
// matrix at `m[ o ]`: the box around its eight transformed corners, as Box3.applyMatrix4 does.
export function expandByTransformedBox( bounds, b, m, o = 0 ) {

	if ( b[ 0 ] > b[ 3 ] || b[ 1 ] > b[ 4 ] || b[ 2 ] > b[ 5 ] ) return;

	for ( let corner = 0; corner < 8; corner ++ ) {

		const x = b[ corner & 1 ? 3 : 0 ], y = b[ corner & 2 ? 4 : 1 ], z = b[ corner & 4 ? 5 : 2 ];
		const px = m[ o ] * x + m[ o + 4 ] * y + m[ o + 8 ] * z + m[ o + 12 ];
		const py = m[ o + 1 ] * x + m[ o + 5 ] * y + m[ o + 9 ] * z + m[ o + 13 ];
		const pz = m[ o + 2 ] * x + m[ o + 6 ] * y + m[ o + 10 ] * z + m[ o + 14 ];
		if ( px < bounds[ 0 ] ) bounds[ 0 ] = px;
		if ( py < bounds[ 1 ] ) bounds[ 1 ] = py;
		if ( pz < bounds[ 2 ] ) bounds[ 2 ] = pz;
		if ( px > bounds[ 3 ] ) bounds[ 3 ] = px;
		if ( py > bounds[ 4 ] ) bounds[ 4 ] = py;
		if ( pz > bounds[ 5 ] ) bounds[ 5 ] = pz;

	}

}

// Sets `box` to six numbers, or empties it when they describe no box.
export function setBox( box, bounds ) {

	if ( bounds[ 0 ] > bounds[ 3 ] || bounds[ 1 ] > bounds[ 4 ] || bounds[ 2 ] > bounds[ 5 ] ) return box.makeEmpty();
	box.min.set( bounds[ 0 ], bounds[ 1 ], bounds[ 2 ] );
	box.max.set( bounds[ 3 ], bounds[ 4 ], bounds[ 5 ] );
	return box;

}

// Constructs an engine class from JavaScript values.
function construct( Base, self, args ) {

	const temporaries = [];
	const value = Reflect.construct( Base, args.map( ( a ) => toEngine( a, temporaries ) ), self );
	free( temporaries );
	return value;

}

export class Plane extends W.Plane {

	constructor( normal = new Vector3( 1, 0, 0 ), constant = 0 ) {

		return construct( W.Plane, new.target, [ normal, constant ] );

	}

}

export class Ray extends W.Ray {

	constructor( origin = new Vector3(), direction = new Vector3( 0, 0, - 1 ) ) {

		return construct( W.Ray, new.target, [ origin, direction ] );

	}

}

export class Frustum extends W.Frustum {

	setFromProjectionMatrix( m, coordinateSystem = WebGLCoordinateSystem, reversedDepth = false ) {

		const temporaries = [];
		super.setFromProjectionMatrix( toEngine( m, temporaries ), coordinateSystem === WebGPUCoordinateSystem, reversedDepth );
		free( temporaries );
		return this;

	}

}

/** A segment from `start` to `end`. `start` and `end` read as copies; change them with `set`. */
export class Line3 extends W.Line3 {

	constructor( start = new Vector3(), end = new Vector3() ) {

		return construct( W.Line3, new.target, [ start, end ] );

	}

}

// A temporary triangle for the static helpers, freed after use.
function withTriangle( a, b, c, use ) {

	const triangle = new Triangle( a, b, c );

	try {

		return use( triangle );

	} finally {

		triangle.free();

	}

}

/** `a`, `b` and `c` read as copies; change them with `set`. */
export class Triangle extends W.Triangle {

	constructor( a = new Vector3(), b = new Vector3(), c = new Vector3() ) {

		return construct( W.Triangle, new.target, [ a, b, c ] );

	}

	setFromPointsAndIndices( points, i0, i1, i2 ) {

		return this.set( points[ i0 ], points[ i1 ], points[ i2 ] );

	}

	setFromAttributeAndIndices( attribute, i0, i1, i2 ) {

		const vertex = ( i ) => new Vector3( attribute.getX( i ), attribute.getY( i ), attribute.getZ( i ) );
		return this.set( vertex( i0 ), vertex( i1 ), vertex( i2 ) );

	}

	/** `v1`, `v2`, `v3` weighted by the barycentric coordinates of `point`; null for a degenerate triangle. */
	getInterpolation( point, v1, v2, v3, target ) {

		const weights = this.getBarycoord( point );
		if ( weights === null ) return null;
		target.setScalar( 0 );
		target.addScaledVector( v1, weights.x );
		target.addScaledVector( v2, weights.y );
		target.addScaledVector( v3, weights.z );
		return target;

	}

	static getNormal( a, b, c, target ) {

		return withTriangle( a, b, c, ( t ) => t.getNormal( target ) );

	}

	static getBarycoord( point, a, b, c, target ) {

		return withTriangle( a, b, c, ( t ) => t.getBarycoord( point, target ) );

	}

	static containsPoint( point, a, b, c ) {

		return withTriangle( a, b, c, ( t ) => t.containsPoint( point ) );

	}

	static getInterpolation( point, p1, p2, p3, v1, v2, v3, target ) {

		return withTriangle( p1, p2, p3, ( t ) => t.getInterpolation( point, v1, v2, v3, target ) );

	}

	static isFrontFacing( a, b, c, direction ) {

		return withTriangle( a, b, c, ( t ) => t.isFrontFacing( direction ) );

	}

}

/** Radius, polar angle `phi` from +Y and azimuth `theta` around Y. */
export class Spherical extends W.Spherical {

	constructor( radius = 1, phi = 0, theta = 0 ) {

		super( radius, phi, theta );

	}

}

/** Distance `radius` from the Y axis, angle `theta` around it, and height `y`. */
export class Cylindrical extends W.Cylindrical {

	constructor( radius = 1, theta = 0, y = 0 ) {

		super( radius, theta, y );

	}

}

for ( const [ Compat, Base, name ] of [
	[ Plane, W.Plane, 'Plane' ], [ Ray, W.Ray, 'Ray' ], [ Frustum, W.Frustum, 'Frustum' ],
	[ Line3, W.Line3, 'Line3' ], [ Triangle, W.Triangle, 'Triangle' ], [ Spherical, W.Spherical, 'Spherical' ], [ Cylindrical, W.Cylindrical, 'Cylindrical' ]
] ) extend( Compat, Base, name );
