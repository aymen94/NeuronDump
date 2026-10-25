// toJSON and fromJSON of the curve types, in three.js's JSON layout; paths and shapes use
// `curveFromJSON` for their segments.
//
// Part of the curves; `../curves.js` gathers them.

import { Vector2, Vector3 } from '../math.js';
import { Curve } from './curve.js';
import { CatmullRomCurve3, CubicBezierCurve, CubicBezierCurve3, Curves, EllipseCurve, LineCurve, LineCurve3, QuadraticBezierCurve, QuadraticBezierCurve3, SplineCurve } from './curve-types.js';

const baseToJSON = Curve.prototype.toJSON, baseFromJSON = Curve.prototype.fromJSON;

// A curve class's own fields: numbers or flags copied as they are, vectors as arrays, and
// lists of points as arrays of arrays.
function serialize( Class, { values = [], vectors = [], points = null, Vector = Vector2 } ) {

	Class.prototype.toJSON = function () {

		const data = baseToJSON.call( this );
		for ( const key of values ) data[ key ] = this[ key ];
		for ( const key of vectors ) data[ key ] = this[ key ].toArray();
		if ( points ) data.points = this.points.map( ( point ) => point.toArray() );
		return data;

	};

	Class.prototype.fromJSON = function ( json ) {

		baseFromJSON.call( this, json );
		for ( const key of values ) this[ key ] = json[ key ];
		for ( const key of vectors ) this[ key ] = new Vector().fromArray( json[ key ] );
		if ( points ) this.points = json.points.map( ( point ) => new Vector().fromArray( point ) );
		return this;

	};

}

serialize( EllipseCurve, { values: [ 'aX', 'aY', 'xRadius', 'yRadius', 'aStartAngle', 'aEndAngle', 'aClockwise', 'aRotation' ] } );
serialize( CatmullRomCurve3, { values: [ 'closed', 'curveType', 'tension' ], points: true, Vector: Vector3 } );
serialize( CubicBezierCurve, { vectors: [ 'v0', 'v1', 'v2', 'v3' ] } );
serialize( CubicBezierCurve3, { vectors: [ 'v0', 'v1', 'v2', 'v3' ], Vector: Vector3 } );
serialize( LineCurve, { vectors: [ 'v1', 'v2' ] } );
serialize( LineCurve3, { vectors: [ 'v1', 'v2' ], Vector: Vector3 } );
serialize( QuadraticBezierCurve, { vectors: [ 'v0', 'v1', 'v2' ] } );
serialize( QuadraticBezierCurve3, { vectors: [ 'v0', 'v1', 'v2' ], Vector: Vector3 } );
serialize( SplineCurve, { points: true } );

/** A curve of any type from its JSON. */
export function curveFromJSON( json ) {

	return new Curves[ json.type ]().fromJSON( json );

}
