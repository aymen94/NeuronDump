// CurvePath, Path and Shape: curves joined into outlines.
//
// Part of the curves, paths and shapes; `../curves.js` gathers them.

import * as W from '../../index.js';
import { Vector2 } from '../values.js';
import { Curve } from './curve.js';
import { CubicBezierCurve, EllipseCurve, LineCurve, LineCurve3, QuadraticBezierCurve, SplineCurve } from './curve-types.js';
import { curveFromJSON } from './json.js';

/** Curves joined end to end, parameterized by arc length over the whole. */
export class CurvePath extends Curve {

	constructor() {

		super();
		this.type = 'CurvePath';
		this.curves = [];
		this.autoClose = false;

	}

	add( curve ) {

		this.curves.push( curve );

	}

	closePath() {

		const start = this.curves[ 0 ].getPoint( 0 ), end = this.curves[ this.curves.length - 1 ].getPoint( 1 );
		if ( ! start.equals( end ) ) this.curves.push( start.isVector2 === true ? new LineCurve( end, start ) : new LineCurve3( end, start ) );
		return this;

	}

	getPoint( t, target ) {

		const d = t * this.getLength(), lengths = this.getCurveLengths();

		for ( let i = 0; i < lengths.length; i ++ ) {

			if ( lengths[ i ] >= d ) {

				const diff = lengths[ i ] - d, curve = this.curves[ i ], length = curve.getLength();
				return curve.getPointAt( length === 0 ? 0 : 1 - diff / length, target );

			}

		}

		return null;

	}

	getLength() {

		const lengths = this.getCurveLengths();
		return lengths[ lengths.length - 1 ];

	}

	updateArcLengths() {

		this.needsUpdate = true;
		this.cacheLengths = null;
		this.getCurveLengths();

	}

	getCurveLengths() {

		if ( this.cacheLengths && this.cacheLengths.length === this.curves.length ) return this.cacheLengths;
		const lengths = [];
		let sums = 0;

		for ( const curve of this.curves ) {

			sums += curve.getLength();
			lengths.push( sums );

		}

		this.cacheLengths = lengths;
		return lengths;

	}

	getSpacedPoints( divisions = 40 ) {

		const points = [];
		for ( let i = 0; i <= divisions; i ++ ) points.push( this.getPoint( i / divisions ) );
		if ( this.autoClose ) points.push( points[ 0 ] );
		return points;

	}

	// Points per curve: two for lines, twice `divisions` for ellipses, per control point for splines.
	getPoints( divisions = 12 ) {

		const points = [];
		let last;

		for ( const curve of this.curves ) {

			const resolution = curve.isEllipseCurve ? divisions * 2 : ( curve.isLineCurve || curve.isLineCurve3 ) ? 1 : curve.isSplineCurve ? divisions * curve.points.length : divisions;

			for ( const point of curve.getPoints( resolution ) ) {

				if ( last && last.equals( point ) ) continue;
				points.push( point );
				last = point;

			}

		}

		if ( this.autoClose && points.length > 1 && ! points[ points.length - 1 ].equals( points[ 0 ] ) ) points.push( points[ 0 ] );
		return points;

	}

	copy( source ) {

		super.copy( source );
		this.curves = source.curves.map( ( curve ) => curve.clone() );
		this.autoClose = source.autoClose;
		return this;

	}

	toJSON() {

		const data = super.toJSON();
		data.autoClose = this.autoClose;
		data.curves = this.curves.map( ( curve ) => curve.toJSON() );
		return data;

	}

	fromJSON( json ) {

		super.fromJSON( json );
		this.autoClose = json.autoClose;
		this.curves = json.curves.map( curveFromJSON );
		return this;

	}

}

/** A 2D path drawn with canvas-like commands. */
export class Path extends CurvePath {

	constructor( points ) {

		super();
		this.type = 'Path';
		this.currentPoint = new Vector2();
		if ( points ) this.setFromPoints( points );

	}

	setFromPoints( points ) {

		this.moveTo( points[ 0 ].x, points[ 0 ].y );
		for ( let i = 1; i < points.length; i ++ ) this.lineTo( points[ i ].x, points[ i ].y );
		return this;

	}

	moveTo( x, y ) {

		this.currentPoint.set( x, y );
		return this;

	}

	lineTo( x, y ) {

		this.curves.push( new LineCurve( this.currentPoint.clone(), new Vector2( x, y ) ) );
		this.currentPoint.set( x, y );
		return this;

	}

	quadraticCurveTo( aCPx, aCPy, aX, aY ) {

		this.curves.push( new QuadraticBezierCurve( this.currentPoint.clone(), new Vector2( aCPx, aCPy ), new Vector2( aX, aY ) ) );
		this.currentPoint.set( aX, aY );
		return this;

	}

	bezierCurveTo( aCP1x, aCP1y, aCP2x, aCP2y, aX, aY ) {

		this.curves.push( new CubicBezierCurve( this.currentPoint.clone(), new Vector2( aCP1x, aCP1y ), new Vector2( aCP2x, aCP2y ), new Vector2( aX, aY ) ) );
		this.currentPoint.set( aX, aY );
		return this;

	}

	splineThru( pts ) {

		this.curves.push( new SplineCurve( [ this.currentPoint.clone() ].concat( pts ) ) );
		this.currentPoint.copy( pts[ pts.length - 1 ] );
		return this;

	}

	arc( aX, aY, aRadius, aStartAngle, aEndAngle, aClockwise ) {

		this.absarc( aX + this.currentPoint.x, aY + this.currentPoint.y, aRadius, aStartAngle, aEndAngle, aClockwise );
		return this;

	}

	absarc( aX, aY, aRadius, aStartAngle, aEndAngle, aClockwise ) {

		this.absellipse( aX, aY, aRadius, aRadius, aStartAngle, aEndAngle, aClockwise );
		return this;

	}

	ellipse( aX, aY, xRadius, yRadius, aStartAngle, aEndAngle, aClockwise, aRotation ) {

		this.absellipse( aX + this.currentPoint.x, aY + this.currentPoint.y, xRadius, yRadius, aStartAngle, aEndAngle, aClockwise, aRotation );
		return this;

	}

	// An ellipse joined to the path by a line when it does not start at the current point.
	absellipse( aX, aY, xRadius, yRadius, aStartAngle, aEndAngle, aClockwise, aRotation ) {

		const curve = new EllipseCurve( aX, aY, xRadius, yRadius, aStartAngle, aEndAngle, aClockwise, aRotation );

		if ( this.curves.length > 0 ) {

			const first = curve.getPoint( 0 );
			if ( ! first.equals( this.currentPoint ) ) this.lineTo( first.x, first.y );

		}

		this.curves.push( curve );
		this.currentPoint.copy( curve.getPoint( 1 ) );
		return this;

	}

	copy( source ) {

		super.copy( source );
		this.currentPoint.copy( source.currentPoint );
		return this;

	}

	toJSON() {

		const data = super.toJSON();
		data.currentPoint = this.currentPoint.toArray();
		return data;

	}

	fromJSON( json ) {

		super.fromJSON( json );
		this.currentPoint.fromArray( json.currentPoint );
		return this;

	}

}

/** A closed 2D outline with optional `holes` (paths), for `ShapeGeometry` and `ExtrudeGeometry`. */
export class Shape extends Path {

	constructor( points ) {

		super( points );
		this.uuid = W.MathUtils.generateUUID();
		this.type = 'Shape';
		this.holes = [];

	}

	getPointsHoles( divisions ) {

		return this.holes.map( ( hole ) => hole.getPoints( divisions ) );

	}

	extractPoints( divisions ) {

		return { shape: this.getPoints( divisions ), holes: this.getPointsHoles( divisions ) };

	}

	copy( source ) {

		super.copy( source );
		this.holes = source.holes.map( ( hole ) => hole.clone() );
		return this;

	}

	toJSON() {

		const data = super.toJSON();
		data.uuid = this.uuid;
		data.holes = this.holes.map( ( hole ) => hole.toJSON() );
		return data;

	}

	fromJSON( json ) {

		super.fromJSON( json );
		this.uuid = json.uuid;
		this.holes = json.holes.map( ( hole ) => new Path().fromJSON( hole ) );
		return this;

	}

}
