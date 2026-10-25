// The curve types: ellipses and arcs, lines, splines, and quadratic and cubic Bezier curves.
//
// Part of the curves, paths and shapes; `../curves.js` gathers them.

import { Vector2, Vector3 } from '../values.js';
import { CatmullRom, CubicBezier, QuadraticBezier } from './interpolation.js';
import { Curve } from './curve.js';

export class EllipseCurve extends Curve {

	constructor( aX = 0, aY = 0, xRadius = 1, yRadius = 1, aStartAngle = 0, aEndAngle = Math.PI * 2, aClockwise = false, aRotation = 0 ) {

		super();
		this.type = 'EllipseCurve';
		this.aX = aX;
		this.aY = aY;
		this.xRadius = xRadius;
		this.yRadius = yRadius;
		this.aStartAngle = aStartAngle;
		this.aEndAngle = aEndAngle;
		this.aClockwise = aClockwise;
		this.aRotation = aRotation;

	}

	getPoint( t, target = new Vector2() ) {

		const twoPi = Math.PI * 2;
		let deltaAngle = this.aEndAngle - this.aStartAngle;
		const samePoints = Math.abs( deltaAngle ) < Number.EPSILON;
		while ( deltaAngle < 0 ) deltaAngle += twoPi;
		while ( deltaAngle > twoPi ) deltaAngle -= twoPi;
		if ( deltaAngle < Number.EPSILON ) deltaAngle = samePoints ? 0 : twoPi;
		if ( this.aClockwise === true && ! samePoints ) deltaAngle = deltaAngle === twoPi ? - twoPi : deltaAngle - twoPi;

		const angle = this.aStartAngle + t * deltaAngle;
		let x = this.aX + this.xRadius * Math.cos( angle ), y = this.aY + this.yRadius * Math.sin( angle );

		if ( this.aRotation !== 0 ) {

			const cos = Math.cos( this.aRotation ), sin = Math.sin( this.aRotation ), tx = x - this.aX, ty = y - this.aY;
			x = tx * cos - ty * sin + this.aX;
			y = tx * sin + ty * cos + this.aY;

		}

		return target.set( x, y );

	}

	copy( source ) {

		super.copy( source );
		this.aX = source.aX;
		this.aY = source.aY;
		this.xRadius = source.xRadius;
		this.yRadius = source.yRadius;
		this.aStartAngle = source.aStartAngle;
		this.aEndAngle = source.aEndAngle;
		this.aClockwise = source.aClockwise;
		this.aRotation = source.aRotation;
		return this;

	}

}

EllipseCurve.prototype.isEllipseCurve = true;

export class ArcCurve extends EllipseCurve {

	constructor( aX, aY, aRadius, aStartAngle, aEndAngle, aClockwise ) {

		super( aX, aY, aRadius, aRadius, aStartAngle, aEndAngle, aClockwise );
		this.type = 'ArcCurve';

	}

}

ArcCurve.prototype.isArcCurve = true;

// A cubic between x1 and x2 with Catmull-Rom tangents, uniform or non-uniform (by the
// chord lengths dt0, dt1, dt2 of the four points).
function cubicPoly() {

	let c0 = 0, c1 = 0, c2 = 0, c3 = 0;

	function init( x0, x1, t0, t1 ) {

		c0 = x0;
		c1 = t0;
		c2 = - 3 * x0 + 3 * x1 - 2 * t0 - t1;
		c3 = 2 * x0 - 2 * x1 + t0 + t1;

	}

	return {

		initCatmullRom( x0, x1, x2, x3, tension ) {

			init( x1, x2, tension * ( x2 - x0 ), tension * ( x3 - x1 ) );

		},

		initNonuniformCatmullRom( x0, x1, x2, x3, dt0, dt1, dt2 ) {

			const t1 = ( ( x1 - x0 ) / dt0 - ( x2 - x0 ) / ( dt0 + dt1 ) + ( x2 - x1 ) / dt1 ) * dt1;
			const t2 = ( ( x2 - x1 ) / dt1 - ( x3 - x1 ) / ( dt1 + dt2 ) + ( x3 - x2 ) / dt2 ) * dt1;
			init( x1, x2, t1, t2 );

		},

		calc( t ) {

			const t2 = t * t;
			return c0 + c1 * t + c2 * t2 + c3 * t2 * t;

		}

	};

}

const _tmp = new Vector3(), _tmp2 = new Vector3(), _px = cubicPoly(), _py = cubicPoly(), _pz = cubicPoly();

/** A smooth curve through `points`; `curveType` is 'centripetal' (default), 'chordal' or 'catmullrom' (with `tension`). */
export class CatmullRomCurve3 extends Curve {

	constructor( points = [], closed = false, curveType = 'centripetal', tension = 0.5 ) {

		super();
		this.type = 'CatmullRomCurve3';
		this.points = points;
		this.closed = closed;
		this.curveType = curveType;
		this.tension = tension;

	}

	getPoint( t, target = new Vector3() ) {

		const points = this.points, l = points.length;
		const p = ( l - ( this.closed ? 0 : 1 ) ) * t;
		let intPoint = Math.floor( p ), weight = p - intPoint;

		if ( this.closed ) {

			intPoint += intPoint > 0 ? 0 : ( Math.floor( Math.abs( intPoint ) / l ) + 1 ) * l;

		} else if ( weight === 0 && intPoint === l - 1 ) {

			intPoint = l - 2;
			weight = 1;

		}

		const p0 = this.closed || intPoint > 0 ? points[ ( intPoint - 1 ) % l ] : _tmp2.subVectors( points[ 0 ], points[ 1 ] ).add( points[ 0 ] );
		const p1 = points[ intPoint % l ], p2 = points[ ( intPoint + 1 ) % l ];
		const p3 = this.closed || intPoint + 2 < l ? points[ ( intPoint + 2 ) % l ] : _tmp.subVectors( points[ l - 1 ], points[ l - 2 ] ).add( points[ l - 1 ] );

		if ( this.curveType === 'centripetal' || this.curveType === 'chordal' ) {

			const pow = this.curveType === 'chordal' ? 0.5 : 0.25;
			let dt0 = Math.pow( p0.distanceToSquared( p1 ), pow ), dt1 = Math.pow( p1.distanceToSquared( p2 ), pow ), dt2 = Math.pow( p2.distanceToSquared( p3 ), pow );
			// safety check for repeated points
			if ( dt1 < 1e-4 ) dt1 = 1.0;
			if ( dt0 < 1e-4 ) dt0 = dt1;
			if ( dt2 < 1e-4 ) dt2 = dt1;
			_px.initNonuniformCatmullRom( p0.x, p1.x, p2.x, p3.x, dt0, dt1, dt2 );
			_py.initNonuniformCatmullRom( p0.y, p1.y, p2.y, p3.y, dt0, dt1, dt2 );
			_pz.initNonuniformCatmullRom( p0.z, p1.z, p2.z, p3.z, dt0, dt1, dt2 );

		} else if ( this.curveType === 'catmullrom' ) {

			_px.initCatmullRom( p0.x, p1.x, p2.x, p3.x, this.tension );
			_py.initCatmullRom( p0.y, p1.y, p2.y, p3.y, this.tension );
			_pz.initCatmullRom( p0.z, p1.z, p2.z, p3.z, this.tension );

		}

		return target.set( _px.calc( weight ), _py.calc( weight ), _pz.calc( weight ) );

	}

	copy( source ) {

		super.copy( source );
		this.points = source.points.map( ( point ) => point.clone() );
		this.closed = source.closed;
		this.curveType = source.curveType;
		this.tension = source.tension;
		return this;

	}

}

CatmullRomCurve3.prototype.isCatmullRomCurve3 = true;

export class CubicBezierCurve extends Curve {

	constructor( v0 = new Vector2(), v1 = new Vector2(), v2 = new Vector2(), v3 = new Vector2() ) {

		super();
		this.type = 'CubicBezierCurve';
		this.v0 = v0;
		this.v1 = v1;
		this.v2 = v2;
		this.v3 = v3;

	}

	getPoint( t, target = new Vector2() ) {

		const { v0, v1, v2, v3 } = this;
		return target.set( CubicBezier( t, v0.x, v1.x, v2.x, v3.x ), CubicBezier( t, v0.y, v1.y, v2.y, v3.y ) );

	}

	copy( source ) {

		super.copy( source );
		this.v0.copy( source.v0 );
		this.v1.copy( source.v1 );
		this.v2.copy( source.v2 );
		this.v3.copy( source.v3 );
		return this;

	}

}

CubicBezierCurve.prototype.isCubicBezierCurve = true;

export class CubicBezierCurve3 extends Curve {

	constructor( v0 = new Vector3(), v1 = new Vector3(), v2 = new Vector3(), v3 = new Vector3() ) {

		super();
		this.type = 'CubicBezierCurve3';
		this.v0 = v0;
		this.v1 = v1;
		this.v2 = v2;
		this.v3 = v3;

	}

	getPoint( t, target = new Vector3() ) {

		const { v0, v1, v2, v3 } = this;
		return target.set( CubicBezier( t, v0.x, v1.x, v2.x, v3.x ), CubicBezier( t, v0.y, v1.y, v2.y, v3.y ), CubicBezier( t, v0.z, v1.z, v2.z, v3.z ) );

	}

	copy( source ) {

		super.copy( source );
		this.v0.copy( source.v0 );
		this.v1.copy( source.v1 );
		this.v2.copy( source.v2 );
		this.v3.copy( source.v3 );
		return this;

	}

}

CubicBezierCurve3.prototype.isCubicBezierCurve3 = true;

export class LineCurve extends Curve {

	constructor( v1 = new Vector2(), v2 = new Vector2() ) {

		super();
		this.type = 'LineCurve';
		this.v1 = v1;
		this.v2 = v2;

	}

	getPoint( t, target = new Vector2() ) {

		if ( t === 1 ) return target.copy( this.v2 );
		return target.copy( this.v2 ).sub( this.v1 ).multiplyScalar( t ).add( this.v1 );

	}

	getPointAt( u, target ) {

		return this.getPoint( u, target );

	}

	getTangent( t, target = new Vector2() ) {

		return target.subVectors( this.v2, this.v1 ).normalize();

	}

	getTangentAt( u, target ) {

		return this.getTangent( u, target );

	}

	copy( source ) {

		super.copy( source );
		this.v1.copy( source.v1 );
		this.v2.copy( source.v2 );
		return this;

	}

}

LineCurve.prototype.isLineCurve = true;

export class LineCurve3 extends Curve {

	constructor( v1 = new Vector3(), v2 = new Vector3() ) {

		super();
		this.type = 'LineCurve3';
		this.v1 = v1;
		this.v2 = v2;

	}

	getPoint( t, target = new Vector3() ) {

		if ( t === 1 ) return target.copy( this.v2 );
		return target.copy( this.v2 ).sub( this.v1 ).multiplyScalar( t ).add( this.v1 );

	}

	getPointAt( u, target ) {

		return this.getPoint( u, target );

	}

	getTangent( t, target = new Vector3() ) {

		return target.subVectors( this.v2, this.v1 ).normalize();

	}

	getTangentAt( u, target ) {

		return this.getTangent( u, target );

	}

	copy( source ) {

		super.copy( source );
		this.v1.copy( source.v1 );
		this.v2.copy( source.v2 );
		return this;

	}

}

LineCurve3.prototype.isLineCurve3 = true;

export class QuadraticBezierCurve extends Curve {

	constructor( v0 = new Vector2(), v1 = new Vector2(), v2 = new Vector2() ) {

		super();
		this.type = 'QuadraticBezierCurve';
		this.v0 = v0;
		this.v1 = v1;
		this.v2 = v2;

	}

	getPoint( t, target = new Vector2() ) {

		const { v0, v1, v2 } = this;
		return target.set( QuadraticBezier( t, v0.x, v1.x, v2.x ), QuadraticBezier( t, v0.y, v1.y, v2.y ) );

	}

	copy( source ) {

		super.copy( source );
		this.v0.copy( source.v0 );
		this.v1.copy( source.v1 );
		this.v2.copy( source.v2 );
		return this;

	}

}

QuadraticBezierCurve.prototype.isQuadraticBezierCurve = true;

export class QuadraticBezierCurve3 extends Curve {

	constructor( v0 = new Vector3(), v1 = new Vector3(), v2 = new Vector3() ) {

		super();
		this.type = 'QuadraticBezierCurve3';
		this.v0 = v0;
		this.v1 = v1;
		this.v2 = v2;

	}

	getPoint( t, target = new Vector3() ) {

		const { v0, v1, v2 } = this;
		return target.set( QuadraticBezier( t, v0.x, v1.x, v2.x ), QuadraticBezier( t, v0.y, v1.y, v2.y ), QuadraticBezier( t, v0.z, v1.z, v2.z ) );

	}

	copy( source ) {

		super.copy( source );
		this.v0.copy( source.v0 );
		this.v1.copy( source.v1 );
		this.v2.copy( source.v2 );
		return this;

	}

}

QuadraticBezierCurve3.prototype.isQuadraticBezierCurve3 = true;

/** A 2D Catmull-Rom spline through `points`. */
export class SplineCurve extends Curve {

	constructor( points = [] ) {

		super();
		this.type = 'SplineCurve';
		this.points = points;

	}

	getPoint( t, target = new Vector2() ) {

		const points = this.points, p = ( points.length - 1 ) * t;
		const intPoint = Math.floor( p ), weight = p - intPoint;
		const p0 = points[ intPoint === 0 ? intPoint : intPoint - 1 ];
		const p1 = points[ intPoint ];
		const p2 = points[ intPoint > points.length - 2 ? points.length - 1 : intPoint + 1 ];
		const p3 = points[ intPoint > points.length - 3 ? points.length - 1 : intPoint + 2 ];
		return target.set( CatmullRom( weight, p0.x, p1.x, p2.x, p3.x ), CatmullRom( weight, p0.y, p1.y, p2.y, p3.y ) );

	}

	copy( source ) {

		super.copy( source );
		this.points = source.points.map( ( point ) => point.clone() );
		return this;

	}

}

SplineCurve.prototype.isSplineCurve = true;

export const Curves = { ArcCurve, CatmullRomCurve3, CubicBezierCurve, CubicBezierCurve3, EllipseCurve, LineCurve, LineCurve3, QuadraticBezierCurve, QuadraticBezierCurve3, SplineCurve };
