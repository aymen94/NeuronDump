// Curve: the base of every curve (points, lengths, tangents, Frenet frames).
//
// Part of the curves, paths and shapes; `../curves.js` gathers them.

import { Matrix4, Vector2, Vector3 } from '../values.js';

const clamp = ( value, min, max ) => Math.max( min, Math.min( max, value ) );

/** A curve over t in 0..1. Subclasses define `getPoint( t, target )`. */
export class Curve {

	constructor() {

		this.type = 'Curve';
		this.arcLengthDivisions = 200;
		this.needsUpdate = false;
		this.cacheArcLengths = null;

	}

	getPoint() {

		console.warn( 'WAMR3D: Curve: .getPoint() not implemented.' );
		return null;

	}

	getPointAt( u, target ) {

		return this.getPoint( this.getUtoTmapping( u ), target );

	}

	getPoints( divisions = 5 ) {

		const points = [];
		for ( let d = 0; d <= divisions; d ++ ) points.push( this.getPoint( d / divisions ) );
		return points;

	}

	getSpacedPoints( divisions = 5 ) {

		const points = [];
		for ( let d = 0; d <= divisions; d ++ ) points.push( this.getPointAt( d / divisions ) );
		return points;

	}

	getLength() {

		const lengths = this.getLengths();
		return lengths[ lengths.length - 1 ];

	}

	getLengths( divisions = this.arcLengthDivisions ) {

		if ( this.cacheArcLengths && this.cacheArcLengths.length === divisions + 1 && ! this.needsUpdate ) return this.cacheArcLengths;
		this.needsUpdate = false;
		const cache = [ 0 ];
		let last = this.getPoint( 0 ), sum = 0;

		for ( let p = 1; p <= divisions; p ++ ) {

			const current = this.getPoint( p / divisions );
			sum += current.distanceTo( last );
			cache.push( sum );
			last = current;

		}

		this.cacheArcLengths = cache;
		return cache;

	}

	updateArcLengths() {

		this.needsUpdate = true;
		this.getLengths();

	}

	// The t at which the arc length is `u` of the whole (or `distance`).
	getUtoTmapping( u, distance = null ) {

		const arcLengths = this.getLengths(), il = arcLengths.length;
		const target = distance ? distance : u * arcLengths[ il - 1 ];
		let low = 0, high = il - 1, i = 0;

		while ( low <= high ) {

			i = Math.floor( low + ( high - low ) / 2 );
			const comparison = arcLengths[ i ] - target;

			if ( comparison < 0 ) {

				low = i + 1;

			} else if ( comparison > 0 ) {

				high = i - 1;

			} else {

				high = i;
				break;

			}

		}

		i = high;
		if ( arcLengths[ i ] === target ) return i / ( il - 1 );
		const before = arcLengths[ i ], after = arcLengths[ i + 1 ];
		return ( i + ( target - before ) / ( after - before ) ) / ( il - 1 );

	}

	getTangent( t, target ) {

		const delta = 0.0001, t1 = Math.max( 0, t - delta ), t2 = Math.min( 1, t + delta );
		const pt1 = this.getPoint( t1 ), pt2 = this.getPoint( t2 );
		const tangent = target || ( pt1.isVector2 ? new Vector2() : new Vector3() );
		return tangent.copy( pt2 ).sub( pt1 ).normalize();

	}

	getTangentAt( u, target ) {

		return this.getTangent( this.getUtoTmapping( u ), target );

	}

	// Tangents, normals and binormals along the curve with the least twist (parallel
	// transport), evened out over a closed curve.
	computeFrenetFrames( segments, closed = false ) {

		const normal = new Vector3(), vec = new Vector3(), mat = new Matrix4();
		const tangents = [], normals = [], binormals = [];
		for ( let i = 0; i <= segments; i ++ ) tangents[ i ] = this.getTangentAt( i / segments, new Vector3() );

		normals[ 0 ] = new Vector3();
		binormals[ 0 ] = new Vector3();
		let min = Number.MAX_VALUE;
		const tx = Math.abs( tangents[ 0 ].x ), ty = Math.abs( tangents[ 0 ].y ), tz = Math.abs( tangents[ 0 ].z );

		if ( tx <= min ) {

			min = tx;
			normal.set( 1, 0, 0 );

		}

		if ( ty <= min ) {

			min = ty;
			normal.set( 0, 1, 0 );

		}

		if ( tz <= min ) normal.set( 0, 0, 1 );

		vec.crossVectors( tangents[ 0 ], normal ).normalize();
		normals[ 0 ].crossVectors( tangents[ 0 ], vec );
		binormals[ 0 ].crossVectors( tangents[ 0 ], normals[ 0 ] );

		for ( let i = 1; i <= segments; i ++ ) {

			normals[ i ] = normals[ i - 1 ].clone();
			binormals[ i ] = binormals[ i - 1 ].clone();
			vec.crossVectors( tangents[ i - 1 ], tangents[ i ] );

			if ( vec.length() > Number.EPSILON ) {

				vec.normalize();
				const theta = Math.acos( clamp( tangents[ i - 1 ].dot( tangents[ i ] ), - 1, 1 ) );
				normals[ i ].applyMatrix4( mat.makeRotationAxis( vec, theta ) );

			}

			binormals[ i ].crossVectors( tangents[ i ], normals[ i ] );

		}

		if ( closed === true ) {

			let theta = Math.acos( clamp( normals[ 0 ].dot( normals[ segments ] ), - 1, 1 ) ) / segments;
			if ( tangents[ 0 ].dot( vec.crossVectors( normals[ 0 ], normals[ segments ] ) ) > 0 ) theta = - theta;

			for ( let i = 1; i <= segments; i ++ ) {

				normals[ i ].applyMatrix4( mat.makeRotationAxis( tangents[ i ], theta * i ) );
				binormals[ i ].crossVectors( tangents[ i ], normals[ i ] );

			}

		}

		return { tangents, normals, binormals };

	}

	clone() {

		return new this.constructor().copy( this );

	}

	copy( source ) {

		this.arcLengthDivisions = source.arcLengthDivisions;
		return this;

	}

	toJSON() {

		return { metadata: { version: 4.7, type: 'Curve', generator: 'Curve.toJSON' }, arcLengthDivisions: this.arcLengthDivisions, type: this.type };

	}

	fromJSON( json ) {

		this.arcLengthDivisions = json.arcLengthDivisions;
		return this;

	}

}
