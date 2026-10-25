// three.js `Sphere` of the object API, kept in JavaScript (see `../values.js`).
//
// Behaviour follows three.js (MIT License, Copyright 2010-2026 three.js authors); numbers
// are 64-bit here, as in three.js.

import { Box3 } from './box3.js';
import { Vector3 } from './vector3.js';

export class Sphere {

	constructor( center = new Vector3(), radius = - 1 ) {

		this.center = center;
		this.radius = radius;

	}

	set( center, radius ) {

		this.center.copy( center );
		this.radius = radius;
		return this;

	}

	setFromPoints( points, optionalCenter ) {

		const center = this.center;
		if ( optionalCenter !== undefined ) center.copy( optionalCenter );
		else ( _box ??= new Box3() ).setFromPoints( points ).getCenter( center );
		let maxRadiusSq = 0;
		for ( let i = 0; i < points.length; i ++ ) maxRadiusSq = Math.max( maxRadiusSq, center.distanceToSquared( points[ i ] ) );
		this.radius = Math.sqrt( maxRadiusSq );
		return this;

	}

	copy( sphere ) {

		this.center.copy( sphere.center );
		this.radius = sphere.radius;
		return this;

	}

	isEmpty() {

		return this.radius < 0;

	}

	makeEmpty() {

		this.center.set( 0, 0, 0 );
		this.radius = - 1;
		return this;

	}

	containsPoint( point ) {

		return point.distanceToSquared( this.center ) <= this.radius * this.radius;

	}

	distanceToPoint( point ) {

		return point.distanceTo( this.center ) - this.radius;

	}

	intersectsSphere( sphere ) {

		const radiusSum = this.radius + sphere.radius;
		return sphere.center.distanceToSquared( this.center ) <= radiusSum * radiusSum;

	}

	intersectsBox( box ) {

		return box.intersectsSphere( this );

	}

	intersectsPlane( plane ) {

		return Math.abs( plane.distanceToPoint( this.center ) ) <= this.radius;

	}

	clampPoint( point, target = new Vector3() ) {

		const deltaLengthSq = this.center.distanceToSquared( point );
		target.copy( point );
		if ( deltaLengthSq > this.radius * this.radius ) target.sub( this.center ).normalize().multiplyScalar( this.radius ).add( this.center );
		return target;

	}

	getBoundingBox( target = new Box3() ) {

		if ( this.isEmpty() ) return target.makeEmpty();
		target.set( this.center, this.center );
		return target.expandByScalar( this.radius );

	}

	applyMatrix4( matrix ) {

		this.center.applyMatrix4( matrix );
		this.radius = this.radius * matrix.getMaxScaleOnAxis();
		return this;

	}

	translate( offset ) {

		this.center.add( offset );
		return this;

	}

	expandByPoint( point ) {

		if ( this.isEmpty() ) {

			this.center.copy( point );
			this.radius = 0;
			return this;

		}

		_v1.subVectors( point, this.center );
		const lengthSq = _v1.lengthSq();

		if ( lengthSq > this.radius * this.radius ) {

			// grow just enough: move the centre half the way the radius grows
			const length = Math.sqrt( lengthSq ), delta = ( length - this.radius ) * 0.5;
			this.center.addScaledVector( _v1, delta / length );
			this.radius += delta;

		}

		return this;

	}

	union( sphere ) {

		if ( sphere.isEmpty() ) return this;
		if ( this.isEmpty() ) return this.copy( sphere );

		if ( this.center.equals( sphere.center ) ) {

			this.radius = Math.max( this.radius, sphere.radius );

		} else {

			_v2.subVectors( sphere.center, this.center ).setLength( sphere.radius );
			this.expandByPoint( _v1.copy( sphere.center ).add( _v2 ) );
			this.expandByPoint( _v1.copy( sphere.center ).sub( _v2 ) );

		}

		return this;

	}

	equals( sphere ) {

		return sphere.center.equals( this.center ) && sphere.radius === this.radius;

	}

	clone() {

		return new this.constructor().copy( this );

	}

	toJSON() {

		return { radius: this.radius, center: this.center.toArray() };

	}

	fromJSON( json ) {

		this.radius = json.radius;
		this.center.fromArray( json.center );
		return this;

	}

}

Sphere.prototype.isSphere = true;

const _v1 = new Vector3();
const _v2 = new Vector3();
let _box; // made on first use: Box3 imports this module
