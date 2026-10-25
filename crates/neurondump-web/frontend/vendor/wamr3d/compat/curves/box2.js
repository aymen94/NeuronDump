// Box2: an axis-aligned rectangle.
//
// Part of the curves, paths and shapes; `../curves.js` gathers them.

import { Vector2 } from '../values.js';

export class Box2 {

	constructor( min = new Vector2( Infinity, Infinity ), max = new Vector2( - Infinity, - Infinity ) ) {

		this.min = min;
		this.max = max;

	}

	set( min, max ) {

		this.min.copy( min );
		this.max.copy( max );
		return this;

	}

	setFromPoints( points ) {

		this.makeEmpty();
		for ( const point of points ) this.expandByPoint( point );
		return this;

	}

	setFromCenterAndSize( center, size ) {

		const hx = size.x / 2, hy = size.y / 2;
		this.min.set( center.x - hx, center.y - hy );
		this.max.set( center.x + hx, center.y + hy );
		return this;

	}

	clone() {

		return new Box2().copy( this );

	}

	copy( box ) {

		this.min.copy( box.min );
		this.max.copy( box.max );
		return this;

	}

	makeEmpty() {

		this.min.x = this.min.y = Infinity;
		this.max.x = this.max.y = - Infinity;
		return this;

	}

	isEmpty() {

		return this.max.x < this.min.x || this.max.y < this.min.y;

	}

	getCenter( target ) {

		return this.isEmpty() ? target.set( 0, 0 ) : target.addVectors( this.min, this.max ).multiplyScalar( 0.5 );

	}

	getSize( target ) {

		return this.isEmpty() ? target.set( 0, 0 ) : target.subVectors( this.max, this.min );

	}

	expandByPoint( point ) {

		this.min.min( point );
		this.max.max( point );
		return this;

	}

	expandByVector( vector ) {

		this.min.sub( vector );
		this.max.add( vector );
		return this;

	}

	expandByScalar( scalar ) {

		this.min.addScalar( - scalar );
		this.max.addScalar( scalar );
		return this;

	}

	containsPoint( point ) {

		return point.x >= this.min.x && point.x <= this.max.x && point.y >= this.min.y && point.y <= this.max.y;

	}

	containsBox( box ) {

		return this.min.x <= box.min.x && box.max.x <= this.max.x && this.min.y <= box.min.y && box.max.y <= this.max.y;

	}

	getParameter( point, target ) {

		return target.set( ( point.x - this.min.x ) / ( this.max.x - this.min.x ), ( point.y - this.min.y ) / ( this.max.y - this.min.y ) );

	}

	intersectsBox( box ) {

		return box.max.x >= this.min.x && box.min.x <= this.max.x && box.max.y >= this.min.y && box.min.y <= this.max.y;

	}

	clampPoint( point, target ) {

		return target.copy( point ).clamp( this.min, this.max );

	}

	distanceToPoint( point ) {

		return this.clampPoint( point, new Vector2() ).distanceTo( point );

	}

	intersect( box ) {

		this.min.max( box.min );
		this.max.min( box.max );
		if ( this.isEmpty() ) this.makeEmpty();
		return this;

	}

	union( box ) {

		this.min.min( box.min );
		this.max.max( box.max );
		return this;

	}

	translate( offset ) {

		this.min.add( offset );
		this.max.add( offset );
		return this;

	}

	equals( box ) {

		return box.min.equals( this.min ) && box.max.equals( this.max );

	}

}

Box2.prototype.isBox2 = true;
