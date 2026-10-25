// three.js `Box3` of the object API, kept in JavaScript (see `../values.js`).
//
// Behaviour follows three.js (MIT License, Copyright 2010-2026 three.js authors); numbers
// are 64-bit here, as in three.js.

import { Sphere } from './sphere.js';
import { Vector3 } from './vector3.js';

export class Box3 {

	constructor( min = new Vector3( Infinity, Infinity, Infinity ), max = new Vector3( - Infinity, - Infinity, - Infinity ) ) {

		this.min = min;
		this.max = max;

	}

	set( min, max ) {

		this.min.copy( min );
		this.max.copy( max );
		return this;

	}

	setFromArray( array ) {

		this.makeEmpty();
		for ( let i = 0; i < array.length; i += 3 ) this.expandByPoint( _vector.fromArray( array, i ) );
		return this;

	}

	setFromBufferAttribute( attribute ) {

		this.makeEmpty();
		for ( let i = 0; i < attribute.count; i ++ ) this.expandByPoint( _vector.fromBufferAttribute( attribute, i ) );
		return this;

	}

	setFromPoints( points ) {

		this.makeEmpty();
		for ( let i = 0; i < points.length; i ++ ) this.expandByPoint( points[ i ] );
		return this;

	}

	setFromCenterAndSize( center, size ) {

		const half = _vector.copy( size ).multiplyScalar( 0.5 );
		this.min.copy( center ).sub( half );
		this.max.copy( center ).add( half );
		return this;

	}

	/**
	 * The world-space box around `object` and its descendants: their geometries' boxes
	 * transformed by their world matrices, or every vertex with `precise`.
	 */
	setFromObject( object, precise = false ) {

		this.makeEmpty();
		return this.expandByObject( object, precise );

	}

	expandByObject( object, precise = false ) {

		object.updateWorldMatrix( false, false );
		const geometry = object.geometry;

		if ( geometry !== undefined && geometry.isBufferGeometry ) {

			const m = object.matrixWorld, position = geometry.attributes.position;

			if ( precise && position !== undefined && ! object.isInstancedMesh ) {

				for ( let i = 0; i < position.count; i ++ ) this.expandByPoint( _vector.fromBufferAttribute( position, i ).applyMatrix4( m ) );

			} else {

				let box;

				if ( object.isInstancedMesh ) {

					if ( object.boundingBox === null ) object.computeBoundingBox();
					box = object.boundingBox;

				} else {

					if ( geometry.boundingBox === null ) geometry.computeBoundingBox();
					box = geometry.boundingBox;

				}

				this.union( _box.copy( box ).applyMatrix4( m ) );

			}

		}

		for ( const child of object.children ) this.expandByObject( child, precise );
		return this;

	}

	clone() {

		return new this.constructor().copy( this );

	}

	copy( box ) {

		this.min.copy( box.min );
		this.max.copy( box.max );
		return this;

	}

	makeEmpty() {

		this.min.x = this.min.y = this.min.z = Infinity;
		this.max.x = this.max.y = this.max.z = - Infinity;
		return this;

	}

	isEmpty() {

		return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;

	}

	getCenter( target = new Vector3() ) {

		return this.isEmpty() ? target.set( 0, 0, 0 ) : target.addVectors( this.min, this.max ).multiplyScalar( 0.5 );

	}

	getSize( target = new Vector3() ) {

		return this.isEmpty() ? target.set( 0, 0, 0 ) : target.subVectors( this.max, this.min );

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

		return point.x >= this.min.x && point.x <= this.max.x && point.y >= this.min.y && point.y <= this.max.y && point.z >= this.min.z && point.z <= this.max.z;

	}

	containsBox( box ) {

		return this.min.x <= box.min.x && box.max.x <= this.max.x && this.min.y <= box.min.y && box.max.y <= this.max.y && this.min.z <= box.min.z && box.max.z <= this.max.z;

	}

	getParameter( point, target ) {

		return target.set( ( point.x - this.min.x ) / ( this.max.x - this.min.x ), ( point.y - this.min.y ) / ( this.max.y - this.min.y ), ( point.z - this.min.z ) / ( this.max.z - this.min.z ) );

	}

	intersectsBox( box ) {

		return box.max.x >= this.min.x && box.min.x <= this.max.x && box.max.y >= this.min.y && box.min.y <= this.max.y && box.max.z >= this.min.z && box.min.z <= this.max.z;

	}

	intersectsSphere( sphere ) {

		this.clampPoint( sphere.center, _vector );
		return _vector.distanceToSquared( sphere.center ) <= sphere.radius * sphere.radius;

	}

	intersectsPlane( plane ) {

		// the box's extent along the normal, against the plane's offset
		const n = plane.normal;
		let min = n.x > 0 ? n.x * this.min.x : n.x * this.max.x, max = n.x > 0 ? n.x * this.max.x : n.x * this.min.x;
		min += n.y > 0 ? n.y * this.min.y : n.y * this.max.y;
		max += n.y > 0 ? n.y * this.max.y : n.y * this.min.y;
		min += n.z > 0 ? n.z * this.min.z : n.z * this.max.z;
		max += n.z > 0 ? n.z * this.max.z : n.z * this.min.z;
		return min <= - plane.constant && max >= - plane.constant;

	}

	intersectsTriangle( triangle ) {

		if ( this.isEmpty() ) return false;
		// separating axes: the edge cross products, the box axes, the triangle normal
		this.getCenter( _center );
		_extents.subVectors( this.max, _center );
		_v0.subVectors( triangle.a, _center );
		_v1.subVectors( triangle.b, _center );
		_v2.subVectors( triangle.c, _center );
		_f0.subVectors( _v1, _v0 );
		_f1.subVectors( _v2, _v1 );
		_f2.subVectors( _v0, _v2 );
		const edges = [
			0, - _f0.z, _f0.y, 0, - _f1.z, _f1.y, 0, - _f2.z, _f2.y,
			_f0.z, 0, - _f0.x, _f1.z, 0, - _f1.x, _f2.z, 0, - _f2.x,
			- _f0.y, _f0.x, 0, - _f1.y, _f1.x, 0, - _f2.y, _f2.x, 0
		];
		if ( ! separated( edges ) ) return false;
		if ( ! separated( [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ] ) ) return false;
		_normal.crossVectors( _f0, _f1 );
		return separated( [ _normal.x, _normal.y, _normal.z ] );

	}

	clampPoint( point, target = new Vector3() ) {

		return target.copy( point ).clamp( this.min, this.max );

	}

	distanceToPoint( point ) {

		return this.clampPoint( point, _vector ).distanceTo( point );

	}

	getBoundingSphere( target = new Sphere() ) {

		if ( this.isEmpty() ) return target.makeEmpty();
		this.getCenter( target.center );
		target.radius = this.getSize( _vector ).length() * 0.5;
		return target;

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

	applyMatrix4( matrix ) {

		if ( this.isEmpty() ) return this;
		const min = this.min, max = this.max;
		for ( let i = 0; i < 8; i ++ ) _points[ i ].set( i & 4 ? max.x : min.x, i & 2 ? max.y : min.y, i & 1 ? max.z : min.z ).applyMatrix4( matrix );
		return this.setFromPoints( _points );

	}

	translate( offset ) {

		this.min.add( offset );
		this.max.add( offset );
		return this;

	}

	equals( box ) {

		return box.min.equals( this.min ) && box.max.equals( this.max );

	}

	toJSON() {

		return { min: this.min.toArray(), max: this.max.toArray() };

	}

	fromJSON( json ) {

		this.min.fromArray( json.min );
		this.max.fromArray( json.max );
		return this;

	}

}

Box3.prototype.isBox3 = true;

// Whether the triangle (_v0, _v1, _v2) and the box of half size _extents overlap along
// every axis of `axes` (xyz triples).
function separated( axes ) {

	for ( let i = 0; i + 2 < axes.length; i += 3 ) {

		_axis.fromArray( axes, i );
		const r = _extents.x * Math.abs( _axis.x ) + _extents.y * Math.abs( _axis.y ) + _extents.z * Math.abs( _axis.z );
		const p0 = _v0.dot( _axis ), p1 = _v1.dot( _axis ), p2 = _v2.dot( _axis );
		if ( Math.max( - Math.max( p0, p1, p2 ), Math.min( p0, p1, p2 ) ) > r ) return false;

	}

	return true;

}

const _vector = new Vector3();
const _v0 = new Vector3();
const _v1 = new Vector3();
const _v2 = new Vector3();
const _f0 = new Vector3();
const _f1 = new Vector3();
const _f2 = new Vector3();
const _center = new Vector3();
const _extents = new Vector3();
const _normal = new Vector3();
const _axis = new Vector3();
const _points = Array.from( { length: 8 }, () => new Vector3() );
const _box = new Box3();
