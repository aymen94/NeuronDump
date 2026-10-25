// `object.position` and the other parts of a transform, and `object.layers`.

import { ORDERS, QUATERNION, ROTATION, flush, mark, ready, w } from './engine.js';
import { Euler, Quaternion, Vector3 } from './math.js';

class Bound {

	constructor( owner, slot, part, other = 0 ) {

		this._o = owner;
		this._s = slot;
		this._p = part;
		this._q = other;

	}

	// Replaces the whole part, so nothing has to be read back first.
	_write( count, a, b, c, d ) {

		const object = this._o, t = object._t, s = this._s;
		if ( ( object._dirty & this._q ) !== 0 ) flush();
		t[ s ] = a;
		t[ s + 1 ] = b;
		t[ s + 2 ] = c;
		if ( count === 4 ) t[ s + 3 ] = d;
		object._stale &= ~ this._p;
		mark( object, this._p );
		return this;

	}

}

function components( Class, names ) {

	names.forEach( ( name, i ) => Object.defineProperty( Class.prototype, name, {
		get() {

			const object = this._o;
			if ( ( object._dirty & this._q ) !== 0 || ( object._stale & this._p ) !== 0 ) ready( object, this._p, this._q );
			return object._t[ this._s + i ];

		},
		set( value ) {

			const object = this._o;
			if ( ( object._dirty & this._q ) !== 0 || ( object._stale & this._p ) !== 0 ) ready( object, this._p, this._q );
			object._t[ this._s + i ] = value;
			mark( object, this._p );

		}
	} ) );

}

function delegate( Class, Value ) {

	for ( const key of Object.getOwnPropertyNames( Value.prototype ) ) {

		if ( key === 'constructor' || key in Class.prototype || typeof Object.getOwnPropertyDescriptor( Value.prototype, key ).value !== 'function' ) continue;

		Class.prototype[ key ] = function ( ...args ) {

			const value = this.clone(), result = value[ key ]( ...args );
			if ( result !== value ) return result;
			return this.copy( value );

		};

	}

}

export class BoundVector3 extends Bound {

	set( x, y, z ) {

		return this._write( 3, x, y, z );

	}

	setScalar( value ) {

		return this._write( 3, value, value, value );

	}

	copy( v ) {

		return this._write( 3, v.x, v.y, v.z );

	}

	// Without the temporary value the delegated methods make: animation writes these every frame.
	fromArray( array, offset = 0 ) {

		return this._write( 3, array[ offset ], array[ offset + 1 ], array[ offset + 2 ] );

	}

	toArray( array = [], offset = 0 ) {

		array[ offset ] = this.x;
		array[ offset + 1 ] = this.y;
		array[ offset + 2 ] = this.z;
		return array;

	}

	clone() {

		return new Vector3( this.x, this.y, this.z );

	}

}

export class BoundEuler extends Bound {

	get order() {

		const object = this._o;
		ready( object, ROTATION, QUATERNION );
		return ORDERS[ object._t[ 16 ] ];

	}

	set order( value ) {

		const object = this._o, index = ORDERS.indexOf( value );
		if ( index === - 1 ) throw new Error( `WAMR3D: unknown Euler order: ${value}` );
		ready( object, ROTATION, QUATERNION );
		object._t[ 16 ] = index;
		mark( object, ROTATION );

	}

	set( x, y, z, order ) {

		if ( order !== undefined ) {

			const index = ORDERS.indexOf( order );
			if ( index === - 1 ) throw new Error( `WAMR3D: unknown Euler order: ${order}` );
			this._o._t[ 16 ] = index;

		}

		return this._write( 3, x, y, z );

	}

	copy( e ) {

		return this.set( e.x, e.y, e.z, e.order );

	}

	clone() {

		return new Euler( this.x, this.y, this.z, this.order );

	}

}

export class BoundQuaternion extends Bound {

	set( x, y, z, w_ ) {

		return this._write( 4, x, y, z, w_ );

	}

	copy( q ) {

		return this._write( 4, q.x, q.y, q.z, q.w );

	}

	fromArray( array, offset = 0 ) {

		return this._write( 4, array[ offset ], array[ offset + 1 ], array[ offset + 2 ], array[ offset + 3 ] );

	}

	toArray( array = [], offset = 0 ) {

		array[ offset ] = this.x;
		array[ offset + 1 ] = this.y;
		array[ offset + 2 ] = this.z;
		array[ offset + 3 ] = this.w;
		return array;

	}

	clone() {

		return new Quaternion( this.x, this.y, this.z, this.w );

	}

}

components( BoundVector3, [ 'x', 'y', 'z' ] );
components( BoundEuler, [ 'x', 'y', 'z' ] );
components( BoundQuaternion, [ 'x', 'y', 'z', 'w' ] );
delegate( BoundVector3, Vector3 );
delegate( BoundEuler, Euler );
delegate( BoundQuaternion, Quaternion );
BoundVector3.prototype.isVector3 = BoundEuler.prototype.isEuler = BoundQuaternion.prototype.isQuaternion = true;

/** three.js `Layers`: a 32-bit mask of the layers an object, camera or raycaster is in. */
export class Layers {

	constructor() {

		this.mask = 1;

	}

	set( layer ) {

		this.mask = ( 1 << layer ) >>> 0;

	}

	enable( layer ) {

		this.mask = ( this.mask | ( 1 << layer ) ) >>> 0;

	}

	enableAll() {

		this.mask = 0xffffffff;

	}

	toggle( layer ) {

		this.mask = ( this.mask ^ ( 1 << layer ) ) >>> 0;

	}

	disable( layer ) {

		this.mask = ( this.mask & ~ ( 1 << layer ) ) >>> 0;

	}

	disableAll() {

		this.mask = 0;

	}

	test( layers ) {

		return ( this.mask & layers.mask ) !== 0;

	}

	isEnabled( layer ) {

		return ( this.mask & ( 1 << layer ) ) !== 0;

	}

}

/** The layers of an engine object: the mask lives in the engine. */
export class BoundLayers extends Layers {

	constructor( owner ) {

		super();
		this._o = owner;
		this._m = undefined;

	}

	// The mask is kept here after the first read, so tests and toggles cost no engine call.
	get mask() {

		return this._m ??= w().layers( this._o.handle );

	}

	set mask( value ) {

		// the base constructor's default is the engine's default
		if ( this._o === undefined ) return;
		w().setLayers( this._o.handle, this._m = value >>> 0 );

	}

}
