// three.js `Euler` of the object API, kept in JavaScript (see `../values.js`).
//
// Behaviour follows three.js (MIT License, Copyright 2010-2026 three.js authors); numbers
// are 64-bit here, as in three.js.

import { clamp } from './common.js';
import { Matrix4 } from './matrix4.js';
import { Quaternion } from './quaternion.js';

const ORDERS = [ 'XYZ', 'YXZ', 'ZXY', 'ZYX', 'YZX', 'XZY' ];

export class Euler {

	constructor( x = 0, y = 0, z = 0, order = Euler.DEFAULT_ORDER ) {

		this.x = x;
		this.y = y;
		this._order = order;
		this.z = z;

	}

	get order() {

		return this._order;

	}

	set order( value ) {

		if ( ! ORDERS.includes( value ) ) throw new Error( `WAMR3D: unknown Euler order: ${value}` );
		this._order = value;

	}

	set( x, y, z, order = this._order ) {

		this.x = x;
		this.y = y;
		this.z = z;
		if ( order !== this._order ) this.order = order;
		return this;

	}

	clone() {

		return new this.constructor( this.x, this.y, this.z, this._order );

	}

	copy( euler ) {

		return this.set( euler.x, euler.y, euler.z, euler.order );

	}

	setFromRotationMatrix( m, order = this._order ) {

		const te = m.elements;
		const m11 = te[ 0 ], m12 = te[ 4 ], m13 = te[ 8 ];
		const m21 = te[ 1 ], m22 = te[ 5 ], m23 = te[ 9 ];
		const m31 = te[ 2 ], m32 = te[ 6 ], m33 = te[ 10 ];
		// near gimbal lock the second angle is ±90° and the third is taken as 0
		const free = 0.9999999;

		switch ( order ) {

			case 'XYZ':
				this.y = Math.asin( clamp( m13, - 1, 1 ) );
				if ( Math.abs( m13 ) < free ) {

					this.x = Math.atan2( - m23, m33 );
					this.z = Math.atan2( - m12, m11 );

				} else {

					this.x = Math.atan2( m32, m22 );
					this.z = 0;

				}

				break;

			case 'YXZ':
				this.x = Math.asin( - clamp( m23, - 1, 1 ) );
				if ( Math.abs( m23 ) < free ) {

					this.y = Math.atan2( m13, m33 );
					this.z = Math.atan2( m21, m22 );

				} else {

					this.y = Math.atan2( - m31, m11 );
					this.z = 0;

				}

				break;

			case 'ZXY':
				this.x = Math.asin( clamp( m32, - 1, 1 ) );
				if ( Math.abs( m32 ) < free ) {

					this.y = Math.atan2( - m31, m33 );
					this.z = Math.atan2( - m12, m22 );

				} else {

					this.y = 0;
					this.z = Math.atan2( m21, m11 );

				}

				break;

			case 'ZYX':
				this.y = Math.asin( - clamp( m31, - 1, 1 ) );
				if ( Math.abs( m31 ) < free ) {

					this.x = Math.atan2( m32, m33 );
					this.z = Math.atan2( m21, m11 );

				} else {

					this.x = 0;
					this.z = Math.atan2( - m12, m22 );

				}

				break;

			case 'YZX':
				this.z = Math.asin( clamp( m21, - 1, 1 ) );
				if ( Math.abs( m21 ) < free ) {

					this.x = Math.atan2( - m23, m22 );
					this.y = Math.atan2( - m31, m11 );

				} else {

					this.x = 0;
					this.y = Math.atan2( m13, m33 );

				}

				break;

			case 'XZY':
				this.z = Math.asin( - clamp( m12, - 1, 1 ) );
				if ( Math.abs( m12 ) < free ) {

					this.x = Math.atan2( m32, m22 );
					this.y = Math.atan2( m13, m11 );

				} else {

					this.x = Math.atan2( - m23, m33 );
					this.y = 0;

				}

				break;

			default: throw new Error( `WAMR3D: unknown Euler order: ${order}` );

		}

		this._order = order;
		return this;

	}

	setFromQuaternion( q, order = this._order ) {

		return this.setFromRotationMatrix( _matrix.makeRotationFromQuaternion( q ), order );

	}

	setFromVector3( v, order = this._order ) {

		return this.set( v.x, v.y, v.z, order );

	}

	reorder( newOrder ) {

		return this.setFromQuaternion( _quaternion.setFromEuler( this ), newOrder );

	}

	equals( euler ) {

		return euler.x === this.x && euler.y === this.y && euler.z === this.z && euler.order === this._order;

	}

	fromArray( array ) {

		return this.set( array[ 0 ], array[ 1 ], array[ 2 ], array[ 3 ] ?? this._order );

	}

	toArray( array = [], offset = 0 ) {

		array[ offset ] = this.x;
		array[ offset + 1 ] = this.y;
		array[ offset + 2 ] = this.z;
		array[ offset + 3 ] = this._order;
		return array;

	}

	* [ Symbol.iterator ]() {

		yield this.x;
		yield this.y;
		yield this.z;
		yield this._order;

	}

}

Euler.DEFAULT_ORDER = 'XYZ';
Euler.prototype.isEuler = true;

const _quaternion = new Quaternion();
const _matrix = new Matrix4();
