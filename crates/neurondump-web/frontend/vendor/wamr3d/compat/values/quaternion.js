// three.js `Quaternion` of the object API, kept in JavaScript (see `../values.js`).
//
// Behaviour follows three.js (MIT License, Copyright 2010-2026 three.js authors); numbers
// are 64-bit here, as in three.js.

import { clamp } from './common.js';

export class Quaternion {

	constructor( x = 0, y = 0, z = 0, w = 1 ) {

		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;

	}

	static slerpFlat( dst, dstOffset, src0, srcOffset0, src1, srcOffset1, t ) {

		let x0 = src0[ srcOffset0 ], y0 = src0[ srcOffset0 + 1 ], z0 = src0[ srcOffset0 + 2 ], w0 = src0[ srcOffset0 + 3 ];
		let x1 = src1[ srcOffset1 ], y1 = src1[ srcOffset1 + 1 ], z1 = src1[ srcOffset1 + 2 ], w1 = src1[ srcOffset1 + 3 ];

		if ( w0 !== w1 || x0 !== x1 || y0 !== y1 || z0 !== z1 ) {

			let dot = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1;

			if ( dot < 0 ) {

				x1 = - x1;
				y1 = - y1;
				z1 = - z1;
				w1 = - w1;
				dot = - dot;

			}

			let s = 1 - t;

			if ( dot < 0.9995 ) {

				const theta = Math.acos( dot ), sin = Math.sin( theta );
				s = Math.sin( s * theta ) / sin;
				t = Math.sin( t * theta ) / sin;
				x0 = x0 * s + x1 * t;
				y0 = y0 * s + y1 * t;
				z0 = z0 * s + z1 * t;
				w0 = w0 * s + w1 * t;

			} else {

				// nearly parallel: a normalized linear blend
				x0 = x0 * s + x1 * t;
				y0 = y0 * s + y1 * t;
				z0 = z0 * s + z1 * t;
				w0 = w0 * s + w1 * t;
				const f = 1 / Math.sqrt( x0 * x0 + y0 * y0 + z0 * z0 + w0 * w0 );
				x0 *= f;
				y0 *= f;
				z0 *= f;
				w0 *= f;

			}

		}

		dst[ dstOffset ] = x0;
		dst[ dstOffset + 1 ] = y0;
		dst[ dstOffset + 2 ] = z0;
		dst[ dstOffset + 3 ] = w0;

	}

	static multiplyQuaternionsFlat( dst, dstOffset, src0, srcOffset0, src1, srcOffset1 ) {

		const x0 = src0[ srcOffset0 ], y0 = src0[ srcOffset0 + 1 ], z0 = src0[ srcOffset0 + 2 ], w0 = src0[ srcOffset0 + 3 ];
		const x1 = src1[ srcOffset1 ], y1 = src1[ srcOffset1 + 1 ], z1 = src1[ srcOffset1 + 2 ], w1 = src1[ srcOffset1 + 3 ];
		dst[ dstOffset ] = x0 * w1 + w0 * x1 + y0 * z1 - z0 * y1;
		dst[ dstOffset + 1 ] = y0 * w1 + w0 * y1 + z0 * x1 - x0 * z1;
		dst[ dstOffset + 2 ] = z0 * w1 + w0 * z1 + x0 * y1 - y0 * x1;
		dst[ dstOffset + 3 ] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;
		return dst;

	}

	set( x, y, z, w ) {

		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
		return this;

	}

	clone() {

		return new this.constructor( this.x, this.y, this.z, this.w );

	}

	copy( q ) {

		return this.set( q.x, q.y, q.z, q.w );

	}

	setFromEuler( euler ) {

		const x = euler.x, y = euler.y, z = euler.z, order = euler.order;
		const c1 = Math.cos( x / 2 ), c2 = Math.cos( y / 2 ), c3 = Math.cos( z / 2 );
		const s1 = Math.sin( x / 2 ), s2 = Math.sin( y / 2 ), s3 = Math.sin( z / 2 );

		switch ( order ) {

			case 'XYZ': return this.set( s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3 );
			case 'YXZ': return this.set( s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3, c1 * c2 * c3 + s1 * s2 * s3 );
			case 'ZXY': return this.set( s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3 );
			case 'ZYX': return this.set( s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3, c1 * c2 * c3 + s1 * s2 * s3 );
			case 'YZX': return this.set( s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 + s1 * c2 * s3, c1 * c2 * s3 - s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3 );
			case 'XZY': return this.set( s1 * c2 * c3 - c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 + s1 * s2 * s3 );
			default: throw new Error( `WAMR3D: unknown Euler order: ${order}` );

		}

	}

	setFromAxisAngle( axis, angle ) {

		const half = angle / 2, s = Math.sin( half );
		return this.set( axis.x * s, axis.y * s, axis.z * s, Math.cos( half ) );

	}

	setFromRotationMatrix( m ) {

		const te = m.elements;
		const m11 = te[ 0 ], m12 = te[ 4 ], m13 = te[ 8 ];
		const m21 = te[ 1 ], m22 = te[ 5 ], m23 = te[ 9 ];
		const m31 = te[ 2 ], m32 = te[ 6 ], m33 = te[ 10 ];
		const trace = m11 + m22 + m33;

		if ( trace > 0 ) {

			const s = 0.5 / Math.sqrt( trace + 1 );
			return this.set( ( m32 - m23 ) * s, ( m13 - m31 ) * s, ( m21 - m12 ) * s, 0.25 / s );

		}

		if ( m11 > m22 && m11 > m33 ) {

			const s = 2 * Math.sqrt( 1 + m11 - m22 - m33 );
			return this.set( 0.25 * s, ( m12 + m21 ) / s, ( m13 + m31 ) / s, ( m32 - m23 ) / s );

		}

		if ( m22 > m33 ) {

			const s = 2 * Math.sqrt( 1 + m22 - m11 - m33 );
			return this.set( ( m12 + m21 ) / s, 0.25 * s, ( m23 + m32 ) / s, ( m13 - m31 ) / s );

		}

		const s = 2 * Math.sqrt( 1 + m33 - m11 - m22 );
		return this.set( ( m13 + m31 ) / s, ( m23 + m32 ) / s, 0.25 * s, ( m21 - m12 ) / s );

	}

	setFromUnitVectors( vFrom, vTo ) {

		let r = vFrom.dot( vTo ) + 1;

		if ( r < 1e-8 ) {

			// opposite vectors: any axis perpendicular to them
			r = 0;
			if ( Math.abs( vFrom.x ) > Math.abs( vFrom.z ) ) this.set( - vFrom.y, vFrom.x, 0, r );
			else this.set( 0, - vFrom.z, vFrom.y, r );

		} else {

			this.set( vFrom.y * vTo.z - vFrom.z * vTo.y, vFrom.z * vTo.x - vFrom.x * vTo.z, vFrom.x * vTo.y - vFrom.y * vTo.x, r );

		}

		return this.normalize();

	}

	angleTo( q ) {

		return 2 * Math.acos( Math.abs( clamp( this.dot( q ), - 1, 1 ) ) );

	}

	rotateTowards( q, step ) {

		const angle = this.angleTo( q );
		if ( angle === 0 ) return this;
		return this.slerp( q, Math.min( 1, step / angle ) );

	}

	identity() {

		return this.set( 0, 0, 0, 1 );

	}

	invert() {

		return this.conjugate();

	}

	conjugate() {

		return this.set( - this.x, - this.y, - this.z, this.w );

	}

	dot( q ) {

		return this.x * q.x + this.y * q.y + this.z * q.z + this.w * q.w;

	}

	lengthSq() {

		return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;

	}

	length() {

		return Math.sqrt( this.lengthSq() );

	}

	normalize() {

		let l = this.length();
		if ( l === 0 ) return this.set( 0, 0, 0, 1 );
		l = 1 / l;
		return this.set( this.x * l, this.y * l, this.z * l, this.w * l );

	}

	multiply( q ) {

		return this.multiplyQuaternions( this, q );

	}

	premultiply( q ) {

		return this.multiplyQuaternions( q, this );

	}

	multiplyQuaternions( a, b ) {

		const ax = a.x, ay = a.y, az = a.z, aw = a.w;
		const bx = b.x, by = b.y, bz = b.z, bw = b.w;
		return this.set( ax * bw + aw * bx + ay * bz - az * by, ay * bw + aw * by + az * bx - ax * bz, az * bw + aw * bz + ax * by - ay * bx, aw * bw - ax * bx - ay * by - az * bz );

	}

	slerp( qb, t ) {

		let x = qb.x, y = qb.y, z = qb.z, w = qb.w, dot = this.dot( qb );

		if ( dot < 0 ) {

			x = - x;
			y = - y;
			z = - z;
			w = - w;
			dot = - dot;

		}

		let s = 1 - t;

		if ( dot < 0.9995 ) {

			const theta = Math.acos( dot ), sin = Math.sin( theta );
			s = Math.sin( s * theta ) / sin;
			t = Math.sin( t * theta ) / sin;
			return this.set( this.x * s + x * t, this.y * s + y * t, this.z * s + z * t, this.w * s + w * t );

		}

		this.set( this.x * s + x * t, this.y * s + y * t, this.z * s + z * t, this.w * s + w * t );
		return this.normalize();

	}

	slerpQuaternions( qa, qb, t ) {

		return this.copy( qa ).slerp( qb, t );

	}

	random() {

		const theta1 = 2 * Math.PI * Math.random(), theta2 = 2 * Math.PI * Math.random();
		const x0 = Math.random(), r1 = Math.sqrt( 1 - x0 ), r2 = Math.sqrt( x0 );
		return this.set( r1 * Math.sin( theta1 ), r1 * Math.cos( theta1 ), r2 * Math.sin( theta2 ), r2 * Math.cos( theta2 ) );

	}

	equals( q ) {

		return q.x === this.x && q.y === this.y && q.z === this.z && q.w === this.w;

	}

	fromArray( array, offset = 0 ) {

		return this.set( array[ offset ], array[ offset + 1 ], array[ offset + 2 ], array[ offset + 3 ] );

	}

	toArray( array = [], offset = 0 ) {

		array[ offset ] = this.x;
		array[ offset + 1 ] = this.y;
		array[ offset + 2 ] = this.z;
		array[ offset + 3 ] = this.w;
		return array;

	}

	fromBufferAttribute( attribute, index ) {

		return this.set( attribute.getX( index ), attribute.getY( index ), attribute.getZ( index ), attribute.getW( index ) );

	}

	toJSON() {

		return this.toArray();

	}

	* [ Symbol.iterator ]() {

		yield this.x;
		yield this.y;
		yield this.z;
		yield this.w;

	}

}

Quaternion.prototype.isQuaternion = true;
