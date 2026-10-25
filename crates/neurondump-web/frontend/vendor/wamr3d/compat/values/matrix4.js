// three.js `Matrix4` of the object API, kept in JavaScript (see `../values.js`).
//
// Behaviour follows three.js (MIT License, Copyright 2010-2026 three.js authors); numbers
// are 64-bit here, as in three.js.

import { Vector3 } from './vector3.js';

export const WebGLCoordinateSystem = 2000, WebGPUCoordinateSystem = 2001;

export class Matrix4 {

	constructor( n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44 ) {

		this.elements = [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ];
		if ( n11 !== undefined ) this.set( n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44 );

	}

	/** Row-major arguments; `elements` are column-major. */
	set( n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44 ) {

		const te = this.elements;
		te[ 0 ] = n11; te[ 4 ] = n12; te[ 8 ] = n13; te[ 12 ] = n14;
		te[ 1 ] = n21; te[ 5 ] = n22; te[ 9 ] = n23; te[ 13 ] = n24;
		te[ 2 ] = n31; te[ 6 ] = n32; te[ 10 ] = n33; te[ 14 ] = n34;
		te[ 3 ] = n41; te[ 7 ] = n42; te[ 11 ] = n43; te[ 15 ] = n44;
		return this;

	}

	identity() {

		return this.set( 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 );

	}

	clone() {

		return new this.constructor().fromArray( this.elements );

	}

	copy( m ) {

		const te = this.elements, me = m.elements;
		te[ 0 ] = me[ 0 ]; te[ 1 ] = me[ 1 ]; te[ 2 ] = me[ 2 ]; te[ 3 ] = me[ 3 ];
		te[ 4 ] = me[ 4 ]; te[ 5 ] = me[ 5 ]; te[ 6 ] = me[ 6 ]; te[ 7 ] = me[ 7 ];
		te[ 8 ] = me[ 8 ]; te[ 9 ] = me[ 9 ]; te[ 10 ] = me[ 10 ]; te[ 11 ] = me[ 11 ];
		te[ 12 ] = me[ 12 ]; te[ 13 ] = me[ 13 ]; te[ 14 ] = me[ 14 ]; te[ 15 ] = me[ 15 ];
		return this;

	}

	copyPosition( m ) {

		const te = this.elements, me = m.elements;
		te[ 12 ] = me[ 12 ];
		te[ 13 ] = me[ 13 ];
		te[ 14 ] = me[ 14 ];
		return this;

	}

	setFromMatrix3( m ) {

		const me = m.elements;
		return this.set( me[ 0 ], me[ 3 ], me[ 6 ], 0, me[ 1 ], me[ 4 ], me[ 7 ], 0, me[ 2 ], me[ 5 ], me[ 8 ], 0, 0, 0, 0, 1 );

	}

	extractBasis( xAxis, yAxis, zAxis ) {

		if ( this.determinantAffine() === 0 ) {

			xAxis.set( 1, 0, 0 );
			yAxis.set( 0, 1, 0 );
			zAxis.set( 0, 0, 1 );
			return this;

		}

		xAxis.setFromMatrixColumn( this, 0 );
		yAxis.setFromMatrixColumn( this, 1 );
		zAxis.setFromMatrixColumn( this, 2 );
		return this;

	}

	makeBasis( xAxis, yAxis, zAxis ) {

		return this.set( xAxis.x, yAxis.x, zAxis.x, 0, xAxis.y, yAxis.y, zAxis.y, 0, xAxis.z, yAxis.z, zAxis.z, 0, 0, 0, 0, 1 );

	}

	extractRotation( m ) {

		if ( m.determinantAffine() === 0 ) return this.identity();
		const te = this.elements, me = m.elements;
		const scaleX = 1 / _v1.setFromMatrixColumn( m, 0 ).length();
		const scaleY = 1 / _v1.setFromMatrixColumn( m, 1 ).length();
		const scaleZ = 1 / _v1.setFromMatrixColumn( m, 2 ).length();
		te[ 0 ] = me[ 0 ] * scaleX; te[ 1 ] = me[ 1 ] * scaleX; te[ 2 ] = me[ 2 ] * scaleX; te[ 3 ] = 0;
		te[ 4 ] = me[ 4 ] * scaleY; te[ 5 ] = me[ 5 ] * scaleY; te[ 6 ] = me[ 6 ] * scaleY; te[ 7 ] = 0;
		te[ 8 ] = me[ 8 ] * scaleZ; te[ 9 ] = me[ 9 ] * scaleZ; te[ 10 ] = me[ 10 ] * scaleZ; te[ 11 ] = 0;
		te[ 12 ] = 0; te[ 13 ] = 0; te[ 14 ] = 0; te[ 15 ] = 1;
		return this;

	}

	makeRotationFromEuler( euler ) {

		const te = this.elements, order = euler.order;
		const x = euler.x, y = euler.y, z = euler.z;
		const a = Math.cos( x ), b = Math.sin( x ), c = Math.cos( y ), d = Math.sin( y ), e = Math.cos( z ), f = Math.sin( z );

		if ( order === 'XYZ' ) {

			const ae = a * e, af = a * f, be = b * e, bf = b * f;
			te[ 0 ] = c * e; te[ 4 ] = - c * f; te[ 8 ] = d;
			te[ 1 ] = af + be * d; te[ 5 ] = ae - bf * d; te[ 9 ] = - b * c;
			te[ 2 ] = bf - ae * d; te[ 6 ] = be + af * d; te[ 10 ] = a * c;

		} else if ( order === 'YXZ' ) {

			const ce = c * e, cf = c * f, de = d * e, df = d * f;
			te[ 0 ] = ce + df * b; te[ 4 ] = de * b - cf; te[ 8 ] = a * d;
			te[ 1 ] = a * f; te[ 5 ] = a * e; te[ 9 ] = - b;
			te[ 2 ] = cf * b - de; te[ 6 ] = df + ce * b; te[ 10 ] = a * c;

		} else if ( order === 'ZXY' ) {

			const ce = c * e, cf = c * f, de = d * e, df = d * f;
			te[ 0 ] = ce - df * b; te[ 4 ] = - a * f; te[ 8 ] = de + cf * b;
			te[ 1 ] = cf + de * b; te[ 5 ] = a * e; te[ 9 ] = df - ce * b;
			te[ 2 ] = - a * d; te[ 6 ] = b; te[ 10 ] = a * c;

		} else if ( order === 'ZYX' ) {

			const ae = a * e, af = a * f, be = b * e, bf = b * f;
			te[ 0 ] = c * e; te[ 4 ] = be * d - af; te[ 8 ] = ae * d + bf;
			te[ 1 ] = c * f; te[ 5 ] = bf * d + ae; te[ 9 ] = af * d - be;
			te[ 2 ] = - d; te[ 6 ] = b * c; te[ 10 ] = a * c;

		} else if ( order === 'YZX' ) {

			const ac = a * c, ad = a * d, bc = b * c, bd = b * d;
			te[ 0 ] = c * e; te[ 4 ] = bd - ac * f; te[ 8 ] = bc * f + ad;
			te[ 1 ] = f; te[ 5 ] = a * e; te[ 9 ] = - b * e;
			te[ 2 ] = - d * e; te[ 6 ] = ad * f + bc; te[ 10 ] = ac - bd * f;

		} else if ( order === 'XZY' ) {

			const ac = a * c, ad = a * d, bc = b * c, bd = b * d;
			te[ 0 ] = c * e; te[ 4 ] = - f; te[ 8 ] = d * e;
			te[ 1 ] = ac * f + bd; te[ 5 ] = a * e; te[ 9 ] = ad * f - bc;
			te[ 2 ] = bc * f - ad; te[ 6 ] = b * e; te[ 10 ] = bd * f + ac;

		}

		te[ 3 ] = 0; te[ 7 ] = 0; te[ 11 ] = 0;
		te[ 12 ] = 0; te[ 13 ] = 0; te[ 14 ] = 0; te[ 15 ] = 1;
		return this;

	}

	makeRotationFromQuaternion( q ) {

		return this.compose( _zero, q, _one );

	}

	lookAt( eye, target, up ) {

		const te = this.elements;
		_z.subVectors( eye, target );
		// eye and target in the same place
		if ( _z.lengthSq() === 0 ) _z.z = 1;
		_z.normalize();
		_x.crossVectors( up, _z );

		if ( _x.lengthSq() === 0 ) {

			// up parallel to the view direction: nudge the direction
			if ( Math.abs( up.z ) === 1 ) _z.x += 0.0001;
			else _z.z += 0.0001;
			_z.normalize();
			_x.crossVectors( up, _z );

		}

		_x.normalize();
		_y.crossVectors( _z, _x );
		te[ 0 ] = _x.x; te[ 4 ] = _y.x; te[ 8 ] = _z.x;
		te[ 1 ] = _x.y; te[ 5 ] = _y.y; te[ 9 ] = _z.y;
		te[ 2 ] = _x.z; te[ 6 ] = _y.z; te[ 10 ] = _z.z;
		return this;

	}

	multiply( m ) {

		return this.multiplyMatrices( this, m );

	}

	premultiply( m ) {

		return this.multiplyMatrices( m, this );

	}

	multiplyMatrices( a, b ) {

		const ae = a.elements, be = b.elements, te = this.elements;
		const a11 = ae[ 0 ], a12 = ae[ 4 ], a13 = ae[ 8 ], a14 = ae[ 12 ];
		const a21 = ae[ 1 ], a22 = ae[ 5 ], a23 = ae[ 9 ], a24 = ae[ 13 ];
		const a31 = ae[ 2 ], a32 = ae[ 6 ], a33 = ae[ 10 ], a34 = ae[ 14 ];
		const a41 = ae[ 3 ], a42 = ae[ 7 ], a43 = ae[ 11 ], a44 = ae[ 15 ];
		const b11 = be[ 0 ], b12 = be[ 4 ], b13 = be[ 8 ], b14 = be[ 12 ];
		const b21 = be[ 1 ], b22 = be[ 5 ], b23 = be[ 9 ], b24 = be[ 13 ];
		const b31 = be[ 2 ], b32 = be[ 6 ], b33 = be[ 10 ], b34 = be[ 14 ];
		const b41 = be[ 3 ], b42 = be[ 7 ], b43 = be[ 11 ], b44 = be[ 15 ];
		te[ 0 ] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
		te[ 4 ] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
		te[ 8 ] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
		te[ 12 ] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;
		te[ 1 ] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
		te[ 5 ] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
		te[ 9 ] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
		te[ 13 ] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;
		te[ 2 ] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
		te[ 6 ] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
		te[ 10 ] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
		te[ 14 ] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;
		te[ 3 ] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
		te[ 7 ] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
		te[ 11 ] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
		te[ 15 ] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;
		return this;

	}

	multiplyScalar( s ) {

		const te = this.elements;
		for ( let i = 0; i < 16; i ++ ) te[ i ] *= s;
		return this;

	}

	determinant() {

		const te = this.elements;
		const n11 = te[ 0 ], n12 = te[ 4 ], n13 = te[ 8 ], n14 = te[ 12 ];
		const n21 = te[ 1 ], n22 = te[ 5 ], n23 = te[ 9 ], n24 = te[ 13 ];
		const n31 = te[ 2 ], n32 = te[ 6 ], n33 = te[ 10 ], n34 = te[ 14 ];
		const n41 = te[ 3 ], n42 = te[ 7 ], n43 = te[ 11 ], n44 = te[ 15 ];
		const t11 = n23 * n34 - n24 * n33, t12 = n22 * n34 - n24 * n32, t13 = n22 * n33 - n23 * n32;
		const t21 = n21 * n34 - n24 * n31, t22 = n21 * n33 - n23 * n31, t23 = n21 * n32 - n22 * n31;
		return n11 * ( n42 * t11 - n43 * t12 + n44 * t13 ) - n12 * ( n41 * t11 - n43 * t21 + n44 * t22 ) + n13 * ( n41 * t12 - n42 * t21 + n44 * t23 ) - n14 * ( n41 * t13 - n42 * t22 + n43 * t23 );

	}

	determinantAffine() {

		const te = this.elements;
		const n11 = te[ 0 ], n12 = te[ 4 ], n13 = te[ 8 ], n21 = te[ 1 ], n22 = te[ 5 ], n23 = te[ 9 ], n31 = te[ 2 ], n32 = te[ 6 ], n33 = te[ 10 ];
		return n11 * ( n22 * n33 - n23 * n32 ) - n12 * ( n21 * n33 - n23 * n31 ) + n13 * ( n21 * n32 - n22 * n31 );

	}

	transpose() {

		const te = this.elements;
		let t = te[ 1 ]; te[ 1 ] = te[ 4 ]; te[ 4 ] = t;
		t = te[ 2 ]; te[ 2 ] = te[ 8 ]; te[ 8 ] = t;
		t = te[ 6 ]; te[ 6 ] = te[ 9 ]; te[ 9 ] = t;
		t = te[ 3 ]; te[ 3 ] = te[ 12 ]; te[ 12 ] = t;
		t = te[ 7 ]; te[ 7 ] = te[ 13 ]; te[ 13 ] = t;
		t = te[ 11 ]; te[ 11 ] = te[ 14 ]; te[ 14 ] = t;
		return this;

	}

	setPosition( x, y, z ) {

		const te = this.elements;

		if ( typeof x === 'object' ) {

			te[ 12 ] = x.x;
			te[ 13 ] = x.y;
			te[ 14 ] = x.z;

		} else {

			te[ 12 ] = x;
			te[ 13 ] = y;
			te[ 14 ] = z;

		}

		return this;

	}

	invert() {

		const te = this.elements;
		const n11 = te[ 0 ], n21 = te[ 1 ], n31 = te[ 2 ], n41 = te[ 3 ];
		const n12 = te[ 4 ], n22 = te[ 5 ], n32 = te[ 6 ], n42 = te[ 7 ];
		const n13 = te[ 8 ], n23 = te[ 9 ], n33 = te[ 10 ], n43 = te[ 11 ];
		const n14 = te[ 12 ], n24 = te[ 13 ], n34 = te[ 14 ], n44 = te[ 15 ];
		const t1 = n11 * n22 - n21 * n12, t2 = n11 * n32 - n31 * n12, t3 = n11 * n42 - n41 * n12;
		const t4 = n21 * n32 - n31 * n22, t5 = n21 * n42 - n41 * n22, t6 = n31 * n42 - n41 * n32;
		const t7 = n13 * n24 - n23 * n14, t8 = n13 * n34 - n33 * n14, t9 = n13 * n44 - n43 * n14;
		const t10 = n23 * n34 - n33 * n24, t11 = n23 * n44 - n43 * n24, t12 = n33 * n44 - n43 * n34;
		const det = t1 * t12 - t2 * t11 + t3 * t10 + t4 * t9 - t5 * t8 + t6 * t7;
		if ( det === 0 ) return this.set( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
		const detInv = 1 / det;
		te[ 0 ] = ( n22 * t12 - n32 * t11 + n42 * t10 ) * detInv;
		te[ 1 ] = ( n31 * t11 - n21 * t12 - n41 * t10 ) * detInv;
		te[ 2 ] = ( n24 * t6 - n34 * t5 + n44 * t4 ) * detInv;
		te[ 3 ] = ( n33 * t5 - n23 * t6 - n43 * t4 ) * detInv;
		te[ 4 ] = ( n32 * t9 - n12 * t12 - n42 * t8 ) * detInv;
		te[ 5 ] = ( n11 * t12 - n31 * t9 + n41 * t8 ) * detInv;
		te[ 6 ] = ( n34 * t3 - n14 * t6 - n44 * t2 ) * detInv;
		te[ 7 ] = ( n13 * t6 - n33 * t3 + n43 * t2 ) * detInv;
		te[ 8 ] = ( n12 * t11 - n22 * t9 + n42 * t7 ) * detInv;
		te[ 9 ] = ( n21 * t9 - n11 * t11 - n41 * t7 ) * detInv;
		te[ 10 ] = ( n14 * t5 - n24 * t3 + n44 * t1 ) * detInv;
		te[ 11 ] = ( n23 * t3 - n13 * t5 - n43 * t1 ) * detInv;
		te[ 12 ] = ( n22 * t8 - n12 * t10 - n32 * t7 ) * detInv;
		te[ 13 ] = ( n11 * t10 - n21 * t8 + n31 * t7 ) * detInv;
		te[ 14 ] = ( n24 * t2 - n14 * t4 - n34 * t1 ) * detInv;
		te[ 15 ] = ( n13 * t4 - n23 * t2 + n33 * t1 ) * detInv;
		return this;

	}

	scale( v ) {

		const te = this.elements, x = v.x, y = v.y, z = v.z;
		te[ 0 ] *= x; te[ 4 ] *= y; te[ 8 ] *= z;
		te[ 1 ] *= x; te[ 5 ] *= y; te[ 9 ] *= z;
		te[ 2 ] *= x; te[ 6 ] *= y; te[ 10 ] *= z;
		te[ 3 ] *= x; te[ 7 ] *= y; te[ 11 ] *= z;
		return this;

	}

	getMaxScaleOnAxis() {

		const te = this.elements;
		const x = te[ 0 ] * te[ 0 ] + te[ 1 ] * te[ 1 ] + te[ 2 ] * te[ 2 ];
		const y = te[ 4 ] * te[ 4 ] + te[ 5 ] * te[ 5 ] + te[ 6 ] * te[ 6 ];
		const z = te[ 8 ] * te[ 8 ] + te[ 9 ] * te[ 9 ] + te[ 10 ] * te[ 10 ];
		return Math.sqrt( Math.max( x, y, z ) );

	}

	makeTranslation( x, y, z ) {

		return typeof x === 'object'
			? this.set( 1, 0, 0, x.x, 0, 1, 0, x.y, 0, 0, 1, x.z, 0, 0, 0, 1 )
			: this.set( 1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1 );

	}

	makeRotationX( theta ) {

		const c = Math.cos( theta ), s = Math.sin( theta );
		return this.set( 1, 0, 0, 0, 0, c, - s, 0, 0, s, c, 0, 0, 0, 0, 1 );

	}

	makeRotationY( theta ) {

		const c = Math.cos( theta ), s = Math.sin( theta );
		return this.set( c, 0, s, 0, 0, 1, 0, 0, - s, 0, c, 0, 0, 0, 0, 1 );

	}

	makeRotationZ( theta ) {

		const c = Math.cos( theta ), s = Math.sin( theta );
		return this.set( c, - s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 );

	}

	makeRotationAxis( axis, angle ) {

		const c = Math.cos( angle ), s = Math.sin( angle ), t = 1 - c;
		const x = axis.x, y = axis.y, z = axis.z, tx = t * x, ty = t * y;
		return this.set(
			tx * x + c, tx * y - s * z, tx * z + s * y, 0,
			tx * y + s * z, ty * y + c, ty * z - s * x, 0,
			tx * z - s * y, ty * z + s * x, t * z * z + c, 0,
			0, 0, 0, 1
		);

	}

	makeScale( x, y, z ) {

		return this.set( x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1 );

	}

	makeShear( xy, xz, yx, yz, zx, zy ) {

		return this.set( 1, yx, zx, 0, xy, 1, zy, 0, xz, yz, 1, 0, 0, 0, 0, 1 );

	}

	compose( position, quaternion, scale ) {

		const te = this.elements;
		const x = quaternion.x, y = quaternion.y, z = quaternion.z, w = quaternion.w;
		const x2 = x + x, y2 = y + y, z2 = z + z;
		const xx = x * x2, xy = x * y2, xz = x * z2, yy = y * y2, yz = y * z2, zz = z * z2;
		const wx = w * x2, wy = w * y2, wz = w * z2;
		const sx = scale.x, sy = scale.y, sz = scale.z;
		te[ 0 ] = ( 1 - ( yy + zz ) ) * sx; te[ 1 ] = ( xy + wz ) * sx; te[ 2 ] = ( xz - wy ) * sx; te[ 3 ] = 0;
		te[ 4 ] = ( xy - wz ) * sy; te[ 5 ] = ( 1 - ( xx + zz ) ) * sy; te[ 6 ] = ( yz + wx ) * sy; te[ 7 ] = 0;
		te[ 8 ] = ( xz + wy ) * sz; te[ 9 ] = ( yz - wx ) * sz; te[ 10 ] = ( 1 - ( xx + yy ) ) * sz; te[ 11 ] = 0;
		te[ 12 ] = position.x; te[ 13 ] = position.y; te[ 14 ] = position.z; te[ 15 ] = 1;
		return this;

	}

	decompose( position, quaternion, scale ) {

		const te = this.elements;
		position.set( te[ 12 ], te[ 13 ], te[ 14 ] );
		const det = this.determinantAffine();

		if ( det === 0 ) {

			scale.set( 1, 1, 1 );
			quaternion.identity();
			return this;

		}

		let sx = _v1.set( te[ 0 ], te[ 1 ], te[ 2 ] ).length();
		const sy = _v1.set( te[ 4 ], te[ 5 ], te[ 6 ] ).length();
		const sz = _v1.set( te[ 8 ], te[ 9 ], te[ 10 ] ).length();
		// a negative determinant means a mirroring: it is put on x
		if ( det < 0 ) sx = - sx;
		const me = _m1.copy( this ).elements;
		me[ 0 ] /= sx; me[ 1 ] /= sx; me[ 2 ] /= sx;
		me[ 4 ] /= sy; me[ 5 ] /= sy; me[ 6 ] /= sy;
		me[ 8 ] /= sz; me[ 9 ] /= sz; me[ 10 ] /= sz;
		quaternion.setFromRotationMatrix( _m1 );
		scale.set( sx, sy, sz );
		return this;

	}

	makePerspective( left, right, top, bottom, near, far, coordinateSystem = WebGLCoordinateSystem, reversedDepth = false ) {

		const te = this.elements;
		const x = 2 * near / ( right - left ), y = 2 * near / ( top - bottom );
		const a = ( right + left ) / ( right - left ), b = ( top + bottom ) / ( top - bottom );
		const [ c, d ] = depthRange( near, far, coordinateSystem, reversedDepth, true );
		te[ 0 ] = x; te[ 4 ] = 0; te[ 8 ] = a; te[ 12 ] = 0;
		te[ 1 ] = 0; te[ 5 ] = y; te[ 9 ] = b; te[ 13 ] = 0;
		te[ 2 ] = 0; te[ 6 ] = 0; te[ 10 ] = c; te[ 14 ] = d;
		te[ 3 ] = 0; te[ 7 ] = 0; te[ 11 ] = - 1; te[ 15 ] = 0;
		return this;

	}

	makeOrthographic( left, right, top, bottom, near, far, coordinateSystem = WebGLCoordinateSystem, reversedDepth = false ) {

		const te = this.elements;
		const x = 2 / ( right - left ), y = 2 / ( top - bottom );
		const a = - ( right + left ) / ( right - left ), b = - ( top + bottom ) / ( top - bottom );
		const [ c, d ] = depthRange( near, far, coordinateSystem, reversedDepth, false );
		te[ 0 ] = x; te[ 4 ] = 0; te[ 8 ] = 0; te[ 12 ] = a;
		te[ 1 ] = 0; te[ 5 ] = y; te[ 9 ] = 0; te[ 13 ] = b;
		te[ 2 ] = 0; te[ 6 ] = 0; te[ 10 ] = c; te[ 14 ] = d;
		te[ 3 ] = 0; te[ 7 ] = 0; te[ 11 ] = 0; te[ 15 ] = 1;
		return this;

	}

	equals( matrix ) {

		const te = this.elements, me = matrix.elements;
		for ( let i = 0; i < 16; i ++ ) if ( te[ i ] !== me[ i ] ) return false;
		return true;

	}

	fromArray( array, offset = 0 ) {

		const te = this.elements;
		for ( let i = 0; i < 16; i ++ ) te[ i ] = array[ i + offset ];
		return this;

	}

	toArray( array = [], offset = 0 ) {

		const te = this.elements;
		for ( let i = 0; i < 16; i ++ ) array[ offset + i ] = te[ i ];
		return array;

	}

}

Matrix4.prototype.isMatrix4 = true;

// The depth terms (third row) of a projection: scale and offset mapping near..far to
// -1..1 (WebGL), 0..1 (WebGPU), or 1..0 with reversed depth.
function depthRange( near, far, coordinateSystem, reversedDepth, perspective ) {

	const range = far - near;
	if ( reversedDepth ) return perspective ? [ near / range, far * near / range ] : [ 1 / range, far / range ];
	if ( coordinateSystem === WebGLCoordinateSystem ) return perspective ? [ - ( far + near ) / range, - 2 * far * near / range ] : [ - 2 / range, - ( far + near ) / range ];
	if ( coordinateSystem === WebGPUCoordinateSystem ) return perspective ? [ - far / range, - far * near / range ] : [ - 1 / range, - near / range ];
	throw new Error( `WAMR3D: invalid coordinate system: ${coordinateSystem}` );

}

const _v1 = new Vector3();
const _x = new Vector3();
const _y = new Vector3();
const _z = new Vector3();
const _zero = new Vector3( 0, 0, 0 );
const _one = new Vector3( 1, 1, 1 );
const _m1 = new Matrix4();
