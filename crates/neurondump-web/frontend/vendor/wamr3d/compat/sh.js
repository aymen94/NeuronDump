// Third-order spherical harmonics with Vector3 coefficients, for light probes.

import { Vector3 } from './values.js';

export class SphericalHarmonics3 {

	constructor() {

		this.coefficients = Array.from( { length: 9 }, () => new Vector3() );

	}

	set( coefficients ) {

		for ( let i = 0; i < 9; i ++ ) this.coefficients[ i ].copy( coefficients[ i ] );
		return this;

	}

	zero() {

		for ( const c of this.coefficients ) c.set( 0, 0, 0 );
		return this;

	}

	/** The radiance in direction `normal` (unit length). */
	getAt( normal, target ) {

		const { x, y, z } = normal, c = this.coefficients;
		target.copy( c[ 0 ] ).multiplyScalar( 0.282095 );
		target.addScaledVector( c[ 1 ], 0.488603 * y );
		target.addScaledVector( c[ 2 ], 0.488603 * z );
		target.addScaledVector( c[ 3 ], 0.488603 * x );
		target.addScaledVector( c[ 4 ], 1.092548 * ( x * y ) );
		target.addScaledVector( c[ 5 ], 1.092548 * ( y * z ) );
		target.addScaledVector( c[ 6 ], 0.315392 * ( 3.0 * z * z - 1.0 ) );
		target.addScaledVector( c[ 7 ], 1.092548 * ( x * z ) );
		target.addScaledVector( c[ 8 ], 0.546274 * ( x * x - y * y ) );
		return target;

	}

	/** The irradiance (radiance convolved with the cosine lobe) around `normal`. */
	getIrradianceAt( normal, target ) {

		const { x, y, z } = normal, c = this.coefficients;
		target.copy( c[ 0 ] ).multiplyScalar( 0.886227 );
		target.addScaledVector( c[ 1 ], 2.0 * 0.511664 * y );
		target.addScaledVector( c[ 2 ], 2.0 * 0.511664 * z );
		target.addScaledVector( c[ 3 ], 2.0 * 0.511664 * x );
		target.addScaledVector( c[ 4 ], 2.0 * 0.429043 * x * y );
		target.addScaledVector( c[ 5 ], 2.0 * 0.429043 * y * z );
		target.addScaledVector( c[ 6 ], 0.743125 * z * z - 0.247708 );
		target.addScaledVector( c[ 7 ], 2.0 * 0.429043 * x * z );
		target.addScaledVector( c[ 8 ], 0.429043 * ( x * x - y * y ) );
		return target;

	}

	add( sh ) {

		for ( let i = 0; i < 9; i ++ ) this.coefficients[ i ].add( sh.coefficients[ i ] );
		return this;

	}

	addScaledSH( sh, s ) {

		for ( let i = 0; i < 9; i ++ ) this.coefficients[ i ].addScaledVector( sh.coefficients[ i ], s );
		return this;

	}

	scale( s ) {

		for ( const c of this.coefficients ) c.multiplyScalar( s );
		return this;

	}

	lerp( sh, alpha ) {

		for ( let i = 0; i < 9; i ++ ) this.coefficients[ i ].lerp( sh.coefficients[ i ], alpha );
		return this;

	}

	equals( sh ) {

		return this.coefficients.every( ( c, i ) => c.equals( sh.coefficients[ i ] ) );

	}

	copy( sh ) {

		return this.set( sh.coefficients );

	}

	clone() {

		return new this.constructor().copy( this );

	}

	fromArray( array, offset = 0 ) {

		for ( let i = 0; i < 9; i ++ ) this.coefficients[ i ].fromArray( array, offset + i * 3 );
		return this;

	}

	toArray( array = [], offset = 0 ) {

		for ( let i = 0; i < 9; i ++ ) this.coefficients[ i ].toArray( array, offset + i * 3 );
		return array;

	}

	/** The nine basis values in direction `normal` (unit length). */
	static getBasisAt( normal, shBasis ) {

		const { x, y, z } = normal;
		shBasis[ 0 ] = 0.282095;
		shBasis[ 1 ] = 0.488603 * y;
		shBasis[ 2 ] = 0.488603 * z;
		shBasis[ 3 ] = 0.488603 * x;
		shBasis[ 4 ] = 1.092548 * x * y;
		shBasis[ 5 ] = 1.092548 * y * z;
		shBasis[ 6 ] = 0.315392 * ( 3 * z * z - 1 );
		shBasis[ 7 ] = 1.092548 * x * z;
		shBasis[ 8 ] = 0.546274 * ( x * x - y * y );

	}

}

SphericalHarmonics3.prototype.isSphericalHarmonics3 = true;
