// ColorManagement and Color.
//
// Part of the math classes; `../math.js` gathers them.

import * as W from '../../index.js';
import { LinearSRGBColorSpace, NoColorSpace, SRGBColorSpace } from '../constants.js';
import { ignored } from '../engine.js';

/**
 * Colours are stored in the linear working colour space (`LinearSRGBColorSpace`), as with
 * three.js colour management. Only `SRGBColorSpace` and `LinearSRGBColorSpace` are known.
 */
export const ColorManagement = {

	get enabled() {

		return true;

	},

	set enabled( value ) {

		if ( ! value ) ignored( 'ColorManagement.enabled = false' );

	},

	workingColorSpace: LinearSRGBColorSpace,

	/** `color` from colour space `source` to `target` (sRGB or linear sRGB; no-colour-space passes through). */
	convert( color, source, target ) {

		if ( source === target || ! source || ! target ) return color;
		if ( linearSpace( source ) && ! linearSpace( target ) ) return color.convertLinearToSRGB();
		if ( ! linearSpace( source ) && linearSpace( target ) ) return color.convertSRGBToLinear();
		return color;

	},

	toWorkingColorSpace( color, colorSpace ) {

		return this.convert( color, colorSpace, this.workingColorSpace );

	},

	fromWorkingColorSpace( color, colorSpace ) {

		return this.convert( color, this.workingColorSpace, colorSpace );

	}

};

function linearSpace( colorSpace ) {

	if ( colorSpace === LinearSRGBColorSpace || colorSpace === NoColorSpace ) return true;
	if ( colorSpace !== SRGBColorSpace ) throw new Error( `WAMR3D: unknown colour space: ${colorSpace}` );
	return false;

}

const srgbToLinear = ( c ) => ( c < 0.04045 ? c * 0.0773993808 : Math.pow( c * 0.9478672986 + 0.0521327014, 2.4 ) );
const linearToSRGB = ( c ) => ( c < 0.0031308 ? c * 12.92 : 1.055 * Math.pow( c, 0.41666 ) - 0.055 );

function hueToRGB( p, q, t ) {

	if ( t < 0 ) t += 1;
	if ( t > 1 ) t -= 1;
	if ( t < 1 / 6 ) return p + ( q - p ) * 6 * t;
	if ( t < 1 / 2 ) return q;
	if ( t < 2 / 3 ) return p + ( q - p ) * 6 * ( 2 / 3 - t );
	return p;

}

// CSS colours (names, `#rgb`, `rgb()`, `hsl()`) are parsed by the engine; it gives the
// linear components.
let parser = null;

/** Components in the linear working colour space; `r`, `g` and `b` are plain numbers. */
export class Color {

	/** `new Color()`, `new Color(0xff8800)`, `new Color('orange')` or `new Color(r, g, b)`. */
	constructor( r, g, b ) {

		this.r = 1;
		this.g = 1;
		this.b = 1;
		if ( r !== undefined ) this.set( r, g, b );

	}

	set( r, g, b ) {

		if ( g === undefined && b === undefined ) {

			if ( r && r.isColor ) this.copy( r );
			else if ( typeof r === 'number' ) this.setHex( r );
			else if ( typeof r === 'string' ) this.setStyle( r );

		} else {

			this.setRGB( r, g, b );

		}

		return this;

	}

	setScalar( s ) {

		this.r = s;
		this.g = s;
		this.b = s;
		return this;

	}

	/** Components in `colorSpace` (default: the linear working space). */
	setRGB( r, g, b, colorSpace = LinearSRGBColorSpace ) {

		this.r = r;
		this.g = g;
		this.b = b;
		if ( ! linearSpace( colorSpace ) ) this.convertSRGBToLinear();
		return this;

	}

	/** A hex number in `colorSpace` (default: sRGB). */
	setHex( hex, colorSpace = SRGBColorSpace ) {

		hex = Math.floor( hex );
		return this.setRGB( ( hex >> 16 & 255 ) / 255, ( hex >> 8 & 255 ) / 255, ( hex & 255 ) / 255, colorSpace );

	}

	/** Hue, saturation and lightness in `colorSpace` (default: the linear working space). */
	setHSL( h, s, l, colorSpace = LinearSRGBColorSpace ) {

		h = ( ( h % 1 ) + 1 ) % 1;
		s = Math.min( Math.max( s, 0 ), 1 );
		l = Math.min( Math.max( l, 0 ), 1 );

		if ( s === 0 ) {

			this.r = this.g = this.b = l;

		} else {

			const p = l <= 0.5 ? l * ( 1 + s ) : l + s - l * s, q = 2 * l - p;
			this.r = hueToRGB( q, p, h + 1 / 3 );
			this.g = hueToRGB( q, p, h );
			this.b = hueToRGB( q, p, h - 1 / 3 );

		}

		if ( ! linearSpace( colorSpace ) ) this.convertSRGBToLinear();
		return this;

	}

	/** A CSS colour whose numbers are in `colorSpace` (default: sRGB). */
	setStyle( style, colorSpace = SRGBColorSpace ) {

		parser ??= new W.Color( 1, 1, 1 );
		parser.setRGB( this.r, this.g, this.b );
		parser.setStyle( style );
		if ( linearSpace( colorSpace ) ) parser.convertLinearToSRGB();
		this.r = parser.r;
		this.g = parser.g;
		this.b = parser.b;
		return this;

	}

	setColorName( style, colorSpace = SRGBColorSpace ) {

		return this.setStyle( style, colorSpace );

	}

	clone() {

		return new this.constructor( this.r, this.g, this.b );

	}

	copy( color ) {

		this.r = color.r;
		this.g = color.g;
		this.b = color.b;
		return this;

	}

	copySRGBToLinear( color ) {

		this.r = srgbToLinear( color.r );
		this.g = srgbToLinear( color.g );
		this.b = srgbToLinear( color.b );
		return this;

	}

	copyLinearToSRGB( color ) {

		this.r = linearToSRGB( color.r );
		this.g = linearToSRGB( color.g );
		this.b = linearToSRGB( color.b );
		return this;

	}

	convertSRGBToLinear() {

		return this.copySRGBToLinear( this );

	}

	convertLinearToSRGB() {

		return this.copyLinearToSRGB( this );

	}

	// The components in `colorSpace`, in `scratchColor`.
	_in( colorSpace ) {

		scratchColor.copy( this );
		if ( ! linearSpace( colorSpace ) ) scratchColor.convertLinearToSRGB();
		return scratchColor;

	}

	getHex( colorSpace = SRGBColorSpace ) {

		const c = this._in( colorSpace ), byte = ( value ) => Math.round( Math.min( Math.max( value * 255, 0 ), 255 ) );
		return byte( c.r ) * 65536 + byte( c.g ) * 256 + byte( c.b );

	}

	getHexString( colorSpace = SRGBColorSpace ) {

		return ( '000000' + this.getHex( colorSpace ).toString( 16 ) ).slice( - 6 );

	}

	getHSL( target = {}, colorSpace = LinearSRGBColorSpace ) {

		const c = this._in( colorSpace ), r = c.r, g = c.g, b = c.b;
		const max = Math.max( r, g, b ), min = Math.min( r, g, b ), lightness = ( min + max ) / 2;
		let hue = 0, saturation = 0;

		if ( min !== max ) {

			const delta = max - min;
			saturation = lightness <= 0.5 ? delta / ( max + min ) : delta / ( 2 - max - min );
			if ( max === r ) hue = ( g - b ) / delta + ( g < b ? 6 : 0 );
			else if ( max === g ) hue = ( b - r ) / delta + 2;
			else hue = ( r - g ) / delta + 4;
			hue /= 6;

		}

		target.h = hue;
		target.s = saturation;
		target.l = lightness;
		return target;

	}

	getRGB( target, colorSpace = LinearSRGBColorSpace ) {

		const c = this._in( colorSpace );
		target.r = c.r;
		target.g = c.g;
		target.b = c.b;
		return target;

	}

	getStyle( colorSpace = SRGBColorSpace ) {

		const c = this._in( colorSpace );
		if ( colorSpace !== SRGBColorSpace ) return `color(${colorSpace} ${c.r.toFixed( 3 )} ${c.g.toFixed( 3 )} ${c.b.toFixed( 3 )})`;
		return `rgb(${Math.round( c.r * 255 )},${Math.round( c.g * 255 )},${Math.round( c.b * 255 )})`;

	}

	offsetHSL( h, s, l ) {

		const hsl = this.getHSL( hslA );
		return this.setHSL( hsl.h + h, hsl.s + s, hsl.l + l );

	}

	add( color ) {

		this.r += color.r;
		this.g += color.g;
		this.b += color.b;
		return this;

	}

	addColors( color1, color2 ) {

		this.r = color1.r + color2.r;
		this.g = color1.g + color2.g;
		this.b = color1.b + color2.b;
		return this;

	}

	addScalar( s ) {

		this.r += s;
		this.g += s;
		this.b += s;
		return this;

	}

	sub( color ) {

		this.r = Math.max( 0, this.r - color.r );
		this.g = Math.max( 0, this.g - color.g );
		this.b = Math.max( 0, this.b - color.b );
		return this;

	}

	multiply( color ) {

		this.r *= color.r;
		this.g *= color.g;
		this.b *= color.b;
		return this;

	}

	multiplyScalar( s ) {

		this.r *= s;
		this.g *= s;
		this.b *= s;
		return this;

	}

	lerp( color, alpha ) {

		this.r += ( color.r - this.r ) * alpha;
		this.g += ( color.g - this.g ) * alpha;
		this.b += ( color.b - this.b ) * alpha;
		return this;

	}

	lerpColors( color1, color2, alpha ) {

		this.r = color1.r + ( color2.r - color1.r ) * alpha;
		this.g = color1.g + ( color2.g - color1.g ) * alpha;
		this.b = color1.b + ( color2.b - color1.b ) * alpha;
		return this;

	}

	lerpHSL( color, alpha ) {

		const a = this.getHSL( hslA ), b = color.getHSL( hslB );
		return this.setHSL( a.h + ( b.h - a.h ) * alpha, a.s + ( b.s - a.s ) * alpha, a.l + ( b.l - a.l ) * alpha );

	}

	setFromVector3( v ) {

		this.r = v.x;
		this.g = v.y;
		this.b = v.z;
		return this;

	}

	applyMatrix3( m ) {

		const r = this.r, g = this.g, b = this.b, e = m.elements;
		this.r = e[ 0 ] * r + e[ 3 ] * g + e[ 6 ] * b;
		this.g = e[ 1 ] * r + e[ 4 ] * g + e[ 7 ] * b;
		this.b = e[ 2 ] * r + e[ 5 ] * g + e[ 8 ] * b;
		return this;

	}

	equals( c ) {

		return c.r === this.r && c.g === this.g && c.b === this.b;

	}

	fromArray( array, offset = 0 ) {

		this.r = array[ offset ];
		this.g = array[ offset + 1 ];
		this.b = array[ offset + 2 ];
		return this;

	}

	toArray( array = [], offset = 0 ) {

		array[ offset ] = this.r;
		array[ offset + 1 ] = this.g;
		array[ offset + 2 ] = this.b;
		return array;

	}

	fromBufferAttribute( attribute, index ) {

		this.r = attribute.getX( index );
		this.g = attribute.getY( index );
		this.b = attribute.getZ( index );
		return this;

	}

	toJSON() {

		return this.getHex();

	}

	* [ Symbol.iterator ]() {

		yield this.r;
		yield this.g;
		yield this.b;

	}

}

Color.prototype.isColor = true;
const scratchColor = new Color(), hslA = { h: 0, s: 0, l: 0 }, hslB = { h: 0, s: 0, l: 0 };


// Min and max of a box as six numbers.
