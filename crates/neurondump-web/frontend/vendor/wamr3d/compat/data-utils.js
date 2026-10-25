// three.js `DataUtils`: conversion between 32-bit floats and the 16-bit half floats of
// HalfFloatType textures. Rounds to nearest (ties to even), keeps infinities and NaN, and
// makes subnormals for the smallest values, as IEEE 754 binary16 does.

const _float = new Float32Array( 1 ), _bits = new Uint32Array( _float.buffer );

function toHalfFloat( value ) {

	_float[ 0 ] = Math.min( Math.max( value, - 65504 ), 65504 ) === value || ! Number.isFinite( value ) ? value : Math.sign( value ) * 65504;
	const bits = _bits[ 0 ], sign = bits >>> 16 & 0x8000, exponent = bits >>> 23 & 0xff, mantissa = bits & 0x7fffff;
	if ( exponent === 0xff ) return sign | 0x7c00 | ( mantissa ? 0x200 : 0 );
	const half = exponent - 127 + 15;
	if ( half >= 0x1f ) return sign | 0x7c00;

	if ( half <= 0 ) {

		if ( half < - 10 ) return sign;
		const m = mantissa | 0x800000, shift = 14 - half, rounded = m >> shift;
		const rest = m & ( ( 1 << shift ) - 1 ), halfway = 1 << shift - 1;
		return sign | ( rounded + ( rest > halfway || ( rest === halfway && rounded & 1 ) ? 1 : 0 ) );

	}

	const rounded = half << 10 | mantissa >> 13, rest = mantissa & 0x1fff;
	return sign | ( rounded + ( rest > 0x1000 || ( rest === 0x1000 && rounded & 1 ) ? 1 : 0 ) );

}

function fromHalfFloat( value ) {

	const sign = value & 0x8000 ? - 1 : 1, exponent = value >> 10 & 0x1f, mantissa = value & 0x3ff;
	if ( exponent === 0 ) return sign * mantissa * 2 ** - 24;
	if ( exponent === 0x1f ) return mantissa ? NaN : sign * Infinity;
	return sign * ( 1 + mantissa / 1024 ) * 2 ** ( exponent - 15 );

}

export const DataUtils = { toHalfFloat, fromHalfFloat };
