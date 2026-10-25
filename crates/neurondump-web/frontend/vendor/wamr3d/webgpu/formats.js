// Formats, filters and pipeline-key bits shared by the WebGPU host's modules.

export const WRAPS = { 1000: 'repeat', 1001: 'clamp-to-edge', 1002: 'mirror-repeat' };
// Texture filter constants: nearest 1003, linear 1006, and their mipmapped variants.
export const LINEAR = new Set( [ 1006, 1007, 1008 ] ), MIPMAPPED = new Set( [ 1004, 1005, 1007, 1008 ] ), MIP_LINEAR = new Set( [ 1005, 1008 ] );
// With tone mapping the scene is drawn into this linear format, then tone-mapped onto the canvas.
export const HDR_FORMAT = 'rgba16float', HDR_KEY = 1 << 20;
// Render targets hold linear colour without encoding (three.js NoColorSpace); RGBA8_KEY
// marks pipelines for such a target, SINGLE_KEY those without multisampling.
export const RGBA8_KEY = 1 << 21, SINGLE_KEY = 1 << 22;

// Blend factors by three.js blending mode (normal, additive, subtractive, multiply,
// custom drawn as normal): colour source and destination, alpha source and destination;
// the first five for straight alpha, the next five for premultiplied alpha. Subtractive
// and multiply need premultiplied alpha in three.js and use those factors either way.
export const BLEND_FACTORS = [
	[ 'src-alpha', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha' ],
	[ 'src-alpha', 'one', 'one', 'one' ],
	[ 'zero', 'one-minus-src', 'zero', 'one' ],
	[ 'dst', 'one-minus-src-alpha', 'zero', 'one' ],
	[ 'src-alpha', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha' ],
	[ 'one', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha' ],
	[ 'one', 'one', 'one', 'one' ],
	[ 'zero', 'one-minus-src', 'zero', 'one' ],
	[ 'dst', 'one-minus-src-alpha', 'zero', 'one' ],
	[ 'one', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha' ]
];

// The versions of a geometry's extra attributes (records of `Renderer.geometryState`
// from `o`), as a key; empty when it has none.
export function auxRecord( state, o, count ) {

	let key = '';
	for ( let a = 0, p = o; a < count; a ++, p += 2 ) if ( state[ p + 1 ] !== 0 ) key += a + ':' + state[ p ] + ':' + state[ p + 1 ] + ',';
	return key;

}

// First instance and instance count of points or segments drawn as quads, from a draw's
// first vertex and vertex count: a point per vertex, a segment per strip pair, or one per
// two vertices of a list.
// three.js's blend factors (from ZeroFactor, 200) and equations (from AddEquation, 100)
// in WebGPU's names; the constant colour has no separate alpha in WebGPU.
const CUSTOM_FACTORS = [ 'zero', 'one', 'src', 'one-minus-src', 'src-alpha', 'one-minus-src-alpha', 'dst-alpha', 'one-minus-dst-alpha', 'dst', 'one-minus-dst', 'src-alpha-saturated', 'constant', 'one-minus-constant', 'constant', 'one-minus-constant' ];
const CUSTOM_EQUATIONS = [ 'add', 'subtract', 'reverse-subtract', 'min', 'max' ];

// The blend state of `CustomBlending` from the material's packed settings (see the
// engine's `Material::custom_blending`). Min and max ignore the factors, as in WebGL;
// WebGPU requires them to be one.
export function customBlend( packed ) {

	const component = ( shift ) => {

		const operation = CUSTOM_EQUATIONS[ packed >> shift + 8 & 7 ] ?? 'add', minMax = operation === 'min' || operation === 'max';
		return { operation, srcFactor: minMax ? 'one' : CUSTOM_FACTORS[ packed >> shift & 15 ] ?? 'one', dstFactor: minMax ? 'one' : CUSTOM_FACTORS[ packed >> shift + 4 & 15 ] ?? 'zero' };

	};

	return { color: component( 0 ), alpha: component( 11 ) };

}

// The rows of RGBA `data` (`width` texels each) in the opposite order, for a data
// texture with `flipY`, as three.js uploads it.
export function flippedRows( data, width, height ) {

	const row = width * 4, out = new data.constructor( data.length );
	for ( let y = 0; y < height; y ++ ) out.set( data.subarray( y * row, ( y + 1 ) * row ), ( height - 1 - y ) * row );
	return out;

}

export function quadInstances( topology, start, count ) {

	if ( topology === 3 ) return [ start, count ];
	if ( topology === 2 ) return [ start >> 1, count >> 1 ];
	return [ start, Math.max( 0, count - 1 ) ];

}

// What a mesh program is specialised for, as `shading,material,state,maps,lights`: the
// material's shading mode, its flags and state flags with only the costly feature bits
// that it lacks cleared, the maps it has, and the light kinds of the frame (1 shadowed
// lights, 2 area lights). The shaders compile out code behind a cleared bit, so draws
// whose material differs in those features use different programs.
const MATERIAL_GATED = 0x3f0000, STATE_GATED = 4 | 32 | 64 | 128, MAPS_FLOAT = 64, MAP_COUNT = 25, STATE_FLOAT = 276;

export function lightFeatures( frame ) {

	let features = 0;
	const count = Math.min( Math.trunc( frame[ 19 ] ), 16 );

	for ( let i = 0, at = 48; i < count; i ++, at += 16 ) {

		if ( frame[ at + 3 ] > 2.5 ) features |= 2;
		else if ( Math.trunc( frame[ at + 14 ] + 0.5 ) > 0 ) features |= 1;

	}

	return features;

}

export function meshFeatures( blocks, at, lights ) {

	const flags = blocks[ at + 7 ] >>> 0, state = blocks[ at + STATE_FLOAT ] >>> 0;
	let maps = 0;
	for ( let k = 0, o = at + MAPS_FLOAT + 3; k < MAP_COUNT; k ++, o += 8 ) if ( blocks[ o ] > 0.5 ) maps |= 1 << k;
	return ( blocks[ at + 6 ] + 0.5 >>> 0 ) + ',' + ( ( flags | ~ MATERIAL_GATED ) >>> 0 ) + ',' + ( ( state | ~ STATE_GATED ) >>> 0 ) + ',' + ( maps >>> 0 ) + ',' + lights;

}

export const FEATURE_NAMES = [ 'SHADING_MODE', 'MATERIAL_FEATURES', 'STATE_FEATURES', 'MAP_FEATURES', 'LIGHT_FEATURES' ];
