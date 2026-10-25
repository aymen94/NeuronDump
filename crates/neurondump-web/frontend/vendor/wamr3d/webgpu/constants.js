// Constants shared by the host's parts.
//
// Part of the WebGPU host; `../webgpu.js` creates the device and draws the frames.

export const UNUSED_FRAMES = 300;
export const warned = new Set();
export const warnOnce = ( what ) => {

	if ( warned.has( what ) ) return;
	warned.add( what );
	console.warn( `WAMR3D (WebGPU): ${what}` );

};

// Vertex inputs with a buffer of their own, in location order: name, floats per vertex.
// They are the first records of `Renderer.geometryState`; tangent, uv1, skinIndex,
// skinWeight and lineDistance follow and are interleaved into one buffer (`AUX_SLOT`).
export const ATTRIBUTES = [ [ 'position', 3 ], [ 'normal', 3 ], [ 'uv', 2 ], [ 'color', 3 ] ];
export const AUX_ATTRIBUTES = 5, AUX_STRIDE = 60, AUX_FLOATS = 15, AUX_SIZES = [ 4, 2, 4, 4, 1 ], AUX_OFFSETS = [ 0, 4, 6, 10, 14 ];
export const TOPOLOGIES = [ 'triangle-list', 'line-strip', 'line-list', 'point-list' ];
export const VERTEX_COLOR_SLOT = 3, MATRIX_SLOT = 4, COLOR_SLOT = 5, AUX_SLOT = 6;

// Engine pipeline-key bits: a line strip closed back to its first vertex, colours of four
// floats, a mesh drawn as the edges of its triangles, interleaved extra attributes,
// premultiplied alpha, and points or lines drawn as quads.
export const LOOP_KEY = 1 << 7, RGBA_COLOR_KEY = 1 << 8, WIREFRAME_KEY = 1 << 9, AUX_KEY = 1 << 10, PREMULTIPLIED_KEY = 1 << 14, QUAD_KEY = 1 << 15;
// Host-only bit of shadow pipelines: the caster has an alpha test.
export const ALPHA_TEST_KEY = 1 << 23;

export const vertexBuffers = ( key ) => [
	{ arrayStride: 12, attributes: [ { shaderLocation: 0, offset: 0, format: 'float32x3' } ] },
	{ arrayStride: 12, attributes: [ { shaderLocation: 1, offset: 0, format: 'float32x3' } ] },
	{ arrayStride: 8, attributes: [ { shaderLocation: 2, offset: 0, format: 'float32x2' } ] },
	key & RGBA_COLOR_KEY ? { arrayStride: 16, attributes: [ { shaderLocation: 3, offset: 0, format: 'float32x4' } ] } : { arrayStride: 12, attributes: [ { shaderLocation: 3, offset: 0, format: 'float32x3' } ] },
	{ arrayStride: 64, stepMode: 'instance', attributes: [ 0, 1, 2, 3 ].map( ( column ) => ( { shaderLocation: 4 + column, offset: column * 16, format: 'float32x4' } ) ) },
	{ arrayStride: 12, stepMode: 'instance', attributes: [ { shaderLocation: 8, offset: 0, format: 'float32x3' } ] },
	// with a stride of 0 every vertex reads the same zeros
	{ arrayStride: key & AUX_KEY ? AUX_STRIDE : 0, attributes: [
		{ shaderLocation: 9, offset: 0, format: 'float32x4' }, { shaderLocation: 10, offset: 16, format: 'float32x2' },
		{ shaderLocation: 11, offset: 24, format: 'float32x4' }, { shaderLocation: 12, offset: 40, format: 'float32x4' },
		{ shaderLocation: 15, offset: 56, format: 'float32' }
	] }
];

// Points and lines as quads read their vertices per instance: a point, a strip's
// consecutive pair, or a list's pair (see `vertex_quad`).
export function quadBuffers( key ) {

	const topology = key >> 5 & 3, components = key & RGBA_COLOR_KEY ? 4 : 3, pairs = topology === 2;
	const second = topology === 3 ? 0 : 12, colorSecond = topology === 3 ? 0 : components * 4;
	return [
		{ arrayStride: pairs ? 24 : 12, stepMode: 'instance', attributes: [ { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 13, offset: second, format: 'float32x3' } ] },
		// the line distances of both ends, for dashes
		{ arrayStride: pairs ? 8 : 4, stepMode: 'instance', attributes: [ { shaderLocation: 1, offset: 0, format: 'float32' }, { shaderLocation: 2, offset: topology === 3 ? 0 : 4, format: 'float32' } ] },
		null,
		{ arrayStride: ( pairs ? 2 : 1 ) * components * 4, stepMode: 'instance', attributes: [ { shaderLocation: 3, offset: 0, format: `float32x${components}` }, { shaderLocation: 14, offset: colorSecond, format: `float32x${components}` } ] }
	];

}

// Copies a depth texture into a float colour target (see `depthCopy`).
export const DEPTH_COPY = `
@group(0) @binding(0) var depth_map: texture_depth_2d;

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
	let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
	return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
	let depth = textureLoad(depth_map, vec2<i32>(position.xy), 0);
	return vec4<f32>(depth, depth, depth, 1.0);
}`;

export const BLIT = `
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
struct Output { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vertex(@builtin(vertex_index) index: u32) -> Output {
    var output: Output;
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    output.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    output.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return output;
}
@fragment fn fragment(input: Output) -> @location(0) vec4<f32> {
    return textureSample(source, source_sampler, input.uv);
}`;

// Indices of a wireframe over `count` triangle vertices: two per edge, three edges per triangle.
export const wireframeCount = ( count ) => Math.floor( count / 3 ) * 6;

// A cube face: an image, `{ data, width, height }`, or a DataTexture (as three.js accepts).
export const cubeFace = ( face ) => ( face?.isDataTexture ? face.image : face );
// Whether all six square faces of a cube texture have arrived.
export const cubeReady = ( images ) => Array.isArray( images ) && images.length === 6 && images.map( cubeFace ).every( ( image, i, faces ) => image && ( image.data || image.width > 0 ) && image.width === faces[ 0 ].width );
// Data textures of float or half-float type are uploaded as rgba16float or rgba32float.
export const dataFormat = ( source ) => ( source?.type === 1015 ? 'rgba32float' : source?.type === 1016 ? 'rgba16float' : null );

// The packed custom blending in a material block: the w of its 13th vector.
export const CUSTOM_BLENDING_FLOAT = 51;

// Main passes draw with a stencil buffer (three.js `stencilWrite`); shadow maps without.
export const DEPTH_FORMAT = 'depth24plus-stencil8';

// Pipeline-key bits of a material's state (see the engine's `pipeline_key`).
export const ALPHA_TO_COVERAGE_KEY = 1 << 24, DEPTH_FUNC_SHIFT = 25, COLOR_WRITE_OFF_KEY = 1 << 28;

// three.js `depthFunc` (0 never ... 7 not equal), stencil functions (from 512) and
// stencil operations as the engine numbers them, in WebGPU's names.
export const DEPTH_FUNCTIONS = [ 'never', 'always', 'less', 'less-equal', 'equal', 'greater-equal', 'greater', 'not-equal' ];
export const STENCIL_FUNCTIONS = [ 'never', 'less', 'equal', 'less-equal', 'greater', 'not-equal', 'greater-equal', 'always' ];
export const STENCIL_OPERATIONS = [ 'zero', 'keep', 'replace', 'increment-clamp', 'decrement-clamp', 'increment-wrap', 'decrement-wrap', 'invert' ];

// Floats of a material block the hosts read: blend colour, then depth function, colour
// write and polygon offset factor and units, then the packed stencil settings with their
// reference and masks, then the state flags (see the engine's `pack_material`).
export const BLEND_COLOR_FLOAT = 264, POLYGON_OFFSET_FLOAT = 270, STENCIL_FLOAT = 272, STATE_FLAGS_FLOAT = 276;
// The packed stencil settings of a material that leaves the stencil alone.
export const DEFAULT_STENCIL = 1182;
