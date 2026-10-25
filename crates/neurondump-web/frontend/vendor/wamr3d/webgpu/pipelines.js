// Render pipelines by state key, and the effect pipelines.
//
// Part of the WebGPU host; `../webgpu.js` creates the device and draws the frames.

import { Renderer } from '../index.js';
import { BLEND_FACTORS, FEATURE_NAMES, HDR_FORMAT, HDR_KEY, RGBA8_KEY, SINGLE_KEY, customBlend } from './formats.js';
import { ALPHA_TO_COVERAGE_KEY, COLOR_WRITE_OFF_KEY, DEPTH_FORMAT, DEPTH_FUNCTIONS, DEPTH_FUNC_SHIFT, PREMULTIPLIED_KEY, QUAD_KEY, STENCIL_FUNCTIONS, STENCIL_OPERATIONS, TOPOLOGIES, quadBuffers, vertexBuffers } from './constants.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installPipelines( ctx ) {

	const { device, format, layout, module, samples } = ctx;

	// Pipelines are created the first time a state is drawn and kept. The key is the
	// engine's pipeline state (see `Renderer.drawInfo`); HDR_KEY, RGBA8_KEY and
	// SINGLE_KEY mark pipelines that draw into the linear target or a render target.
	const pipelines = new Map();

	function blend( key, custom = 0 ) {

		if ( ( key & 4 ) === 0 ) return undefined;
		if ( ( key >> 11 & 7 ) === 5 ) return customBlend( custom );
		const factors = BLEND_FACTORS[ ( key & PREMULTIPLIED_KEY ? 5 : 0 ) + Math.max( 1, key >> 11 & 7 ) - 1 ];
		return { color: { srcFactor: factors[ 0 ], dstFactor: factors[ 1 ] }, alpha: { srcFactor: factors[ 2 ], dstFactor: factors[ 3 ] } };

	}

	// Depth and stencil state: the depth function of the key (always without a depth test),
	// and from `state` (see `materialState` in ../webgpu.js), when given, the polygon offset
	// and the stencil test. Lines and points take no depth bias, as WebGPU requires.
	function depthStencil( key, state, triangles ) {

		const value = { format: DEPTH_FORMAT, depthWriteEnabled: ( key & 8 ) !== 0, depthCompare: key & 16 ? DEPTH_FUNCTIONS[ key >> DEPTH_FUNC_SHIFT & 7 ] : 'always' };
		if ( ! state ) return value;
		const [ factor, units, bits, readMask, writeMask ] = state.split( ',' ).map( Number );

		if ( triangles ) {

			value.depthBias = Math.round( units );
			value.depthBiasSlopeScale = factor;

		}

		if ( bits & 1 ) {

			const face = { compare: STENCIL_FUNCTIONS[ bits >> 1 & 7 ], failOp: STENCIL_OPERATIONS[ bits >> 4 & 7 ], depthFailOp: STENCIL_OPERATIONS[ bits >> 7 & 7 ], passOp: STENCIL_OPERATIONS[ bits >> 10 & 7 ] };
			value.stencilFront = face;
			value.stencilBack = face;
			value.stencilReadMask = readMask;
			value.stencilWriteMask = writeMask;

		}

		return value;

	}

	// The pipeline of a full-screen effect (`custom` 1 to 6), with the draw's blending and depth.
	let effectModule = null;

	function effectPipeline( key, custom ) {

		const id = 'effect:' + key + ':' + custom;
		let value = pipelines.get( id );
		if ( value ) return value;
		effectModule ??= device.createShaderModule( { code: Renderer.effectShaderSource() } );
		value = device.createRenderPipeline( {
			layout,
			vertex: { module: effectModule, entryPoint: 'vertex_effect', buffers: vertexBuffers( key ) },
			fragment: { module: effectModule, entryPoint: Renderer.effectEntryPoint( custom ), targets: [ { format: key & HDR_KEY ? HDR_FORMAT : key & RGBA8_KEY ? 'rgba8unorm' : format, blend: blend( key ) } ] },
			primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: 'none' },
			depthStencil: depthStencil( key, null, true ),
			multisample: { count: key & SINGLE_KEY ? 1 : samples }
		} );
		pipelines.set( id, value );
		return value;

	}

	// `custom` is the material's packed custom blending, read only with CustomBlending;
	// `state` its polygon offset and stencil settings ('' when they are the defaults);
	// `features` the `meshFeatures` key the fragment stage is specialised for.
	function descriptor( key, custom, state, features ) {

		const constants = {};
		if ( features ) features.split( ',' ).forEach( ( v, i ) => constants[ FEATURE_NAMES[ i ] ] = Number( v ) );
		const quad = ( key & QUAD_KEY ) !== 0, topology = quad ? 'triangle-list' : TOPOLOGIES[ key >> 5 & 3 ];

		return {
			layout,
			vertex: quad ? { module, entryPoint: 'vertex_quad', buffers: quadBuffers( key ) } : { module, entryPoint: 'vertex', buffers: vertexBuffers( key ) },
			fragment: { module, entryPoint: 'fragment', constants, targets: [ { format: key & HDR_KEY ? HDR_FORMAT : key & RGBA8_KEY ? 'rgba8unorm' : format, blend: blend( key, custom ), writeMask: key & COLOR_WRITE_OFF_KEY ? 0 : GPUColorWrite.ALL } ] },
			primitive: { topology, frontFace: 'ccw', cullMode: quad ? 'none' : [ 'back', 'front', 'none' ][ key & 3 ], stripIndexFormat: topology === 'line-strip' ? 'uint32' : undefined },
			depthStencil: depthStencil( key, state, topology === 'triangle-list' ),
			multisample: { count: key & SINGLE_KEY ? 1 : samples, alphaToCoverageEnabled: ( key & ALPHA_TO_COVERAGE_KEY ) !== 0 && ! ( key & SINGLE_KEY ) && samples > 1 }
		};

	}

	function pipeline( key, custom = 0, state = '', features = '' ) {

		const id = key + ':' + custom + ':' + state + ':' + features;
		let value = pipelines.get( id );
		if ( value ) return value;
		value = device.createRenderPipeline( descriptor( key, custom, state, features ) );
		pipelines.set( id, value );
		return value;

	}

	// The same pipeline built off the main thread; `pipeline` finds it once it is ready.
	const compiling = new Map();

	function pipelineAsync( key, custom = 0, state = '', features = '' ) {

		const id = key + ':' + custom + ':' + state + ':' + features;
		if ( pipelines.has( id ) ) return Promise.resolve();
		let promise = compiling.get( id );

		if ( ! promise ) {

			promise = device.createRenderPipelineAsync( descriptor( key, custom, state, features ) ).then( ( value ) => {

				compiling.delete( id );
				if ( ! pipelines.has( id ) ) pipelines.set( id, value );

			}, ( error ) => {

				compiling.delete( id );
				throw error;

			} );
			compiling.set( id, promise );

		}

		return promise;

	}

	Object.assign( ctx, { effectPipeline, pipeline, pipelineAsync, pipelines } );

}
