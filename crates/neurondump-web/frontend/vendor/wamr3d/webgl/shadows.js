// The bone texture and the shadow maps (with the VSM blur).
//
// Part of the WebGL 2 host; `../webgl.js` creates the context and draws the frames.

import { memory } from '../index.js';
import { BONE_UNIT, MATERIAL_FLOATS, STATE_FLAGS_FLOAT, MATERIAL_STRIDE, OBJECT_STRIDE } from './constants.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installShadows( ctx ) {

	const { BONES_PER_ROW, DRAW_FIELDS, MAX_SHADOWS, bindGeometry, bindInstances, bindMaps, bindRange, distinct, draw, floatTargets, frameBuffer, gl, instances, materialBuffer, newTexture, program, ranges, setTextures, shadowObjectBuffer, unbindArrays, upload, warn } = ctx;

	// ---------------------------------------------------------------- bones

	ctx.boneTexture = null;
	let boneRows = 0;

	// The frame's skinning matrices in a float texture 1024 texels wide.
	function uploadBones( renderer ) {

		const length = renderer.skinMatricesLength(), width = BONES_PER_ROW * 4;
		const matrices = length / 16, rows = Math.max( 1, Math.ceil( matrices / BONES_PER_ROW ) );

		if ( ! ctx.boneTexture || boneRows < rows ) {

			if ( ctx.boneTexture ) gl.deleteTexture( ctx.boneTexture );
			ctx.boneTexture = newTexture();
			boneRows = Math.max( rows, boneRows * 2 );
			gl.texStorage2D( gl.TEXTURE_2D, 1, gl.RGBA32F, width, boneRows );
			gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST );
			gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST );

		}

		if ( length === 0 ) return;
		const data = new Float32Array( memory().buffer, renderer.skinMatricesPtr(), length );
		gl.bindTexture( gl.TEXTURE_2D, ctx.boneTexture );
		gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, false );
		const full = Math.floor( matrices / BONES_PER_ROW ), rest = matrices - full * BONES_PER_ROW;
		if ( full > 0 ) gl.texSubImage2D( gl.TEXTURE_2D, 0, 0, 0, width, full, gl.RGBA, gl.FLOAT, data, 0 );
		if ( rest > 0 ) gl.texSubImage2D( gl.TEXTURE_2D, 0, 0, full, rest * 4, 1, gl.RGBA, gl.FLOAT, data, full * BONES_PER_ROW * 16 );

	}

	// ---------------------------------------------------------------- shadows

	let shadowWidth = 0, shadowHeight = 0, shadowLayers = 0;
	ctx.shadowTexture = null;
	ctx.vsmTexture = null;
	ctx.vsmScratch = null;
	const shadowFramebuffer = gl.createFramebuffer(), vsmFramebuffer = gl.createFramebuffer();
	// Reads depth as numbers (the shadow texture compares by itself).
	const depthSampler = gl.createSampler();
	gl.samplerParameteri( depthSampler, gl.TEXTURE_COMPARE_MODE, gl.NONE );
	gl.samplerParameteri( depthSampler, gl.TEXTURE_MIN_FILTER, gl.NEAREST );
	gl.samplerParameteri( depthSampler, gl.TEXTURE_MAG_FILTER, gl.NEAREST );

	// One layer per shadow map (a point light has six), as many as frames have needed so far.
	function ensureShadowTexture( width, height, layers, vsm ) {

		const sameSize = ctx.shadowTexture && shadowWidth === width && shadowHeight === height;

		if ( ! sameSize || shadowLayers < layers ) {

			if ( ctx.shadowTexture ) gl.deleteTexture( ctx.shadowTexture );
			shadowLayers = Math.min( MAX_SHADOWS, sameSize ? Math.max( layers, shadowLayers ) : layers );
			ctx.shadowTexture = gl.createTexture();
			gl.bindTexture( gl.TEXTURE_2D_ARRAY, ctx.shadowTexture );
			gl.texImage3D( gl.TEXTURE_2D_ARRAY, 0, gl.DEPTH_COMPONENT24, width, height, shadowLayers, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_MODE, gl.COMPARE_REF_TO_TEXTURE );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_COMPARE_FUNC, gl.LEQUAL );
			shadowWidth = width;
			shadowHeight = height;
			if ( ctx.vsmTexture ) gl.deleteTexture( ctx.vsmTexture );
			ctx.vsmTexture = null;

		}

		if ( vsm && ! ctx.vsmTexture ) {

			if ( ! floatTargets ) warn( 'VSM shadows need EXT_color_buffer_float' );
			ctx.vsmTexture = gl.createTexture();
			gl.bindTexture( gl.TEXTURE_2D_ARRAY, ctx.vsmTexture );
			gl.texStorage3D( gl.TEXTURE_2D_ARRAY, 1, gl.RGBA16F, width, height, shadowLayers );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
			gl.texParameteri( gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
			if ( ctx.vsmScratch ) gl.deleteTexture( ctx.vsmScratch );
			ctx.vsmScratch = newTexture();
			gl.texStorage2D( gl.TEXTURE_2D, 1, gl.RGBA16F, width, height );
			gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST );
			gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST );

		}

	}

	// The VSM blur of every pass: vertical from the depth layer into the scratch moments,
	// horizontal from them into the layer of the moments array.
	function blurShadows( renderer, passes ) {

		const vertical = program( 'vsm:vertical', 'fullscreen.vert', 'vsm.frag', '#define VERTICAL 1' ), horizontal = program( 'vsm:horizontal', 'fullscreen.vert', 'vsm.frag' );
		gl.bindFramebuffer( gl.FRAMEBUFFER, vsmFramebuffer );
		gl.drawBuffers( [ gl.COLOR_ATTACHMENT0 ] );
		gl.disable( gl.DEPTH_TEST );
		gl.disable( gl.CULL_FACE );
		unbindArrays();

		for ( let p = 0; p < passes; p ++ ) {

			const passInfo = renderer.shadowPassInfo( p ), blur = renderer.shadowBlur( p ), layer = passInfo[ 0 ];
			gl.viewport( 0, 0, passInfo[ 1 ], passInfo[ 2 ] );
			gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, ctx.vsmScratch, 0 );
			gl.useProgram( vertical.id );
			gl.uniform4f( vertical.uniform( 'settings' ), blur[ 0 ], blur[ 1 ], layer, 0 );
			gl.activeTexture( gl.TEXTURE0 );
			gl.bindTexture( gl.TEXTURE_2D_ARRAY, ctx.shadowTexture );
			gl.bindSampler( 0, depthSampler );
			gl.drawArrays( gl.TRIANGLES, 0, 3 );
			gl.bindSampler( 0, null );
			gl.framebufferTextureLayer( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, ctx.vsmTexture, 0, layer );
			gl.useProgram( horizontal.id );
			gl.uniform4f( horizontal.uniform( 'settings' ), blur[ 0 ], blur[ 1 ], layer, 0 );
			gl.activeTexture( gl.TEXTURE1 );
			gl.bindTexture( gl.TEXTURE_2D, ctx.vsmScratch );
			gl.drawArrays( gl.TRIANGLES, 0, 3 );

		}

		gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0 );
		gl.bindFramebuffer( gl.FRAMEBUFFER, null );
		gl.enable( gl.DEPTH_TEST );

	}

	// Depth only, one layer per shadow map, casters with the main pass's buffers, materials and textures.
	function drawShadows( renderer, instanceSources, frame ) {

		const passes = renderer.shadowCount();
		if ( passes === 0 ) return 0;
		const [ width, height ] = renderer.shadowMapSize();
		const variance = renderer.shadowFilter() === 2 && floatTargets;
		ensureShadowTexture( width, height, passes, variance );
		gl.bindFramebuffer( gl.FRAMEBUFFER, shadowFramebuffer );
		// the main passes bind it later; a first frame would draw no shadows without it
		gl.bindBufferBase( gl.UNIFORM_BUFFER, 0, frameBuffer );
		gl.drawBuffers( [ gl.NONE ] );
		gl.disable( gl.BLEND );
		gl.depthMask( true );
		gl.depthFunc( gl.LEQUAL );
		gl.activeTexture( gl.TEXTURE0 + BONE_UNIT );
		gl.bindTexture( gl.TEXTURE_2D, ctx.boneTexture );
		let calls = 0, current = null;

		for ( let p = 0; p < passes; p ++ ) {

			const passInfo = renderer.shadowPassInfo( p ), draws = renderer.shadowDrawInfo( p ), slots = renderer.shadowDrawGeometrySlots( p );
			const data = renderer.shadowObjectUniforms( p );
			gl.framebufferTextureLayer( gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, ctx.shadowTexture, 0, passInfo[ 0 ] );
			gl.viewport( 0, 0, passInfo[ 1 ], passInfo[ 2 ] );
			gl.clear( gl.DEPTH_BUFFER_BIT );
			if ( ranges ) upload( gl.UNIFORM_BUFFER, data, shadowObjectBuffer );
			let bound = null, boundInstances, set = null;

			for ( let i = 0, o = 0; i < slots.length; i ++, o += DRAW_FIELDS ) {

				const entry = distinct[ slots[ i ] ], key = draws[ o ], start = draws[ o + 2 ], vertices = draws[ o + 3 ], material = draws[ o + 6 ];
				// a fragment stage for alpha tests and for clipping planes that clip shadows
				const alphaTest = frame.materials[ material * MATERIAL_FLOATS + 23 ] > 0 || ( frame.materials[ material * MATERIAL_FLOATS + STATE_FLAGS_FLOAT ] & 256 ) !== 0;
				const shader = alphaTest ? program( 'shadow:alpha', 'shadow.vert', 'shadow.frag', '#define ALPHA_TEST 1' ) : program( 'shadow', 'shadow.vert', 'shadow.frag' );

				if ( shader !== current ) {

					gl.useProgram( shader.id );
					current = shader;

				}

				gl.uniformMatrix4fv( shader.uniform( 'shadow_view_projection' ), false, passInfo.subarray( 3, 19 ) );
				const instance = instances( draws[ o + 4 ], instanceSources ), copies = instance ? draws[ o + 5 ] : 1;
				const side = key & 3;
				if ( side === 2 ) gl.disable( gl.CULL_FACE );
				else {

					gl.enable( gl.CULL_FACE );
					gl.cullFace( side === 0 ? gl.BACK : gl.FRONT );

				}

				if ( instance !== boundInstances ) {

					bindInstances( instance );
					boundInstances = instance;

				}

				if ( entry !== bound ) {

					bindGeometry( entry );
					bound = entry;

				}

				const maps = setTextures[ draws[ o + 1 ] ];

				if ( maps !== set ) {

					bindMaps( maps );
					set = maps;

				}

				bindRange( 1, shadowObjectBuffer, i, OBJECT_STRIDE, data );
				bindRange( 3, materialBuffer, material, MATERIAL_STRIDE, frame.materials );
				draw( 0, entry, start, vertices, copies );
				calls ++;

			}

		}

		gl.framebufferTextureLayer( gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, null, 0, 0 );
		gl.bindFramebuffer( gl.FRAMEBUFFER, null );
		if ( variance ) blurShadows( renderer, passes );
		return calls;

	}

	Object.assign( ctx, { depthSampler, drawShadows, ensureShadowTexture, shadowFramebuffer, uploadBones, vsmFramebuffer } );

}
