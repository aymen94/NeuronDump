// WebGL 2 host for the WAMR3D renderer.
//
// The same frames as the WebGPU host (`webgpu.js`): the engine decides what is drawn and
// prepares every uniform; this file turns that into WebGL 2 calls with the engine's GLSL
// (`Renderer.glslSource`). It is used when WebGPU is missing or when asked for. The
// uniform blocks are uploaded as the engine packs them: their std140 layout matches.

import { Renderer, memory } from './index.js';
import { lightFeatures, meshFeatures, quadInstances } from './webgpu/formats.js';
import { BONE_UNIT, CUSTOM_BLENDING_FLOAT, ENVIRONMENT_UNIT, LOOP_KEY, MATERIAL_FLOATS, MATERIAL_STRIDE, OBJECT_STRIDE, QUAD_KEY, SHADOW_UNIT, TRANSMISSION_UNIT, UNUSED_FRAMES, VSM_UNIT, WIREFRAME_KEY } from './webgl/constants.js';
import { installPrograms } from './webgl/programs.js';
import { installBuffers } from './webgl/buffers.js';
import { installTextures } from './webgl/textures.js';
import { installShadows } from './webgl/shadows.js';
import { installEnvironment } from './webgl/environment.js';

export async function createWebGLBackend( canvas, { antialias = true, powerPreference = 'high-performance' } = {} ) {

	if ( Renderer.materialUniformSize() !== MATERIAL_STRIDE ) throw new Error( 'WAMR3D: the WebGL 2 host does not match the uniform layout of the engine' );

	const gl = canvas.getContext( 'webgl2', { antialias, alpha: false, depth: true, stencil: true, premultipliedAlpha: false, powerPreference } );
	if ( ! gl ) throw new Error( 'WAMR3D: WebGL 2 is not available in this browser' );

	// Half-float targets carry tone mapping, render targets of float type and the
	// prefiltered environments; without them those fall back as `warn` reports.
	const floatTargets = gl.getExtension( 'EXT_color_buffer_float' ) !== null;
	const floatLinear = gl.getExtension( 'OES_texture_float_linear' ) !== null;
	const anisotropic = gl.getExtension( 'EXT_texture_filter_anisotropic' );
	const samples = antialias ? Math.min( 4, gl.getParameter( gl.MAX_SAMPLES ) ) : 0;
	const MAX_SHADOWS = Renderer.maxShadows(), TEXTURE_SLOTS = Renderer.textureSlots();
	const GEOMETRY_FIELDS = Renderer.geometryStateSize(), DRAW_FIELDS = Renderer.drawInfoSize(), BONES_PER_ROW = Renderer.boneTextureMatricesPerRow();
	const TONE_MAPPING = Renderer.toneMappingIndex(), EXPOSURE = Renderer.toneMappingExposureIndex();
	// Object and material uniforms are bound as ranges of one buffer when the device allows 256-byte offsets.
	const ranges = OBJECT_STRIDE % gl.getParameter( gl.UNIFORM_BUFFER_OFFSET_ALIGNMENT ) === 0;
	const warned = new Set();
	const warn = ( what ) => {

		if ( warned.has( what ) ) return;
		warned.add( what );
		console.warn( `WAMR3D (WebGL 2): ${what}` );

	};

	const WRAP = { 1000: gl.REPEAT, 1001: gl.CLAMP_TO_EDGE, 1002: gl.MIRRORED_REPEAT };
	const MIN_FILTER = { 1003: gl.NEAREST, 1004: gl.NEAREST_MIPMAP_NEAREST, 1005: gl.NEAREST_MIPMAP_LINEAR, 1006: gl.LINEAR, 1007: gl.LINEAR_MIPMAP_NEAREST, 1008: gl.LINEAR_MIPMAP_LINEAR };
	const FACTORS = { 'zero': gl.ZERO, 'one': gl.ONE, 'src-alpha': gl.SRC_ALPHA, 'one-minus-src-alpha': gl.ONE_MINUS_SRC_ALPHA, 'one-minus-src': gl.ONE_MINUS_SRC_COLOR, 'dst': gl.DST_COLOR };

	// The parts of the host share these through `ctx`.
	const ctx = { BONES_PER_ROW, DRAW_FIELDS, EXPOSURE, FACTORS, GEOMETRY_FIELDS, MAX_SHADOWS, MIN_FILTER, TEXTURE_SLOTS, TONE_MAPPING, WRAP, anisotropic, floatLinear, floatTargets, gl, ranges, samples, warn };
	installPrograms( ctx );
	installBuffers( ctx );
	installTextures( ctx );
	installShadows( ctx );
	installEnvironment( ctx );
	const { applyMaterialState, applyState, resetState, background, backgroundBuffer, backgroundProgram, bindCustomAttributes, bindGeometry, bindInstances, bindMaps, bindQuads, bindRange, depthSampler, distinct, draw, drawShadows, drawWireframe, effectProgram, ensureShadowTexture, finishPrograms, entries, environment, environmentFramebuffer, environments, frameBuffer, geometries, geometry, hdrTarget, info, instanced, instances, levelBuffers, materialBuffer, meshProgram, multiply, normalMatrix, objectBuffer, panoramaSampler, perInstance, program, programs, quadSource, release, renderTarget, renderTargets, scratchBuffer, screenSampler, setTextures, setUniforms, shaderProgram, shadowFramebuffer, startProgram, shadowObjectBuffer, stackFramebuffer, texture, textures, transmissionTarget, unbindArrays, upload, uploadBones, vsmFramebuffer, white } = ctx;
	// ---------------------------------------------------------------- frame

	gl.enable( gl.DEPTH_TEST );
	gl.frontFace( gl.CCW );
	const linearToSRGB = ( c ) => ( c <= 0.0031308 ? c * 12.92 : 1.055 * c ** ( 1 / 2.4 ) - 0.055 );

	function drawBackground( backdrop, srgb, flipY ) {

		const shader = backgroundProgram( srgb );
		gl.useProgram( shader.id );
		gl.uniform1f( shader.uniform( 'flip_y' ), flipY );
		gl.bindBufferBase( gl.UNIFORM_BUFFER, 2, backgroundBuffer );
		gl.activeTexture( gl.TEXTURE0 );
		gl.bindTexture( gl.TEXTURE_2D, backdrop.texture );
		gl.bindSampler( 0, backdrop.sampler );
		gl.disable( gl.BLEND );
		gl.disable( gl.CULL_FACE );
		gl.depthMask( false );
		gl.depthFunc( gl.ALWAYS );
		unbindArrays();
		gl.drawArrays( gl.TRIANGLES, 0, 3 );
		gl.bindSampler( 0, null );

	}

	// A draw's packed custom blending, when its material uses CustomBlending.
	// The frame's object and material uniforms in the module's memory. A call into the
	// module (an upload, a draw list) can grow its memory, which detaches the views; they
	// are taken again then.
	function frameData( renderer, count ) {

		let objects = null, materials = null;
		const take = () => {

			objects = count > 0 ? new Float32Array( memory().buffer, renderer.objectUniformsPtr(), renderer.objectUniformsLength() ) : new Float32Array( 0 );
			materials = new Float32Array( memory().buffer, renderer.materialUniformsPtr(), renderer.materialUniformsLength() );

		};

		take();
		return {
			get objects() {

				if ( count > 0 && objects.buffer.byteLength === 0 ) take();
				return objects;

			},
			get materials() {

				if ( materials.buffer.byteLength === 0 ) take();
				return materials;

			}
		};

	}

	// The feature key of each draw's mesh program. Per material of the frame: its feature
	// key, which of its programs were started (bit 0 meshes, bit 1 quads), and the program
	// of each kind for the current pass. Draws are ordered by depth, so materials alternate
	// from draw to draw; these keep the key and program lookups to once per material.
	const drawFeatures = [], materialFeatures = [], materialStarted = [], materialPrograms = [];

	// Starts the mesh programs of draws `from` to `to`, all before any is waited for.
	// `data` is from `frameData`: building a program can grow the module's memory.
	function startPrograms( draws, from, to, srgb, data, extras ) {

		const started = [];
		if ( drawFeatures.length < to ) drawFeatures.length = to;

		materialFeatures.length = materialStarted.length = 0;

		for ( let i = from, o = from * DRAW_FIELDS; i < to; i ++, o += DRAW_FIELDS ) {

			const custom = draws[ o + 7 ];
			if ( custom > 0 && ( custom < 16 || extras?.shaderMaterials?.get( custom ) ) ) continue;
			const material = draws[ o + 6 ], quad = ( draws[ o ] & QUAD_KEY ) !== 0, kind = quad ? 2 : 1;
			let key = materialFeatures[ material ];

			if ( key === undefined ) {

				key = materialFeatures[ material ] = meshFeatures( data.materials, material * MATERIAL_FLOATS, ctx.lightFeatures );
				materialStarted[ material ] = 0;

			}

			drawFeatures[ i ] = key;
			if ( materialStarted[ material ] & kind ) continue;
			materialStarted[ material ] |= kind;
			started.push( meshProgram( srgb, quad, key, true ) );

		}

		return started;

	}

	const customBlending = ( key, materialData, material ) => ( key & 4 && ( key >> 11 & 7 ) === 5 ? materialData[ material * MATERIAL_FLOATS + CUSTOM_BLENDING_FLOAT ] : 0 );

	// Draws `from` to `to` of the prepared frame with the mesh programs, the frame-wide
	// textures bound to their units.
	function drawRange( world, draws, from, to, srgb, flipY, data, environmentTexture, copy, sources, extras ) {

		gl.bindBufferBase( gl.UNIFORM_BUFFER, 0, frameBuffer );
		gl.activeTexture( gl.TEXTURE0 + SHADOW_UNIT );
		gl.bindTexture( gl.TEXTURE_2D_ARRAY, ctx.shadowTexture ?? ( ensureShadowTexture( 1, 1, 1, false ), ctx.shadowTexture ) );
		gl.activeTexture( gl.TEXTURE0 + ENVIRONMENT_UNIT );
		gl.bindTexture( gl.TEXTURE_2D, environmentTexture ?? white );
		gl.activeTexture( gl.TEXTURE0 + TRANSMISSION_UNIT );
		gl.bindTexture( gl.TEXTURE_2D, copy ?? white );
		gl.activeTexture( gl.TEXTURE0 + VSM_UNIT );
		gl.bindTexture( gl.TEXTURE_2D_ARRAY, ctx.vsmTexture ?? ctx.shadowTexture );
		gl.bindSampler( VSM_UNIT, ctx.vsmTexture ? null : depthSampler );
		gl.activeTexture( gl.TEXTURE0 + BONE_UNIT );
		gl.bindTexture( gl.TEXTURE_2D, ctx.boneTexture );
		let state = - 1, blendState = 0, appliedMaterial = - 1, bound = null, set = null, triangles = 0, boundInstances, material = - 1, current = null;

		startPrograms( draws, from, to, srgb, data, extras );
		const features = drawFeatures;
		materialPrograms.length = 0;

		for ( let i = from, o = from * DRAW_FIELDS; i < to; i ++, o += DRAW_FIELDS ) {

			const entry = entries[ i ], key = draws[ o ], start = draws[ o + 2 ], vertices = draws[ o + 3 ], custom = draws[ o + 7 ];
			const instance = perInstance[ i ], copies = instance ? draws[ o + 5 ] : 1;
			const quad = ( key & QUAD_KEY ) !== 0;
			const own = custom >= 16 ? extras?.shaderMaterials?.get( custom ) : null;

			if ( own ) {

				// a GLSL shader material: its program, the built-in uniforms of this draw, its uniforms
				const lights = own.lights === true ? extras.lights : null, fog = own.fog === true ? extras.fog : null;
				const shader = shaderProgram( own, instance !== null, srgb, lights, fog );
				if ( shader !== current ) gl.useProgram( shader.id );
				current = shader;
				if ( shader.flip ) gl.uniform1f( shader.flip, flipY );
				const model = data.objects.subarray( i * 64, i * 64 + 16 ), view = extras.view, modelView = multiply( view, model, new Float32Array( 16 ) );
				setUniforms( shader, own, { modelMatrix: model, viewMatrix: view, projectionMatrix: extras.projection, modelViewMatrix: modelView, normalMatrix: normalMatrix( modelView, new Float32Array( 9 ) ), cameraPosition: extras.position, isOrthographic: extras.orthographic, ...lights?.uniforms, ...fog?.uniforms }, sources );
				applyState( key, customBlending( key, data.materials, draws[ o + 6 ] ) );
				applyMaterialState( key, data.materials, draws[ o + 6 ] * MATERIAL_FLOATS );
				appliedMaterial = - 1;
				state = - 1;
				set = null;
				bindInstances( instance );
				boundInstances = instance;
				bindGeometry( entry );
				bindCustomAttributes( world, entry, shader );
				bound = null;
				draw( key, entry, start, vertices, copies );
				if ( ( key >> 5 & 3 ) === 0 ) triangles += vertices / 3 * copies;
				continue;

			}

			const shader = custom > 0 && custom < 16 ? effectProgram( custom, srgb ) : ( materialPrograms[ draws[ o + 6 ] * 2 + ( quad ? 1 : 0 ) ] ??= meshProgram( srgb, quad, features[ i ] ) );

			if ( shader !== current ) {

				gl.useProgram( shader.id );
				gl.uniform1f( shader.uniform( 'flip_y' ), flipY );
				current = shader;

			}

			// nothing below calls into the module before the next draw, so the view stays valid
			const materialData = data.materials;
			const blending = customBlending( key, materialData, draws[ o + 6 ] );
			if ( key !== state || blending !== blendState ) {

				applyState( state = key, blendState = blending );
				appliedMaterial = - 1;

			}

			// polygon offset, stencil and blend colour follow the material
			if ( draws[ o + 6 ] !== appliedMaterial ) applyMaterialState( key, materialData, ( appliedMaterial = draws[ o + 6 ] ) * MATERIAL_FLOATS );

			const maps = setTextures[ draws[ o + 1 ] ];

			if ( maps !== set ) {

				bindMaps( maps );
				set = maps;

			}

			bindRange( 1, objectBuffer, i, OBJECT_STRIDE, ranges ? null : data.objects );

			if ( draws[ o + 6 ] !== material ) {

				material = draws[ o + 6 ];
				bindRange( 3, materialBuffer, material, MATERIAL_STRIDE, materialData );

			}

			if ( quad ) {

				const topology = key >> 5 & 3, loop = ( key & LOOP_KEY ) !== 0 && vertices > 1;
				const [ first, count ] = loop ? [ 0, vertices ] : quadInstances( topology, start, vertices );
				bindQuads( quadSource( world, entry, start, vertices, loop ), topology, first );
				if ( count > 0 ) gl.drawArraysInstanced( gl.TRIANGLES, 0, 6, count );
				bound = null;
				boundInstances = undefined;
				continue;

			}

			if ( instance !== boundInstances ) {

				bindInstances( instance );
				boundInstances = instance;

			}

			if ( entry !== bound ) {

				bindGeometry( entry );
				bound = entry;

			}

			if ( key & WIREFRAME_KEY ) {

				drawWireframe( world, entry, start, vertices, copies );
				bound = null;

			} else draw( key, entry, start, vertices, copies );

			if ( ( key >> 5 & 3 ) === 0 ) triangles += vertices / 3 * copies;

		}

		gl.bindSampler( VSM_UNIT, null );
		resetState();
		return triangles;

	}

	return {

		name: 'webgl', isWebGLBackend: true, gl, info,

		/**
		 * Builds the mesh programs that `render` with the same scene, camera and target
		 * would use, without drawing; resolves once they are ready.
		 */
		compile( world, renderer, root, camera, target = null, extras = null ) {

			const count = renderer.prepare( world, root, camera );
			ctx.lightFeatures = lightFeatures( renderer.frameUniforms() );
			const draws = renderer.drawInfo(), data = frameData( renderer, count );
			const started = startPrograms( draws, 0, count, ! target && ! floatTargets, data, extras );
			if ( renderer.transmissiveCount() > 0 ) started.push( ...startPrograms( draws, 0, renderer.opaqueCount(), false, data, extras ) );
			// the pass that tone-maps and encodes canvas frames drawn through a linear target
			if ( ! target && floatTargets ) started.push( startProgram( 'output', 'fullscreen.vert', 'output.frag' ) );
			return finishPrograms( started );

		},

		/**
		 * Same arguments as the WebGPU host's `render`. `extras` carries what GLSL shader
		 * materials need: `{ shaderMaterials, view, projection, position, orthographic }`
		 * (the materials by custom id, the camera's view and projection matrices with WebGL
		 * depth, its world position).
		 */
		render( world, renderer, root, camera, clear, sources, instanceSources, target = null, pixelRatio = 1, extras = null ) {

			ctx.frameNumber ++;
			const offscreen = target ? renderTarget( target ) : null;
			ctx.drawingInto = target ? target.id : 0;
			const width = offscreen ? offscreen.width : Math.max( 1, canvas.width ), height = offscreen ? offscreen.height : Math.max( 1, canvas.height );
			renderer.setViewport( width, height, pixelRatio );
			const count = renderer.prepare( world, root, camera );
			info.candidates = renderer.candidateCount();
			const frameUniforms = renderer.frameUniforms();
			upload( gl.UNIFORM_BUFFER, frameUniforms, frameBuffer );
			ctx.lightFeatures = lightFeatures( frameUniforms );

			const geometryState = renderer.geometryState(), slots = renderer.drawGeometrySlots(), draws = renderer.drawInfo();
			const records = geometryState.length / GEOMETRY_FIELDS;
			entries.length = perInstance.length = count;
			distinct.length = records;
			for ( let r = 0; r < records; r ++ ) distinct[ r ] = geometry( world, geometryState, r * GEOMETRY_FIELDS );

			const sets = renderer.textureSets();
			setTextures.length = sets.length / TEXTURE_SLOTS;
			for ( let set = 0; set < setTextures.length; set ++ ) setTextures[ set ] = Array.from( sets.subarray( set * TEXTURE_SLOTS, ( set + 1 ) * TEXTURE_SLOTS ), ( id ) => texture( id, sources ) );

			for ( let i = 0, o = 0; i < count; i ++, o += DRAW_FIELDS ) {

				entries[ i ] = distinct[ slots[ i ] ];
				perInstance[ i ] = instances( draws[ o + 4 ], instanceSources );

			}

			const environmentTexture = environment( renderer.environmentInfo(), sources );
			const backdrop = background( renderer, sources );
			uploadBones( renderer );

			// Taken after the uploads above, which may have grown the module's memory.
			const data = frameData( renderer, count );
			const objectData = data.objects, materialData = data.materials;

			if ( ranges ) {

				if ( count > ctx.objectCapacity ) ctx.objectCapacity = Math.max( 64, Math.ceil( count * 1.5 ) );
				gl.bindBuffer( gl.UNIFORM_BUFFER, objectBuffer );
				gl.bufferData( gl.UNIFORM_BUFFER, ctx.objectCapacity * OBJECT_STRIDE, gl.DYNAMIC_DRAW );
				if ( count > 0 ) gl.bufferSubData( gl.UNIFORM_BUFFER, 0, objectData );
				const materials = materialData.length / MATERIAL_FLOATS;
				if ( materials > ctx.materialCapacity ) ctx.materialCapacity = Math.max( 16, Math.ceil( materials * 1.5 ) );
				gl.bindBuffer( gl.UNIFORM_BUFFER, materialBuffer );
				gl.bufferData( gl.UNIFORM_BUFFER, ctx.materialCapacity * MATERIAL_STRIDE, gl.DYNAMIC_DRAW );
				if ( materialData.length > 0 ) gl.bufferSubData( gl.UNIFORM_BUFFER, 0, materialData );

			}

			const shadowCalls = drawShadows( renderer, instanceSources, data );

			// Tone mapping applies only to what is drawn to the canvas, through a linear target.
			let toneMapping = offscreen ? 0 : frameUniforms[ TONE_MAPPING ];

			if ( toneMapping !== 0 && ! floatTargets ) {

				warn( 'tone mapping needs EXT_color_buffer_float; drawn without it' );
				toneMapping = 0;

			}

			// Render targets are drawn upside down (see `flip_y` in the GLSL), which also
			// turns the winding of front faces.
			const flipY = offscreen ? - 1 : 1;
			const transmissive = renderer.transmissiveCount();
			let copy = null;

			if ( transmissive > 0 ) {

				// the background and the opaque draws, into the copy transmissive materials read
				const copyTarget = transmissionTarget( width, height );
				gl.bindFramebuffer( gl.FRAMEBUFFER, copyTarget.framebuffer );
				gl.drawBuffers( [ gl.COLOR_ATTACHMENT0 ] );
				gl.viewport( 0, 0, width, height );
				gl.depthMask( true );
				gl.clearColor( clear.r, clear.g, clear.b, clear.a );
				gl.clearDepth( 1 );
				gl.clearStencil( 0 );
				resetState();
				gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT );
				gl.frontFace( gl.CW );
				if ( backdrop ) drawBackground( backdrop, false, - 1 );
				drawRange( world, draws, 0, renderer.opaqueCount(), false, - 1, data, environmentTexture, null, sources, extras );
				gl.bindTexture( gl.TEXTURE_2D, copyTarget.color );
				gl.generateMipmap( gl.TEXTURE_2D );
				copy = copyTarget.color;

			}

			// Canvas frames go through a linear half-float target when possible, so that
			// blending happens in linear light as on WebGPU; the output pass then tone-maps
			// (if asked) and encodes to sRGB.
			const linearTarget = ! offscreen && floatTargets ? hdrTarget( width, height ) : null;
			const srgb = ! offscreen && ! linearTarget;
			gl.bindFramebuffer( gl.FRAMEBUFFER, offscreen ? offscreen.draw : linearTarget ? linearTarget.draw : null );
			// extra colour attachments of a render target are not written
			if ( offscreen || linearTarget ) gl.drawBuffers( [ gl.COLOR_ATTACHMENT0 ] );
			gl.viewport( 0, 0, width, height );
			gl.depthMask( true );
			// the clear colour is linear; the canvas holds sRGB
			const encode = srgb ? linearToSRGB : ( c ) => c;
			gl.clearColor( encode( clear.r ), encode( clear.g ), encode( clear.b ), clear.a );
			gl.clearDepth( 1 );
			gl.clearStencil( 0 );
			resetState();
			if ( extras?.load !== true ) gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT );
			gl.frontFace( offscreen ? gl.CW : gl.CCW );
			if ( backdrop ) drawBackground( backdrop, srgb, flipY );
			const triangles = drawRange( world, draws, 0, count, srgb, flipY, data, environmentTexture, copy, sources, extras );

			if ( offscreen && offscreen.draw !== offscreen.framebuffer ) {

				gl.bindFramebuffer( gl.READ_FRAMEBUFFER, offscreen.draw );
				gl.bindFramebuffer( gl.DRAW_FRAMEBUFFER, offscreen.framebuffer );
				gl.blitFramebuffer( 0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.NEAREST );

			}

			if ( linearTarget ) {

				if ( linearTarget.draw !== linearTarget.resolve ) {

					gl.bindFramebuffer( gl.READ_FRAMEBUFFER, linearTarget.draw );
					gl.bindFramebuffer( gl.DRAW_FRAMEBUFFER, linearTarget.resolve );
					gl.blitFramebuffer( 0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.NEAREST );

				}

				const output = program( 'output', 'fullscreen.vert', 'output.frag' );
				gl.bindFramebuffer( gl.FRAMEBUFFER, null );
				gl.viewport( 0, 0, width, height );
				gl.useProgram( output.id );
				gl.uniform4f( output.uniform( 'settings' ), toneMapping, frameUniforms[ EXPOSURE ], 0, 0 );
				gl.activeTexture( gl.TEXTURE0 );
				gl.bindTexture( gl.TEXTURE_2D, linearTarget.color );
				gl.disable( gl.BLEND );
				gl.disable( gl.CULL_FACE );
				gl.depthFunc( gl.ALWAYS );
				unbindArrays();
				gl.drawArrays( gl.TRIANGLES, 0, 3 );

			}

			gl.bindFramebuffer( gl.FRAMEBUFFER, null );
			gl.frontFace( gl.CCW );
			ctx.drawingInto = 0;
			info.calls = count;
			info.shadowCalls = shadowCalls;
			info.triangles = triangles;
			info.geometries = geometries.size;
			info.textures = textures.size;
			info.pipelines = programs.size;

			if ( ctx.frameNumber % UNUSED_FRAMES === 0 ) {

				const stale = ( entry ) => ctx.frameNumber - entry.checked >= UNUSED_FRAMES;
				for ( const [ slot, entry ] of geometries ) if ( stale( entry ) ) {

					release( entry );
					geometries.delete( slot );

				}

				for ( const [ id, entry ] of textures ) if ( stale( entry ) ) {

					gl.deleteTexture( entry.texture );
					textures.delete( id );

				}

				for ( const [ id, entry ] of instanced ) if ( stale( entry ) ) {

					if ( entry.matrix ) gl.deleteBuffer( entry.matrix );
					if ( entry.color ) gl.deleteBuffer( entry.color );
					instanced.delete( id );

				}

			}

		},

		/** Frees the GPU side of a render target (by its texture id). */
		disposeRenderTarget( id ) {

			renderTargets.get( id )?.release();
			renderTargets.delete( id );

		},

		dispose() {

			for ( const entry of renderTargets.values() ) entry.release();
			renderTargets.clear();
			for ( const entry of geometries.values() ) release( entry );
			geometries.clear();
			for ( const entry of textures.values() ) gl.deleteTexture( entry.texture );
			textures.clear();
			for ( const entry of instanced.values() ) {

				if ( entry.matrix ) gl.deleteBuffer( entry.matrix );
				if ( entry.color ) gl.deleteBuffer( entry.color );

			}

			instanced.clear();
			for ( const entry of environments.values() ) gl.deleteTexture( entry.texture );
			environments.clear();
			for ( const buffer of levelBuffers.values() ) gl.deleteBuffer( buffer );
			ctx.hdr?.release();
			ctx.transmission?.release();
			for ( const texture_ of [ ctx.shadowTexture, ctx.vsmTexture, ctx.vsmScratch, ctx.boneTexture, ctx.stack?.texture ] ) if ( texture_ ) gl.deleteTexture( texture_ );
			for ( const buffer of [ frameBuffer, objectBuffer, materialBuffer, shadowObjectBuffer, backgroundBuffer, scratchBuffer ] ) gl.deleteBuffer( buffer );
			for ( const framebuffer of [ shadowFramebuffer, environmentFramebuffer, stackFramebuffer, vsmFramebuffer ] ) gl.deleteFramebuffer( framebuffer );
			for ( const sampler of [ panoramaSampler, screenSampler, depthSampler ] ) gl.deleteSampler( sampler );
			gl.deleteTexture( white );
			for ( const { id } of programs.values() ) gl.deleteProgram( id );
			programs.clear();

		}

	};

}
