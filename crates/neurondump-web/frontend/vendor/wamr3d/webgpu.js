// WebGPU host for the WAMR3D renderer.
//
// The engine decides what is drawn (culling, order, pipeline state, uniform data,
// shader); this file only turns that into WebGPU calls and keeps the GPU copies of
// geometry and textures up to date.

/**
 * `fallbackAdapter` asks for the browser's software adapter, where it has one: slow, but
 * it draws without a usable GPU.
 */

import { Renderer, memory } from './index.js';
import { HDR_FORMAT, HDR_KEY, SINGLE_KEY, lightFeatures, meshFeatures, quadInstances } from './webgpu/formats.js';
import { BLEND_COLOR_FLOAT, COLOR_SLOT, CUSTOM_BLENDING_FLOAT, LOOP_KEY, POLYGON_OFFSET_FLOAT, STENCIL_FLOAT, MATRIX_SLOT, QUAD_KEY, UNUSED_FRAMES, VERTEX_COLOR_SLOT, WIREFRAME_KEY, warnOnce, wireframeCount } from './webgpu/constants.js';
import { installState } from './webgpu/state.js';
import { installPipelines } from './webgpu/pipelines.js';
import { installTextures } from './webgpu/textures.js';
import { installBuffers } from './webgpu/buffers.js';
import { installShadows } from './webgpu/shadows.js';

export async function createBackend( canvas, { antialias = true, powerPreference = 'high-performance', fallbackAdapter = false } = {} ) {

	if ( ! globalThis.navigator?.gpu ) throw new Error( 'WAMR3D: WebGPU is not available in this browser' );
	const adapter = await navigator.gpu.requestAdapter( fallbackAdapter ? { forceFallbackAdapter: true } : { powerPreference } );
	if ( ! adapter ) throw new Error( 'WAMR3D: no WebGPU adapter found' );
	const features = [ 'float32-filterable' ].filter( ( name ) => adapter.features.has( name ) );
	const device = await adapter.requestDevice( { requiredFeatures: features } );

	const context = canvas.getContext( 'webgpu' );
	// The shader writes linear colour into an sRGB view of the canvas, so blending and
	// the multisample resolve happen in linear light and the GPU encodes on write.
	const canvasFormat = navigator.gpu.getPreferredCanvasFormat(), format = canvasFormat + '-srgb';
	context.configure( { device, format: canvasFormat, viewFormats: [ format ], alphaMode: 'opaque' } );

	const samples = antialias ? 4 : 1;
	const stride = Renderer.objectUniformSize(), objectBindingBytes = Renderer.objectBindingSize(), materialStride = Renderer.materialUniformSize();
	const GEOMETRY_FIELDS = Renderer.geometryStateSize(), DRAW_FIELDS = Renderer.drawInfoSize();
	const module = device.createShaderModule( { code: Renderer.shaderSource() } );
	const TONE_MAPPING = Renderer.toneMappingIndex(), EXPOSURE = Renderer.toneMappingExposureIndex();
	const MATERIAL_FLOATS = materialStride / 4;

	const frameLayout = device.createBindGroupLayout( { entries: [
		{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} },
		{ binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: {} }
	] } );
	// The split-sum table of the specular lobe, read by standard and physical materials;
	// it never changes.
	const dfgData = new Float32Array( Renderer.dfgTable() );
	const dfgTable = device.createBuffer( { size: dfgData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( dfgTable, 0, dfgData );
	const objectLayout = device.createBindGroupLayout( { entries: [
		{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { hasDynamicOffset: true, minBindingSize: objectBindingBytes } },
		{ binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { hasDynamicOffset: true, minBindingSize: materialStride } }
	] } );
	// A texture and its sampler per slot (the order of `Renderer.textureSets`). Only the
	// displacement slot is read by the vertex stage.
	const TEXTURE_SLOTS = Renderer.textureSlots(), DISPLACEMENT_SLOT = Renderer.displacementSlot();
	const stages = ( binding ) => ( binding >> 1 === DISPLACEMENT_SLOT ? GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT : GPUShaderStage.FRAGMENT );
	const textureLayout = device.createBindGroupLayout( { entries: Array.from( { length: TEXTURE_SLOTS * 2 }, ( _, binding ) => ( binding % 2 === 0
		? { binding, visibility: stages( binding ), texture: {} }
		: { binding, visibility: stages( binding ), sampler: {} } ) ) } );
	// Group 3 holds what the whole frame samples: the shadow maps, the environments, the
	// bones, the copy transmissive materials read, and the blurred VSM moments.
	const frameTexturesLayout = device.createBindGroupLayout( { entries: [
		{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
		{ binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
		{ binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
		{ binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
		{ binding: 4, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } },
		{ binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },
		{ binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: '2d-array' } }
	] } );
	const layout = device.createPipelineLayout( { bindGroupLayouts: [ frameLayout, objectLayout, textureLayout, frameTexturesLayout ] } );

	// Shadow maps: one layer per map (a point light has six), depth only, as many layers as
	// the frames have needed so far. Bilinear comparison sampling, as three.js uses for
	// PCFShadowMap when the device supports it; basic filtering samples texel centres.
	const MAX_SHADOWS = Renderer.maxShadows(), SHADOW_FORMAT = 'depth32float';
	const shadowSampler = device.createSampler( { compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' } );
	const shadowModule = device.createShaderModule( { code: Renderer.shadowShaderSource() } );
	const shadowCameraLayout = device.createBindGroupLayout( { entries: [ { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { hasDynamicOffset: true, minBindingSize: 64 } } ] } );
	const boneLayout = device.createBindGroupLayout( { entries: [ { binding: 0, visibility: GPUShaderStage.VERTEX, texture: { sampleType: 'unfilterable-float' } } ] } );
	const shadowPipelineLayout = device.createPipelineLayout( { bindGroupLayouts: [ shadowCameraLayout, objectLayout, textureLayout, boneLayout ] } );
	const shadowPipelines = new Map();

	// Bones: a float texture 1024 texels wide, four texels per matrix, grown as needed.
	const BONES_PER_ROW = Renderer.boneTextureMatricesPerRow();
	// The transmission copy and the VSM moments; small placeholders until a frame needs them.
	const emptyFloat = device.createTexture( { size: [ 1, 1 ], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING } );
	const emptyFloatView = emptyFloat.createView();
	const emptyVsm = device.createTexture( { size: [ 1, 1, 1 ], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING } );
	const emptyVsmView = emptyVsm.createView( { dimension: '2d-array' } );

	// The parts of the host share these through `ctx`.
	const ctx = { canvas, BONES_PER_ROW, DRAW_FIELDS, EXPOSURE, GEOMETRY_FIELDS, MATERIAL_FLOATS, MAX_SHADOWS, SHADOW_FORMAT, TEXTURE_SLOTS, TONE_MAPPING, boneLayout, context, device, emptyFloat, emptyFloatView, emptyVsm, emptyVsmView, format, frameLayout, frameTexturesLayout, layout, materialStride, module, objectBindingBytes, objectLayout, samples, shadowCameraLayout, shadowModule, shadowPipelineLayout, shadowPipelines, shadowSampler, stride, textureLayout };
	installState( ctx );
	installPipelines( ctx );
	installTextures( ctx );
	installBuffers( ctx );
	installShadows( ctx );
	const { auxZeros, background, backgroundPipeline, bindGeometry, derivedIndex, distinct, drawShadows, effectPipeline, ensureHdrTargets, ensureTargets, ensureTransmission, entries, environment, environmentPasses, generateMipmaps, geometries, geometry, groups, identity, info, instanced, instances, noEnvironmentView, perInstance, pipeline, pipelineAsync, pipelines, quadSource, rebuildFrameTextures, release, renderTarget, renderTargets, setGroups, textureSetGroup, textureSets, textures, uploadBones, white } = ctx;

	// A material's polygon offset and stencil settings as a pipeline-cache key: '' for the
	// defaults (no offset, stencil untouched), else factor, units, packed stencil settings,
	// read and write masks.
	function materialState( blocks, at ) {

		const factor = blocks[ at + POLYGON_OFFSET_FLOAT ], units = blocks[ at + POLYGON_OFFSET_FLOAT + 1 ];
		const bits = blocks[ at + STENCIL_FLOAT ], readMask = blocks[ at + STENCIL_FLOAT + 2 ], writeMask = blocks[ at + STENCIL_FLOAT + 3 ];
		if ( factor === 0 && units === 0 && ( bits & 1 ) === 0 ) return '';
		return factor + ',' + units + ',' + bits + ',' + readMask + ',' + writeMask;

	}

	// The dynamic offsets of group 1, passed as a typed array: the form of `setBindGroup`
	// that takes an array converts and allocates a sequence on every draw.
	const offsets = new Uint32Array( 2 );

	// Per material of the frame, the draw key of the pipeline last chosen for it and that
	// pipeline. Draws are ordered by depth, so materials alternate from draw to draw;
	// this keeps the feature key and pipeline lookup to once per material and key.
	const materialKeys = [], materialPipelines = [];

	// Records the draws `from` to `to` of the prepared frame into `pass`.
	function drawRange( pass, world, draws, from, to, variant, frameTextures ) {

		pass.setBindGroup( 0, ctx.frameGroup );
		pass.setBindGroup( 3, frameTextures );
		materialKeys.length = materialPipelines.length = 0;
		let current = null, reference = 0, constant = null, bound = null, map = null, triangles = 0, boundInstances = undefined;

		for ( let i = from, o = from * DRAW_FIELDS; i < to; i ++, o += DRAW_FIELDS ) {

			const entry = entries[ i ], key = draws[ o ], start = draws[ o + 2 ], vertices = draws[ o + 3 ], material = draws[ o + 6 ], custom = draws[ o + 7 ];
			const instance = perInstance[ i ], copies = instance ? draws[ o + 5 ] : 1;

			if ( custom > 0 && custom < 16 ) {

				const effect = effectPipeline( key | variant, custom );
				if ( effect !== current ) pass.setPipeline( current = effect );

			} else {

				// three.js's WebGPURenderer draws no GLSL shader materials either; they draw with their colour
				if ( custom >= 16 ) warnOnce( 'ShaderMaterial and RawShaderMaterial (GLSL) draw only on WebGL 2; drawn unlit' );
				const blocks = ctx.materials.view, at = material * MATERIAL_FLOATS;
				const blending = ( key & 4 ) && ( key >> 11 & 7 ) === 5 ? blocks[ at + CUSTOM_BLENDING_FLOAT ] : 0;
				let chosen = materialKeys[ material ] === key ? materialPipelines[ material ] : null;

				if ( ! chosen ) {

					chosen = pipeline( key | variant, blending, materialState( blocks, at ), meshFeatures( blocks, at, ctx.lightFeatures ) );
					materialKeys[ material ] = key;
					materialPipelines[ material ] = chosen;

				}

				if ( chosen !== current ) pass.setPipeline( current = chosen );
				const stencilReference = blocks[ at + STENCIL_FLOAT + 1 ];
				if ( stencilReference !== reference ) pass.setStencilReference( reference = stencilReference );

				if ( blending ) {

					// the constant colour of custom blending (three.js `blendColor`, `blendAlpha`)
					const r = blocks[ at + BLEND_COLOR_FLOAT ], g = blocks[ at + BLEND_COLOR_FLOAT + 1 ], b = blocks[ at + BLEND_COLOR_FLOAT + 2 ], a = blocks[ at + BLEND_COLOR_FLOAT + 3 ];
					if ( ! constant || constant.r !== r || constant.g !== g || constant.b !== b || constant.a !== a ) pass.setBlendConstant( constant = { r, g, b, a } );

				}

			}
			if ( groups[ i ] !== map ) pass.setBindGroup( 2, map = groups[ i ] );
			offsets[ 0 ] = i * stride;
			offsets[ 1 ] = material * materialStride;
			pass.setBindGroup( 1, ctx.objectGroup, offsets, 0, 2 );

			if ( key & QUAD_KEY ) {

				// points or segments as instances of a six-vertex quad
				const topology = key >> 5 & 3, loop = ( key & LOOP_KEY ) !== 0 && vertices > 1;
				const source = quadSource( world, entry, start, vertices, loop );
				const [ first, count ] = loop ? [ 0, vertices ] : quadInstances( topology, start, vertices );
				pass.setVertexBuffer( 0, source.buffers[ 0 ] ?? ctx.zeros );
				pass.setVertexBuffer( VERTEX_COLOR_SLOT, source.buffers[ VERTEX_COLOR_SLOT ] ?? ctx.zeros );
				pass.setVertexBuffer( 1, source.distance ?? ctx.zeros );
				bound = null;
				if ( count > 0 ) pass.draw( 6, count, 0, first );
				continue;

			}

			if ( instance !== boundInstances ) {

				pass.setVertexBuffer( MATRIX_SLOT, instance ? instance.matrix : identity );
				pass.setVertexBuffer( COLOR_SLOT, instance?.color ?? ctx.ones );
				boundInstances = instance;

			}

			if ( entry !== bound ) {

				bindGeometry( pass, entry );
				bound = entry;

			}

			if ( key & ( LOOP_KEY | WIREFRAME_KEY ) ) {

				const wireframe = ( key & WIREFRAME_KEY ) !== 0, indices = wireframe ? wireframeCount( vertices ) : vertices + 1;

				if ( indices > 1 ) {

					pass.setIndexBuffer( derivedIndex( world, entry, start, vertices, wireframe ), 'uint32' );
					pass.drawIndexed( indices, copies, 0 );
					// the geometry's own index buffer is bound again by the next draw
					bound = null;

				}

			} else if ( entry.index ) pass.drawIndexed( vertices, copies, start );
			else pass.draw( vertices, copies, start );
			if ( ( key >> 5 & 3 ) === 0 ) triangles += vertices / 3 * copies;

		}

		return triangles;

	}

	return {

		name: fallbackAdapter ? 'webgpu-software' : 'webgpu', isWebGPUBackend: true, device, format, info,

		/**
		 * Builds the mesh pipelines that `render` with the same scene, camera and target
		 * would use, without drawing; resolves once they are ready.
		 */
		compile( world, renderer, root, camera, target = null ) {

			const offscreen = target ? renderTarget( target ) : null;
			const count = renderer.prepare( world, root, camera );
			const frameUniforms = renderer.frameUniforms();
			ctx.lightFeatures = lightFeatures( frameUniforms );
			const variant = offscreen ? offscreen.variant : frameUniforms[ TONE_MAPPING ] !== 0 ? HDR_KEY : 0;
			const draws = renderer.drawInfo(), blocks = new Float32Array( memory().buffer, renderer.materialUniformsPtr(), renderer.materialUniformsLength() );
			const pending = [];
			const start = ( to, passVariant ) => {

				for ( let i = 0, o = 0; i < to; i ++, o += DRAW_FIELDS ) {

					const key = draws[ o ], custom = draws[ o + 7 ], at = draws[ o + 6 ] * MATERIAL_FLOATS;
					if ( custom > 0 && custom < 16 ) continue;
					const blending = ( key & 4 ) && ( key >> 11 & 7 ) === 5 ? blocks[ at + CUSTOM_BLENDING_FLOAT ] : 0;
					pending.push( pipelineAsync( key | passVariant, blending, materialState( blocks, at ), meshFeatures( blocks, at, ctx.lightFeatures ) ) );

				}

			};

			start( count, variant );
			if ( renderer.transmissiveCount() > 0 ) start( renderer.opaqueCount(), HDR_KEY | SINGLE_KEY );
			return Promise.all( pending ).then( () => {} );

		},

		/**
		 * `instanceSources` maps instance ids to `{ matrix, matrixVersion, color, colorVersion }`
		 * (16 floats and optionally 3 floats per instance).
		 * `clear` is a linear colour `{ r, g, b, a }`. `sources` maps texture ids to `{ image, version, flipY, magFilter, minFilter, wrapS, wrapT, colorSpace, generateMipmaps, type }`.
		 * `target`, when given, is a render target `{ id, width, height, samples, type, texture, textures, depthTexture }`
		 * (`type` 'byte' or 'float'): the frame is drawn into it, without tone mapping.
		 * `pixelRatio` sizes points and lines drawn as quads. */
		render( world, renderer, root, camera, clear, sources, instanceSources, target = null, pixelRatio = 1, extras = null ) {

			const load = extras?.load === true;

			ctx.frameNumber ++;
			const offscreen = target ? renderTarget( target ) : null;
			ctx.drawingInto = target ? target.id : 0;
			if ( ! offscreen ) ensureTargets();
			const width = offscreen ? offscreen.width : ctx.targetWidth, height = offscreen ? offscreen.height : ctx.targetHeight;
			renderer.setViewport( width, height, pixelRatio );

			const count = renderer.prepare( world, root, camera );
			info.candidates = renderer.candidateCount();

			const frameUniforms = renderer.frameUniforms();

			if ( ! ctx.frameBuffer ) {

				ctx.frameBuffer = device.createBuffer( { size: frameUniforms.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
				ctx.frameGroup = device.createBindGroup( { layout: frameLayout, entries: [ { binding: 0, resource: { buffer: ctx.frameBuffer } }, { binding: 1, resource: { buffer: dfgTable } } ] } );

			}

			device.queue.writeBuffer( ctx.frameBuffer, 0, frameUniforms );
			ctx.lightFeatures = lightFeatures( frameUniforms );

			const materialCount = renderer.materialUniformsLength() / MATERIAL_FLOATS;

			if ( count > ctx.objectCapacity || materialCount > ctx.materialCapacity || ! ctx.objectGroup ) {

				if ( count > ctx.objectCapacity || ! ctx.objectBuffer ) {

					ctx.objectBuffer?.destroy();
					ctx.objectCapacity = Math.max( 64, Math.ceil( count * 1.5 ) );
					ctx.objectBuffer = device.createBuffer( { size: ctx.objectCapacity * stride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );

				}

				if ( materialCount > ctx.materialCapacity || ! ctx.materialBuffer ) {

					ctx.materialBuffer?.destroy();
					ctx.materialCapacity = Math.max( 16, Math.ceil( materialCount * 1.5 ) );
					ctx.materialBuffer = device.createBuffer( { size: ctx.materialCapacity * materialStride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
					ctx.shadowObjectGroup = null;

				}

				ctx.objectGroup = device.createBindGroup( { layout: objectLayout, entries: [ { binding: 0, resource: { buffer: ctx.objectBuffer, size: objectBindingBytes } }, { binding: 1, resource: { buffer: ctx.materialBuffer, size: materialStride } } ] } );

			}

			const geometryState = renderer.geometryState(), slots = renderer.drawGeometrySlots(), draws = renderer.drawInfo();
			const records = geometryState.length / GEOMETRY_FIELDS;
			entries.length = groups.length = perInstance.length = count;
			distinct.length = records;
			for ( let r = 0; r < records; r ++ ) distinct[ r ] = geometry( world, geometryState, r * GEOMETRY_FIELDS );

			const sets = renderer.textureSets();
			setGroups.length = sets.length / TEXTURE_SLOTS;
			for ( let set = 0; set < setGroups.length; set ++ ) setGroups[ set ] = textureSetGroup( sets, set, sources );

			for ( let i = 0, o = 0; i < count; i ++, o += DRAW_FIELDS ) {

				entries[ i ] = distinct[ slots[ i ] ];
				groups[ i ] = setGroups[ draws[ o + 1 ] ];
				perInstance[ i ] = instances( draws[ o + 4 ], draws[ o + 5 ], instanceSources );

			}

			if ( ! ctx.ones || ctx.ones.size < ctx.oneCount * 12 ) {

				ctx.ones?.destroy();
				ctx.ones = device.createBuffer( { size: ctx.oneCount * 12, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST } );
				device.queue.writeBuffer( ctx.ones, 0, new Float32Array( ctx.oneCount * 3 ).fill( 1 ) );

			}

			if ( ! ctx.zeros || ctx.zeros.size < ctx.zeroBytes ) {

				ctx.zeros?.destroy();
				ctx.zeros = device.createBuffer( { size: ctx.zeroBytes, usage: GPUBufferUsage.VERTEX } );

			}

			const nextEnvironment = environment( renderer.environmentInfo(), sources ) ?? noEnvironmentView;

			if ( nextEnvironment !== ctx.environmentView ) {

				ctx.environmentView = nextEnvironment;
				rebuildFrameTextures();

			}

			const backdrop = background( renderer, sources );
			uploadBones( renderer );
			const transmissive = renderer.transmissiveCount(), opaque = renderer.opaqueCount();
			if ( transmissive > 0 ) ensureTransmission( width, height );

			// Taken after the uploads above, which may have grown the module's memory.
			// The frame's material blocks in the module's memory. A call into the module (an
			// upload, a draw list) can grow its memory, which detaches the view; it is taken
			// again then.
			let materialView = new Float32Array( memory().buffer, renderer.materialUniformsPtr(), renderer.materialUniformsLength() );
			if ( materialView.length > 0 ) device.queue.writeBuffer( ctx.materialBuffer, 0, materialView );
			ctx.materials = {
				get view() {

					if ( materialView.buffer.byteLength === 0 ) materialView = new Float32Array( memory().buffer, renderer.materialUniformsPtr(), renderer.materialUniformsLength() );
					return materialView;

				}
			};
			if ( count > 0 ) device.queue.writeBuffer( ctx.objectBuffer, 0, new Float32Array( memory().buffer, renderer.objectUniformsPtr(), renderer.objectUniformsLength() ) );

			const encoder = device.createCommandEncoder();
			const shadowCalls = drawShadows( encoder, renderer, instanceSources, ctx.materials );
			// Tone mapping applies only to what is drawn to the canvas, as in three.js.
			const toneMapping = offscreen ? 0 : frameUniforms[ TONE_MAPPING ], hdr = toneMapping !== 0;
			if ( hdr ) ensureHdrTargets();
			else if ( ! offscreen ) ctx.ensureColorTarget();

			if ( transmissive > 0 ) {

				// the background and the opaque draws, into the copy transmissive materials read
				const copy = encoder.beginRenderPass( {
					colorAttachments: [ { view: ctx.transmission.target, clearValue: clear, loadOp: 'clear', storeOp: 'store' } ],
					depthStencilAttachment: { view: ctx.transmission.depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard', stencilClearValue: 0, stencilLoadOp: 'clear', stencilStoreOp: 'discard' }
				} );

				if ( backdrop ) {

					copy.setPipeline( backgroundPipeline( HDR_KEY | SINGLE_KEY ) );
					copy.setBindGroup( 0, backdrop );
					copy.draw( 3 );

				}

				drawRange( copy, world, draws, 0, opaque, HDR_KEY | SINGLE_KEY, ctx.frameTexturesGroupNoCopy );
				copy.end();
				generateMipmaps( ctx.transmission.texture, encoder );

			}

			const view = offscreen ? null : context.getCurrentTexture().createView( { format } );
			const variant = offscreen ? offscreen.variant : hdr ? HDR_KEY : 0;
			const colorAttachments = offscreen
				? offscreen.views.map( ( resolved, i ) => {

					const multisampled = offscreen.multisampleViews[ i ];
					return { view: multisampled ?? resolved, resolveTarget: multisampled ? resolved : undefined, clearValue: clear, loadOp: load ? 'load' : 'clear', storeOp: multisampled && ! load ? 'discard' : 'store' };

				} )
				: [ { view: ( hdr ? ctx.hdrMultisampleView : ctx.colorView ) ?? ( hdr ? ctx.hdrView : view ), resolveTarget: ( hdr ? ctx.hdrMultisampleView : ctx.colorView ) ? ( hdr ? ctx.hdrView : view ) : undefined, clearValue: clear, loadOp: load ? 'load' : 'clear', storeOp: ( hdr ? ctx.hdrMultisampleView : ctx.colorView ) && ! load ? 'discard' : 'store' } ];
			// a render target's extra colour attachments repeat the first output
			if ( offscreen && offscreen.count > 1 ) colorAttachments.length = 1;
			const pass = encoder.beginRenderPass( {
				colorAttachments,
				depthStencilAttachment: { view: offscreen ? offscreen.depthView : ctx.depthView, stencilClearValue: 0, stencilLoadOp: load ? 'load' : 'clear', stencilStoreOp: 'store', depthClearValue: 1, depthLoadOp: load ? 'load' : 'clear', depthStoreOp: offscreen?.depthId || load ? 'store' : 'discard' }
			} );

			if ( backdrop ) {

				pass.setPipeline( backgroundPipeline( variant ) );
				pass.setBindGroup( 0, backdrop );
				pass.draw( 3 );

			}

			const triangles = drawRange( pass, world, draws, 0, count, variant, ctx.frameTexturesGroup );
			pass.end();
			if ( offscreen ) ctx.depthDrawn( offscreen );

			if ( hdr ) {

				device.queue.writeBuffer( ctx.outputSettings, 0, new Float32Array( [ toneMapping, frameUniforms[ EXPOSURE ], 0, 0 ] ) );
				const output = encoder.beginRenderPass( { colorAttachments: [ { view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } } ] } );
				output.setPipeline( ctx.outputPipeline );
				output.setBindGroup( 0, ctx.outputGroup );
				output.draw( 3 );
				output.end();

			}

			device.queue.submit( [ encoder.finish() ] );
			ctx.drawingInto = 0;

			info.calls = count;
			info.shadowCalls = shadowCalls;
			info.triangles = triangles;
			info.geometries = geometries.size;
			info.textures = textures.size;
			info.pipelines = pipelines.size;

			if ( ctx.frameNumber % UNUSED_FRAMES === 0 ) {

				for ( const [ handle, entry ] of geometries ) {

					if ( ctx.frameNumber - entry.checked < UNUSED_FRAMES ) continue;
					release( entry );
					geometries.delete( handle );

				}

				for ( const [ id, entry ] of textures ) {

					if ( ctx.frameNumber - entry.checked < UNUSED_FRAMES ) continue;
					entry.texture?.destroy();
					textures.delete( id );

				}

				for ( const [ key, entry ] of textureSets ) {

					if ( ctx.frameNumber - entry.checked >= UNUSED_FRAMES ) textureSets.delete( key );

				}

				for ( const [ id, entry ] of instanced ) {

					if ( ctx.frameNumber - entry.checked < UNUSED_FRAMES ) continue;
					entry.matrix?.destroy();
					entry.color?.destroy();
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
			for ( const entry of textures.values() ) entry.texture?.destroy();
			textures.clear();
			for ( const entry of instanced.values() ) {

				entry.matrix?.destroy();
				entry.color?.destroy();

			}

			instanced.clear();
			for ( const resource of [ white, identity, dfgTable, auxZeros, ctx.ones, ctx.zeros, ctx.objectBuffer, ctx.materialBuffer, ctx.frameBuffer, ctx.depthTexture, ctx.colorTexture, ctx.hdrTexture, ctx.hdrMultisample, ctx.outputSettings, ctx.shadowTexture, ctx.shadowCameras, ctx.shadowObjects, ctx.boneTexture, emptyFloat, emptyVsm, ctx.vsmTexture, ctx.vsmScratch, ctx.transmission?.texture, ctx.transmission?.depth, ctx.vsm?.settings, ctx.vsm?.placeholder ] ) resource?.destroy();
			environmentPasses.dispose();
			context.unconfigure();
			// Destroying the device while frames are still queued on the GPU can block the page
			// for tens of seconds in Chrome (seen when an app swaps renderers under heavy load),
			// so the device goes once its submitted work is done.
			device.queue.onSubmittedWorkDone().then( () => device.destroy(), () => device.destroy() );

		}

	};

}
