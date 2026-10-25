// Group 3 (frame textures), bones, shadow maps and the VSM blur.
//
// Part of the WebGPU host; `../webgpu.js` creates the device and draws the frames.

import { Renderer, memory } from '../index.js';
import { HDR_FORMAT } from './formats.js';
import { ALPHA_TEST_KEY, COLOR_SLOT, STATE_FLAGS_FLOAT, MATRIX_SLOT, vertexBuffers } from './constants.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installShadows( ctx ) {

	const { BONES_PER_ROW, DRAW_FIELDS, MATERIAL_FLOATS, MAX_SHADOWS, SHADOW_FORMAT, bindGeometry, boneLayout, device, distinct, emptyFloatView, emptyVsmView, environmentSampler, frameTexturesLayout, identity, instances, materialStride, objectBindingBytes, objectLayout, setGroups, shadowCameraLayout, shadowModule, shadowPipelineLayout, shadowPipelines, shadowSampler, stride } = ctx;
	// group 1 offsets of a shadow draw, reused (see `drawRange` in ../webgpu.js)
	const offsets = new Uint32Array( 2 );

	function rebuildFrameTextures() {

		const entries = ( copy ) => [
			{ binding: 0, resource: ctx.shadowTexture.createView( { dimension: '2d-array' } ) },
			{ binding: 1, resource: shadowSampler },
			{ binding: 2, resource: ctx.environmentView },
			{ binding: 3, resource: environmentSampler },
			{ binding: 4, resource: ctx.boneView },
			{ binding: 5, resource: copy },
			{ binding: 6, resource: ctx.vsmView ?? emptyVsmView }
		];
		ctx.frameTexturesGroup = device.createBindGroup( { layout: frameTexturesLayout, entries: entries( ctx.transmission?.view ?? emptyFloatView ) } );
		ctx.frameTexturesGroupNoCopy = ctx.transmission ? device.createBindGroup( { layout: frameTexturesLayout, entries: entries( emptyFloatView ) } ) : ctx.frameTexturesGroup;
		ctx.boneGroup = device.createBindGroup( { layout: boneLayout, entries: [ { binding: 0, resource: ctx.boneView } ] } );

	}

	rebuildFrameTextures();

	// Uploads the frame's skinning matrices, growing the bone texture when needed.
	function uploadBones( renderer ) {

		const length = renderer.skinMatricesLength();
		if ( length === 0 ) return;
		const matrices = length / 16, rows = Math.ceil( matrices / BONES_PER_ROW ), width = BONES_PER_ROW * 4;

		if ( ctx.boneTexture.width !== width || ctx.boneTexture.height < rows ) {

			ctx.boneTexture.destroy();
			ctx.boneTexture = device.createTexture( { size: [ width, Math.max( rows, ctx.boneTexture.height * 2, 1 ) ], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
			ctx.boneView = ctx.boneTexture.createView();
			rebuildFrameTextures();

		}

		// whole rows; the last one is padded with what the module's memory holds after it
		const data = new Float32Array( memory().buffer, renderer.skinMatricesPtr(), length );
		const full = Math.floor( matrices / BONES_PER_ROW );
		if ( full > 0 ) device.queue.writeTexture( { texture: ctx.boneTexture }, data, { bytesPerRow: width * 16, offset: 0 }, [ width, full ] );
		const rest = matrices - full * BONES_PER_ROW;
		if ( rest > 0 ) device.queue.writeTexture( { texture: ctx.boneTexture, origin: [ 0, full ] }, data, { bytesPerRow: width * 16, offset: full * BONES_PER_ROW * 64 }, [ rest * 4, 1 ] );

	}

	let shadowCameraGroup = null, shadowObjectCapacity = 0;
	ctx.shadowCameras = null;
	ctx.shadowObjects = null;
	ctx.shadowObjectGroup = null;

	// Casters draw their back faces (three.js draws the opposite of `material.side`). Only
	// casters with an alpha test run a fragment stage.
	function shadowPipeline( key, vsm ) {

		const pipelineKey = key | ( vsm ? 1 << 24 : 0 );
		let value = shadowPipelines.get( pipelineKey );
		if ( value ) return value;
		value = device.createRenderPipeline( {
			layout: shadowPipelineLayout,
			vertex: { module: shadowModule, entryPoint: 'vertex', buffers: vertexBuffers( key ) },
			fragment: key & ALPHA_TEST_KEY ? { module: shadowModule, entryPoint: 'fragment', targets: [] } : undefined,
			primitive: { topology: 'triangle-list', frontFace: 'ccw', cullMode: [ 'back', 'front', 'none' ][ key & 3 ] },
			depthStencil: { format: SHADOW_FORMAT, depthWriteEnabled: true, depthCompare: 'less-equal' }
		} );
		shadowPipelines.set( pipelineKey, value );
		return value;

	}

	function ensureShadowTexture( width, height, layers, vsm ) {

		const sameSize = ctx.shadowTexture.width === width && ctx.shadowTexture.height === height;

		if ( ! sameSize || ctx.shadowTexture.depthOrArrayLayers < layers ) {

			const count = Math.min( MAX_SHADOWS, sameSize ? Math.max( layers, ctx.shadowTexture.depthOrArrayLayers ) : layers );
			ctx.shadowTexture.destroy();
			ctx.shadowTexture = device.createTexture( { size: [ width, height, count ], format: SHADOW_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT } );
			rebuildFrameTextures();

		}

		if ( vsm && ( ! ctx.vsmTexture || ctx.vsmTexture.width !== width || ctx.vsmTexture.height !== height || ctx.vsmTexture.depthOrArrayLayers !== ctx.shadowTexture.depthOrArrayLayers ) ) {

			ctx.vsmTexture?.destroy();
			ctx.vsmScratch?.destroy();
			ctx.vsmTexture = device.createTexture( { size: [ width, height, ctx.shadowTexture.depthOrArrayLayers ], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT } );
			ctx.vsmScratch = device.createTexture( { size: [ width, height ], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT } );
			ctx.vsmView = ctx.vsmTexture.createView( { dimension: '2d-array' } );
			rebuildFrameTextures();

		}

	}

	// The VSM blur: the depth of each pass's layer, vertically into a scratch texture of
	// moments, then horizontally into the layer of the moments array.
	ctx.vsm = null;

	function blurShadows( encoder, renderer, passes ) {

		if ( ! ctx.vsm ) {

			const vsmModule = device.createShaderModule( { code: Renderer.vsmShaderSource() } );
			const vsmLayout = device.createBindGroupLayout( { entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { hasDynamicOffset: true, minBindingSize: 16 } },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth', viewDimension: '2d-array' } },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }
			] } );
			const pipelineLayout = device.createPipelineLayout( { bindGroupLayouts: [ vsmLayout ] } );
			const make = ( entryPoint ) => device.createRenderPipeline( { layout: pipelineLayout, vertex: { module: vsmModule, entryPoint: 'vertex' }, fragment: { module: vsmModule, entryPoint, targets: [ { format: HDR_FORMAT } ] } } );
			ctx.vsm = { layout: vsmLayout, vertical: make( 'vertical' ), horizontal: make( 'horizontal' ), settings: device.createBuffer( { size: 256 * MAX_SHADOWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } ), placeholder: device.createTexture( { size: [ 1, 1 ], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING } ) };

		}

		const settings = new Float32Array( 64 * passes );
		const group = ( depthView, momentsView ) => device.createBindGroup( { layout: ctx.vsm.layout, entries: [
			{ binding: 0, resource: { buffer: ctx.vsm.settings, size: 16 } },
			{ binding: 1, resource: depthView },
			{ binding: 2, resource: momentsView }
		] } );
		const vertical = group( ctx.shadowTexture.createView( { dimension: '2d-array' } ), ctx.vsm.placeholder.createView() );
		const horizontal = group( ctx.shadowTexture.createView( { dimension: '2d-array' } ), ctx.vsmScratch.createView() );

		for ( let p = 0; p < passes; p ++ ) {

			const info = renderer.shadowPassInfo( p ), blur = renderer.shadowBlur( p );
			settings.set( [ blur[ 0 ], blur[ 1 ], info[ 0 ], 0 ], p * 64 );

		}

		device.queue.writeBuffer( ctx.vsm.settings, 0, settings );

		for ( let p = 0; p < passes; p ++ ) {

			const info = renderer.shadowPassInfo( p );
			const layer = info[ 0 ], w = info[ 1 ], h = info[ 2 ];
			const first = encoder.beginRenderPass( { colorAttachments: [ { view: ctx.vsmScratch.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 0, b: 0, a: 1 } } ] } );
			first.setViewport( 0, 0, w, h, 0, 1 );
			first.setPipeline( ctx.vsm.vertical );
			first.setBindGroup( 0, vertical, [ p * 256 ] );
			first.draw( 3 );
			first.end();
			const second = encoder.beginRenderPass( { colorAttachments: [ { view: ctx.vsmTexture.createView( { dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1 } ), loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 0, b: 0, a: 1 } } ] } );
			second.setViewport( 0, 0, w, h, 0, 1 );
			second.setPipeline( ctx.vsm.horizontal );
			second.setBindGroup( 0, horizontal, [ p * 256 ] );
			second.draw( 3 );
			second.end();

		}

	}

	// Draws each shadow pass of the prepared frame into its layer. Casters use the same
	// geometry, instance, material and texture buffers as the main pass.
	function drawShadows( encoder, renderer, instanceSources, materials ) {

		const passes = renderer.shadowCount();
		if ( passes === 0 ) return 0;
		const [ width, height ] = renderer.shadowMapSize();
		const variance = renderer.shadowFilter() === 2;
		ensureShadowTexture( width, height, passes, variance );

		if ( ! ctx.shadowCameras ) {

			ctx.shadowCameras = device.createBuffer( { size: 256 * MAX_SHADOWS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			shadowCameraGroup = device.createBindGroup( { layout: shadowCameraLayout, entries: [ { binding: 1, resource: { buffer: ctx.shadowCameras, size: 64 } } ] } );

		}

		const infos = [], uniforms = [];
		let total = 0;

		for ( let p = 0; p < passes; p ++ ) {

			const info = renderer.shadowPassInfo( p );
			infos.push( { layer: info[ 0 ], width: info[ 1 ], height: info[ 2 ], draws: renderer.shadowDrawInfo( p ), slots: renderer.shadowDrawGeometrySlots( p ), first: total } );
			device.queue.writeBuffer( ctx.shadowCameras, p * 256, new Float32Array( info.subarray( 3, 19 ) ) );
			const objectData = renderer.shadowObjectUniforms( p );
			uniforms.push( objectData );
			total += objectData.length / ( stride / 4 );

		}

		if ( total > shadowObjectCapacity ) {

			ctx.shadowObjects?.destroy();
			shadowObjectCapacity = Math.max( 64, Math.ceil( total * 1.5 ) );
			ctx.shadowObjects = device.createBuffer( { size: shadowObjectCapacity * stride, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
			ctx.shadowObjectGroup = null;

		}

		ctx.shadowObjectGroup ??= device.createBindGroup( { layout: objectLayout, entries: [ { binding: 0, resource: { buffer: ctx.shadowObjects, size: objectBindingBytes } }, { binding: 1, resource: { buffer: ctx.materialBuffer, size: materialStride } } ] } );
		let calls = 0;

		for ( let p = 0; p < passes; p ++ ) {

			const { layer, width: w, height: h, draws, slots, first } = infos[ p ];
			if ( uniforms[ p ].length > 0 ) device.queue.writeBuffer( ctx.shadowObjects, first * stride, uniforms[ p ] );

			const pass = encoder.beginRenderPass( {
				colorAttachments: [],
				depthStencilAttachment: { view: ctx.shadowTexture.createView( { dimension: '2d', baseArrayLayer: layer, arrayLayerCount: 1 } ), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' }
			} );
			pass.setViewport( 0, 0, w, h, 0, 1 );
			pass.setBindGroup( 0, shadowCameraGroup, [ p * 256 ] );
			pass.setBindGroup( 3, ctx.boneGroup );
			let state = - 1, bound = null, boundInstances, map = null;

			for ( let i = 0, o = 0; i < slots.length; i ++, o += DRAW_FIELDS ) {

				const entry = distinct[ slots[ i ] ], material = draws[ o + 6 ], start = draws[ o + 2 ], vertices = draws[ o + 3 ];
				// surface.w of the material block is its alpha test
				// a fragment stage for alpha tests and for clipping planes that clip shadows
				const block = materials.view, at = material * MATERIAL_FLOATS;
				const key = draws[ o ] | ( block[ at + 23 ] > 0 || ( block[ at + STATE_FLAGS_FLOAT ] & 256 ) ? ALPHA_TEST_KEY : 0 );
				const instance = instances( draws[ o + 4 ], draws[ o + 5 ], instanceSources ), copies = instance ? draws[ o + 5 ] : 1;
				if ( key !== state ) pass.setPipeline( shadowPipeline( state = key, variance ) );

				if ( instance !== boundInstances ) {

					pass.setVertexBuffer( MATRIX_SLOT, instance ? instance.matrix : identity );
					pass.setVertexBuffer( COLOR_SLOT, instance?.color ?? ctx.ones );
					boundInstances = instance;

				}

				if ( entry !== bound ) {

					bindGeometry( pass, entry );
					bound = entry;

				}

				const set = setGroups[ draws[ o + 1 ] ];
				if ( set !== map ) pass.setBindGroup( 2, map = set );
				offsets[ 0 ] = ( first + i ) * stride;
				offsets[ 1 ] = material * materialStride;
				pass.setBindGroup( 1, ctx.shadowObjectGroup, offsets, 0, 2 );
				if ( entry.index ) pass.drawIndexed( vertices, copies, start );
				else pass.draw( vertices, copies, start );
				calls ++;

			}

			pass.end();

		}

		if ( variance ) blurShadows( encoder, renderer, passes );
		return calls;

	}

	Object.assign( ctx, { drawShadows, rebuildFrameTextures, uploadBones } );

}
