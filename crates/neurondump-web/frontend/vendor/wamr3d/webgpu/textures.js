// Textures, mipmaps, cube textures, texture-set bind groups, render targets and the environment passes.
//
// Part of the WebGPU host; `../webgpu.js` creates the device and draws the frames.

import { Renderer } from '../index.js';
import { HDR_FORMAT, HDR_KEY, LINEAR, MIPMAPPED, MIP_LINEAR, RGBA8_KEY, SINGLE_KEY, WRAPS, flippedRows } from './formats.js';
import { createEnvironmentPasses } from './environment.js';
import { BLIT, DEPTH_COPY, DEPTH_FORMAT, cubeFace, cubeReady, dataFormat } from './constants.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installTextures( ctx ) {

	const { TEXTURE_SLOTS, blits, canvas, device, format, samples, textureLayout, textures } = ctx;

	// The view and sampler of a texture, made once per upload. Bind groups are made per
	// set of textures (see `textureSetGroup`) and remember the bindings they were made from.
	function textureBinding( gpuTexture, source ) {

		const filterable = gpuTexture.format !== 'rgba32float' || device.features.has( 'float32-filterable' );
		const filter = ( value ) => ( filterable && LINEAR.has( value ?? 1006 ) ? 'linear' : 'nearest' );
		const sampler = device.createSampler( {
			magFilter: filter( source?.magFilter ), minFilter: filter( source?.minFilter ),
			mipmapFilter: filterable && MIP_LINEAR.has( source?.minFilter ) ? 'linear' : 'nearest',
			addressModeU: WRAPS[ source?.wrapS ] ?? 'clamp-to-edge', addressModeV: WRAPS[ source?.wrapT ] ?? 'clamp-to-edge',
			maxAnisotropy: filterable && ( source?.anisotropy ?? 1 ) > 1 && LINEAR.has( source?.magFilter ?? 1006 ) && MIP_LINEAR.has( source?.minFilter ) ? Math.min( 16, source.anisotropy ) : 1
		} );
		return { view: gpuTexture.createView(), sampler };

	}

	function createTexture( width, height, textureFormat, mipLevelCount = 1 ) {

		return device.createTexture( { size: [ width, height ], mipLevelCount, format: textureFormat, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT } );

	}

	// Fills every mip level from the one above it with a filtered draw.
	function generateMipmaps( gpuTexture, encoder = null ) {

		let blit = blits.get( gpuTexture.format );

		if ( ! blit ) {

			const blitModule = device.createShaderModule( { code: BLIT } );
			blit = {
				pipeline: device.createRenderPipeline( { layout: 'auto', vertex: { module: blitModule, entryPoint: 'vertex' }, fragment: { module: blitModule, entryPoint: 'fragment', targets: [ { format: gpuTexture.format } ] } } ),
				sampler: device.createSampler( { magFilter: 'linear', minFilter: 'linear' } )
			};
			blits.set( gpuTexture.format, blit );

		}

		const own = encoder === null, commands = encoder ?? device.createCommandEncoder();

		for ( let level = 1; level < gpuTexture.mipLevelCount; level ++ ) {

			const group = device.createBindGroup( { layout: blit.pipeline.getBindGroupLayout( 0 ), entries: [
				{ binding: 0, resource: gpuTexture.createView( { baseMipLevel: level - 1, mipLevelCount: 1 } ) },
				{ binding: 1, resource: blit.sampler }
			] } );
			const pass = commands.beginRenderPass( { colorAttachments: [ { view: gpuTexture.createView( { baseMipLevel: level, mipLevelCount: 1 } ), loadOp: 'clear', storeOp: 'store' } ] } );
			pass.setPipeline( blit.pipeline );
			pass.setBindGroup( 0, group );
			pass.draw( 3 );
			pass.end();

		}

		if ( own ) device.queue.submit( [ commands.finish() ] );

	}

	const white = createTexture( 1, 1, 'rgba8unorm' );
	device.queue.writeTexture( { texture: white }, new Uint8Array( [ 255, 255, 255, 255 ] ), {}, [ 1, 1 ] );
	const whiteBinding = textureBinding( white, null );

	// A depth texture read as a material map: the material slots take filterable floats,
	// which a depth texture is not, so its depth is copied into a float texture (red, green
	// and blue the depth, alpha one, as three.js samples a depth texture) after each time
	// its target was drawn.
	let depthCopyPipeline = null;

	function depthCopy( binding ) {

		if ( binding.copy && ! binding.drawn ) return binding.copy;
		const { width, height } = binding;
		const format = device.features.has( 'float32-filterable' ) ? 'r32float' : 'r16float';

		if ( ! binding.copy || binding.copy.texture.width !== width || binding.copy.texture.height !== height ) {

			binding.copy?.texture.destroy();
			const copy = device.createTexture( { size: [ width, height ], format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT } );
			binding.copy = { texture: copy, view: copy.createView(), sampler: binding.sampler };

		}

		depthCopyPipeline ??= device.createRenderPipeline( {
			layout: 'auto',
			vertex: { module: device.createShaderModule( { code: DEPTH_COPY } ), entryPoint: 'vertex' },
			fragment: { module: device.createShaderModule( { code: DEPTH_COPY } ), entryPoint: 'fragment', targets: [ { format } ] },
			primitive: { topology: 'triangle-list' }
		} );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginRenderPass( { colorAttachments: [ { view: binding.copy.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 1, g: 1, b: 1, a: 1 } } ] } );
		pass.setPipeline( depthCopyPipeline );
		pass.setBindGroup( 0, device.createBindGroup( { layout: depthCopyPipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: binding.view } ] } ) );
		pass.draw( 3 );
		pass.end();
		device.queue.submit( [ encoder.finish() ] );
		binding.drawn = false;
		return binding.copy;

	}

	// A material's texture. Uploaded when the source's version changes; a source that is
	// not ready yet (an image still loading) draws as white.
	function texture( id, sources ) {

		const source = id === 0 ? null : sources?.get( id );

		if ( source?.isRenderTargetTexture || source?.isDepthTexture ) {

			// Not while drawing into the same target; not before it was drawn once.
			if ( id === ctx.drawingInto ) return whiteBinding;
			const binding = renderTargetBindings.get( id );
			if ( ! binding ) return whiteBinding;
			return binding.depth ? depthCopy( binding ) : binding;

		}

		if ( source?.isCubeTexture ) return cubeTexture( id, source );
		const image = source?.image;
		if ( ! image || ! ( image.data || image.width > 0 ) ) return whiteBinding;

		let entry = textures.get( id );
		if ( ! entry ) textures.set( id, entry = { texture: null, binding: null, version: - 1, checked: 0 } );
		entry.checked = ctx.frameNumber;
		if ( entry.version === source.version ) return entry.binding;

		const { width, height } = image, float = image.data ? dataFormat( source ) : null;
		const textureFormat = float ?? ( source.colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm' );
		const levels = ! float && source.generateMipmaps !== false && MIPMAPPED.has( source.minFilter ) ? Math.floor( Math.log2( Math.max( width, height ) ) ) + 1 : 1;

		if ( ! entry.texture || entry.texture.width !== width || entry.texture.height !== height || entry.texture.format !== textureFormat || entry.texture.mipLevelCount !== levels ) {

			entry.texture?.destroy();
			entry.texture = createTexture( width, height, textureFormat, levels );

		}

		if ( image.data ) {

			// float data is RGBA of 32-bit floats; half floats arrive as Uint16Array
			const bytes = textureFormat === 'rgba32float' ? 16 : textureFormat === 'rgba16float' ? 8 : 4;
			const data = source.flipY === true ? flippedRows( image.data, width, height ) : image.data;
			device.queue.writeTexture( { texture: entry.texture }, data, { bytesPerRow: width * bytes }, [ width, height ] );

		} else {

			device.queue.copyExternalImageToTexture( { source: image, flipY: source.flipY !== false }, { texture: entry.texture }, [ width, height ] );

		}

		if ( levels > 1 ) generateMipmaps( entry.texture );
		entry.binding = textureBinding( entry.texture, source );
		entry.version = source.version;
		return entry.binding;

	}

	// A cube texture becomes an equirectangular image (four faces wide, two high) on each
	// upload, so environments and backgrounds take it like any panorama. Float faces
	// (three.js HDR cube maps) keep a float format. White until its six images have arrived.
	let cubePass = null;

	function cubeTexture( id, source ) {

		if ( ! cubeReady( source.image ) ) return whiteBinding;

		let entry = textures.get( id );
		if ( ! entry ) textures.set( id, entry = { texture: null, binding: null, version: - 1, checked: 0 } );
		entry.checked = ctx.frameNumber;
		if ( entry.version === source.version ) return entry.binding;

		const images = source.image.map( cubeFace ), size = images[ 0 ].width;
		const faceSource = source.image[ 0 ]?.isDataTexture ? source.image[ 0 ] : source;
		const float = images[ 0 ].data ? dataFormat( faceSource ) : null;
		// faces are sampled filtered; rgba32float may not be, so float faces go to rgba16float
		const faceFormat = float ?? ( source.colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm' ), format_ = float ? HDR_FORMAT : faceFormat;
		const faces = device.createTexture( { size: [ size, size, 6 ], format: faceFormat, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT } );
		const bytes = faceFormat === 'rgba32float' ? 16 : faceFormat === 'rgba16float' ? 8 : 4;

		images.forEach( ( image, layer ) => {

			if ( image.data ) device.queue.writeTexture( { texture: faces, origin: [ 0, 0, layer ] }, image.data, { bytesPerRow: size * bytes, rowsPerImage: size }, [ size, size, 1 ] );
			else device.queue.copyExternalImageToTexture( { source: image, flipY: false }, { texture: faces, origin: [ 0, 0, layer ] }, [ size, size ] );

		} );

		const width = Math.min( 4 * size, device.limits.maxTextureDimension2D ), height = width / 2;

		if ( ! entry.texture || entry.texture.width !== width || entry.texture.format !== format_ ) {

			entry.texture?.destroy();
			entry.texture = device.createTexture( { size: [ width, height ], format: format_, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT } );

		}

		if ( ! cubePass ) {

			const cubeModule = device.createShaderModule( { code: Renderer.cubeToEquirectShaderSource() } );
			cubePass = { module: cubeModule, pipelines: new Map(), sampler: device.createSampler( { magFilter: 'linear', minFilter: 'linear' } ), nearest: device.createSampler( {} ) };

		}

		let converter = cubePass.pipelines.get( format_ );
		if ( ! converter ) cubePass.pipelines.set( format_, converter = device.createRenderPipeline( { layout: 'auto', vertex: { module: cubePass.module, entryPoint: 'vertex' }, fragment: { module: cubePass.module, entryPoint: 'fragment', targets: [ { format: format_ } ] } } ) );

		const filterable = faceFormat !== 'rgba32float' || device.features.has( 'float32-filterable' );
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginRenderPass( { colorAttachments: [ { view: entry.texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } } ] } );
		pass.setPipeline( converter );
		pass.setBindGroup( 0, device.createBindGroup( { layout: converter.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: faces.createView( { dimension: 'cube' } ) }, { binding: 1, resource: filterable ? cubePass.sampler : cubePass.nearest } ] } ) );
		pass.draw( 3 );
		pass.end();
		device.queue.submit( [ encoder.finish() ] );
		faces.destroy();

		entry.binding = { view: entry.texture.createView(), sampler: cubePass.sampler };
		entry.version = source.version;
		return entry.binding;

	}

	// One bind group per distinct set of textures, kept while each of its textures keeps
	// the binding it was made with.
	const textureSets = new Map(), setGroups = [];

	function textureSetGroup( sets, set, sources ) {

		const ids = sets.subarray( set * TEXTURE_SLOTS, ( set + 1 ) * TEXTURE_SLOTS ), key = ids.join( ',' );
		const bindings = Array.from( ids, ( id ) => texture( id, sources ) );
		let entry = textureSets.get( key );

		if ( ! entry || entry.bindings.some( ( binding, i ) => binding !== bindings[ i ] ) ) {

			const group = device.createBindGroup( { layout: textureLayout, entries: bindings.flatMap( ( binding, i ) => [ { binding: 2 * i, resource: binding.view }, { binding: 2 * i + 1, resource: binding.sampler } ] ) } );
			textureSets.set( key, entry = { bindings, group, checked: 0 } );

		}

		entry.checked = ctx.frameNumber;
		return entry.group;

	}

	function ensureHdrTargets() {

		if ( ctx.hdrTexture && ctx.hdrTexture.width === ctx.targetWidth && ctx.hdrTexture.height === ctx.targetHeight ) return;
		ctx.hdrTexture?.destroy();
		ctx.hdrMultisample?.destroy();
		ctx.hdrTexture = device.createTexture( { size: [ ctx.targetWidth, ctx.targetHeight ], format: HDR_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING } );
		ctx.hdrMultisample = samples > 1 ? device.createTexture( { size: [ ctx.targetWidth, ctx.targetHeight ], format: HDR_FORMAT, sampleCount: samples, usage: GPUTextureUsage.RENDER_ATTACHMENT } ) : null;
		ctx.hdrView = ctx.hdrTexture.createView();
		ctx.hdrMultisampleView = ctx.hdrMultisample ? ctx.hdrMultisample.createView() : null;

		if ( ! ctx.outputPipeline ) {

			const outputModule = device.createShaderModule( { code: Renderer.outputShaderSource() } );
			ctx.outputPipeline = device.createRenderPipeline( { layout: 'auto', vertex: { module: outputModule, entryPoint: 'vertex' }, fragment: { module: outputModule, entryPoint: 'fragment', targets: [ { format } ] } } );
			ctx.outputSettings = device.createBuffer( { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );

		}

		ctx.outputGroup = device.createBindGroup( { layout: ctx.outputPipeline.getBindGroupLayout( 0 ), entries: [ { binding: 0, resource: { buffer: ctx.outputSettings } }, { binding: 1, resource: ctx.hdrView } ] } );

	}

	// The mipmapped copy of the opaque scene that transmissive materials read, as large as
	// the frame.
	function ensureTransmission( width, height ) {

		if ( ctx.transmission && ctx.transmission.width === width && ctx.transmission.height === height ) return ctx.transmission;
		ctx.transmission?.texture.destroy();
		ctx.transmission?.depth.destroy();
		const levels = Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;
		const texture_ = device.createTexture( { size: [ width, height ], format: HDR_FORMAT, mipLevelCount: levels, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING } );
		const depth = device.createTexture( { size: [ width, height ], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT } );
		ctx.transmission = { width, height, texture: texture_, depth, view: texture_.createView(), target: texture_.createView( { baseMipLevel: 0, mipLevelCount: 1 } ), depthView: depth.createView() };
		ctx.rebuildFrameTextures();
		return ctx.transmission;

	}

	// GPU side of render targets, by the id of their texture: colour attachments
	// (multisampled when asked), depth (sampled when the target has a depth texture), and
	// the bindings that read them as material maps.
	const renderTargets = new Map(), renderTargetBindings = new Map();

	function renderTarget( target ) {

		let entry = renderTargets.get( target.id );
		// Multisampling needs pipelines for 4 samples, which exist only with antialias.
		const { width, height } = target, float = target.type !== 'byte', multisampled = target.samples > 1 && samples > 1;
		const count = target.textures?.length ?? 1, depthTexture_ = target.depthTexture ?? null;
		if ( entry && entry.width === width && entry.height === height && entry.float === float && entry.multisampled === multisampled && entry.count === count && entry.depthId === ( depthTexture_?.id ?? 0 ) ) return entry;
		if ( entry ) entry.release();

		const colorFormat = float ? HDR_FORMAT : 'rgba8unorm';
		const colors = [], multisampledColors = [];

		for ( let i = 0; i < count; i ++ ) {

			colors.push( device.createTexture( { size: [ width, height ], format: colorFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING } ) );
			if ( multisampled ) multisampledColors.push( device.createTexture( { size: [ width, height ], format: colorFormat, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT } ) );

		}

		// with a depth texture, the depth is resolved into a single-sampled copy for sampling
		const depth = device.createTexture( { size: [ width, height ], format: DEPTH_FORMAT, sampleCount: multisampled && ! depthTexture_ ? 4 : 1, usage: GPUTextureUsage.RENDER_ATTACHMENT | ( depthTexture_ ? GPUTextureUsage.TEXTURE_BINDING : 0 ) } );
		entry = {
			width, height, float, multisampled: multisampled && ! depthTexture_, count, depthId: depthTexture_?.id ?? 0, colors, multisampledColors, depth,
			views: colors.map( ( c ) => c.createView() ), multisampleViews: multisampledColors.map( ( c ) => c.createView() ), depthView: depth.createView(),
			variant: ( float ? HDR_KEY : RGBA8_KEY ) | ( multisampled && ! depthTexture_ ? 0 : SINGLE_KEY ),
			release() {

				for ( const c of colors ) c.destroy();
				for ( const c of multisampledColors ) c.destroy();
				depth.destroy();
				for ( const texture_ of target.textures ?? [ target.texture ] ) renderTargetBindings.delete( texture_.id );
				if ( depthTexture_ ) renderTargetBindings.delete( depthTexture_.id );

			}
		};
		// with a depth texture the target is single-sampled: the colour is read as drawn
		if ( depthTexture_ ) entry.multisampleViews = [];
		( target.textures ?? [ target.texture ] ).forEach( ( texture_, i ) => renderTargetBindings.set( texture_.id, { ...textureBinding( colors[ i ], texture_ ), texture: colors[ i ] } ) );
		if ( depthTexture_ ) renderTargetBindings.set( depthTexture_.id, { view: depth.createView( { aspect: 'depth-only' } ), sampler: device.createSampler( {} ), depth: true, drawn: true, width, height } );
		renderTargets.set( target.id, entry );
		return entry;

	}

	function ensureTargets() {

		const width = Math.max( 1, canvas.width ), height = Math.max( 1, canvas.height );
		if ( width === ctx.targetWidth && height === ctx.targetHeight ) return;
		ctx.depthTexture?.destroy();
		ctx.colorTexture?.destroy();
		ctx.depthTexture = device.createTexture( { size: [ width, height ], format: DEPTH_FORMAT, sampleCount: samples, usage: GPUTextureUsage.RENDER_ATTACHMENT } );
		ctx.colorTexture = null;
		ctx.depthView = ctx.depthTexture.createView();
		ctx.colorView = null;
		ctx.targetWidth = width;
		ctx.targetHeight = height;

	}

	// The multisampled canvas-format target, made on the first frame drawn without tone
	// mapping: tone-mapped frames draw into the linear target instead, and at 3840x2160
	// with four samples this one alone is 127 MiB.
	function ensureColorTarget() {

		if ( samples === 1 || ctx.colorTexture ) return;
		ctx.colorTexture = device.createTexture( { size: [ ctx.targetWidth, ctx.targetHeight ], format, sampleCount: samples, usage: GPUTextureUsage.RENDER_ATTACHMENT } );
		ctx.colorView = ctx.colorTexture.createView();

	}

	const environmentPasses = createEnvironmentPasses( device, { format, samples, texture, textureOf: ( id ) => textures.get( id )?.texture ?? renderTargetBindings.get( id )?.texture ?? null, frame: () => ctx.frameNumber } );
	const { sampler: environmentSampler, empty: noEnvironmentView, environment, background, backgroundPipeline } = environmentPasses;
	ctx.environmentView = noEnvironmentView;

	// A target with a depth texture was drawn: its depth copy is out of date.
	function depthDrawn( target ) {

		const binding = target.depthId ? renderTargetBindings.get( target.depthId ) : null;
		if ( binding ) binding.drawn = true;

	}

	Object.assign( ctx, { depthDrawn, background, backgroundPipeline, ensureColorTarget, ensureHdrTargets, ensureTargets, ensureTransmission, environment, environmentPasses, environmentSampler, generateMipmaps, noEnvironmentView, renderTarget, renderTargets, setGroups, texture, textureSetGroup, textureSets, white } );

}
