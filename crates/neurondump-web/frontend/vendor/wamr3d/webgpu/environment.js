// Environment maps and texture backgrounds for the WebGPU host: prefiltering an
// equirectangular image into three.js's cube-UV layout (PMREM), and the background pass.

import { Renderer } from '../index.js';
import { HDR_FORMAT, HDR_KEY, LINEAR, RGBA8_KEY, SINGLE_KEY, WRAPS } from './formats.js';

// Prefiltered environments (three.js PMREM): smallest face of the detailed levels, and
// the extra, ever blurrier levels kept at that size.
export const LOD_MIN = 4, EXTRA_LODS = 6;
const FACE_ORDER = [ 3, 1, 5, 0, 4, 2 ];

// The quads of one level of the cube-UV layout, with the direction each texel stands
// for: positions are clip coordinates, then the direction (6 floats per vertex).
export function cubeUVLevel( size ) {

	const texel = 1 / ( size - 2 ), min = - texel, max = 1 + texel;
	const uv = [ min, min, max, min, max, max, min, min, max, max, min, max ];
	const data = new Float32Array( 6 * 6 * 6 );

	for ( let face = 0; face < 6; face ++ ) {

		const x = ( face % 3 ) * 2 / 3 - 1, y = face > 2 ? 0 : - 1;
		const corners = [ x, y, x + 2 / 3, y, x + 2 / 3, y + 1, x, y, x + 2 / 3, y + 1, x, y + 1 ];
		const index = FACE_ORDER[ face ];

		for ( let vertex = 0; vertex < 6; vertex ++ ) {

			const u = uv[ vertex * 2 ] * 2 - 1, v = uv[ vertex * 2 + 1 ] * 2 - 1;
			const direction = [ [ 1, v, u ], [ - u, 1, - v ], [ - u, v, 1 ], [ - 1, v, - u ], [ - u, - 1, v ], [ u, v, - 1 ] ][ index ];
			data.set( [ corners[ vertex * 2 ], corners[ vertex * 2 + 1 ], 0, ...direction ], ( index * 6 + vertex ) * 6 );

		}

	}

	return data;

}

/**
 * `texture( id, sources )` makes sure a source texture is on the GPU, `textureOf( id )`
 * gives its GPUTexture (or null) and `frame()` the host's frame number, which marks the
 * prefiltered maps in use.
 */
export function createEnvironmentPasses( device, { format, samples, texture, textureOf, frame } ) {

	// trilinear: the transmission copy, which shares it, has mip levels
	const environmentSampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' } );
	const noEnvironment = device.createTexture( { size: [ 1, 1 ], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING } );
	const noEnvironmentView = noEnvironment.createView();

	// Prefiltered environment maps by source texture id, rebuilt when the source changes.
	const environments = new Map();
	let pmrem = null;

	function prefilter( source, gpuSource, lodMax ) {

		if ( ! pmrem ) {

			const pmremModule = device.createShaderModule( { code: Renderer.pmremShaderSource() } );
			const vertex = { module: pmremModule, entryPoint: 'vertex', buffers: [ { arrayStride: 24, attributes: [ { shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' } ] } ] };
			// One explicit layout for both passes: an automatic one would leave out the
			// settings the equirectangular pass does not read.
			const pmremLayout = device.createBindGroupLayout( { entries: [
				{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
				{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
				{ binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
			] } );
			const layout = device.createPipelineLayout( { bindGroupLayouts: [ pmremLayout ] } );
			pmrem = {
				layout: pmremLayout,
				equirect: device.createRenderPipeline( { layout, vertex, fragment: { module: pmremModule, entryPoint: 'equirect', targets: [ { format: HDR_FORMAT } ] } } ),
				ggx: device.createRenderPipeline( { layout, vertex, fragment: { module: pmremModule, entryPoint: 'ggx', targets: [ { format: HDR_FORMAT } ] } } ),
				sampler: device.createSampler( { magFilter: 'linear', minFilter: 'linear' } ),
				levels: new Map()
			};

		}

		const cubeSize = 2 ** lodMax, width = 3 * Math.max( cubeSize, 16 * 7 ), height = 4 * cubeSize;
		const total = lodMax - LOD_MIN + 1 + EXTRA_LODS, sizes = [];
		for ( let i = 0, lod = lodMax; i < total; i ++ ) {

			sizes.push( 2 ** lod );
			if ( lod > LOD_MIN ) lod --;

		}

		const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
		const cubeUV = device.createTexture( { size: [ width, height ], format: HDR_FORMAT, usage } );
		const pingPong = device.createTexture( { size: [ width, height ], format: HDR_FORMAT, usage } );
		const cubeView = cubeUV.createView(), pingView = pingPong.createView();
		const passes = 1 + 2 * ( total - 1 );
		const settings = device.createBuffer( { size: 256 * passes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
		const filter = ( value ) => ( LINEAR.has( value ?? 1006 ) ? 'linear' : 'nearest' );
		const sourceSampler = device.createSampler( { magFilter: filter( source.magFilter ), minFilter: filter( source.minFilter ), addressModeU: WRAPS[ source.wrapS ] ?? 'clamp-to-edge', addressModeV: WRAPS[ source.wrapT ] ?? 'clamp-to-edge' } );

		const level = ( size ) => {

			let buffer = pmrem.levels.get( size );
			if ( ! buffer ) {

				const data = cubeUVLevel( size );
				buffer = device.createBuffer( { size: data.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST } );
				device.queue.writeBuffer( buffer, 0, data );
				pmrem.levels.set( size, buffer );

			}

			return buffer;

		};

		const encoder = device.createCommandEncoder();
		const draw = ( pass, pipeline, sourceView, sampler, levelSize, target, clear, x, y, w, h ) => {

			const group = device.createBindGroup( { layout: pmrem.layout, entries: [
				{ binding: 0, resource: { buffer: settings, offset: pass * 256, size: 32 } },
				{ binding: 1, resource: sourceView },
				{ binding: 2, resource: sampler }
			] } );
			const renderPass = encoder.beginRenderPass( { colorAttachments: [ { view: target, loadOp: clear ? 'clear' : 'load', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } } ] } );
			renderPass.setViewport( x, y, w, h, 0, 1 );
			renderPass.setScissorRect( x, y, w, h );
			renderPass.setPipeline( pipeline );
			renderPass.setBindGroup( 0, group );
			renderPass.setVertexBuffer( 0, level( levelSize ) );
			renderPass.draw( 36 );
			renderPass.end();

		};

		const uniforms = new Float32Array( 64 * passes );
		const set = ( pass, roughness, mip ) => uniforms.set( [ roughness, mip, 1 / width, 1 / height, lodMax, 0, 0, 0 ], pass * 64 );
		set( 0, 0, 0 );
		draw( 0, pmrem.equirect, gpuSource.createView(), sourceSampler, sizes[ 0 ], cubeView, true, 0, 0, 3 * cubeSize, 2 * cubeSize );

		// Each level is convolved from the one before into the ping-pong target, then
		// copied back (roughness 0 reads one texel), as three.js does.
		for ( let out = 1, pass = 1; out < total; out ++, pass += 2 ) {

			const target = out / ( total - 1 ), previous = ( out - 1 ) / ( total - 1 );
			const roughness = Math.sqrt( target * target - previous * previous ) * target * 1.25;
			const size = sizes[ out ];
			const x = 3 * size * ( out > lodMax - LOD_MIN ? out - lodMax + LOD_MIN : 0 ), y = 4 * ( cubeSize - size );
			set( pass, roughness, lodMax - ( out - 1 ) );
			set( pass + 1, 0, lodMax - out );
			draw( pass, pmrem.ggx, cubeView, pmrem.sampler, size, pingView, out === 1, x, y, 3 * size, 2 * size );
			draw( pass + 1, pmrem.ggx, pingView, pmrem.sampler, size, cubeView, false, x, y, 3 * size, 2 * size );

		}

		device.queue.writeBuffer( settings, 0, uniforms );
		device.queue.submit( [ encoder.finish() ] );
		pingPong.destroy();
		settings.destroy();
		return cubeUV;

	}

	// The prefiltered map of one environment at `lodMax`, or null while its source is not ready.
	function prefiltered( id, lodMax, sources ) {

		const source = sources?.get( id );
		texture( id, sources );
		const gpuSource = textureOf( id );
		if ( ! source || ! gpuSource ) return null;
		const key = id + ':' + lodMax;
		let entry = environments.get( key );

		if ( entry && entry.version === source.version ) {

			entry.checked = frame();
			return entry;

		}

		entry?.texture.destroy();
		const cubeUV = prefilter( source, gpuSource, lodMax );
		entry = { version: source.version, lodMax, texture: cubeUV, view: cubeUV.createView(), checked: frame(), stamp: ++ stamps };
		environments.set( key, entry );
		return entry;

	}

	let stamps = 0, stack = null;

	// The frame's environments stacked in one texture, in the order of `info` (the
	// renderer's `environmentInfo`: lodMax, width and height of one map, texture ids). A
	// single environment is its own map; several are copied into bands of one texture,
	// again only when one of them changed. The empty map while none is ready.
	function environment( info, sources ) {

		if ( info.length === 0 ) return noEnvironmentView;
		const lodMax = info[ 0 ], width = info[ 1 ], height = info[ 2 ], ids = info.subarray ? info.subarray( 3 ) : info.slice( 3 );
		const maps = Array.from( ids, ( id ) => prefiltered( id, lodMax, sources ) );
		if ( maps.some( ( map ) => map === null ) ) return null;
		if ( maps.length === 1 ) return maps[ 0 ].view;

		const key = ids.join( ',' ) + ':' + lodMax + ':' + maps.map( ( map ) => map.stamp ).join( ',' );
		if ( stack && stack.key === key ) return stack.view;
		const size = [ width, height * maps.length ];

		if ( ! stack || stack.texture.width !== size[ 0 ] || stack.texture.height !== size[ 1 ] ) {

			stack?.texture.destroy();
			const texture_ = device.createTexture( { size, format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
			stack = { texture: texture_, view: texture_.createView(), key: '' };

		}

		const encoder = device.createCommandEncoder();
		maps.forEach( ( map, k ) => encoder.copyTextureToTexture( { texture: map.texture }, { texture: stack.texture, origin: [ 0, k * height, 0 ] }, [ width, height, 1 ] ) );
		device.queue.submit( [ encoder.finish() ] );
		stack.key = key;
		return stack.view;

	}

	// The background pass (a texture as three.js `scene.background`): one pipeline per kind
	// of target, and a bind group that follows the texture shown.
	const backgroundLayout = device.createBindGroupLayout( { entries: [
		{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
		{ binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
		{ binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} }
	] } );
	const backgroundPipelines = new Map();
	// Equirectangular images wrap around horizontally.
	const panoramaSampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat' } );
	const screenSampler = device.createSampler( { magFilter: 'linear', minFilter: 'linear' } );
	let backgroundModule = null, backgroundBuffer = null, backgroundGroup = null, backgroundBound = null;
	let backgroundSource = null, backgroundSourceView = null;

	function backgroundPipeline( variant ) {

		let value = backgroundPipelines.get( variant );
		if ( value ) return value;
		backgroundModule ??= device.createShaderModule( { code: Renderer.backgroundShaderSource() } );
		value = device.createRenderPipeline( {
			layout: device.createPipelineLayout( { bindGroupLayouts: [ backgroundLayout ] } ),
			vertex: { module: backgroundModule, entryPoint: 'vertex' },
			fragment: { module: backgroundModule, entryPoint: 'fragment', targets: [ { format: variant & HDR_KEY ? HDR_FORMAT : variant & RGBA8_KEY ? 'rgba8unorm' : format } ] },
			depthStencil: { format: 'depth24plus-stencil8', depthWriteEnabled: false, depthCompare: 'always' },
			multisample: { count: variant & SINGLE_KEY ? 1 : samples }
		} );
		backgroundPipelines.set( variant, value );
		return value;

	}

	// The bind group of the prepared frame's background, or null to keep the clear colour
	// (no background, or its image is not ready yet).
	function background( renderer, sources ) {

		const info = renderer.backgroundInfo();
		if ( info.length === 0 ) return null;
		const [ id, kind, lodMax ] = info;
		let view, sampler = kind === 2 ? screenSampler : panoramaSampler;

		if ( kind === 1 ) {

			view = prefiltered( id, lodMax, sources )?.view ?? null;
			sampler = environmentSampler;

		} else {

			texture( id, sources );
			const gpuTexture = textureOf( id );

			if ( gpuTexture !== backgroundSource ) {

				backgroundSource = gpuTexture;
				backgroundSourceView = gpuTexture?.createView() ?? null;

			}

			view = backgroundSourceView;

		}

		if ( ! view || view === noEnvironmentView ) return null;
		const uniforms = renderer.backgroundUniforms();
		backgroundBuffer ??= device.createBuffer( { size: uniforms.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST } );
		device.queue.writeBuffer( backgroundBuffer, 0, uniforms );

		if ( backgroundBound === null || backgroundBound.view !== view || backgroundBound.sampler !== sampler ) {

			backgroundGroup = device.createBindGroup( { layout: backgroundLayout, entries: [
				{ binding: 0, resource: { buffer: backgroundBuffer } },
				{ binding: 1, resource: view },
				{ binding: 2, resource: sampler }
			] } );
			backgroundBound = { view, sampler };

		}

		return backgroundGroup;

	}

	return {

		sampler: environmentSampler,
		/** Bound while there is no environment. */
		empty: noEnvironmentView,
		environment,
		background,
		backgroundPipeline,

		dispose() {

			noEnvironment.destroy();
			backgroundBuffer?.destroy();
			for ( const entry of environments.values() ) entry.texture.destroy();
			environments.clear();
			stack?.texture.destroy();
			if ( pmrem ) for ( const buffer of pmrem.levels.values() ) buffer.destroy();

		}

	};

}
