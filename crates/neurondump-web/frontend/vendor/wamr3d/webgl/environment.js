// Prefiltered environments and backgrounds.
//
// Part of the WebGL 2 host; `../webgl.js` creates the context and draws the frames.

import { EXTRA_LODS, LOD_MIN, cubeUVLevel } from '../webgpu/environment.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installEnvironment( ctx ) {

	const { backgroundBuffer, floatTargets, gl, newTexture, parameters, program, texture, unbindArrays, upload, warn, white } = ctx;

	// ---------------------------------------------------------------- environment

	// Prefiltered environments in three.js's cube-UV layout, by source texture id and
	// lodMax, and the frame's stack of them. They need half-float targets.
	const environments = new Map(), levelBuffers = new Map();
	const environmentFramebuffer = gl.createFramebuffer(), stackFramebuffer = gl.createFramebuffer();
	ctx.stack = null;

	function floatTexture( width, height ) {

		const id = newTexture();
		gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null );
		parameters( null, 1 );
		return id;

	}

	function prefilter( gpuSource, lodMax ) {

		const cubeSize = 2 ** lodMax, width = 3 * Math.max( cubeSize, 16 * 7 ), height = 4 * cubeSize;
		const total = lodMax - LOD_MIN + 1 + EXTRA_LODS, sizes = [];
		for ( let i = 0, lod = lodMax; i < total; i ++ ) {

			sizes.push( 2 ** lod );
			if ( lod > LOD_MIN ) lod --;

		}

		const cubeUV = floatTexture( width, height ), pingPong = floatTexture( width, height );
		const equirect = program( 'pmrem:equirect', 'pmrem.vert', 'pmrem.equirect' ), ggx = program( 'pmrem:ggx', 'pmrem.vert', 'pmrem.ggx' );
		gl.disable( gl.DEPTH_TEST );
		gl.disable( gl.BLEND );
		gl.disable( gl.CULL_FACE );
		gl.enable( gl.SCISSOR_TEST );
		gl.bindFramebuffer( gl.FRAMEBUFFER, environmentFramebuffer );
		gl.drawBuffers( [ gl.COLOR_ATTACHMENT0 ] );
		unbindArrays();

		const level = ( size ) => {

			let buffer = levelBuffers.get( size );
			if ( ! buffer ) levelBuffers.set( size, buffer = upload( gl.ARRAY_BUFFER, cubeUVLevel( size ) ) );
			ctx.arrayInput( 0, buffer, 3, 24, 0, 0 );
			ctx.arrayInput( 1, buffer, 3, 24, 12, 0 );

		};

		const pass = ( shader, source, target, size, clear, x, y, w, h, roughness, mip ) => {

			gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0 );

			if ( clear ) {

				gl.disable( gl.SCISSOR_TEST );
				gl.clearColor( 0, 0, 0, 1 );
				gl.clear( gl.COLOR_BUFFER_BIT );
				gl.enable( gl.SCISSOR_TEST );

			}

			gl.viewport( x, y, w, h );
			gl.scissor( x, y, w, h );
			gl.useProgram( shader.id );
			gl.uniform4f( shader.uniform( 'params' ), roughness, mip, 1 / width, 1 / height );
			gl.uniform4f( shader.uniform( 'extra' ), lodMax, 0, 0, 0 );
			gl.activeTexture( gl.TEXTURE0 );
			gl.bindTexture( gl.TEXTURE_2D, source );
			level( size );
			gl.drawArrays( gl.TRIANGLES, 0, 36 );

		};

		pass( equirect, gpuSource, cubeUV, sizes[ 0 ], true, 0, 0, 3 * cubeSize, 2 * cubeSize, 0, 0 );

		// Each level is convolved from the one before into the ping-pong target, then
		// copied back (roughness 0 reads one texel).
		for ( let out = 1; out < total; out ++ ) {

			const target = out / ( total - 1 ), previous = ( out - 1 ) / ( total - 1 );
			const roughness = Math.sqrt( target * target - previous * previous ) * target * 1.25;
			const size = sizes[ out ];
			const x = 3 * size * ( out > lodMax - LOD_MIN ? out - lodMax + LOD_MIN : 0 ), y = 4 * ( cubeSize - size );
			pass( ggx, cubeUV, pingPong, size, out === 1, x, y, 3 * size, 2 * size, roughness, lodMax - ( out - 1 ) );
			pass( ggx, pingPong, cubeUV, size, false, x, y, 3 * size, 2 * size, 0, lodMax - out );

		}

		gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0 );
		gl.bindFramebuffer( gl.FRAMEBUFFER, null );
		gl.disable( gl.SCISSOR_TEST );
		gl.enable( gl.DEPTH_TEST );
		unbindArrays();
		gl.deleteTexture( pingPong );
		return cubeUV;

	}

	let stamps = 0;

	function prefiltered( id, lodMax, sources ) {

		if ( ! floatTargets ) {

			warn( 'environment maps need EXT_color_buffer_float; drawn without them' );
			return null;

		}

		const source = sources?.get( id ), gpuSource = texture( id, sources );
		if ( ! source || gpuSource === white ) return null;
		const key = id + ':' + lodMax;
		let entry = environments.get( key );

		if ( entry && entry.version === source.version ) {

			entry.checked = ctx.frameNumber;
			return entry;

		}

		if ( entry ) gl.deleteTexture( entry.texture );
		entry = { version: source.version, lodMax, texture: prefilter( gpuSource, lodMax ), checked: ctx.frameNumber, stamp: ++ stamps };
		environments.set( key, entry );
		return entry;

	}

	// The frame's environments stacked in bands of one texture (see `Renderer.environmentInfo`).
	function environment( info, sources ) {

		if ( info.length === 0 ) return null;
		const lodMax = info[ 0 ], width = info[ 1 ], height = info[ 2 ], ids = Array.from( info.subarray( 3 ) );
		const maps = ids.map( ( id ) => prefiltered( id, lodMax, sources ) );
		if ( maps.some( ( map ) => map === null ) ) return null;
		if ( maps.length === 1 ) return maps[ 0 ].texture;
		const key = ids.join( ',' ) + ':' + lodMax + ':' + maps.map( ( map ) => map.stamp ).join( ',' );
		if ( ctx.stack && ctx.stack.key === key ) return ctx.stack.texture;

		if ( ! ctx.stack || ctx.stack.width !== width || ctx.stack.height !== height * maps.length ) {

			if ( ctx.stack ) gl.deleteTexture( ctx.stack.texture );
			ctx.stack = { texture: floatTexture( width, height * maps.length ), width, height: height * maps.length, key: '' };

		}

		gl.bindFramebuffer( gl.DRAW_FRAMEBUFFER, stackFramebuffer );
		gl.framebufferTexture2D( gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, ctx.stack.texture, 0 );
		gl.bindFramebuffer( gl.READ_FRAMEBUFFER, environmentFramebuffer );

		maps.forEach( ( map, k ) => {

			gl.framebufferTexture2D( gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, map.texture, 0 );
			gl.blitFramebuffer( 0, 0, width, height, 0, k * height, width, ( k + 1 ) * height, gl.COLOR_BUFFER_BIT, gl.NEAREST );

		} );

		gl.framebufferTexture2D( gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0 );
		gl.bindFramebuffer( gl.READ_FRAMEBUFFER, null );
		gl.bindFramebuffer( gl.DRAW_FRAMEBUFFER, null );
		ctx.stack.key = key;
		return ctx.stack.texture;

	}

	// Background: one sampler per kind, overriding the texture's own settings.
	const panoramaSampler = gl.createSampler(), screenSampler = gl.createSampler();

	for ( const [ sampler, wrapS ] of [ [ panoramaSampler, gl.REPEAT ], [ screenSampler, gl.CLAMP_TO_EDGE ] ] ) {

		gl.samplerParameteri( sampler, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
		gl.samplerParameteri( sampler, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
		gl.samplerParameteri( sampler, gl.TEXTURE_WRAP_S, wrapS );
		gl.samplerParameteri( sampler, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );

	}

	// The texture and sampler of the prepared frame's background, or null to keep the clear colour.
	function background( renderer, sources ) {

		const backgroundInfo = renderer.backgroundInfo();
		if ( backgroundInfo.length === 0 ) return null;
		const [ id, kind, lodMax ] = backgroundInfo;
		const source = kind === 1 ? prefiltered( id, lodMax, sources )?.texture : texture( id, sources );
		if ( ! source || source === white ) return null;
		upload( gl.UNIFORM_BUFFER, renderer.backgroundUniforms(), backgroundBuffer );
		return { texture: source, sampler: kind === 0 ? panoramaSampler : screenSampler };

	}

	Object.assign( ctx, { background, environment, environmentFramebuffer, environments, levelBuffers, panoramaSampler, screenSampler, stackFramebuffer } );

}
