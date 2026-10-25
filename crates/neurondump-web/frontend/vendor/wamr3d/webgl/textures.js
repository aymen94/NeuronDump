// Textures, render targets and the half-float and transmission targets.
//
// Part of the WebGL 2 host; `../webgl.js` creates the context and draws the frames.

import { MIPMAPPED, flippedRows } from '../webgpu/formats.js';
import { cubeFace, cubeReady } from './constants.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installTextures( ctx ) {

	const { MIN_FILTER, WRAP, anisotropic, floatLinear, floatTargets, gl, program, samples, unbindArrays, warn } = ctx;

	// ---------------------------------------------------------------- textures

	function newTexture() {

		const id = gl.createTexture();
		gl.bindTexture( gl.TEXTURE_2D, id );
		return id;

	}

	function parameters( source, levels, float = false ) {

		const min = source?.minFilter ?? 1006, mag = source?.magFilter ?? 1006;
		let minFilter = MIN_FILTER[ min ] ?? gl.LINEAR;
		if ( levels === 1 ) minFilter = MIPMAPPED.has( min ) ? ( min === 1004 || min === 1005 ? gl.NEAREST : gl.LINEAR ) : minFilter;
		// 32-bit float textures filter only with OES_texture_float_linear
		const nearest = float && ! floatLinear;
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, nearest ? gl.NEAREST : minFilter );
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, nearest || mag === 1003 ? gl.NEAREST : gl.LINEAR );
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, WRAP[ source?.wrapS ] ?? gl.CLAMP_TO_EDGE );
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, WRAP[ source?.wrapT ] ?? gl.CLAMP_TO_EDGE );
		if ( anisotropic && ( source?.anisotropy ?? 1 ) > 1 && levels > 1 ) gl.texParameterf( gl.TEXTURE_2D, anisotropic.TEXTURE_MAX_ANISOTROPY_EXT, Math.min( source.anisotropy, gl.getParameter( anisotropic.MAX_TEXTURE_MAX_ANISOTROPY_EXT ) ) );

	}

	const white = newTexture();
	gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array( [ 255, 255, 255, 255 ] ) );
	parameters( null, 1 );

	// WebGL ignores UNPACK_FLIP_Y for ImageBitmap, so a bitmap is turned over on a canvas.
	function upright( bitmap ) {

		const flipped = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas( bitmap.width, bitmap.height ) : Object.assign( document.createElement( 'canvas' ), { width: bitmap.width, height: bitmap.height } );
		const context = flipped.getContext( '2d' );
		context.translate( 0, bitmap.height );
		context.scale( 1, - 1 );
		context.drawImage( bitmap, 0, 0 );
		return flipped;

	}

	const textures = new Map();
	ctx.frameNumber = 0;
	ctx.drawingInto = 0;

	// The internal format, format and type of a data texture's pixels (RGBA).
	function dataFormat( source, srgb ) {

		if ( source?.type === 1015 ) return [ gl.RGBA32F, gl.FLOAT, true ];
		if ( source?.type === 1016 ) return [ gl.RGBA16F, gl.HALF_FLOAT, false ];
		return [ srgb ? gl.SRGB8_ALPHA8 : gl.RGBA8, gl.UNSIGNED_BYTE, false ];

	}

	// A material's texture, uploaded when the source's version changes; white while an
	// image is still loading.
	function texture( id, sources ) {

		const source = id === 0 ? null : sources?.get( id );

		if ( source?.isRenderTargetTexture || source?.isDepthTexture ) {

			if ( id === ctx.drawingInto ) return white;
			return renderTargetTextures.get( id ) ?? white;

		}

		if ( source?.isCubeTexture ) return cubeTexture( id, source );
		const image = source?.image;
		if ( ! image || ! ( image.data || image.width > 0 ) ) return white;

		let entry = textures.get( id );
		if ( ! entry ) textures.set( id, entry = { texture: null, version: - 1, checked: 0 } );
		entry.checked = ctx.frameNumber;
		if ( entry.version === source.version ) return entry.texture;

		const { width, height } = image, srgb = source.colorSpace === 'srgb';
		const [ internal, type, float32 ] = image.data ? dataFormat( source, srgb ) : [ srgb ? gl.SRGB8_ALPHA8 : gl.RGBA8, gl.UNSIGNED_BYTE, false ];
		const levels = type === gl.UNSIGNED_BYTE && source.generateMipmaps !== false && MIPMAPPED.has( source.minFilter ) ? 2 : 1;
		if ( entry.texture ) gl.deleteTexture( entry.texture );
		entry.texture = newTexture();

		if ( image.data ) {

			gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, false );
			gl.texImage2D( gl.TEXTURE_2D, 0, internal, width, height, 0, gl.RGBA, type, source.flipY === true ? flippedRows( image.data, width, height ) : image.data );

		} else {

			let pixels = image, flip = source.flipY !== false;

			if ( flip && typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap ) {

				pixels = upright( image );
				flip = false;

			}

			gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, flip );
			gl.texImage2D( gl.TEXTURE_2D, 0, internal, gl.RGBA, gl.UNSIGNED_BYTE, pixels );
			gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, false );

		}

		if ( levels > 1 ) gl.generateMipmap( gl.TEXTURE_2D );
		parameters( source, levels, float32 );
		entry.version = source.version;
		return entry.texture;

	}

	// A cube texture becomes an equirectangular image (four faces wide, two high) on each
	// upload, so environments and backgrounds take it like any panorama. Float faces keep
	// a half-float image. White until its six images have arrived.
	function cubeTexture( id, source ) {

		if ( ! cubeReady( source.image ) ) return white;

		let entry = textures.get( id );
		if ( ! entry ) textures.set( id, entry = { texture: null, version: - 1, checked: 0 } );
		entry.checked = ctx.frameNumber;
		if ( entry.version === source.version ) return entry.texture;

		const images = source.image.map( cubeFace ), size = images[ 0 ].width, srgb = source.colorSpace === 'srgb';
		const faceSource = source.image[ 0 ]?.isDataTexture ? source.image[ 0 ] : source;
		const [ internal, type ] = images[ 0 ].data ? dataFormat( faceSource, srgb ) : [ srgb ? gl.SRGB8_ALPHA8 : gl.RGBA8, gl.UNSIGNED_BYTE ];
		const float = type !== gl.UNSIGNED_BYTE;
		if ( float && ! floatTargets ) warn( 'float cube textures need EXT_color_buffer_float; drawn with 8 bits per channel' );
		const faces = gl.createTexture();
		gl.bindTexture( gl.TEXTURE_CUBE_MAP, faces );
		gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, false );

		images.forEach( ( image, face ) => {

			const target = gl.TEXTURE_CUBE_MAP_POSITIVE_X + face;
			if ( image.data ) gl.texImage2D( target, 0, internal, size, size, 0, gl.RGBA, type, image.data );
			else gl.texImage2D( target, 0, internal, gl.RGBA, gl.UNSIGNED_BYTE, image );

		} );

		const filter = type === gl.FLOAT && ! floatLinear ? gl.NEAREST : gl.LINEAR;
		gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, filter );
		gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, filter );
		gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
		gl.texParameteri( gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );

		const width = Math.min( 4 * size, gl.getParameter( gl.MAX_TEXTURE_SIZE ) ), height = width / 2;
		if ( entry.texture ) gl.deleteTexture( entry.texture );
		entry.texture = newTexture();
		const hdr = float && floatTargets;
		gl.texImage2D( gl.TEXTURE_2D, 0, hdr ? gl.RGBA16F : srgb ? gl.SRGB8_ALPHA8 : gl.RGBA8, width, height, 0, gl.RGBA, hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null );
		parameters( null, 1 );

		const shader = program( 'cube:equirect', 'fullscreen.vert', 'cube.equirect' );
		gl.bindFramebuffer( gl.FRAMEBUFFER, ctx.environmentFramebuffer );
		gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, entry.texture, 0 );
		gl.drawBuffers( [ gl.COLOR_ATTACHMENT0 ] );
		gl.viewport( 0, 0, width, height );
		gl.disable( gl.DEPTH_TEST );
		gl.disable( gl.BLEND );
		gl.disable( gl.CULL_FACE );
		gl.useProgram( shader.id );
		gl.uniform1i( shader.uniform( 'faces' ), 0 );
		gl.activeTexture( gl.TEXTURE0 );
		gl.bindTexture( gl.TEXTURE_CUBE_MAP, faces );
		unbindArrays();
		gl.drawArrays( gl.TRIANGLES, 0, 3 );
		gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0 );
		gl.bindFramebuffer( gl.FRAMEBUFFER, null );
		gl.bindTexture( gl.TEXTURE_CUBE_MAP, null );
		gl.deleteTexture( faces );
		gl.enable( gl.DEPTH_TEST );

		entry.version = source.version;
		return entry.texture;

	}

	// ---------------------------------------------------------------- render targets

	// GL side of render targets by their texture id; `renderTargetTextures` gives the
	// textures (colour attachments and the depth texture) materials can read.
	const renderTargets = new Map(), renderTargetTextures = new Map();

	function renderTarget( target ) {

		let entry = renderTargets.get( target.id );
		const float = target.type !== 'byte' && floatTargets;
		if ( target.type !== 'byte' && ! floatTargets ) warn( 'float render targets need EXT_color_buffer_float; drawn with 8 bits per channel' );
		const count = target.textures?.length ?? 1, depthTexture = target.depthTexture ?? null;
		const multisampled = target.samples > 1 && samples > 0 && ! depthTexture;
		if ( entry && entry.width === target.width && entry.height === target.height && entry.float === float && entry.count === count && entry.multisampled === multisampled && entry.depthId === ( depthTexture?.id ?? 0 ) ) return entry;
		if ( entry ) entry.release();

		const { width, height } = target, colors = [], buffers = [];
		const framebuffer = gl.createFramebuffer();
		gl.bindFramebuffer( gl.FRAMEBUFFER, framebuffer );

		( target.textures ?? [ target.texture ] ).forEach( ( texture_, i ) => {

			const color = newTexture();
			gl.texImage2D( gl.TEXTURE_2D, 0, float ? gl.RGBA16F : gl.RGBA8, width, height, 0, gl.RGBA, float ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null );
			parameters( texture_, 1 );
			gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, color, 0 );
			colors.push( color );
			renderTargetTextures.set( texture_.id, color );

		} );

		let depth = null;

		if ( depthTexture ) {

			depth = newTexture();
			gl.texImage2D( gl.TEXTURE_2D, 0, gl.DEPTH24_STENCIL8, width, height, 0, gl.DEPTH_STENCIL, gl.UNSIGNED_INT_24_8, null );
			gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST );
			gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST );
			gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.TEXTURE_2D, depth, 0 );
			renderTargetTextures.set( depthTexture.id, depth );

		} else {

			const buffer = gl.createRenderbuffer();
			gl.bindRenderbuffer( gl.RENDERBUFFER, buffer );
			gl.renderbufferStorage( gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, width, height );
			gl.framebufferRenderbuffer( gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, buffer );
			buffers.push( buffer );

		}

		// Multisampled targets draw into renderbuffers and are resolved into the textures.
		let draw = framebuffer;

		if ( multisampled ) {

			draw = gl.createFramebuffer();
			gl.bindFramebuffer( gl.FRAMEBUFFER, draw );
			const storage = ( format, attachment ) => {

				const buffer = gl.createRenderbuffer();
				gl.bindRenderbuffer( gl.RENDERBUFFER, buffer );
				gl.renderbufferStorageMultisample( gl.RENDERBUFFER, samples, format, width, height );
				gl.framebufferRenderbuffer( gl.FRAMEBUFFER, attachment, gl.RENDERBUFFER, buffer );
				buffers.push( buffer );

			};

			storage( float ? gl.RGBA16F : gl.RGBA8, gl.COLOR_ATTACHMENT0 );
			storage( gl.DEPTH24_STENCIL8, gl.DEPTH_STENCIL_ATTACHMENT );

		}

		gl.bindFramebuffer( gl.FRAMEBUFFER, null );
		entry = {
			width, height, float, count, multisampled, depthId: depthTexture?.id ?? 0, colors, framebuffer, draw, release() {

				for ( const color of colors ) gl.deleteTexture( color );
				if ( depth ) gl.deleteTexture( depth );
				for ( const buffer of buffers ) gl.deleteRenderbuffer( buffer );
				gl.deleteFramebuffer( framebuffer );
				if ( draw !== framebuffer ) gl.deleteFramebuffer( draw );
				for ( const texture_ of target.textures ?? [ target.texture ] ) renderTargetTextures.delete( texture_.id );
				if ( depthTexture ) renderTargetTextures.delete( depthTexture.id );

			}
		};
		renderTargets.set( target.id, entry );
		return entry;

	}

	// The linear target of a tone-mapped frame (multisampled when antialiased, then
	// resolved), and the output pass that tone-maps it onto the canvas.
	ctx.hdr = null;

	function hdrTarget( width, height ) {

		if ( ctx.hdr && ctx.hdr.width === width && ctx.hdr.height === height ) return ctx.hdr;
		ctx.hdr?.release();
		const color = newTexture();
		gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null );
		parameters( null, 1 );
		const resolve = gl.createFramebuffer();
		gl.bindFramebuffer( gl.FRAMEBUFFER, resolve );
		gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0 );
		const buffers = [];
		const storage = ( format, count ) => {

			const buffer = gl.createRenderbuffer();
			gl.bindRenderbuffer( gl.RENDERBUFFER, buffer );
			if ( count > 0 ) gl.renderbufferStorageMultisample( gl.RENDERBUFFER, count, format, width, height );
			else gl.renderbufferStorage( gl.RENDERBUFFER, format, width, height );
			buffers.push( buffer );
			return buffer;

		};

		let draw = resolve;

		if ( samples > 0 ) {

			draw = gl.createFramebuffer();
			gl.bindFramebuffer( gl.FRAMEBUFFER, draw );
			gl.framebufferRenderbuffer( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, storage( gl.RGBA16F, samples ) );

		}

		gl.framebufferRenderbuffer( gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, storage( gl.DEPTH24_STENCIL8, samples ) );
		gl.bindFramebuffer( gl.FRAMEBUFFER, null );
		ctx.hdr = {
			width, height, color, draw, resolve, release() {

				gl.deleteTexture( color );
				for ( const buffer of buffers ) gl.deleteRenderbuffer( buffer );
				gl.deleteFramebuffer( resolve );
				if ( draw !== resolve ) gl.deleteFramebuffer( draw );

			}
		};
		return ctx.hdr;

	}

	// The mipmapped copy of the opaque scene that transmissive materials read, drawn like a
	// render target (rows top first).
	ctx.transmission = null;

	function transmissionTarget( width, height ) {

		if ( ctx.transmission && ctx.transmission.width === width && ctx.transmission.height === height ) return ctx.transmission;
		ctx.transmission?.release();
		const color = newTexture();
		const levels = Math.floor( Math.log2( Math.max( width, height ) ) ) + 1;
		gl.texStorage2D( gl.TEXTURE_2D, levels, floatTargets ? gl.RGBA16F : gl.RGBA8, width, height );
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR );
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
		const depth = gl.createRenderbuffer();
		gl.bindRenderbuffer( gl.RENDERBUFFER, depth );
		gl.renderbufferStorage( gl.RENDERBUFFER, gl.DEPTH24_STENCIL8, width, height );
		const framebuffer = gl.createFramebuffer();
		gl.bindFramebuffer( gl.FRAMEBUFFER, framebuffer );
		gl.framebufferTexture2D( gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0 );
		gl.framebufferRenderbuffer( gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.RENDERBUFFER, depth );
		gl.bindFramebuffer( gl.FRAMEBUFFER, null );
		ctx.transmission = { width, height, color, framebuffer, release() {

			gl.deleteTexture( color );
			gl.deleteRenderbuffer( depth );
			gl.deleteFramebuffer( framebuffer );

		} };
		return ctx.transmission;

	}

	Object.assign( ctx, { hdrTarget, newTexture, parameters, renderTarget, renderTargets, texture, textures, transmissionTarget, white } );

}
