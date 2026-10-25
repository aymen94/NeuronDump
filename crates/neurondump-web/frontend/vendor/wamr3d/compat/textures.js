// Textures, render targets and loading images.

import { ClampToEdgeWrapping, CubeReflectionMapping, DepthFormat, LinearFilter, LinearMipmapLinearFilter, NearestFilter, NoColorSpace, RGBAFormat, SRGBColorSpace, UVMapping, UnsignedByteType, UnsignedIntType } from './constants.js';
import { ignored, lazyUUID } from './engine.js';
import { Vector2, Vector4 } from './math.js';
import { withEvents } from './events.js';

// Images live in the browser, so the engine only carries a texture's id and the
// renderer's host looks the image up here. A texture is kept until dispose().
export const textureSources = new Map();
let nextTextureId = 1;

export class Texture {

	constructor( image = null ) {

		this.id = nextTextureId ++;
		this.name = '';
		this.image = image;
		this.version = 0;
		this.flipY = true;
		this.wrapS = this.wrapT = ClampToEdgeWrapping;
		this.magFilter = LinearFilter;
		this.minFilter = LinearMipmapLinearFilter;
		this.generateMipmaps = true;
		this.colorSpace = NoColorSpace;
		this.offset = new Vector2( 0, 0 );
		this.repeat = new Vector2( 1, 1 );
		this.center = new Vector2( 0, 0 );
		this.rotation = 0;
		this.mapping = UVMapping;
		// the uv set the texture is read with (0 `uv`, 1 `uv1`), and anisotropic filtering
		this.channel = 0;
		this.anisotropy = 1;
		this.format = RGBAFormat;
		this.type = UnsignedByteType;
		this.userData = {};
		textureSources.set( this.id, this );

	}

	/** Set to true after changing `image` or its pixels. */
	set needsUpdate( value ) {

		if ( value ) this.version ++;

	}

	dispose() {

		textureSources.delete( this.id );
		this.dispatchEvent( { type: 'dispose' } );

	}

}

Texture.prototype.isTexture = true;
withEvents( Texture );
lazyUUID( Texture );

// Backends that may hold the GPU side of render targets.
export const backends = new Set();

/**
 * Something to draw into instead of the canvas, with `texture` usable as a material's
 * map, an environment or a background. As in three.js, what is drawn into it is linear
 * colour without tone mapping. `type` UnsignedByteType (default) stores 8 bits per
 * channel, HalfFloatType or FloatType 16-bit floats. `samples` above 1 multisamples
 * when the renderer is antialiased (not together with a depth texture). `depthTexture`
 * keeps the depth readable as a texture. With `count` above 1 there are as many
 * textures (`textures`); the built-in materials write the first. No stencil.
 */
export class RenderTarget {

	constructor( width = 1, height = 1, options = {} ) {

		this.width = width;
		this.height = height;
		this.depth = 1;
		this.samples = options.samples ?? 0;
		this.depthBuffer = options.depthBuffer ?? true;
		this.stencilBuffer = options.stencilBuffer ?? false;
		this.scissor = new Vector4( 0, 0, width, height );
		this.scissorTest = false;
		this.viewport = new Vector4( 0, 0, width, height );
		this.textures = [];

		for ( let i = 0; i < ( options.count ?? 1 ); i ++ ) {

			const texture = new Texture( { width, height, depth: 1 } );
			texture.isRenderTargetTexture = true;
			texture.renderTarget = this;
			texture.flipY = false;
			texture.generateMipmaps = options.generateMipmaps ?? false;
			texture.minFilter = options.minFilter ?? LinearFilter;
			texture.magFilter = options.magFilter ?? LinearFilter;
			texture.wrapS = options.wrapS ?? ClampToEdgeWrapping;
			texture.wrapT = options.wrapT ?? ClampToEdgeWrapping;
			texture.type = options.type ?? UnsignedByteType;
			texture.colorSpace = options.colorSpace ?? NoColorSpace;
			// drawn into once the version is above 0
			texture.version = 0;
			this.textures.push( texture );

		}

		this.depthTexture = options.depthTexture ?? null;
		if ( this.depthTexture ) this.depthTexture.renderTarget = this;

	}

	get texture() {

		return this.textures[ 0 ];

	}

	set texture( value ) {

		this.textures[ 0 ] = value;

	}

	get count() {

		return this.textures.length;

	}

	setSize( width, height ) {

		this.width = width;
		this.height = height;
		for ( const texture of this.textures ) texture.image = { width, height, depth: 1 };
		if ( this.depthTexture ) this.depthTexture.image = { width, height, depth: 1 };
		this.viewport.set( 0, 0, width, height );
		this.scissor.set( 0, 0, width, height );

	}

	dispose() {

		for ( const backend of backends ) backend.disposeRenderTarget( this.texture.id );

	}

}

RenderTarget.prototype.isRenderTarget = true;

/** The same as `RenderTarget`; the name three.js code written for WebGL uses. */
export class WebGLRenderTarget extends RenderTarget {}

WebGLRenderTarget.prototype.isWebGLRenderTarget = true;

/**
 * Pixels from an array, rows bottom to top: RGBA of one byte per channel, or with `type`
 * FloatType (a Float32Array) or HalfFloatType (a Uint16Array of half floats) for HDR.
 */
export class DataTexture extends Texture {

	constructor( data = null, width = 1, height = 1, format = RGBAFormat, type = UnsignedByteType ) {

		super( { data, width, height } );
		if ( format !== RGBAFormat ) ignored( 'DataTexture formats other than RGBAFormat' );
		this.format = format;
		this.type = type;
		this.flipY = false;
		this.generateMipmaps = false;
		this.magFilter = this.minFilter = NearestFilter;

	}

}

DataTexture.prototype.isDataTexture = true;

export class CanvasTexture extends Texture {

	constructor( canvas ) {

		super( canvas );
		this.needsUpdate = true;

	}

}

CanvasTexture.prototype.isCanvasTexture = true;

/** The depth of a render target as a texture (set as the target's `depthTexture`). */
export class DepthTexture extends Texture {

	constructor( width, height, type = UnsignedIntType ) {

		super( { width, height, depth: 1 } );
		this.isRenderTargetTexture = true;
		this.format = DepthFormat;
		this.type = type;
		this.flipY = false;
		this.generateMipmaps = false;
		this.magFilter = this.minFilter = NearestFilter;
		this.compareFunction = null;

	}

}

DepthTexture.prototype.isDepthTexture = true;

export class TextureLoader {

	load( url, onLoad, onProgress, onError ) {

		const texture = new Texture();

		fetch( url ).then( ( response ) => {

			if ( ! response.ok ) throw new Error( `WAMR3D: could not load ${url} (${response.status})` );
			return response.blob();

		} ).then( ( blob ) => createImageBitmap( blob ) ).then( ( image ) => {

			texture.image = image;
			texture.needsUpdate = true;
			if ( onLoad ) onLoad( texture );

		} ).catch( ( error ) => {

			if ( onError ) onError( error );
			else console.error( error );

		} );

		return texture;

	}

}

/**
 * Six square images in the order +x, -x, +y, -y, +z, -z, for `scene.environment`,
 * `envMap` and `scene.background`. The renderer turns them into an equirectangular image
 * (four faces wide) once per upload, so they draw like an equirectangular map.
 */
export class CubeTexture extends Texture {

	constructor( images = [], mapping = CubeReflectionMapping ) {

		super( images );
		this.mapping = mapping;
		this.flipY = false;

	}

	get images() {

		return this.image;

	}

	set images( value ) {

		this.image = value;

	}

}

CubeTexture.prototype.isCubeTexture = true;

/** Loads the six faces of a `CubeTexture` (sRGB), calling `onLoad` once all have arrived. */
export class CubeTextureLoader {

	constructor() {

		this.path = '';

	}

	setPath( path ) {

		this.path = path;
		return this;

	}

	load( urls, onLoad, onProgress, onError ) {

		const texture = new CubeTexture();
		texture.colorSpace = SRGBColorSpace;
		let loaded = 0;

		urls.forEach( ( url, face ) => {

			fetch( this.path + url ).then( ( response ) => {

				if ( ! response.ok ) throw new Error( `WAMR3D: could not load ${this.path + url} (${response.status})` );
				return response.blob();

			} ).then( ( blob ) => createImageBitmap( blob ) ).then( ( image ) => {

				texture.images[ face ] = image;
				if ( ++ loaded !== urls.length ) return;
				texture.needsUpdate = true;
				if ( onLoad ) onLoad( texture );

			} ).catch( ( error ) => {

				if ( onError ) onError( error );
				else console.error( error );

			} );

		} );

		return texture;

	}

}
