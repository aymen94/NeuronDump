// Image, image-bitmap, texture, cube-texture and data-texture loaders.
//
// Part of the loaders; `../loaders.js` gathers them.

import { CubeTexture, DataTexture, Texture } from '../textures.js';
import { Cache, Loader } from './manager.js';
import { FileLoader } from './files.js';

/** three.js `ImageLoader`: an `HTMLImageElement` for a URL. */
export class ImageLoader extends Loader {

	load( url, onLoad, onProgress, onError ) {

		if ( this.path !== undefined ) url = this.path + url;
		url = this.manager.resolveURL( url );
		const cached = Cache.get( `image:${url}` );

		if ( cached !== undefined ) {

			this.manager.itemStart( url );
			setTimeout( () => {

				if ( onLoad ) onLoad( cached );
				this.manager.itemEnd( url );

			}, 0 );
			return cached;

		}

		const image = document.createElementNS( 'http://www.w3.org/1999/xhtml', 'img' );
		const loaded = () => {

			Cache.add( `image:${url}`, image );
			if ( onLoad ) onLoad( image );
			this.manager.itemEnd( url );

		};

		const failed = ( event ) => {

			if ( onError ) onError( event );
			this.manager.itemError( url );
			this.manager.itemEnd( url );

		};

		image.addEventListener( 'load', loaded, { once: true } );
		image.addEventListener( 'error', failed, { once: true } );
		if ( url.slice( 0, 5 ) !== 'data:' && this.crossOrigin !== undefined ) image.crossOrigin = this.crossOrigin;
		this.manager.itemStart( url );
		image.src = url;
		return image;

	}

}

/** three.js `ImageBitmapLoader`: an `ImageBitmap` for a URL (`setOptions` passes `createImageBitmap` options). */
export class ImageBitmapLoader extends Loader {

	constructor( manager ) {

		super( manager );
		this.isImageBitmapLoader = true;
		this.options = { premultiplyAlpha: 'none' };

	}

	setOptions( options ) {

		this.options = options;
		return this;

	}

	load( url, onLoad, onProgress, onError ) {

		if ( url === undefined ) url = '';
		if ( this.path !== undefined ) url = this.path + url;
		url = this.manager.resolveURL( url );
		const cached = Cache.get( `image-bitmap:${url}` );

		if ( cached !== undefined ) {

			this.manager.itemStart( url );
			Promise.resolve( cached ).then( ( bitmap ) => {

				if ( onLoad ) onLoad( bitmap );
				this.manager.itemEnd( url );

			} );
			return;

		}

		this.manager.itemStart( url );
		const promise = fetch( url, { credentials: this.crossOrigin === 'anonymous' ? 'same-origin' : 'include', headers: this.requestHeader } )
			.then( ( response ) => response.blob() )
			.then( ( blob ) => createImageBitmap( blob, Object.assign( this.options, { colorSpaceConversion: 'none' } ) ) )
			.then( ( bitmap ) => {

				Cache.add( `image-bitmap:${url}`, bitmap );
				if ( onLoad ) onLoad( bitmap );
				this.manager.itemEnd( url );
				return bitmap;

			} ).catch( ( error ) => {

				if ( onError ) onError( error );
				Cache.remove( `image-bitmap:${url}` );
				this.manager.itemError( url );
				this.manager.itemEnd( url );

			} );
		Cache.add( `image-bitmap:${url}`, promise );

	}

}

/** Loads an image into a `Texture` (three.js `TextureLoader`). */
export class TextureLoader extends Loader {

	load( url, onLoad, onProgress, onError ) {

		const texture = new Texture();
		const loader = new ImageLoader( this.manager );
		loader.setCrossOrigin( this.crossOrigin );
		loader.setPath( this.path );
		loader.load( url, ( image ) => {

			texture.image = image;
			texture.needsUpdate = true;
			if ( onLoad !== undefined ) onLoad( texture );

		}, onProgress, onError );
		return texture;

	}

}

/** Six images into a `CubeTexture` (+x, -x, +y, -y, +z, -z), as three.js `CubeTextureLoader`. */
export class CubeTextureLoader extends Loader {

	load( urls, onLoad, onProgress, onError ) {

		const texture = new CubeTexture();
		texture.colorSpace = 'srgb';
		const loader = new ImageLoader( this.manager );
		loader.setCrossOrigin( this.crossOrigin );
		loader.setPath( this.path );
		let loaded = 0;
		const images = new Array( 6 ).fill( null );

		urls.forEach( ( url, i ) => loader.load( url, ( image ) => {

			images[ i ] = image;
			loaded ++;

			if ( loaded === 6 ) {

				texture.image = images;
				texture.needsUpdate = true;
				if ( onLoad ) onLoad( texture );

			}

		}, undefined, onError ) );

		return texture;

	}

}

/**
 * three.js `DataTextureLoader`: subclasses implement `parse( buffer )` returning `{ data,
 * width, height, type, ... }` (as the RGBE and EXR loaders do), loaded into a DataTexture.
 */
export class DataTextureLoader extends Loader {

	load( url, onLoad, onProgress, onError ) {

		const texture = new DataTexture();
		const loader = new FileLoader( this.manager );
		loader.setResponseType( 'arraybuffer' );
		loader.setRequestHeader( this.requestHeader );
		loader.setPath( this.path );
		loader.setWithCredentials( this.withCredentials );

		loader.load( url, ( buffer ) => {

			let data;

			try {

				data = this.parse( buffer );

			} catch ( error ) {

				if ( onError !== undefined ) onError( error );
				else console.error( error );
				return;

			}

			if ( data.image !== undefined ) texture.image = data.image;
			else if ( data.data !== undefined ) texture.image = { width: data.width, height: data.height, data: data.data };
			for ( const key of [ 'wrapS', 'wrapT', 'magFilter', 'minFilter', 'anisotropy', 'colorSpace', 'flipY', 'format', 'type', 'mapping', 'generateMipmaps' ] ) if ( data[ key ] !== undefined ) texture[ key ] = data[ key ];
			if ( data.minFilter === undefined ) texture.minFilter = 1006;
			if ( data.magFilter === undefined ) texture.magFilter = 1006;
			texture.needsUpdate = true;
			if ( onLoad ) onLoad( texture, data );

		}, onProgress, onError );

		return texture;

	}

}
