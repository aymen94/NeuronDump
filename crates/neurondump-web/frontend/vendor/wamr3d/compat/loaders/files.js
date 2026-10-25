// FileLoader and LoaderUtils: fetching files, resolving URLs.
//
// Part of the loaders; `../loaders.js` gathers them.

import { Cache, Loader } from './manager.js';

const loading = {};

/** three.js `FileLoader`: text, JSON, ArrayBuffer, Blob or documents by `responseType`. */
export class FileLoader extends Loader {

	constructor( manager ) {

		super( manager );
		this.mimeType = '';
		this.responseType = '';
		this._abortController = new AbortController();

	}

	load( url, onLoad, onProgress, onError ) {

		if ( url === undefined ) url = '';
		if ( this.path !== undefined ) url = this.path + url;
		url = this.manager.resolveURL( url );
		const cached = Cache.get( `file:${url}` );

		if ( cached !== undefined ) {

			this.manager.itemStart( url );
			setTimeout( () => {

				if ( onLoad ) onLoad( cached );
				this.manager.itemEnd( url );

			}, 0 );
			return cached;

		}

		if ( loading[ url ] !== undefined ) {

			loading[ url ].push( { onLoad, onProgress, onError } );
			return;

		}

		loading[ url ] = [ { onLoad, onProgress, onError } ];
		const request = new Request( url, { headers: new Headers( this.requestHeader ), credentials: this.withCredentials ? 'include' : 'same-origin', signal: this.manager.abortController.signal } );
		const mimeType = this.mimeType, responseType = this.responseType;
		this.manager.itemStart( url );

		fetch( request ).then( ( response ) => {

			if ( response.status !== 200 && response.status !== 0 ) throw new HttpError( `fetch for "${response.url}" responded with ${response.status}: ${response.statusText}`, response );
			const callbacks = loading[ url ], total = Number( response.headers.get( 'Content-Length' ) ) || 0;

			// progress as the body arrives, when anyone listens and the size is known
			if ( typeof ReadableStream === 'undefined' || response.body === null || response.body.getReader === undefined || ! callbacks.some( ( c ) => c.onProgress ) ) return response;
			const reader = response.body.getReader();
			let loaded = 0;
			const stream = new ReadableStream( {
				start( controller ) {

					const read = () => reader.read().then( ( { done, value } ) => {

						if ( done ) {

							controller.close();
							return;

						}

						loaded += value.byteLength;
						const event = new ProgressEvent( 'progress', { lengthComputable: total > 0, loaded, total } );
						for ( const callback of callbacks ) if ( callback.onProgress ) callback.onProgress( event );
						controller.enqueue( value );
						read();

					}, ( e ) => controller.error( e ) );
					read();

				}
			} );
			return new Response( stream );

		} ).then( ( response ) => {

			switch ( responseType ) {

				case 'arraybuffer': return response.arrayBuffer();
				case 'blob': return response.blob();
				case 'document': return response.text().then( ( text ) => new DOMParser().parseFromString( text, mimeType ) );
				case 'json': return response.json();
				default: return response.text();

			}

		} ).then( ( data ) => {

			Cache.add( `file:${url}`, data );
			const callbacks = loading[ url ];
			delete loading[ url ];
			for ( const callback of callbacks ) if ( callback.onLoad ) callback.onLoad( data );

		} ).catch( ( error ) => {

			const callbacks = loading[ url ];

			if ( callbacks === undefined ) {

				this.manager.itemError( url );
				throw error;

			}

			delete loading[ url ];
			for ( const callback of callbacks ) if ( callback.onError ) callback.onError( error );
			this.manager.itemError( url );

		} ).finally( () => this.manager.itemEnd( url ) );

	}

	setResponseType( value ) {

		this.responseType = value;
		return this;

	}

	setMimeType( value ) {

		this.mimeType = value;
		return this;

	}

}

class HttpError extends Error {

	constructor( message, response ) {

		super( message );
		this.response = response;

	}

}

/** three.js `LoaderUtils`. */
export const LoaderUtils = {

	extractUrlBase( url ) {

		const index = url.lastIndexOf( '/' );
		return index === - 1 ? './' : url.slice( 0, index + 1 );

	},

	resolveURL( url, path ) {

		if ( typeof url !== 'string' || url === '' ) return '';
		if ( /^https?:\/\//i.test( path ) && /^\//.test( url ) ) path = path.replace( /(^https?:\/\/[^/]+).*/i, '$1' );
		if ( /^(https?:)?\/\//i.test( url ) ) return url;
		if ( /^data:.*,.*$/i.test( url ) ) return url;
		if ( /^blob:.*$/i.test( url ) ) return url;
		return path + url;

	}

};
