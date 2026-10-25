// Cache, LoadingManager and the Loader base class.
//
// Part of the loaders; `../loaders.js` gathers them.

/** three.js `Cache`: loaded files by URL, used by FileLoader and ImageLoader when enabled. */
export const Cache = {

	enabled: false,
	files: {},

	add( key, file ) {

		if ( this.enabled ) this.files[ key ] = file;

	},

	get( key ) {

		return this.enabled ? this.files[ key ] : undefined;

	},

	remove( key ) {

		delete this.files[ key ];

	},

	clear() {

		this.files = {};

	}

};

/** three.js `LoadingManager`: counts loads in progress and reports start, progress, end and errors. */
export class LoadingManager {

	constructor( onLoad, onProgress, onError ) {

		let isLoading = false, itemsLoaded = 0, itemsTotal = 0, urlModifier;
		const handlers = [], scope = this;
		this.onStart = undefined;
		this.onLoad = onLoad;
		this.onProgress = onProgress;
		this.onError = onError;
		this.abortController = new AbortController();

		this.itemStart = function ( url ) {

			itemsTotal ++;
			if ( isLoading === false && scope.onStart !== undefined ) scope.onStart( url, itemsLoaded, itemsTotal );
			isLoading = true;

		};

		this.itemEnd = function ( url ) {

			itemsLoaded ++;
			if ( scope.onProgress !== undefined ) scope.onProgress( url, itemsLoaded, itemsTotal );

			if ( itemsLoaded === itemsTotal ) {

				isLoading = false;
				if ( scope.onLoad !== undefined ) scope.onLoad();

			}

		};

		this.itemError = function ( url ) {

			if ( scope.onError !== undefined ) scope.onError( url );

		};

		this.resolveURL = function ( url ) {

			return urlModifier ? urlModifier( url ) : url;

		};

		this.setURLModifier = function ( transform ) {

			urlModifier = transform;
			return this;

		};

		this.addHandler = function ( regex, loader ) {

			handlers.push( regex, loader );
			return this;

		};

		this.removeHandler = function ( regex ) {

			const index = handlers.indexOf( regex );
			if ( index !== - 1 ) handlers.splice( index, 2 );
			return this;

		};

		this.getHandler = function ( file ) {

			for ( let i = 0; i < handlers.length; i += 2 ) {

				const regex = handlers[ i ];
				if ( regex.global ) regex.lastIndex = 0;
				if ( regex.test( file ) ) return handlers[ i + 1 ];

			}

			return null;

		};

		this.abort = function () {

			this.abortController.abort();
			this.abortController = new AbortController();
			return this;

		};

	}

}

export const DefaultLoadingManager = new LoadingManager();

/** three.js `Loader`: path, resource path, credentials and request headers shared by loaders. */
export class Loader {

	constructor( manager ) {

		this.manager = manager !== undefined ? manager : DefaultLoadingManager;
		this.crossOrigin = 'anonymous';
		this.withCredentials = false;
		this.path = '';
		this.resourcePath = '';
		this.requestHeader = {};

	}

	load() {}

	loadAsync( url, onProgress ) {

		return new Promise( ( resolve, reject ) => this.load( url, resolve, onProgress, reject ) );

	}

	parse() {}

	setCrossOrigin( crossOrigin ) {

		this.crossOrigin = crossOrigin;
		return this;

	}

	setWithCredentials( value ) {

		this.withCredentials = value;
		return this;

	}

	setPath( path ) {

		this.path = path;
		return this;

	}

	setResourcePath( resourcePath ) {

		this.resourcePath = resourcePath;
		return this;

	}

	setRequestHeader( requestHeader ) {

		this.requestHeader = requestHeader;
		return this;

	}

	abort() {

		return this;

	}

}

Loader.DEFAULT_MATERIAL_NAME = '__DEFAULT';
