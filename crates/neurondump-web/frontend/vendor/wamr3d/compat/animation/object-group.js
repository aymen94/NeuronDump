// AnimationObjectGroup: objects animated by the same actions.
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import * as W from '../../index.js';
import { PropertyBinding } from './binding.js';

/**
 * Objects animated together by one action. Removed objects stay cached at the front of
 * the arrays (below `nCachedObjects_`) until `uncache`.
 */
export class AnimationObjectGroup {

	constructor( ...objects ) {

		this.uuid = W.MathUtils.generateUUID();
		this._objects = objects;
		this.nCachedObjects_ = 0;
		this._indicesByUUID = {};
		objects.forEach( ( object, i ) => {

			this._indicesByUUID[ object.uuid ] = i;

		} );
		this._paths = [];
		this._parsedPaths = [];
		this._bindings = [];
		this._bindingsIndicesByPath = {};
		const scope = this;
		this.stats = {
			objects: { get total() { return scope._objects.length; }, get inUse() { return this.total - scope.nCachedObjects_; } },
			get bindingsPerObject() { return scope._bindings.length; }
		};

	}

	add( ...added ) {

		const objects = this._objects, indicesByUUID = this._indicesByUUID, paths = this._paths, parsedPaths = this._parsedPaths, bindings = this._bindings;
		let knownObject, nObjects = objects.length, nCachedObjects = this.nCachedObjects_;

		for ( const object of added ) {

			const uuid = object.uuid;
			let index = indicesByUUID[ uuid ];

			if ( index === undefined ) {

				index = nObjects ++;
				indicesByUUID[ uuid ] = index;
				objects.push( object );
				for ( let j = 0; j !== bindings.length; ++ j ) bindings[ j ].push( new PropertyBinding( object, paths[ j ], parsedPaths[ j ] ) );

			} else if ( index < nCachedObjects ) {

				knownObject = objects[ index ];
				// moved from the cached part to the active part
				const firstActiveIndex = -- nCachedObjects, lastCachedObject = objects[ firstActiveIndex ];
				indicesByUUID[ lastCachedObject.uuid ] = index;
				objects[ index ] = lastCachedObject;
				indicesByUUID[ uuid ] = firstActiveIndex;
				objects[ firstActiveIndex ] = object;

				for ( let j = 0; j !== bindings.length; ++ j ) {

					const forPath = bindings[ j ], lastCached = forPath[ firstActiveIndex ];
					let binding = forPath[ index ];
					forPath[ index ] = lastCached;
					if ( binding === undefined ) binding = new PropertyBinding( object, paths[ j ], parsedPaths[ j ] );
					forPath[ firstActiveIndex ] = binding;

				}

			} else if ( objects[ index ] !== knownObject ) {

				console.error( 'WAMR3D: AnimationObjectGroup: different objects with the same UUID detected. Clean the caches or recreate your infrastructure when reloading scenes.' );

			}

		}

		this.nCachedObjects_ = nCachedObjects;

	}

	remove( ...removed ) {

		const objects = this._objects, indicesByUUID = this._indicesByUUID, bindings = this._bindings;
		let nCachedObjects = this.nCachedObjects_;

		for ( const object of removed ) {

			const uuid = object.uuid, index = indicesByUUID[ uuid ];
			if ( index === undefined || index < nCachedObjects ) continue;
			const lastCachedIndex = nCachedObjects ++, firstActiveObject = objects[ lastCachedIndex ];
			indicesByUUID[ firstActiveObject.uuid ] = index;
			objects[ index ] = firstActiveObject;
			indicesByUUID[ uuid ] = lastCachedIndex;
			objects[ lastCachedIndex ] = object;

			for ( let j = 0; j !== bindings.length; ++ j ) {

				const forPath = bindings[ j ], firstActive = forPath[ lastCachedIndex ], binding = forPath[ index ];
				forPath[ index ] = firstActive;
				forPath[ lastCachedIndex ] = binding;

			}

		}

		this.nCachedObjects_ = nCachedObjects;

	}

	uncache( ...uncached ) {

		const objects = this._objects, indicesByUUID = this._indicesByUUID, bindings = this._bindings;
		let nCachedObjects = this.nCachedObjects_, nObjects = objects.length;

		for ( const object of uncached ) {

			const uuid = object.uuid, index = indicesByUUID[ uuid ];
			if ( index === undefined ) continue;
			delete indicesByUUID[ uuid ];

			if ( index < nCachedObjects ) {

				// a cached object: fill its place with the last cached one, and that place with the last object
				const firstActiveIndex = -- nCachedObjects, lastCachedObject = objects[ firstActiveIndex ], lastIndex = -- nObjects, lastObject = objects[ lastIndex ];
				if ( index !== firstActiveIndex ) indicesByUUID[ lastCachedObject.uuid ] = index;
				objects[ index ] = lastCachedObject;
				if ( firstActiveIndex !== lastIndex ) indicesByUUID[ lastObject.uuid ] = firstActiveIndex;
				objects[ firstActiveIndex ] = lastObject;
				objects.pop();

				for ( let j = 0; j !== bindings.length; ++ j ) {

					const forPath = bindings[ j ], lastCached = forPath[ firstActiveIndex ], last = forPath[ lastIndex ];
					forPath[ index ] = lastCached;
					forPath[ firstActiveIndex ] = last;
					forPath.pop();

				}

			} else {

				const lastIndex = -- nObjects, lastObject = objects[ lastIndex ];
				if ( index !== lastIndex ) indicesByUUID[ lastObject.uuid ] = index;
				objects[ index ] = lastObject;
				objects.pop();

				for ( let j = 0; j !== bindings.length; ++ j ) {

					const forPath = bindings[ j ];
					forPath[ index ] = forPath[ lastIndex ];
					forPath.pop();

				}

			}

		}

		this.nCachedObjects_ = nCachedObjects;

	}

	subscribe_( path, parsedPath ) {

		let index = this._bindingsIndicesByPath[ path ];
		if ( index !== undefined ) return this._bindings[ index ];
		const objects = this._objects, forPath = new Array( objects.length );
		index = this._bindings.length;
		this._bindingsIndicesByPath[ path ] = index;
		this._paths.push( path );
		this._parsedPaths.push( parsedPath );
		this._bindings.push( forPath );
		for ( let i = this.nCachedObjects_; i !== objects.length; ++ i ) forPath[ i ] = new PropertyBinding( objects[ i ], path, parsedPath );
		return forPath;

	}

	unsubscribe_( path ) {

		const index = this._bindingsIndicesByPath[ path ];
		if ( index === undefined ) return;
		const paths = this._paths, parsedPaths = this._parsedPaths, bindings = this._bindings, last = bindings.length - 1;
		this._bindingsIndicesByPath[ paths[ last ] ] = index;
		bindings[ index ] = bindings[ last ];
		bindings.pop();
		parsedPaths[ index ] = parsedPaths[ last ];
		parsedPaths.pop();
		paths[ index ] = paths[ last ];
		paths.pop();

	}

}

AnimationObjectGroup.prototype.isAnimationObjectGroup = true;
