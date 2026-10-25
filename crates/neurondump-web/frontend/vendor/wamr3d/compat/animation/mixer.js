// AnimationMixer: the actions of a root, updated together.
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import { NormalAnimationBlendMode } from '../constants.js';
import { EventDispatcher } from '../events.js';
import { LinearInterpolant } from './interpolants.js';
import { AnimationClip } from './clip.js';
import { PropertyBinding } from './binding.js';
import { PropertyMixer } from './property-mixer.js';
import { AnimationAction } from './action.js';

const controlResultBuffer = new Float32Array( 1 );

/**
 * Plays clips on the objects below `root`. Call `update( delta )` once per frame. Actions
 * and bindings are kept in arrays with the active ones first, so an update walks only
 * those.
 */
export class AnimationMixer extends EventDispatcher {

	constructor( root ) {

		super();
		this._root = root;
		this._initMemoryManager();
		this._accuIndex = 0;
		this.time = 0;
		this.timeScale = 1.0;

	}

	_bindAction( action, prototypeAction ) {

		const root = action._localRoot || this._root, tracks = action._clip.tracks, bindings = action._propertyBindings, interpolants = action._interpolants;
		const rootUuid = root.uuid, bindingsByName = this._bindingsByRootAndName[ rootUuid ] ??= {};

		for ( let i = 0; i !== tracks.length; ++ i ) {

			const track = tracks[ i ], trackName = track.name;
			let binding = bindingsByName[ trackName ];

			if ( binding !== undefined ) {

				++ binding.referenceCount;
				bindings[ i ] = binding;

			} else {

				binding = bindings[ i ];

				if ( binding !== undefined ) {

					// an existing binding re-entering the cache
					if ( binding._cacheIndex === null ) {

						++ binding.referenceCount;
						this._addInactiveBinding( binding, rootUuid, trackName );

					}

					continue;

				}

				const path = prototypeAction && prototypeAction._propertyBindings[ i ].binding.parsedPath;
				binding = new PropertyMixer( PropertyBinding.create( root, trackName, path ), track.ValueTypeName, track.getValueSize() );
				++ binding.referenceCount;
				this._addInactiveBinding( binding, rootUuid, trackName );
				bindings[ i ] = binding;

			}

			interpolants[ i ].resultBuffer = binding.buffer;

		}

	}

	_activateAction( action ) {

		if ( this._isActiveAction( action ) ) return;

		if ( action._cacheIndex === null ) {

			const rootUuid = ( action._localRoot || this._root ).uuid, clipUuid = action._clip.uuid, actionsForClip = this._actionsByClip[ clipUuid ];
			this._bindAction( action, actionsForClip && actionsForClip.knownActions[ 0 ] );
			this._addInactiveAction( action, clipUuid, rootUuid );

		}

		for ( const binding of action._propertyBindings ) {

			if ( binding.useCount ++ === 0 ) {

				this._lendBinding( binding );
				binding.saveOriginalState();

			}

		}

		this._lendAction( action );

	}

	_deactivateAction( action ) {

		if ( ! this._isActiveAction( action ) ) return;

		for ( const binding of action._propertyBindings ) {

			if ( -- binding.useCount === 0 ) {

				binding.restoreOriginalState();
				this._takeBackBinding( binding );

			}

		}

		this._takeBackAction( action );

	}

	_initMemoryManager() {

		this._actions = [];
		this._nActiveActions = 0;
		this._actionsByClip = {};
		this._bindings = [];
		this._nActiveBindings = 0;
		this._bindingsByRootAndName = {};
		this._controlInterpolants = [];
		this._nActiveControlInterpolants = 0;
		const scope = this;
		this.stats = {
			actions: { get total() { return scope._actions.length; }, get inUse() { return scope._nActiveActions; } },
			bindings: { get total() { return scope._bindings.length; }, get inUse() { return scope._nActiveBindings; } },
			controlInterpolants: { get total() { return scope._controlInterpolants.length; }, get inUse() { return scope._nActiveControlInterpolants; } }
		};

	}

	_isActiveAction( action ) {

		const index = action._cacheIndex;
		return index !== null && index < this._nActiveActions;

	}

	_addInactiveAction( action, clipUuid, rootUuid ) {

		const actions = this._actions;
		let actionsForClip = this._actionsByClip[ clipUuid ];

		if ( actionsForClip === undefined ) {

			actionsForClip = { knownActions: [ action ], actionByRoot: {} };
			action._byClipCacheIndex = 0;
			this._actionsByClip[ clipUuid ] = actionsForClip;

		} else {

			action._byClipCacheIndex = actionsForClip.knownActions.length;
			actionsForClip.knownActions.push( action );

		}

		action._cacheIndex = actions.length;
		actions.push( action );
		actionsForClip.actionByRoot[ rootUuid ] = action;

	}

	_removeInactiveAction( action ) {

		const actions = this._actions, last = actions[ actions.length - 1 ], cacheIndex = action._cacheIndex;
		last._cacheIndex = cacheIndex;
		actions[ cacheIndex ] = last;
		actions.pop();
		action._cacheIndex = null;

		const clipUuid = action._clip.uuid, actionsForClip = this._actionsByClip[ clipUuid ], known = actionsForClip.knownActions;
		const lastKnown = known[ known.length - 1 ], byClipCacheIndex = action._byClipCacheIndex;
		lastKnown._byClipCacheIndex = byClipCacheIndex;
		known[ byClipCacheIndex ] = lastKnown;
		known.pop();
		action._byClipCacheIndex = null;
		delete actionsForClip.actionByRoot[ ( action._localRoot || this._root ).uuid ];
		if ( known.length === 0 ) delete this._actionsByClip[ clipUuid ];
		this._removeInactiveBindingsForAction( action );

	}

	_removeInactiveBindingsForAction( action ) {

		for ( const binding of action._propertyBindings ) if ( -- binding.referenceCount === 0 ) this._removeInactiveBinding( binding );

	}

	_lendAction( action ) {

		const actions = this._actions, prevIndex = action._cacheIndex, lastActiveIndex = this._nActiveActions ++, firstInactive = actions[ lastActiveIndex ];
		action._cacheIndex = lastActiveIndex;
		actions[ lastActiveIndex ] = action;
		firstInactive._cacheIndex = prevIndex;
		actions[ prevIndex ] = firstInactive;

	}

	_takeBackAction( action ) {

		const actions = this._actions, prevIndex = action._cacheIndex, firstInactiveIndex = -- this._nActiveActions, lastActive = actions[ firstInactiveIndex ];
		action._cacheIndex = firstInactiveIndex;
		actions[ firstInactiveIndex ] = action;
		lastActive._cacheIndex = prevIndex;
		actions[ prevIndex ] = lastActive;

	}

	_addInactiveBinding( binding, rootUuid, trackName ) {

		( this._bindingsByRootAndName[ rootUuid ] ??= {} )[ trackName ] = binding;
		binding._cacheIndex = this._bindings.length;
		this._bindings.push( binding );

	}

	_removeInactiveBinding( binding ) {

		const bindings = this._bindings, propertyBinding = binding.binding, rootUuid = propertyBinding.rootNode.uuid;
		const byName = this._bindingsByRootAndName[ rootUuid ], last = bindings[ bindings.length - 1 ], cacheIndex = binding._cacheIndex;
		last._cacheIndex = cacheIndex;
		bindings[ cacheIndex ] = last;
		bindings.pop();
		delete byName[ propertyBinding.path ];
		if ( Object.keys( byName ).length === 0 ) delete this._bindingsByRootAndName[ rootUuid ];

	}

	_lendBinding( binding ) {

		const bindings = this._bindings, prevIndex = binding._cacheIndex, lastActiveIndex = this._nActiveBindings ++, firstInactive = bindings[ lastActiveIndex ];
		binding._cacheIndex = lastActiveIndex;
		bindings[ lastActiveIndex ] = binding;
		firstInactive._cacheIndex = prevIndex;
		bindings[ prevIndex ] = firstInactive;

	}

	_takeBackBinding( binding ) {

		const bindings = this._bindings, prevIndex = binding._cacheIndex, firstInactiveIndex = -- this._nActiveBindings, lastActive = bindings[ firstInactiveIndex ];
		binding._cacheIndex = firstInactiveIndex;
		bindings[ firstInactiveIndex ] = binding;
		lastActive._cacheIndex = prevIndex;
		bindings[ prevIndex ] = lastActive;

	}

	// Two-key interpolants for fading and warping, reused.
	_lendControlInterpolant() {

		const interpolants = this._controlInterpolants, lastActiveIndex = this._nActiveControlInterpolants ++;
		let interpolant = interpolants[ lastActiveIndex ];

		if ( interpolant === undefined ) {

			interpolant = new LinearInterpolant( new Float32Array( 2 ), new Float32Array( 2 ), 1, controlResultBuffer );
			interpolant.__cacheIndex = lastActiveIndex;
			interpolants[ lastActiveIndex ] = interpolant;

		}

		return interpolant;

	}

	_takeBackControlInterpolant( interpolant ) {

		const interpolants = this._controlInterpolants, prevIndex = interpolant.__cacheIndex, firstInactiveIndex = -- this._nActiveControlInterpolants, lastActive = interpolants[ firstInactiveIndex ];
		interpolant.__cacheIndex = firstInactiveIndex;
		interpolants[ firstInactiveIndex ] = interpolant;
		lastActive.__cacheIndex = prevIndex;
		interpolants[ prevIndex ] = lastActive;

	}

	/** The action of `clip` (or the clip named so) on `optionalRoot` or the mixer's root, made once. */
	clipAction( clip, optionalRoot, blendMode ) {

		const root = optionalRoot || this._root, rootUuid = root.uuid;
		let clipObject = typeof clip === 'string' ? AnimationClip.findByName( root, clip ) : clip;
		const clipUuid = clipObject !== null ? clipObject.uuid : clip, actionsForClip = this._actionsByClip[ clipUuid ];
		let prototypeAction = null;
		if ( blendMode === undefined ) blendMode = clipObject !== null ? clipObject.blendMode : NormalAnimationBlendMode;

		if ( actionsForClip !== undefined ) {

			const existing = actionsForClip.actionByRoot[ rootUuid ];
			if ( existing !== undefined && existing.blendMode === blendMode ) return existing;
			prototypeAction = actionsForClip.knownActions[ 0 ];
			if ( clipObject === null ) clipObject = prototypeAction._clip;

		}

		if ( clipObject === null ) return null;
		const action = new AnimationAction( this, clipObject, optionalRoot, blendMode );
		this._bindAction( action, prototypeAction );
		this._addInactiveAction( action, clipUuid, rootUuid );
		return action;

	}

	existingAction( clip, optionalRoot ) {

		const root = optionalRoot || this._root;
		const clipObject = typeof clip === 'string' ? AnimationClip.findByName( root, clip ) : clip;
		const actionsForClip = this._actionsByClip[ clipObject ? clipObject.uuid : clip ];
		return actionsForClip !== undefined ? actionsForClip.actionByRoot[ root.uuid ] || null : null;

	}

	stopAllAction() {

		for ( let i = this._nActiveActions - 1; i >= 0; -- i ) this._actions[ i ].stop();
		return this;

	}

	update( deltaTime ) {

		deltaTime *= this.timeScale;
		const actions = this._actions, nActions = this._nActiveActions, time = this.time += deltaTime;
		const timeDirection = Math.sign( deltaTime ), accuIndex = this._accuIndex ^= 1;
		for ( let i = 0; i !== nActions; ++ i ) actions[ i ]._update( time, deltaTime, timeDirection, accuIndex );
		const bindings = this._bindings, nBindings = this._nActiveBindings;
		for ( let i = 0; i !== nBindings; ++ i ) bindings[ i ].apply( accuIndex );
		return this;

	}

	setTime( time ) {

		this.time = 0;
		for ( const action of this._actions ) action.time = 0;
		return this.update( time );

	}

	getRoot() {

		return this._root;

	}

	uncacheClip( clip ) {

		const actions = this._actions, clipUuid = clip.uuid, actionsForClip = this._actionsByClip[ clipUuid ];
		if ( actionsForClip === undefined ) return;

		for ( const action of actionsForClip.knownActions ) {

			this._deactivateAction( action );
			const cacheIndex = action._cacheIndex, last = actions[ actions.length - 1 ];
			action._cacheIndex = null;
			action._byClipCacheIndex = null;
			last._cacheIndex = cacheIndex;
			actions[ cacheIndex ] = last;
			actions.pop();
			this._removeInactiveBindingsForAction( action );

		}

		delete this._actionsByClip[ clipUuid ];

	}

	uncacheRoot( root ) {

		const rootUuid = root.uuid;

		for ( const clipUuid in this._actionsByClip ) {

			const action = this._actionsByClip[ clipUuid ].actionByRoot[ rootUuid ];

			if ( action !== undefined ) {

				this._deactivateAction( action );
				this._removeInactiveAction( action );

			}

		}

		const byName = this._bindingsByRootAndName[ rootUuid ];

		if ( byName !== undefined ) {

			for ( const trackName in byName ) {

				const binding = byName[ trackName ];
				binding.restoreOriginalState();
				this._removeInactiveBinding( binding );

			}

		}

	}

	uncacheAction( clip, optionalRoot ) {

		const action = this.existingAction( clip, optionalRoot );

		if ( action !== null ) {

			this._deactivateAction( action );
			this._removeInactiveAction( action );

		}

	}

}
