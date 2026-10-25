// AnimationAction: playback state of one clip on one root (time, weight, looping, fading).
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import { AdditiveAnimationBlendMode, LoopOnce, LoopPingPong, LoopRepeat, WrapAroundEnding, ZeroCurvatureEnding, ZeroSlopeEnding } from '../constants.js';

/** One clip playing on one root: time, loop, weight, fading and warping. Made by `mixer.clipAction`. */
export class AnimationAction {

	constructor( mixer, clip, localRoot = null, blendMode = clip.blendMode ) {

		this._mixer = mixer;
		this._clip = clip;
		this._localRoot = localRoot;
		this.blendMode = blendMode;
		const interpolantSettings = { endingStart: ZeroCurvatureEnding, endingEnd: ZeroCurvatureEnding };
		this._interpolants = clip.tracks.map( ( track ) => {

			const interpolant = track.createInterpolant( null );
			interpolant.settings = interpolantSettings;
			return interpolant;

		} );
		this._interpolantSettings = interpolantSettings;
		this._propertyBindings = new Array( clip.tracks.length );
		this._cacheIndex = null;
		this._byClipCacheIndex = null;
		this._timeScaleInterpolant = null;
		this._restoreTimeScale = null;
		this._weightInterpolant = null;
		this.loop = LoopRepeat;
		this._loopCount = - 1;
		this._startTime = null;
		this.time = 0;
		this.timeScale = 1;
		this._effectiveTimeScale = 1;
		this.weight = 1;
		this._effectiveWeight = 1;
		this.repetitions = Infinity;
		this.paused = false;
		this.enabled = true;
		this.clampWhenFinished = false;
		this.zeroSlopeAtStart = true;
		this.zeroSlopeAtEnd = true;

	}

	play() {

		this._mixer._activateAction( this );
		return this;

	}

	stop() {

		this._mixer._deactivateAction( this );
		return this.reset();

	}

	reset() {

		this.paused = false;
		this.enabled = true;
		this.time = 0;
		this._loopCount = - 1;
		this._startTime = null;
		return this.stopFading().stopWarping();

	}

	isRunning() {

		return this.enabled && ! this.paused && this.timeScale !== 0 && this._startTime === null && this._mixer._isActiveAction( this );

	}

	isScheduled() {

		return this._mixer._isActiveAction( this );

	}

	startAt( time ) {

		this._startTime = time;
		return this;

	}

	setLoop( mode, repetitions ) {

		this.loop = mode;
		this.repetitions = repetitions;
		return this;

	}

	setEffectiveWeight( weight ) {

		this.weight = weight;
		this._effectiveWeight = this.enabled ? weight : 0;
		return this.stopFading();

	}

	getEffectiveWeight() {

		return this._effectiveWeight;

	}

	fadeIn( duration ) {

		return this._scheduleFading( duration, 0, 1 );

	}

	fadeOut( duration ) {

		return this._scheduleFading( duration, 1, 0 );

	}

	crossFadeFrom( fadeOutAction, duration, warp = false ) {

		fadeOutAction.fadeOut( duration );
		this.fadeIn( duration );

		if ( warp === true ) {

			const fadeInDuration = this._clip.duration, fadeOutDuration = fadeOutAction._clip.duration;
			fadeOutAction._restoreTimeScale = fadeOutAction.timeScale;
			this._restoreTimeScale = this.timeScale;
			fadeOutAction.warp( 1.0, fadeOutDuration / fadeInDuration, duration );
			this.warp( fadeInDuration / fadeOutDuration, 1.0, duration );

		}

		return this;

	}

	crossFadeTo( fadeInAction, duration, warp = false ) {

		return fadeInAction.crossFadeFrom( this, duration, warp );

	}

	stopFading() {

		const interpolant = this._weightInterpolant;

		if ( interpolant !== null ) {

			this._weightInterpolant = null;
			this._mixer._takeBackControlInterpolant( interpolant );

		}

		return this;

	}

	setEffectiveTimeScale( timeScale ) {

		this.timeScale = timeScale;
		this._effectiveTimeScale = this.paused ? 0 : timeScale;
		return this.stopWarping();

	}

	getEffectiveTimeScale() {

		return this._effectiveTimeScale;

	}

	setDuration( duration ) {

		this.timeScale = this._clip.duration / duration;
		return this.stopWarping();

	}

	syncWith( action ) {

		this.time = action.time;
		this.timeScale = action.timeScale;
		return this.stopWarping();

	}

	halt( duration ) {

		return this.warp( this._effectiveTimeScale, 0, duration );

	}

	warp( startTimeScale, endTimeScale, duration ) {

		const now = this._mixer.time, timeScale = this.timeScale;
		const interpolant = this._timeScaleInterpolant ??= this._mixer._lendControlInterpolant();
		const times = interpolant.parameterPositions, values = interpolant.sampleValues;
		times[ 0 ] = now;
		times[ 1 ] = now + duration;
		values[ 0 ] = startTimeScale / timeScale;
		values[ 1 ] = endTimeScale / timeScale;
		return this;

	}

	stopWarping() {

		const interpolant = this._timeScaleInterpolant;

		if ( interpolant !== null ) {

			this._timeScaleInterpolant = null;
			this._mixer._takeBackControlInterpolant( interpolant );

		}

		this._restoreTimeScale = null;
		return this;

	}

	getMixer() {

		return this._mixer;

	}

	getClip() {

		return this._clip;

	}

	getRoot() {

		return this._localRoot || this._mixer._root;

	}

	_update( time, deltaTime, timeDirection, accuIndex ) {

		if ( ! this.enabled ) {

			this._updateWeight( time );
			return;

		}

		const startTime = this._startTime;

		if ( startTime !== null ) {

			// a scheduled start: nothing moves until it is reached
			const timeRunning = ( time - startTime ) * timeDirection;

			if ( timeRunning < 0 || timeDirection === 0 ) {

				deltaTime = 0;

			} else {

				this._startTime = null;
				deltaTime = timeDirection * timeRunning;

			}

		}

		deltaTime *= this._updateTimeScale( time );
		const clipTime = this._updateTime( deltaTime );
		const weight = this._updateWeight( time );
		if ( weight <= 0 ) return;
		const interpolants = this._interpolants, mixers = this._propertyBindings;

		if ( this.blendMode === AdditiveAnimationBlendMode ) {

			for ( let j = 0; j !== interpolants.length; ++ j ) {

				interpolants[ j ].evaluate( clipTime );
				mixers[ j ].accumulateAdditive( weight );

			}

		} else {

			for ( let j = 0; j !== interpolants.length; ++ j ) {

				interpolants[ j ].evaluate( clipTime );
				mixers[ j ].accumulate( accuIndex, weight );

			}

		}

	}

	_updateWeight( time ) {

		let weight = 0;

		if ( this.enabled ) {

			weight = this.weight;
			const interpolant = this._weightInterpolant;

			if ( interpolant !== null ) {

				const value = interpolant.evaluate( time )[ 0 ];
				weight *= value;

				if ( time > interpolant.parameterPositions[ 1 ] ) {

					this.stopFading();
					if ( value === 0 ) this.enabled = false;

				}

			}

		}

		this._effectiveWeight = weight;
		return weight;

	}

	_updateTimeScale( time ) {

		let timeScale = 0;

		if ( ! this.paused ) {

			timeScale = this.timeScale;
			const interpolant = this._timeScaleInterpolant;

			if ( interpolant !== null ) {

				timeScale *= interpolant.evaluate( time )[ 0 ];

				if ( time > interpolant.parameterPositions[ 1 ] ) {

					if ( timeScale === 0 ) {

						this.paused = true;

					} else {

						if ( this._restoreTimeScale !== null ) timeScale = this._restoreTimeScale;
						this.timeScale = timeScale;

					}

					this.stopWarping();

				}

			}

		}

		this._effectiveTimeScale = timeScale;
		return timeScale;

	}

	// Advances the time, handles looping and ending, and returns the time in the clip.
	_updateTime( deltaTime ) {

		const duration = this._clip.duration, loop = this.loop;
		let time = this.time + deltaTime, loopCount = this._loopCount;
		const pingPong = loop === LoopPingPong;

		if ( deltaTime === 0 ) {

			if ( loopCount === - 1 ) return time;
			return pingPong && ( loopCount & 1 ) === 1 ? duration - time : time;

		}

		if ( loop === LoopOnce ) {

			if ( loopCount === - 1 ) {

				this._loopCount = 0;
				this._setEndings( true, true, false );

			}

			if ( time >= duration || time < 0 ) {

				time = time >= duration ? duration : 0;
				if ( this.clampWhenFinished ) this.paused = true;
				else this.enabled = false;
				this.time = time;
				this._mixer.dispatchEvent( { type: 'finished', action: this, direction: deltaTime < 0 ? - 1 : 1 } );

			} else {

				this.time = time;

			}

		} else {

			if ( loopCount === - 1 ) {

				if ( deltaTime >= 0 ) {

					loopCount = 0;
					this._setEndings( true, this.repetitions === 0, pingPong );

				} else {

					this._setEndings( this.repetitions === 0, true, pingPong );

				}

			}

			if ( time >= duration || time < 0 ) {

				const loopDelta = Math.floor( time / duration );
				time -= duration * loopDelta;
				loopCount += Math.abs( loopDelta );
				const pending = this.repetitions - loopCount;

				if ( pending <= 0 ) {

					if ( this.clampWhenFinished ) this.paused = true;
					else this.enabled = false;
					time = deltaTime > 0 ? duration : 0;
					this.time = time;
					this._mixer.dispatchEvent( { type: 'finished', action: this, direction: deltaTime > 0 ? 1 : - 1 } );

				} else {

					if ( pending === 1 ) {

						const atStart = deltaTime < 0;
						this._setEndings( atStart, ! atStart, pingPong );

					} else {

						this._setEndings( false, false, pingPong );

					}

					this._loopCount = loopCount;
					this.time = time;
					this._mixer.dispatchEvent( { type: 'loop', action: this, loopDelta } );

				}

			} else {

				this._loopCount = loopCount;
				this.time = time;

			}

			if ( pingPong && ( loopCount & 1 ) === 1 ) return duration - time;

		}

		return time;

	}

	_setEndings( atStart, atEnd, pingPong ) {

		const settings = this._interpolantSettings;

		if ( pingPong ) {

			settings.endingStart = settings.endingEnd = ZeroSlopeEnding;

		} else {

			settings.endingStart = atStart ? ( this.zeroSlopeAtStart ? ZeroSlopeEnding : ZeroCurvatureEnding ) : WrapAroundEnding;
			settings.endingEnd = atEnd ? ( this.zeroSlopeAtEnd ? ZeroSlopeEnding : ZeroCurvatureEnding ) : WrapAroundEnding;

		}

	}

	_scheduleFading( duration, weightNow, weightThen ) {

		const now = this._mixer.time;
		const interpolant = this._weightInterpolant ??= this._mixer._lendControlInterpolant();
		const times = interpolant.parameterPositions, values = interpolant.sampleValues;
		times[ 0 ] = now;
		values[ 0 ] = weightNow;
		times[ 1 ] = now + duration;
		values[ 1 ] = weightThen;
		return this;

	}

}
