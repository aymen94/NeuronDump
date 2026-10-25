// The shared Web Audio context and the AudioListener.
//
// Part of the audio classes; `../audio.js` gathers them.

import { Object3D } from '../object3d.js';
import { Quaternion, Vector3 } from '../values.js';

let context;

/** three.js `AudioContext`: the one Web Audio context shared by all audio objects. */
export const AudioContext = {

	getContext() {

		if ( context === undefined ) context = new ( globalThis.AudioContext || globalThis.webkitAudioContext )();
		return context;

	},

	setContext( value ) {

		context = value;

	}

};

export const _position = new Vector3(), _quaternion = new Quaternion(), _scale = new Vector3(), _orientation = new Vector3();

/** The listener's ears, usually a child of the camera. */
export class AudioListener extends Object3D {

	constructor() {

		super();
		this.type = 'AudioListener';
		this.context = AudioContext.getContext();
		this.gain = this.context.createGain();
		this.gain.connect( this.context.destination );
		this.filter = null;
		this.timeDelta = 0;
		this._clock = performance.now();

	}

	getInput() {

		return this.gain;

	}

	removeFilter() {

		if ( this.filter !== null ) {

			this.gain.disconnect( this.filter );
			this.filter.disconnect( this.context.destination );
			this.gain.connect( this.context.destination );
			this.filter = null;

		}

		return this;

	}

	getFilter() {

		return this.filter;

	}

	setFilter( value ) {

		if ( this.filter !== null ) {

			this.gain.disconnect( this.filter );
			this.filter.disconnect( this.context.destination );

		} else {

			this.gain.disconnect( this.context.destination );

		}

		this.filter = value;
		this.gain.connect( this.filter );
		this.filter.connect( this.context.destination );
		return this;

	}

	getMasterVolume() {

		return this.gain.gain.value;

	}

	setMasterVolume( value ) {

		this.gain.gain.setTargetAtTime( value, this.context.currentTime, 0.01 );
		return this;

	}

	// Places the Web Audio listener at the object, facing its -z with its +y up.
	_onFrame() {

		const now = performance.now();
		this.timeDelta = ( now - this._clock ) / 1000;
		this._clock = now;
		const listener = this.context.listener, up = this.up;
		this.matrixWorld.decompose( _position, _quaternion, _scale );
		_orientation.set( 0, 0, - 1 ).applyQuaternion( _quaternion );
		const upWorld = new Vector3( up.x, up.y, up.z ).applyQuaternion( _quaternion );

		if ( listener.positionX ) {

			const endTime = this.context.currentTime + this.timeDelta;
			listener.positionX.linearRampToValueAtTime( _position.x, endTime );
			listener.positionY.linearRampToValueAtTime( _position.y, endTime );
			listener.positionZ.linearRampToValueAtTime( _position.z, endTime );
			listener.forwardX.linearRampToValueAtTime( _orientation.x, endTime );
			listener.forwardY.linearRampToValueAtTime( _orientation.y, endTime );
			listener.forwardZ.linearRampToValueAtTime( _orientation.z, endTime );
			listener.upX.linearRampToValueAtTime( upWorld.x, endTime );
			listener.upY.linearRampToValueAtTime( upWorld.y, endTime );
			listener.upZ.linearRampToValueAtTime( upWorld.z, endTime );

		} else {

			listener.setPosition( _position.x, _position.y, _position.z );
			listener.setOrientation( _orientation.x, _orientation.y, _orientation.z, upWorld.x, upWorld.y, upWorld.z );

		}

	}

}

AudioListener.prototype.isAudioListener = true;
