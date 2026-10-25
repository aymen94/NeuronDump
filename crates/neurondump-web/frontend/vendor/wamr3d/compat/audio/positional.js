// PositionalAudio, AudioAnalyser and AudioLoader.
//
// Part of the audio classes; `../audio.js` gathers them.

import { FileLoader, Loader } from '../loaders.js';
import { AudioContext, _orientation, _position, _quaternion, _scale } from './listener.js';
import { Audio } from './audio.js';

/** A sound from the object's place, panned and attenuated by distance (an HRTF panner). */
export class PositionalAudio extends Audio {

	constructor( listener ) {

		super( listener );
		this.type = 'PositionalAudio';
		this.panner = this.context.createPanner();
		this.panner.panningModel = 'HRTF';
		this.panner.connect( this.gain );

	}

	connect() {

		super.connect();
		this.panner.connect( this.gain );
		return this;

	}

	disconnect() {

		super.disconnect();
		this.panner.disconnect( this.gain );
		return this;

	}

	getOutput() {

		return this.panner;

	}

	getRefDistance() {

		return this.panner.refDistance;

	}

	setRefDistance( value ) {

		this.panner.refDistance = value;
		return this;

	}

	getRolloffFactor() {

		return this.panner.rolloffFactor;

	}

	setRolloffFactor( value ) {

		this.panner.rolloffFactor = value;
		return this;

	}

	getDistanceModel() {

		return this.panner.distanceModel;

	}

	setDistanceModel( value ) {

		this.panner.distanceModel = value;
		return this;

	}

	getMaxDistance() {

		return this.panner.maxDistance;

	}

	setMaxDistance( value ) {

		this.panner.maxDistance = value;
		return this;

	}

	setDirectionalCone( coneInnerAngle, coneOuterAngle, coneOuterGain ) {

		this.panner.coneInnerAngle = coneInnerAngle;
		this.panner.coneOuterAngle = coneOuterAngle;
		this.panner.coneOuterGain = coneOuterGain;
		return this;

	}

	// The panner at the object, facing its +z.
	_onFrame() {

		if ( this.hasPlaybackControl === true && this.isPlaying === false ) return;
		this.matrixWorld.decompose( _position, _quaternion, _scale );
		_orientation.set( 0, 0, 1 ).applyQuaternion( _quaternion );
		const panner = this.panner;

		if ( panner.positionX ) {

			const endTime = this.context.currentTime + this.listener.timeDelta;
			panner.positionX.linearRampToValueAtTime( _position.x, endTime );
			panner.positionY.linearRampToValueAtTime( _position.y, endTime );
			panner.positionZ.linearRampToValueAtTime( _position.z, endTime );
			panner.orientationX.linearRampToValueAtTime( _orientation.x, endTime );
			panner.orientationY.linearRampToValueAtTime( _orientation.y, endTime );
			panner.orientationZ.linearRampToValueAtTime( _orientation.z, endTime );

		} else {

			panner.setPosition( _position.x, _position.y, _position.z );
			panner.setOrientation( _orientation.x, _orientation.y, _orientation.z );

		}

	}

}

/** Frequency data of a sound (three.js `AudioAnalyser`). */
export class AudioAnalyser {

	constructor( audio, fftSize = 2048 ) {

		this.analyser = audio.context.createAnalyser();
		this.analyser.fftSize = fftSize;
		this.data = new Uint8Array( this.analyser.frequencyBinCount );
		audio.getOutput().connect( this.analyser );

	}

	getFrequencyData() {

		this.analyser.getByteFrequencyData( this.data );
		return this.data;

	}

	getAverageFrequency() {

		let value = 0;
		const data = this.getFrequencyData();
		for ( let i = 0; i < data.length; i ++ ) value += data[ i ];
		return value / data.length;

	}

}

/** Decodes a sound file into an `AudioBuffer` (three.js `AudioLoader`). */
export class AudioLoader extends Loader {

	load( url, onLoad, onProgress, onError ) {

		const loader = new FileLoader( this.manager );
		loader.setResponseType( 'arraybuffer' );
		loader.setPath( this.path );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		const failed = ( error ) => {

			if ( onError ) onError( error );
			else console.error( error );
			this.manager.itemError( url );

		};

		loader.load( url, ( buffer ) => {

			try {

				// decodeAudioData detaches the buffer; a copy keeps the cached file usable
				AudioContext.getContext().decodeAudioData( buffer.slice( 0 ), onLoad ).catch( failed );

			} catch ( error ) {

				failed( error );

			}

		}, onProgress, onError );

	}

}
