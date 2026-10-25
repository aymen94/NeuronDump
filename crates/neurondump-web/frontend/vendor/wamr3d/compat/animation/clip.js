// AnimationClip: a named set of tracks.
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import * as W from '../../index.js';
import { NormalAnimationBlendMode } from '../constants.js';
import { getKeyframeOrder, sortedArray } from './utils.js';
import { KeyframeTrack, NumberKeyframeTrack, parseKeyframeTrack } from './tracks.js';

/** A named set of tracks played together. `duration` -1 means the last key of any track. */
export class AnimationClip {

	constructor( name = '', duration = - 1, tracks = [], blendMode = NormalAnimationBlendMode ) {

		this.name = name;
		this.tracks = tracks;
		this.duration = duration;
		this.blendMode = blendMode;
		this.uuid = W.MathUtils.generateUUID();
		this.userData = {};
		if ( this.duration < 0 ) this.resetDuration();

	}

	static parse( json ) {

		const frameTime = 1.0 / ( json.fps || 1.0 );
		const tracks = json.tracks.map( ( track ) => parseKeyframeTrack( track ).scale( frameTime ) );
		const clip = new this( json.name, json.duration, tracks, json.blendMode );
		clip.uuid = json.uuid;
		clip.userData = JSON.parse( json.userData || '{}' );
		return clip;

	}

	static toJSON( clip ) {

		return { name: clip.name, duration: clip.duration, tracks: clip.tracks.map( ( track ) => KeyframeTrack.toJSON( track ) ), uuid: clip.uuid, blendMode: clip.blendMode, userData: JSON.stringify( clip.userData ) };

	}

	/** Plays the morph targets one after another, each at full weight on its frame. */
	static CreateFromMorphTargetSequence( name, morphTargetSequence, fps, noLoop ) {

		const n = morphTargetSequence.length, tracks = [];

		for ( let i = 0; i < n; i ++ ) {

			let times = [ ( i + n - 1 ) % n, i, ( i + 1 ) % n ], values = [ 0, 1, 0 ];
			const order = getKeyframeOrder( times );
			times = sortedArray( times, 1, order );
			values = sortedArray( values, 1, order );

			// the first target is also the last, for a seamless loop
			if ( ! noLoop && times[ 0 ] === 0 ) {

				times.push( n );
				values.push( values[ 0 ] );

			}

			tracks.push( new NumberKeyframeTrack( '.morphTargetInfluences[' + morphTargetSequence[ i ].name + ']', times, values ).scale( 1.0 / fps ) );

		}

		return new this( name, - 1, tracks );

	}

	static findByName( objectOrClipArray, name ) {

		let clips = objectOrClipArray;

		if ( ! Array.isArray( objectOrClipArray ) ) {

			const o = objectOrClipArray;
			clips = ( o.geometry && o.geometry.animations ) || o.animations;

		}

		for ( const clip of clips ) if ( clip.name === name ) return clip;
		return null;

	}

	/** One clip per name prefix of morph targets numbered like `walk001`, `walk002`. */
	static CreateClipsFromMorphTargetSequences( morphTargets, fps, noLoop ) {

		const byName = {}, pattern = /^([\w-]*?)([\d]+)$/;

		for ( const morphTarget of morphTargets ) {

			const parts = morphTarget.name.match( pattern );
			if ( parts && parts.length > 1 ) ( byName[ parts[ 1 ] ] ??= [] ).push( morphTarget );

		}

		return Object.keys( byName ).map( ( name ) => this.CreateFromMorphTargetSequence( name, byName[ name ], fps, noLoop ) );

	}

	resetDuration() {

		let duration = 0;
		for ( const track of this.tracks ) duration = Math.max( duration, track.times[ track.times.length - 1 ] );
		this.duration = duration;
		return this;

	}

	trim() {

		for ( const track of this.tracks ) track.trim( 0, this.duration );
		return this;

	}

	validate() {

		let valid = true;
		for ( const track of this.tracks ) valid = valid && track.validate();
		return valid;

	}

	optimize() {

		for ( const track of this.tracks ) track.optimize();
		return this;

	}

	clone() {

		const clip = new this.constructor( this.name, this.duration, this.tracks.map( ( track ) => track.clone() ), this.blendMode );
		clip.userData = JSON.parse( JSON.stringify( this.userData ) );
		return clip;

	}

	toJSON() {

		return this.constructor.toJSON( this );

	}

}
