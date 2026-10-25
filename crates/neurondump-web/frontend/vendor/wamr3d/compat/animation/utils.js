// AnimationUtils: array conversion, key ordering, additive clips and sub-clips.
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import { AdditiveAnimationBlendMode } from '../constants.js';
import { Quaternion } from '../values.js';

export const isTypedArray = ( object ) => ArrayBuffer.isView( object ) && ! ( object instanceof DataView );

export function convertArray( array, type ) {

	if ( ! array || array.constructor === type ) return array;
	if ( typeof type.BYTES_PER_ELEMENT === 'number' ) return new type( array );
	return Array.prototype.slice.call( array );

}

export function hasTangents( settings ) {

	return settings !== undefined && settings !== null && settings.inTangents !== undefined && settings.outTangents !== undefined;

}

export function getKeyframeOrder( times ) {

	const result = Array.from( { length: times.length }, ( _, i ) => i );
	result.sort( ( i, j ) => times[ i ] - times[ j ] );
	return result;

}

export function sortedArray( values, stride, order ) {

	const result = new values.constructor( values.length );

	for ( let i = 0, dst = 0; dst !== values.length; ++ i ) {

		const src = order[ i ] * stride;
		for ( let j = 0; j !== stride; ++ j ) result[ dst ++ ] = values[ src + j ];

	}

	return result;

}

export function flattenJSON( jsonKeys, times, values, valuePropertyName ) {

	let i = 1, key = jsonKeys[ 0 ];
	while ( key !== undefined && key[ valuePropertyName ] === undefined ) key = jsonKeys[ i ++ ];
	if ( key === undefined ) return;
	let value = key[ valuePropertyName ];
	if ( value === undefined ) return;
	const push = Array.isArray( value ) ? ( v ) => values.push( ...v ) : value.toArray !== undefined ? ( v ) => v.toArray( values, values.length ) : ( v ) => values.push( v );

	do {

		value = key[ valuePropertyName ];

		if ( value !== undefined ) {

			times.push( key.time );
			push( value );

		}

		key = jsonKeys[ i ++ ];

	} while ( key !== undefined );

}

export const AnimationUtils = {

	convertArray,
	isTypedArray,
	hasTangents,
	getKeyframeOrder,
	sortedArray,
	flattenJSON,

	/** The keys of `sourceClip` from frame `startFrame` up to `endFrame`, shifted to start at 0. */
	subclip( sourceClip, name, startFrame, endFrame, fps = 30 ) {

		const clip = sourceClip.clone();
		clip.name = name;
		const tracks = [];

		for ( const track of clip.tracks ) {

			const valueSize = track.getValueSize(), times = [], values = [];

			for ( let j = 0; j < track.times.length; ++ j ) {

				const frame = track.times[ j ] * fps;
				if ( frame < startFrame || frame >= endFrame ) continue;
				times.push( track.times[ j ] );
				for ( let k = 0; k < valueSize; ++ k ) values.push( track.values[ j * valueSize + k ] );

			}

			if ( times.length === 0 ) continue;
			track.times = convertArray( times, track.times.constructor );
			track.values = convertArray( values, track.values.constructor );
			tracks.push( track );

		}

		clip.tracks = tracks;
		let minStartTime = Infinity;
		for ( const track of clip.tracks ) minStartTime = Math.min( minStartTime, track.times[ 0 ] );
		for ( const track of clip.tracks ) track.shift( - 1 * minStartTime );
		clip.resetDuration();
		return clip;

	},

	/** Makes `targetClip` relative to the pose of `referenceClip` at `referenceFrame`, for additive blending. */
	makeClipAdditive( targetClip, referenceFrame = 0, referenceClip = targetClip, fps = 30 ) {

		if ( fps <= 0 ) fps = 30;
		const referenceTime = referenceFrame / fps;

		for ( const referenceTrack of referenceClip.tracks ) {

			const type = referenceTrack.ValueTypeName;
			if ( type === 'bool' || type === 'string' ) continue;
			const targetTrack = targetClip.tracks.find( ( track ) => track.name === referenceTrack.name && track.ValueTypeName === type );
			if ( targetTrack === undefined ) continue;

			const referenceValueSize = referenceTrack.getValueSize(), targetValueSize = targetTrack.getValueSize();
			// glTF cubic splines store in-tangent, value, out-tangent per key
			const referenceOffset = referenceTrack.createInterpolant.isInterpolantFactoryMethodGLTFCubicSpline ? referenceValueSize / 3 : 0;
			const targetOffset = targetTrack.createInterpolant.isInterpolantFactoryMethodGLTFCubicSpline ? targetValueSize / 3 : 0;
			const lastIndex = referenceTrack.times.length - 1;
			let referenceValue;

			if ( referenceTime <= referenceTrack.times[ 0 ] ) {

				referenceValue = referenceTrack.values.slice( referenceOffset, referenceValueSize - referenceOffset );

			} else if ( referenceTime >= referenceTrack.times[ lastIndex ] ) {

				const start = lastIndex * referenceValueSize + referenceOffset;
				referenceValue = referenceTrack.values.slice( start, start + referenceValueSize - referenceOffset );

			} else {

				const interpolant = referenceTrack.createInterpolant();
				interpolant.evaluate( referenceTime );
				referenceValue = interpolant.resultBuffer.slice( referenceOffset, referenceValueSize - referenceOffset );

			}

			if ( type === 'quaternion' ) new Quaternion().fromArray( referenceValue ).normalize().conjugate().toArray( referenceValue );

			for ( let j = 0; j < targetTrack.times.length; ++ j ) {

				const valueStart = j * targetValueSize + targetOffset;

				if ( type === 'quaternion' ) {

					Quaternion.multiplyQuaternionsFlat( targetTrack.values, valueStart, referenceValue, 0, targetTrack.values, valueStart );

				} else {

					const valueEnd = targetValueSize - targetOffset * 2;
					for ( let k = 0; k < valueEnd; ++ k ) targetTrack.values[ valueStart + k ] -= referenceValue[ k ];

				}

			}

		}

		targetClip.blendMode = AdditiveAnimationBlendMode;
		return targetClip;

	}

};
