// Keyframe tracks of every value type, and parsing them from JSON.
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import { InterpolateBezier, InterpolateDiscrete, InterpolateLinear, InterpolateSmooth } from '../constants.js';
import { BezierInterpolant, CubicInterpolant, DiscreteInterpolant, LinearInterpolant, QuaternionLinearInterpolant } from './interpolants.js';
import { convertArray, flattenJSON, hasTangents, isTypedArray } from './utils.js';

function scaleTangentTimes( tangents, timeScale ) {

	for ( let i = 0; i < tangents.length; i += 2 ) tangents[ i ] *= timeScale;

}

/** Values of one property over time. `name` is a binding path such as `'bone.quaternion'` or `'.material.opacity'`. */
export class KeyframeTrack {

	constructor( name, times, values, interpolation ) {

		if ( name === undefined ) throw new Error( 'WAMR3D: KeyframeTrack: track name is undefined' );
		if ( times === undefined || times.length === 0 ) throw new Error( 'WAMR3D: KeyframeTrack: no keyframes in track named ' + name );
		this.name = name;
		this.times = convertArray( times, this.TimeBufferType );
		this.values = convertArray( values, this.ValueBufferType );
		this.setInterpolation( interpolation || this.DefaultInterpolation );

	}

	static toJSON( track ) {

		const trackType = track.constructor;
		let json;

		if ( trackType.toJSON !== this.toJSON ) {

			json = trackType.toJSON( track );

		} else {

			json = { name: track.name, times: convertArray( track.times, Array ), values: convertArray( track.values, Array ) };
			const interpolation = track.getInterpolation();
			if ( interpolation !== track.DefaultInterpolation ) json.interpolation = interpolation;
			if ( hasTangents( track.settings ) ) json.settings = { inTangents: convertArray( track.settings.inTangents, Array ), outTangents: convertArray( track.settings.outTangents, Array ) };

		}

		json.type = track.ValueTypeName;
		return json;

	}

	InterpolantFactoryMethodDiscrete( result ) {

		return new DiscreteInterpolant( this.times, this.values, this.getValueSize(), result );

	}

	InterpolantFactoryMethodLinear( result ) {

		return new LinearInterpolant( this.times, this.values, this.getValueSize(), result );

	}

	InterpolantFactoryMethodSmooth( result ) {

		return new CubicInterpolant( this.times, this.values, this.getValueSize(), result );

	}

	InterpolantFactoryMethodBezier( result ) {

		const interpolant = new BezierInterpolant( this.times, this.values, this.getValueSize(), result );

		if ( this.settings ) {

			interpolant.inTangents = this.settings.inTangents;
			interpolant.outTangents = this.settings.outTangents;

		}

		return interpolant;

	}

	setInterpolation( interpolation ) {

		const factories = {
			[ InterpolateDiscrete ]: this.InterpolantFactoryMethodDiscrete,
			[ InterpolateLinear ]: this.InterpolantFactoryMethodLinear,
			[ InterpolateSmooth ]: this.InterpolantFactoryMethodSmooth,
			[ InterpolateBezier ]: this.InterpolantFactoryMethodBezier
		};
		const factoryMethod = factories[ interpolation ];

		if ( factoryMethod === undefined ) {

			const message = 'unsupported interpolation for ' + this.ValueTypeName + ' keyframe track named ' + this.name;

			if ( this.createInterpolant === undefined ) {

				if ( interpolation !== this.DefaultInterpolation ) this.setInterpolation( this.DefaultInterpolation );
				else throw new Error( message );

			}

			console.warn( 'WAMR3D: KeyframeTrack:', message );
			return this;

		}

		this.createInterpolant = factoryMethod;
		return this;

	}

	getInterpolation() {

		switch ( this.createInterpolant ) {

			case this.InterpolantFactoryMethodDiscrete: return InterpolateDiscrete;
			case this.InterpolantFactoryMethodLinear: return InterpolateLinear;
			case this.InterpolantFactoryMethodSmooth: return InterpolateSmooth;
			case this.InterpolantFactoryMethodBezier: return InterpolateBezier;

		}

		return undefined;

	}

	getValueSize() {

		return this.values.length / this.times.length;

	}

	shift( timeOffset ) {

		if ( timeOffset !== 0 ) for ( let i = 0; i < this.times.length; ++ i ) this.times[ i ] += timeOffset;
		return this;

	}

	scale( timeScale ) {

		if ( timeScale !== 1 ) {

			for ( let i = 0; i < this.times.length; ++ i ) this.times[ i ] *= timeScale;

			if ( hasTangents( this.settings ) ) {

				scaleTangentTimes( this.settings.inTangents, timeScale );
				scaleTangentTimes( this.settings.outTangents, timeScale );

			}

		}

		return this;

	}

	/** Keeps the keys from `startTime` to `endTime`, at least one. */
	trim( startTime, endTime ) {

		const times = this.times, nKeys = times.length;
		let from = 0, to = nKeys - 1;
		while ( from !== nKeys && times[ from ] < startTime ) ++ from;
		while ( to !== - 1 && times[ to ] > endTime ) -- to;
		++ to;

		if ( from !== 0 || to !== nKeys ) {

			if ( from >= to ) {

				to = Math.max( to, 1 );
				from = to - 1;

			}

			const stride = this.getValueSize();
			this.times = times.slice( from, to );
			this.values = this.values.slice( from * stride, to * stride );

		}

		return this;

	}

	validate() {

		let valid = true;
		const valueSize = this.getValueSize();

		if ( valueSize - Math.floor( valueSize ) !== 0 ) {

			console.error( 'WAMR3D: KeyframeTrack: invalid value size in track.', this );
			valid = false;

		}

		const times = this.times, values = this.values;

		if ( times.length === 0 ) {

			console.error( 'WAMR3D: KeyframeTrack: track is empty.', this );
			valid = false;

		}

		let previous = null;

		for ( let i = 0; i !== times.length; i ++ ) {

			const current = times[ i ];

			if ( typeof current === 'number' && isNaN( current ) ) {

				console.error( 'WAMR3D: KeyframeTrack: time is not a valid number.', this, i, current );
				valid = false;
				break;

			}

			if ( previous !== null && previous > current ) {

				console.error( 'WAMR3D: KeyframeTrack: out of order keys.', this, i, current, previous );
				valid = false;
				break;

			}

			previous = current;

		}

		if ( values !== undefined && isTypedArray( values ) ) {

			for ( let i = 0; i !== values.length; ++ i ) {

				if ( isNaN( values[ i ] ) ) {

					console.error( 'WAMR3D: KeyframeTrack: value is not a valid number.', this, i, values[ i ] );
					valid = false;
					break;

				}

			}

		}

		return valid;

	}

	/** Removes keys equal to both neighbours (and duplicate times); smooth tracks keep every key. */
	optimize() {

		const times = this.times.slice(), values = this.values.slice(), stride = this.getValueSize();
		const smooth = this.getInterpolation() === InterpolateSmooth, lastIndex = times.length - 1;
		let writeIndex = 1;

		for ( let i = 1; i < lastIndex; ++ i ) {

			let keep = false;
			const time = times[ i ], timeNext = times[ i + 1 ];

			if ( time !== timeNext && ( i !== 1 || time !== times[ 0 ] ) ) {

				if ( ! smooth ) {

					const offset = i * stride, offsetP = offset - stride, offsetN = offset + stride;

					for ( let j = 0; j !== stride; ++ j ) {

						const value = values[ offset + j ];

						if ( value !== values[ offsetP + j ] || value !== values[ offsetN + j ] ) {

							keep = true;
							break;

						}

					}

				} else {

					keep = true;

				}

			}

			if ( keep ) {

				if ( i !== writeIndex ) {

					times[ writeIndex ] = times[ i ];
					for ( let j = 0; j !== stride; ++ j ) values[ writeIndex * stride + j ] = values[ i * stride + j ];

				}

				++ writeIndex;

			}

		}

		if ( lastIndex > 0 ) {

			times[ writeIndex ] = times[ lastIndex ];
			for ( let j = 0; j !== stride; ++ j ) values[ writeIndex * stride + j ] = values[ lastIndex * stride + j ];
			++ writeIndex;

		}

		if ( writeIndex !== times.length ) {

			this.times = times.slice( 0, writeIndex );
			this.values = values.slice( 0, writeIndex * stride );

		} else {

			this.times = times;
			this.values = values;

		}

		return this;

	}

	clone() {

		const track = new this.constructor( this.name, this.times.slice(), this.values.slice() );
		track.createInterpolant = this.createInterpolant;
		if ( hasTangents( this.settings ) ) track.settings = { inTangents: this.settings.inTangents.slice(), outTangents: this.settings.outTangents.slice() };
		return track;

	}

}

KeyframeTrack.prototype.ValueTypeName = '';
KeyframeTrack.prototype.TimeBufferType = Float32Array;
KeyframeTrack.prototype.ValueBufferType = Float32Array;
KeyframeTrack.prototype.DefaultInterpolation = InterpolateLinear;

export class BooleanKeyframeTrack extends KeyframeTrack {

	constructor( name, times, values ) {

		super( name, times, values );

	}

}

BooleanKeyframeTrack.prototype.ValueTypeName = 'bool';
BooleanKeyframeTrack.prototype.ValueBufferType = Array;
BooleanKeyframeTrack.prototype.DefaultInterpolation = InterpolateDiscrete;
BooleanKeyframeTrack.prototype.InterpolantFactoryMethodLinear = undefined;
BooleanKeyframeTrack.prototype.InterpolantFactoryMethodSmooth = undefined;

export class ColorKeyframeTrack extends KeyframeTrack {}

ColorKeyframeTrack.prototype.ValueTypeName = 'color';

export class NumberKeyframeTrack extends KeyframeTrack {}

NumberKeyframeTrack.prototype.ValueTypeName = 'number';

export class QuaternionKeyframeTrack extends KeyframeTrack {

	InterpolantFactoryMethodLinear( result ) {

		return new QuaternionLinearInterpolant( this.times, this.values, this.getValueSize(), result );

	}

}

QuaternionKeyframeTrack.prototype.ValueTypeName = 'quaternion';
QuaternionKeyframeTrack.prototype.InterpolantFactoryMethodSmooth = undefined;

export class StringKeyframeTrack extends KeyframeTrack {

	constructor( name, times, values ) {

		super( name, times, values );

	}

}

StringKeyframeTrack.prototype.ValueTypeName = 'string';
StringKeyframeTrack.prototype.ValueBufferType = Array;
StringKeyframeTrack.prototype.DefaultInterpolation = InterpolateDiscrete;
StringKeyframeTrack.prototype.InterpolantFactoryMethodLinear = undefined;
StringKeyframeTrack.prototype.InterpolantFactoryMethodSmooth = undefined;

export class VectorKeyframeTrack extends KeyframeTrack {}

VectorKeyframeTrack.prototype.ValueTypeName = 'vector';

function trackTypeForValueTypeName( typeName ) {

	switch ( typeName.toLowerCase() ) {

		case 'scalar': case 'double': case 'float': case 'number': case 'integer': return NumberKeyframeTrack;
		case 'vector': case 'vector2': case 'vector3': case 'vector4': return VectorKeyframeTrack;
		case 'color': return ColorKeyframeTrack;
		case 'quaternion': return QuaternionKeyframeTrack;
		case 'bool': case 'boolean': return BooleanKeyframeTrack;
		case 'string': return StringKeyframeTrack;

	}

	throw new Error( 'WAMR3D: KeyframeTrack: unsupported typeName: ' + typeName );

}

export function parseKeyframeTrack( json ) {

	if ( json.type === undefined ) throw new Error( 'WAMR3D: KeyframeTrack: track type undefined, can not parse' );
	const trackType = trackTypeForValueTypeName( json.type );

	if ( json.times === undefined ) {

		const times = [], values = [];
		flattenJSON( json.keys, times, values, 'value' );
		json.times = times;
		json.values = values;

	}

	const track = trackType.parse !== undefined ? trackType.parse( json ) : new trackType( json.name, json.times, json.values, json.interpolation );
	if ( hasTangents( json.settings ) ) track.settings = { inTangents: convertArray( json.settings.inTangents, Float32Array ), outTangents: convertArray( json.settings.outTangents, Float32Array ) };
	return track;

}
