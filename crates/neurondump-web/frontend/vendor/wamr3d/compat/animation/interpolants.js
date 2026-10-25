// Interpolants: sampling keyframe values between keys (linear, discrete, cubic, Bezier, quaternion slerp).
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import { WrapAroundEnding, ZeroCurvatureEnding, ZeroSlopeEnding } from '../constants.js';
import { Quaternion } from '../values.js';

/** Samples `sampleValues` (`sampleSize` per key) at a time between the `parameterPositions`. */
export class Interpolant {

	constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {

		this.parameterPositions = parameterPositions;
		this._cachedIndex = 0;
		this.resultBuffer = resultBuffer !== undefined ? resultBuffer : new sampleValues.constructor( sampleSize );
		this.sampleValues = sampleValues;
		this.valueSize = sampleSize;
		this.settings = null;
		this.DefaultSettings_ = {};

	}

	// Finds the interval around `t`, starting from the last one: a short linear scan
	// either way covers playback, a binary search covers jumps.
	evaluate( t ) {

		const pp = this.parameterPositions;
		let i1 = this._cachedIndex, t1 = pp[ i1 ], t0 = pp[ i1 - 1 ];

		validate_interval: {

			seek: {

				let right;

				linear_scan: {

					forward_scan: if ( ! ( t < t1 ) ) {

						for ( let giveUpAt = i1 + 2; ; ) {

							if ( t1 === undefined ) {

								if ( t < t0 ) break forward_scan;
								// after the end
								i1 = pp.length;
								this._cachedIndex = i1;
								return this.copySampleValue_( i1 - 1 );

							}

							if ( i1 === giveUpAt ) break;
							t0 = t1;
							t1 = pp[ ++ i1 ];
							if ( t < t1 ) break seek;

						}

						right = pp.length;
						break linear_scan;

					}

					if ( ! ( t >= t0 ) ) {

						const t1global = pp[ 1 ];

						if ( t < t1global ) {

							i1 = 2;
							t0 = t1global;

						}

						for ( let giveUpAt = i1 - 2; ; ) {

							if ( t0 === undefined ) {

								// before the start
								this._cachedIndex = 0;
								return this.copySampleValue_( 0 );

							}

							if ( i1 === giveUpAt ) break;
							t1 = t0;
							t0 = pp[ -- i1 - 1 ];
							if ( t >= t0 ) break seek;

						}

						right = i1;
						i1 = 0;
						break linear_scan;

					}

					break validate_interval;

				}

				while ( i1 < right ) {

					const mid = ( i1 + right ) >>> 1;
					if ( t < pp[ mid ] ) right = mid;
					else i1 = mid + 1;

				}

				t1 = pp[ i1 ];
				t0 = pp[ i1 - 1 ];

				if ( t0 === undefined ) {

					this._cachedIndex = 0;
					return this.copySampleValue_( 0 );

				}

				if ( t1 === undefined ) {

					i1 = pp.length;
					this._cachedIndex = i1;
					return this.copySampleValue_( i1 - 1 );

				}

			}

			this._cachedIndex = i1;
			this.intervalChanged_( i1, t0, t1 );

		}

		return this.interpolate_( i1, t0, t, t1 );

	}

	getSettings_() {

		return this.settings || this.DefaultSettings_;

	}

	copySampleValue_( index ) {

		const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize, offset = index * stride;
		for ( let i = 0; i !== stride; ++ i ) result[ i ] = values[ offset + i ];
		return result;

	}

	interpolate_() {

		throw new Error( 'WAMR3D: Interpolant: call to abstract method' );

	}

	intervalChanged_() {}

}

export class DiscreteInterpolant extends Interpolant {

	interpolate_( i1 ) {

		return this.copySampleValue_( i1 - 1 );

	}

}

export class LinearInterpolant extends Interpolant {

	interpolate_( i1, t0, t, t1 ) {

		const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize;
		const offset1 = i1 * stride, offset0 = offset1 - stride, weight1 = ( t - t0 ) / ( t1 - t0 ), weight0 = 1 - weight1;
		for ( let i = 0; i !== stride; ++ i ) result[ i ] = values[ offset0 + i ] * weight0 + values[ offset1 + i ] * weight1;
		return result;

	}

}

export class QuaternionLinearInterpolant extends Interpolant {

	interpolate_( i1, t0, t, t1 ) {

		const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize, alpha = ( t - t0 ) / ( t1 - t0 );
		let offset = i1 * stride;
		for ( let end = offset + stride; offset !== end; offset += 4 ) Quaternion.slerpFlat( result, 0, values, offset - stride, values, offset, alpha );
		return result;

	}

}

/** Cubic Hermite spline through the keys; the slopes at the first and last key follow `settings.endingStart` and `endingEnd`. */
export class CubicInterpolant extends Interpolant {

	constructor( parameterPositions, sampleValues, sampleSize, resultBuffer ) {

		super( parameterPositions, sampleValues, sampleSize, resultBuffer );
		this._weightPrev = - 0;
		this._offsetPrev = - 0;
		this._weightNext = - 0;
		this._offsetNext = - 0;
		this.DefaultSettings_ = { endingStart: ZeroCurvatureEnding, endingEnd: ZeroCurvatureEnding };

	}

	intervalChanged_( i1, t0, t1 ) {

		const pp = this.parameterPositions;
		let iPrev = i1 - 2, iNext = i1 + 1, tPrev = pp[ iPrev ], tNext = pp[ iNext ];

		if ( tPrev === undefined ) {

			switch ( this.getSettings_().endingStart ) {

				case ZeroSlopeEnding:
					iPrev = i1;
					tPrev = 2 * t0 - t1;
					break;
				case WrapAroundEnding:
					iPrev = pp.length - 2;
					tPrev = t0 + pp[ iPrev ] - pp[ iPrev + 1 ];
					break;
				default:
					iPrev = i1;
					tPrev = t1;

			}

		}

		if ( tNext === undefined ) {

			switch ( this.getSettings_().endingEnd ) {

				case ZeroSlopeEnding:
					iNext = i1;
					tNext = 2 * t1 - t0;
					break;
				case WrapAroundEnding:
					iNext = 1;
					tNext = t1 + pp[ 1 ] - pp[ 0 ];
					break;
				default:
					iNext = i1 - 1;
					tNext = t0;

			}

		}

		const halfDt = ( t1 - t0 ) * 0.5, stride = this.valueSize;
		this._weightPrev = halfDt / ( t0 - tPrev );
		this._weightNext = halfDt / ( tNext - t1 );
		this._offsetPrev = iPrev * stride;
		this._offsetNext = iNext * stride;

	}

	interpolate_( i1, t0, t, t1 ) {

		const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize;
		const o1 = i1 * stride, o0 = o1 - stride, oP = this._offsetPrev, oN = this._offsetNext, wP = this._weightPrev, wN = this._weightNext;
		const p = ( t - t0 ) / ( t1 - t0 ), pp = p * p, ppp = pp * p;
		const sP = - wP * ppp + 2 * wP * pp - wP * p;
		const s0 = ( 1 + wP ) * ppp + ( - 1.5 - 2 * wP ) * pp + ( - 0.5 + wP ) * p + 1;
		const s1 = ( - 1 - wN ) * ppp + ( 1.5 + wN ) * pp + 0.5 * p;
		const sN = wN * ppp - wN * pp;
		for ( let i = 0; i !== stride; ++ i ) result[ i ] = sP * values[ oP + i ] + s0 * values[ o0 + i ] + s1 * values[ o1 + i ] + sN * values[ oN + i ];
		return result;

	}

}

function cubicBezier( s, p0, p1, p2, p3 ) {

	const k = 1 - s;
	return k * k * k * p0 + 3 * k * k * s * p1 + 3 * k * s * s * p2 + s * s * s * p3;

}

function cubicBezierSlope( s, p0, p1, p2, p3 ) {

	const k = 1 - s;
	return 3 * k * k * ( p1 - p0 ) + 6 * k * s * ( p2 - p1 ) + 3 * s * s * ( p3 - p2 );

}

// The Bezier parameter whose time is `x`, by Newton steps from the linear guess.
function solveBezierParameter( x, x0, x1, x2, x3 ) {

	let s = ( x - x0 ) / ( x3 - x0 );

	for ( let i = 0; i < 8; i ++ ) {

		const error = cubicBezier( s, x0, x1, x2, x3 ) - x;
		if ( Math.abs( error ) < 1e-10 ) break;
		const slope = cubicBezierSlope( s, x0, x1, x2, x3 );
		if ( Math.abs( slope ) < 1e-10 ) break;
		s = Math.max( 0, Math.min( 1, s - error / slope ) );

	}

	return s;

}

/** Bezier segments with (time, value) tangent handles per key and component; linear without tangents. */
export class BezierInterpolant extends Interpolant {

	interpolate_( i1, t0, t, t1 ) {

		const result = this.resultBuffer, values = this.sampleValues, stride = this.valueSize;
		const offset1 = i1 * stride, offset0 = offset1 - stride, inTangents = this.inTangents, outTangents = this.outTangents;

		if ( ! inTangents || ! outTangents ) {

			const weight1 = ( t - t0 ) / ( t1 - t0 ), weight0 = 1 - weight1;
			for ( let i = 0; i !== stride; ++ i ) result[ i ] = values[ offset0 + i ] * weight0 + values[ offset1 + i ] * weight1;
			return result;

		}

		const tangentStride = stride * 2, i0 = i1 - 1;

		for ( let i = 0; i !== stride; ++ i ) {

			const out = i0 * tangentStride + i * 2, into = i1 * tangentStride + i * 2;
			const s = solveBezierParameter( t, t0, outTangents[ out ], inTangents[ into ], t1 );
			result[ i ] = cubicBezier( s, values[ offset0 + i ], outTangents[ out + 1 ], inTangents[ into + 1 ], values[ offset1 + i ] );

		}

		return result;

	}

}
