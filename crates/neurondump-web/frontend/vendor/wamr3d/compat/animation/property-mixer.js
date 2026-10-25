// PropertyMixer: blending the values that several actions write into one property.
//
// Part of the keyframe animation system; `../animation.js` gathers it.

import { Quaternion } from '../values.js';

// Blends the values the actions write to one binding. The buffer holds, `valueSize` each:
// the incoming value, two accumulators (alternating between frames), the original value,
// the additive accumulator, and for quaternions a work area.
export class PropertyMixer {

	constructor( binding, typeName, valueSize ) {

		this.binding = binding;
		this.valueSize = valueSize;
		let mixFunction, mixFunctionAdditive, setIdentity;

		switch ( typeName ) {

			case 'quaternion':
				mixFunction = this._slerp;
				mixFunctionAdditive = this._slerpAdditive;
				setIdentity = this._setAdditiveIdentityQuaternion;
				this.buffer = new Float64Array( valueSize * 6 );
				this._workIndex = 5;
				break;

			case 'string':
			case 'bool':
				mixFunction = mixFunctionAdditive = this._select;
				setIdentity = this._setAdditiveIdentityOther;
				this.buffer = new Array( valueSize * 5 );
				break;

			default:
				mixFunction = this._lerp;
				mixFunctionAdditive = this._lerpAdditive;
				setIdentity = this._setAdditiveIdentityNumeric;
				this.buffer = new Float64Array( valueSize * 5 );

		}

		this._mixBufferRegion = mixFunction;
		this._mixBufferRegionAdditive = mixFunctionAdditive;
		this._setIdentity = setIdentity;
		this._origIndex = 3;
		this._addIndex = 4;
		this.cumulativeWeight = 0;
		this.cumulativeWeightAdditive = 0;
		this.useCount = 0;
		this.referenceCount = 0;

	}

	accumulate( accuIndex, weight ) {

		const buffer = this.buffer, stride = this.valueSize, offset = accuIndex * stride + stride;
		let currentWeight = this.cumulativeWeight;

		if ( currentWeight === 0 ) {

			for ( let i = 0; i !== stride; ++ i ) buffer[ offset + i ] = buffer[ i ];
			currentWeight = weight;

		} else {

			currentWeight += weight;
			this._mixBufferRegion( buffer, offset, 0, weight / currentWeight, stride );

		}

		this.cumulativeWeight = currentWeight;

	}

	accumulateAdditive( weight ) {

		const buffer = this.buffer, stride = this.valueSize, offset = stride * this._addIndex;
		if ( this.cumulativeWeightAdditive === 0 ) this._setIdentity();
		this._mixBufferRegionAdditive( buffer, offset, 0, weight, stride );
		this.cumulativeWeightAdditive += weight;

	}

	// Writes the blend, filled up with the original value below full weight, and only when
	// it differs from the other accumulator's (the last frame's) value.
	apply( accuIndex ) {

		const stride = this.valueSize, buffer = this.buffer, offset = accuIndex * stride + stride;
		const weight = this.cumulativeWeight, weightAdditive = this.cumulativeWeightAdditive, binding = this.binding;
		this.cumulativeWeight = 0;
		this.cumulativeWeightAdditive = 0;
		if ( weight < 1 ) this._mixBufferRegion( buffer, offset, stride * this._origIndex, 1 - weight, stride );
		if ( weightAdditive > 0 ) this._mixBufferRegionAdditive( buffer, offset, this._addIndex * stride, 1, stride );

		for ( let i = stride, e = stride + stride; i !== e; ++ i ) {

			if ( buffer[ i ] !== buffer[ i + stride ] ) {

				binding.setValue( buffer, offset );
				break;

			}

		}

	}

	saveOriginalState() {

		const buffer = this.buffer, stride = this.valueSize, original = stride * this._origIndex;
		this.binding.getValue( buffer, original );
		// both accumulators start at the original value
		for ( let i = stride, e = original; i !== e; ++ i ) buffer[ i ] = buffer[ original + ( i % stride ) ];
		this._setIdentity();
		this.cumulativeWeight = 0;
		this.cumulativeWeightAdditive = 0;

	}

	restoreOriginalState() {

		this.binding.setValue( this.buffer, this.valueSize * 3 );

	}

	_setAdditiveIdentityNumeric() {

		const start = this._addIndex * this.valueSize;
		this.buffer.fill( 0, start, start + this.valueSize );

	}

	_setAdditiveIdentityQuaternion() {

		this._setAdditiveIdentityNumeric();
		this.buffer[ this._addIndex * this.valueSize + 3 ] = 1;

	}

	_setAdditiveIdentityOther() {

		const start = this._origIndex * this.valueSize, target = this._addIndex * this.valueSize;
		for ( let i = 0; i < this.valueSize; i ++ ) this.buffer[ target + i ] = this.buffer[ start + i ];

	}

	_select( buffer, dstOffset, srcOffset, t, stride ) {

		if ( t >= 0.5 ) for ( let i = 0; i !== stride; ++ i ) buffer[ dstOffset + i ] = buffer[ srcOffset + i ];

	}

	_slerp( buffer, dstOffset, srcOffset, t ) {

		Quaternion.slerpFlat( buffer, dstOffset, buffer, dstOffset, buffer, srcOffset, t );

	}

	_slerpAdditive( buffer, dstOffset, srcOffset, t, stride ) {

		const work = this._workIndex * stride;
		Quaternion.multiplyQuaternionsFlat( buffer, work, buffer, dstOffset, buffer, srcOffset );
		Quaternion.slerpFlat( buffer, dstOffset, buffer, dstOffset, buffer, work, t );

	}

	_lerp( buffer, dstOffset, srcOffset, t, stride ) {

		const s = 1 - t;

		for ( let i = 0; i !== stride; ++ i ) {

			const j = dstOffset + i;
			buffer[ j ] = buffer[ j ] * s + buffer[ srcOffset + i ] * t;

		}

	}

	_lerpAdditive( buffer, dstOffset, srcOffset, t, stride ) {

		for ( let i = 0; i !== stride; ++ i ) {

			const j = dstOffset + i;
			buffer[ j ] = buffer[ j ] + buffer[ srcOffset + i ] * t;

		}

	}

}
