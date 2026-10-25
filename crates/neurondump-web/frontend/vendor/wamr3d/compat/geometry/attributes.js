// Buffer attributes, interleaved buffers and the typed attribute classes.
//
// Part of the geometry classes; `../geometry.js` gathers them.

import * as W from '../../index.js';
import { lazyUUID, w } from '../engine.js';
import { withEvents } from '../events.js';
import { FloatType, StaticDrawUsage } from '../constants.js';

function toFloat32( array ) {

	return array instanceof Float32Array ? array : Float32Array.from( array );

}

export function toUint32( array ) {

	return array instanceof Uint32Array ? array : Uint32Array.from( array );

}

// Integer arrays marked `normalized` read as 0..1 (unsigned) or -1..1 (signed), as the GPU
// reads them: the divisor, and whether the type is signed.
const NORMALIZED = new Map( [
	[ Uint8Array, [ 255, false ] ], [ Uint8ClampedArray, [ 255, false ] ], [ Int8Array, [ 127, true ] ],
	[ Uint16Array, [ 65535, false ] ], [ Int16Array, [ 32767, true ] ], [ Uint32Array, [ 4294967295, false ] ], [ Int32Array, [ 2147483647, true ] ]
] );

function denormalize( value, array ) {

	const [ divisor, signed ] = NORMALIZED.get( array.constructor ) ?? [ 1, false ];
	return signed ? Math.max( value / divisor, - 1 ) : value / divisor;

}

function normalize( value, array ) {

	const [ divisor ] = NORMALIZED.get( array.constructor ) ?? [ 1 ];
	return Math.round( value * divisor );

}

// The floats the engine stores for an attribute: its items in order (gathered from an
// interleaved buffer), normalized values mapped as the GPU would.
export function attributeFloats( attribute ) {

	const interleaved = attribute.isInterleavedBufferAttribute === true;
	const source = interleaved ? attribute.data.array : attribute.array;
	const scale = attribute.normalized && NORMALIZED.has( source.constructor );
	if ( ! interleaved && ! scale ) return toFloat32( source );

	const { itemSize, count } = attribute, out = new Float32Array( count * itemSize );
	const stride = interleaved ? attribute.data.stride : itemSize, offset = interleaved ? attribute.offset : 0;

	for ( let i = 0, o = 0; i < count; i ++ ) {

		for ( let k = 0, j = i * stride + offset; k < itemSize; k ++, j ++ ) out[ o ++ ] = scale ? denormalize( source[ j ], source ) : source[ j ];

	}

	return out;

}

// Component accessors shared by `BufferAttribute` and `InterleavedBufferAttribute`;
// `_at( index )` is where an item starts in `array`. Normalized values are converted as
// in three.js.
const ACCESSORS = {

	getComponent( index, component ) {

		const value = this.array[ this._at( index ) + component ];
		return this.normalized ? denormalize( value, this.array ) : value;

	},

	setComponent( index, component, value ) {

		this.array[ this._at( index ) + component ] = this.normalized ? normalize( value, this.array ) : value;
		return this;

	},

	getX( index ) {

		return this.getComponent( index, 0 );

	},

	getY( index ) {

		return this.getComponent( index, 1 );

	},

	getZ( index ) {

		return this.getComponent( index, 2 );

	},

	getW( index ) {

		return this.getComponent( index, 3 );

	},

	setX( index, x ) {

		return this.setComponent( index, 0, x );

	},

	setY( index, y ) {

		return this.setComponent( index, 1, y );

	},

	setZ( index, z ) {

		return this.setComponent( index, 2, z );

	},

	setW( index, w_ ) {

		return this.setComponent( index, 3, w_ );

	},

	setXY( index, x, y ) {

		return this.setComponent( index, 0, x ).setComponent( index, 1, y );

	},

	setXYZ( index, x, y, z ) {

		return this.setComponent( index, 0, x ).setComponent( index, 1, y ).setComponent( index, 2, z );

	},

	setXYZW( index, x, y, z, w_ ) {

		return this.setComponent( index, 0, x ).setComponent( index, 1, y ).setComponent( index, 2, z ).setComponent( index, 3, w_ );

	},

	applyMatrix4( m ) {

		return transformItems( this, m.elements, 4, 1 );

	},

	applyNormalMatrix( m ) {

		transformItems( this, m.elements, 3, 0 );
		return normalizeItems( this );

	},

	transformDirection( m ) {

		transformItems( this, m.elements, 4, 0 );
		return normalizeItems( this );

	}

};

// Items as points (`w_` 1) or directions (`w_` 0) through a 4x4 or 3x3 column-major
// matrix; a 4x4 point is divided by its w, as Vector3.applyMatrix4 does.
function transformItems( attribute, e, size, w_ ) {

	for ( let i = 0; i < attribute.count; i ++ ) {

		const x = attribute.getX( i ), y = attribute.getY( i ), z = attribute.getZ( i );
		let nx, ny, nz;

		if ( size === 3 ) {

			nx = e[ 0 ] * x + e[ 3 ] * y + e[ 6 ] * z;
			ny = e[ 1 ] * x + e[ 4 ] * y + e[ 7 ] * z;
			nz = e[ 2 ] * x + e[ 5 ] * y + e[ 8 ] * z;

		} else {

			const d = w_ === 1 ? 1 / ( e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] ) : 1;
			nx = ( e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] * w_ ) * d;
			ny = ( e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] * w_ ) * d;
			nz = ( e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] * w_ ) * d;

		}

		attribute.setXYZ( i, nx, ny, nz );

	}

	return attribute;

}

function normalizeItems( attribute ) {

	for ( let i = 0; i < attribute.count; i ++ ) {

		const x = attribute.getX( i ), y = attribute.getY( i ), z = attribute.getZ( i ), length = Math.hypot( x, y, z ) || 1;
		attribute.setXYZ( i, x / length, y / length, z / length );

	}

	return attribute;

}

// Usage and update ranges, shared by attributes and interleaved buffers. The engine
// receives the whole array on `needsUpdate`; ranges are kept for code that reads them.
const UPDATE_RANGES = {

	setUsage( value ) {

		this.usage = value;
		return this;

	},

	addUpdateRange( start, count ) {

		this.updateRanges.push( { start, count } );

	},

	clearUpdateRanges() {

		this.updateRanges.length = 0;

	},

	onUpload( callback ) {

		this.onUploadCallback = callback;
		return this;

	},

	onUploadCallback() {}

};

// Sends an attribute's current data to the geometry it is set on.
function pushAttribute( attribute ) {

	if ( attribute._geometry === null ) return;
	const handle = attribute._geometry.handle;

	if ( attribute._name === 'index' ) {

		w().setIndex( handle, toUint32( attribute.array ) );

	} else {

		W.attributeView( w(), handle, attribute._name ).set( attributeFloats( attribute ) );
		w().markAttributeNeedsUpdate( handle, attribute._name );

	}

}

export class BufferAttribute {

	constructor( array, itemSize, normalized = false ) {

		if ( Array.isArray( array ) ) throw new TypeError( 'WAMR3D: BufferAttribute: array should be a Typed Array.' );
		this.name = '';
		this.array = array;
		this.itemSize = itemSize;
		this.normalized = normalized;
		this.count = array !== undefined ? array.length / itemSize : 0;
		this.usage = StaticDrawUsage;
		this.updateRanges = [];
		this.gpuType = FloatType;
		this.version = 0;
		this._geometry = null;
		this._name = '';

	}

	/** Set to true after changing `array` to send the new data to the engine. */
	set needsUpdate( value ) {

		if ( ! value ) return;
		this.version ++;
		pushAttribute( this );
		this.onUploadCallback();

	}

	_at( index ) {

		return index * this.itemSize;

	}

	copy( source ) {

		this.name = source.name;
		this.array = new source.array.constructor( source.array );
		this.itemSize = source.itemSize;
		this.count = source.count;
		this.normalized = source.normalized;
		this.usage = source.usage;
		this.gpuType = source.gpuType;
		return this;

	}

	copyAt( index1, attribute, index2 ) {

		index1 *= this.itemSize;
		index2 *= attribute.itemSize;
		for ( let i = 0; i < this.itemSize; i ++ ) this.array[ index1 + i ] = attribute.array[ index2 + i ];
		return this;

	}

	copyArray( array ) {

		this.array.set( array );
		return this;

	}

	applyMatrix3( m ) {

		if ( this.itemSize === 2 ) {

			const e = m.elements;

			for ( let i = 0; i < this.count; i ++ ) {

				const x = this.getX( i ), y = this.getY( i );
				this.setXY( i, e[ 0 ] * x + e[ 3 ] * y + e[ 6 ], e[ 1 ] * x + e[ 4 ] * y + e[ 7 ] );

			}

		} else if ( this.itemSize === 3 ) {

			transformItems( this, m.elements, 3, 0 );

		}

		return this;

	}

	set( value, offset = 0 ) {

		this.array.set( value, offset );
		return this;

	}

	/** A new attribute with a copy of the array, not attached to any geometry. */
	clone() {

		return new this.constructor( this.array, this.itemSize ).copy( this );

	}

	toJSON() {

		const data = { itemSize: this.itemSize, type: this.array.constructor.name, array: Array.from( this.array ), normalized: this.normalized };
		if ( this.name !== '' ) data.name = this.name;
		if ( this.usage !== StaticDrawUsage ) data.usage = this.usage;
		return data;

	}

	dispose() {

		this.dispatchEvent( { type: 'dispose' } );

	}

}

Object.assign( BufferAttribute.prototype, ACCESSORS, UPDATE_RANGES );
BufferAttribute.prototype.isBufferAttribute = true;
withEvents( BufferAttribute );

/**
 * Several attributes packed in one array, `stride` numbers per vertex. The engine keeps
 * each attribute separately: setting `needsUpdate` sends every attribute that uses it again.
 */
export class InterleavedBuffer {

	constructor( array, stride ) {

		this.array = array;
		this.stride = stride;
		this.count = array !== undefined ? array.length / stride : 0;
		this.usage = StaticDrawUsage;
		this.updateRanges = [];
		this.version = 0;
		this._users = new Set();

	}

	set needsUpdate( value ) {

		if ( ! value ) return;
		this.version ++;
		for ( const attribute of this._users ) pushAttribute( attribute );
		this.onUploadCallback();

	}

	copy( source ) {

		this.array = new source.array.constructor( source.array );
		this.count = source.count;
		this.stride = source.stride;
		this.usage = source.usage;
		return this;

	}

	copyAt( index1, attribute, index2 ) {

		index1 *= this.stride;
		index2 *= attribute.stride;
		for ( let i = 0; i < this.stride; i ++ ) this.array[ index1 + i ] = attribute.array[ index2 + i ];
		return this;

	}

	set( value, offset = 0 ) {

		this.array.set( value, offset );
		return this;

	}

	clone() {

		return new this.constructor( this.array.slice(), this.stride ).copy( this );

	}

	toJSON( data = {} ) {

		data.arrayBuffers ??= {};
		const uuid = this.uuid, array = this.array;
		if ( data.arrayBuffers[ uuid ] === undefined ) data.arrayBuffers[ uuid ] = Array.from( new Uint32Array( array.buffer.slice( array.byteOffset, array.byteOffset + array.byteLength ) ) );
		return { uuid, buffer: uuid, type: array.constructor.name, stride: this.stride };

	}

}

Object.assign( InterleavedBuffer.prototype, UPDATE_RANGES );
InterleavedBuffer.prototype.isInterleavedBuffer = true;
lazyUUID( InterleavedBuffer );

/** `itemSize` numbers per vertex, starting `offset` numbers into each vertex of `interleavedBuffer`. */
export class InterleavedBufferAttribute {

	constructor( interleavedBuffer, itemSize, offset, normalized = false ) {

		this.name = '';
		this.data = interleavedBuffer;
		this.itemSize = itemSize;
		this.offset = offset;
		this.normalized = normalized;
		this._geometry = null;
		this._name = '';

	}

	get count() {

		return this.data.count;

	}

	get array() {

		return this.data.array;

	}

	set needsUpdate( value ) {

		this.data.needsUpdate = value;

	}

	_at( index ) {

		return index * this.data.stride + this.offset;

	}

	/** A plain `BufferAttribute` with this attribute's values, as three.js's `clone()` without arguments gives. */
	clone() {

		const array = new this.array.constructor( this.count * this.itemSize );
		for ( let i = 0, o = 0; i < this.count; i ++ ) for ( let k = 0; k < this.itemSize; k ++ ) array[ o ++ ] = this.array[ this._at( i ) + k ];
		return new BufferAttribute( array, this.itemSize, this.normalized );

	}

	/** Without `data`, the values as a plain attribute's JSON; with it, a reference to the shared buffer. */
	toJSON( data ) {

		if ( data === undefined ) return this.clone().toJSON();
		data.interleavedBuffers ??= {};
		if ( data.interleavedBuffers[ this.data.uuid ] === undefined ) data.interleavedBuffers[ this.data.uuid ] = this.data.toJSON( data );
		return { isInterleavedBufferAttribute: true, itemSize: this.itemSize, data: this.data.uuid, offset: this.offset, normalized: this.normalized };

	}

}

Object.assign( InterleavedBufferAttribute.prototype, ACCESSORS );
InterleavedBufferAttribute.prototype.isInterleavedBufferAttribute = true;

function typedAttribute( TypedArray ) {

	return class extends BufferAttribute {

		constructor( array, itemSize, normalized ) {

			super( array instanceof TypedArray ? array : new TypedArray( array ), itemSize, normalized );

		}

	};

}

export const Float32BufferAttribute = typedAttribute( Float32Array );
export const Int8BufferAttribute = typedAttribute( Int8Array ), Uint8BufferAttribute = typedAttribute( Uint8Array ), Uint8ClampedBufferAttribute = typedAttribute( Uint8ClampedArray );
export const Int16BufferAttribute = typedAttribute( Int16Array ), Uint16BufferAttribute = typedAttribute( Uint16Array );
export const Int32BufferAttribute = typedAttribute( Int32Array ), Uint32BufferAttribute = typedAttribute( Uint32Array );

/** Per-instance values of an `InstancedBufferGeometry` (or an instanced mesh's matrices and colours). */
export class InstancedBufferAttribute extends BufferAttribute {

	constructor( array, itemSize, normalized, meshPerAttribute = 1 ) {

		super( array, itemSize, normalized );
		this.meshPerAttribute = meshPerAttribute;

	}

	copy( source ) {

		super.copy( source );
		this.meshPerAttribute = source.meshPerAttribute;
		return this;

	}

	toJSON() {

		const data = super.toJSON();
		data.meshPerAttribute = this.meshPerAttribute;
		data.isInstancedBufferAttribute = true;
		return data;

	}

}

InstancedBufferAttribute.prototype.isInstancedBufferAttribute = true;

export class InstancedInterleavedBuffer extends InterleavedBuffer {

	constructor( array, stride, meshPerAttribute = 1 ) {

		super( array, stride );
		this.meshPerAttribute = meshPerAttribute;

	}

	copy( source ) {

		super.copy( source );
		this.meshPerAttribute = source.meshPerAttribute;
		return this;

	}

	toJSON( data ) {

		const json = super.toJSON( data );
		json.isInstancedInterleavedBuffer = true;
		json.meshPerAttribute = this.meshPerAttribute;
		return json;

	}

}

InstancedInterleavedBuffer.prototype.isInstancedInterleavedBuffer = true;
