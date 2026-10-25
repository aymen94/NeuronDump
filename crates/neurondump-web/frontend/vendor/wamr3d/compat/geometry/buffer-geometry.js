// BufferGeometry: attributes, index, groups, bounds and transforms of an engine geometry.
//
// Part of the geometry classes; `../geometry.js` gathers them.

import * as W from '../../index.js';
import { GEOMETRY, finalizer, lazyUUID, w } from '../engine.js';
import { Box3, Matrix4, Sphere, Vector3 } from '../math.js';
import { withEvents } from '../events.js';
import { BufferAttribute, Float32BufferAttribute, attributeFloats, toUint32 } from './attributes.js';

export const ADOPT = Symbol( 'adopt' );
const _origin = new Vector3(), _up = new Vector3( 0, 1, 0 );

export class BufferGeometry {

	constructor( key, handle ) {

		this.handle = key === ADOPT ? handle : w().createGeometry();
		this.type = 'BufferGeometry';
		this.name = '';
		this.drawRange = { start: 0, count: Infinity };
		this.attributes = {};
		this.index = null;
		this.groups = [];
		this.boundingBox = null;
		this.boundingSphere = null;
		this.userData = {};
		// morph targets by attribute name (position, normal, color), as three.js
		this.morphAttributes = {};
		this.morphTargetsRelative = false;
		// three.js's indirect draw arguments, kept for code that sets them (WebGPU only)
		this.indirect = null;
		this.indirectOffset = 0;
		this._morphSent = new Map();
		this._morphStamp = 0;
		finalizer.register( this, { kind: GEOMETRY, handle: this.handle }, this );
		if ( key === ADOPT ) this._pull();

	}

	// Mirrors the engine's data into `attributes` and `index`.
	_pull( names = [ 'position', 'normal', 'uv', 'tangent' ] ) {

		for ( const name of names ) {

			if ( ! w().hasAttribute( this.handle, name ) ) continue;
			const data = w().getAttribute( this.handle, name );
			const attribute = new BufferAttribute( data, data.length / w().attributeCount( this.handle, name ) );
			attribute._geometry = this;
			attribute._name = name;
			this.attributes[ name ] = attribute;

		}

		const groups = w().getGroups( this.handle );
		this.groups = [];
		for ( let i = 0; i < groups.length; i += 3 ) this.groups.push( { start: groups[ i ], count: groups[ i + 1 ], materialIndex: groups[ i + 2 ] } );

		if ( this.index === null ) {

			const index = w().getIndex( this.handle );

			if ( index.length > 0 ) {

				this.index = new BufferAttribute( index, 1 );
				this.index._geometry = this;
				this.index._name = 'index';

			}

		}

	}

	/** Integer data is stored as floats (normalized when the attribute says so); interleaved data is copied out per attribute. */
	setAttribute( name, attribute ) {

		w().setAttribute( this.handle, name, attributeFloats( attribute ), attribute.itemSize );
		attribute._geometry = this;
		attribute._name = name;
		if ( attribute.isInterleavedBufferAttribute ) attribute.data._users.add( attribute );
		this.attributes[ name ] = attribute;
		return this;

	}

	// Whether the geometry has morph targets for the engine to blend.
	_hasMorphs() {

		const m = this.morphAttributes;
		return ( m.position?.length ?? 0 ) + ( m.normal?.length ?? 0 ) + ( m.color?.length ?? 0 ) > 0;

	}

	// Sends the morph targets that changed to the engine (as `morph.<name>.<i>`
	// attributes); `_morphStamp` changes when any did.
	_syncMorphAttributes() {

		const sent = this._morphSent;

		for ( const name of [ 'position', 'normal', 'color' ] ) {

			const list = this.morphAttributes[ name ] ?? [];

			for ( let i = 0; i < list.length; i ++ ) {

				const attribute = list[ i ], key = `morph.${name}.${i}`, last = sent.get( key );
				if ( last && last[ 0 ] === attribute && last[ 1 ] === attribute.version ) continue;
				w().setAttribute( this.handle, key, attributeFloats( attribute ), attribute.itemSize );
				sent.set( key, [ attribute, attribute.version ] );
				this._morphStamp ++;

			}

			for ( let i = list.length; sent.has( `morph.${name}.${i}` ); i ++ ) {

				w().deleteAttribute( this.handle, `morph.${name}.${i}` );
				sent.delete( `morph.${name}.${i}` );
				this._morphStamp ++;

			}

		}

	}

	getAttribute( name ) {

		return this.attributes[ name ];

	}

	hasAttribute( name ) {

		return this.attributes[ name ] !== undefined;

	}

	setIndex( index ) {

		if ( index === null ) {

			w().clearIndex( this.handle );
			this.index = null;
			return this;

		}

		const attribute = Array.isArray( index ) || ArrayBuffer.isView( index ) ? new BufferAttribute( Uint32Array.from( index ), 1 ) : index;
		w().setIndex( this.handle, toUint32( attribute.array ) );
		attribute._geometry = this;
		attribute._name = 'index';
		this.index = attribute;
		return this;

	}

	getIndex() {

		return this.index;

	}

	setIndirect( indirect, indirectOffset = 0 ) {

		this.indirect = indirect;
		this.indirectOffset = indirectOffset;
		return this;

	}

	getIndirect() {

		return this.indirect;

	}

	addGroup( start, count, materialIndex = 0 ) {

		w().addGroup( this.handle, start, count, materialIndex );
		this.groups.push( { start, count, materialIndex } );

	}

	setDrawRange( start, count ) {

		this.drawRange.start = start;
		this.drawRange.count = count;
		// the engine counts in 32 bits; Infinity means everything
		w().setDrawRange( this.handle, start, Math.min( count, 0xffffffff ) );

	}

	deleteAttribute( name ) {

		w().deleteAttribute( this.handle, name );
		delete this.attributes[ name ];
		return this;

	}

	clearGroups() {

		w().clearGroups( this.handle );
		this.groups = [];

	}

	/** Positions from an array of `Vector2` or `Vector3`, as three.js does. */
	setFromPoints( points ) {

		const position = this.attributes.position;

		if ( position === undefined ) {

			const array = new Float32Array( points.length * 3 );
			points.forEach( ( p, i ) => array.set( [ p.x, p.y, p.z || 0 ], i * 3 ) );
			return this.setAttribute( 'position', new Float32BufferAttribute( array, 3 ) );

		}

		const count = Math.min( points.length, position.count );
		for ( let i = 0; i < count; i ++ ) position.setXYZ( i, points[ i ].x, points[ i ].y, points[ i ].z || 0 );
		if ( points.length > position.count ) console.warn( 'WAMR3D: BufferGeometry: buffer size too small for points data. Use .dispose() and create a new geometry.' );
		position.needsUpdate = true;
		return this;

	}

	/** Copies attributes (as new arrays), index, groups, draw range, bounding volumes and `userData`. */
	copy( source ) {

		if ( source.index === null && this.index !== null ) this.setIndex( null );
		for ( const name of Object.keys( this.attributes ) ) if ( source.attributes[ name ] === undefined ) this.deleteAttribute( name );
		this.name = source.name;
		if ( source.index !== null ) this.setIndex( source.index.clone() );
		for ( const [ name, attribute ] of Object.entries( source.attributes ) ) this.setAttribute( name, attribute.clone() );
		this.clearGroups();
		for ( const group of source.groups ) this.addGroup( group.start, group.count, group.materialIndex );
		this.boundingBox = source.boundingBox && source.boundingBox.clone();
		this.boundingSphere = source.boundingSphere && source.boundingSphere.clone();
		this.setDrawRange( source.drawRange.start, source.drawRange.count );
		this.userData = JSON.parse( JSON.stringify( source.userData ) );
		if ( source.parameters ) this.parameters = { ...source.parameters };
		this.morphTargetsRelative = source.morphTargetsRelative;
		this.morphAttributes = {};
		for ( const [ name, list ] of Object.entries( source.morphAttributes ) ) this.morphAttributes[ name ] = list.map( ( attribute ) => attribute.clone() );
		this.indirect = source.indirect;
		this.indirectOffset = source.indirectOffset;
		return this;

	}

	/** A copy; built-in shapes are rebuilt from their `parameters` and keep their class. */
	clone() {

		const copy = this.parameters ? new this.constructor( ...Object.values( this.parameters ) ) : new this.constructor();
		return copy.copy( this );

	}

	computeVertexNormals() {

		w().computeVertexNormals( this.handle );
		this._pull( [ 'normal' ] );

	}

	computeTangents() {

		w().computeTangents( this.handle );
		this._pull( [ 'tangent' ] );

	}

	computeBoundingBox() {

		const b = w().computeBoundingBox( this.handle );
		this.boundingBox = new Box3( new Vector3( b[ 0 ], b[ 1 ], b[ 2 ] ), new Vector3( b[ 3 ], b[ 4 ], b[ 5 ] ) );

	}

	computeBoundingSphere() {

		const s = w().computeBoundingSphere( this.handle );
		this.boundingSphere = new Sphere( new Vector3( s[ 0 ], s[ 1 ], s[ 2 ] ), s[ 3 ] );

	}

	applyMatrix4( matrix ) {

		const engineMatrix = new W.Matrix4();
		engineMatrix.fromArray( matrix.elements );
		w().applyMatrix4ToGeometry( this.handle, engineMatrix );
		engineMatrix.free();
		this._pull( [ 'position', 'normal', 'tangent' ] );
		if ( this.boundingBox !== null ) this.computeBoundingBox();
		if ( this.boundingSphere !== null ) this.computeBoundingSphere();
		return this;

	}

	applyQuaternion( q ) {

		return this.applyMatrix4( new Matrix4().makeRotationFromQuaternion( q ) );

	}

	/** Rotates the geometry to face `vector`, as an object's `lookAt` turns it. */
	lookAt( vector ) {

		return this.applyMatrix4( new Matrix4().lookAt( vector, _origin, _up ) );

	}

	normalizeNormals() {

		const normals = this.attributes.normal;

		for ( let i = 0; i < normals.count; i ++ ) {

			const x = normals.getX( i ), y = normals.getY( i ), z = normals.getZ( i ), length = Math.hypot( x, y, z ) || 1;
			normals.setXYZ( i, x / length, y / length, z / length );

		}

		normals.needsUpdate = true;

	}

	toJSON() {

		const data = { metadata: { version: 4.7, type: 'BufferGeometry', generator: 'BufferGeometry.toJSON' } };
		data.uuid = this.uuid;
		data.type = this.type;
		if ( this.name !== '' ) data.name = this.name;
		if ( Object.keys( this.userData ).length > 0 ) data.userData = this.userData;

		if ( this.parameters !== undefined ) {

			for ( const key in this.parameters ) if ( this.parameters[ key ] !== undefined ) data[ key ] = this.parameters[ key ];
			return data;

		}

		data.data = { attributes: {} };
		const index = this.index;
		if ( index !== null ) data.data.index = { type: index.array.constructor.name, array: Array.from( index.array ) };
		for ( const [ name, attribute ] of Object.entries( this.attributes ) ) data.data.attributes[ name ] = attribute.toJSON( data.data );
		const morphAttributes = {};
		let hasMorphAttributes = false;

		for ( const [ name, list ] of Object.entries( this.morphAttributes ) ) {

			morphAttributes[ name ] = list.map( ( attribute ) => attribute.toJSON( data.data ) );
			if ( list.length > 0 ) hasMorphAttributes = true;

		}

		if ( hasMorphAttributes ) {

			data.data.morphAttributes = morphAttributes;
			data.data.morphTargetsRelative = this.morphTargetsRelative;

		}

		if ( this.groups.length > 0 ) data.data.groups = JSON.parse( JSON.stringify( this.groups ) );
		const sphere = this.boundingSphere;
		if ( sphere !== null ) data.data.boundingSphere = { center: sphere.center.toArray(), radius: sphere.radius };
		return data;

	}

	translate( x, y, z ) {

		return this.applyMatrix4( new Matrix4().makeTranslation( x, y, z ) );

	}

	scale( x, y, z ) {

		return this.applyMatrix4( new Matrix4().makeScale( x, y, z ) );

	}

	rotateX( angle ) {

		return this.applyMatrix4( new Matrix4().makeRotationX( angle ) );

	}

	rotateY( angle ) {

		return this.applyMatrix4( new Matrix4().makeRotationY( angle ) );

	}

	rotateZ( angle ) {

		return this.applyMatrix4( new Matrix4().makeRotationZ( angle ) );

	}

	center() {

		w().centerGeometry( this.handle );
		this._pull( [ 'position' ] );
		return this;

	}

	toNonIndexed() {

		return new BufferGeometry( ADOPT, w().toNonIndexed( this.handle ) );

	}

	dispose() {

		finalizer.unregister( this );
		w().disposeGeometry( this.handle );
		this.dispatchEvent( { type: 'dispose' } );

	}

}

BufferGeometry.prototype.isBufferGeometry = true;
withEvents( BufferGeometry );
lazyUUID( BufferGeometry );

/** A geometry drawn `instanceCount` times; its `InstancedBufferAttribute`s change per instance. */
export class InstancedBufferGeometry extends BufferGeometry {

	constructor() {

		super();
		this.type = 'InstancedBufferGeometry';
		this.instanceCount = Infinity;

	}

	copy( source ) {

		super.copy( source );
		this.instanceCount = source.instanceCount;
		return this;

	}

	toJSON() {

		const data = super.toJSON();
		data.instanceCount = this.instanceCount;
		data.isInstancedBufferGeometry = true;
		return data;

	}

}

InstancedBufferGeometry.prototype.isInstancedBufferGeometry = true;
