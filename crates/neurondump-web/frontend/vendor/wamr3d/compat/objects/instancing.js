// InstancedMesh and its per-instance matrices and colours.
//
// Part of the drawable objects; `../objects.js` gathers them.

import { hosted, instanceSources, w } from '../engine.js';
import { Box3, Matrix4, Sphere, boxNumbers, expandByTransformedBox, setBox } from '../math.js';
import { InstancedBufferAttribute } from '../geometry.js';
import { DataTexture } from '../textures.js';
import { FloatType, RedFormat } from '../constants.js';
import { Mesh } from './mesh.js';

export { InstancedBufferAttribute };

// What the renderer reads for an instanced mesh. It does not refer to the mesh, so the
// mesh can still be collected.
class InstanceData {

	constructor( count ) {

		const matrices = new Float32Array( count * 16 );
		for ( let i = 0; i < matrices.length; i += 16 ) matrices[ i ] = matrices[ i + 5 ] = matrices[ i + 10 ] = matrices[ i + 15 ] = 1;
		this.matrices = new InstancedBufferAttribute( matrices, 16 );
		this.colors = null;

	}

	get matrix() {

		return this.matrices.array;

	}

	get matrixVersion() {

		return this.matrices.version;

	}

	get color() {

		return this.colors ? this.colors.array : null;

	}

	get colorVersion() {

		return this.colors ? this.colors.version : 0;

	}

}

let nextInstanceId = 1;
const _matrix = new Matrix4();

/**
 * Draws one geometry `count` times, each with its own matrix (and optionally colour),
 * in a single draw call. Instances are not culled one by one. `Raycaster` tests each
 * instance and reports its `instanceId`.
 */
export class InstancedMesh extends Mesh {

	constructor( geometry, material, count ) {

		super( geometry, material );
		this.type = 'InstancedMesh';
		this._instances = new InstanceData( count );
		this._capacity = count;
		this._count = count;
		this.boundingBox = null;
		this.boundingSphere = null;
		// per-instance morph weights (`setMorphAt`), as three.js stores them
		this.morphTexture = null;
		hosted.count ++;
		this._token.instances = nextInstanceId ++;
		instanceSources.set( this._token.instances, this._instances );
		w().setInstances( this.handle, this._token.instances, count );

	}

	get instanceMatrix() {

		return this._instances.matrices;

	}

	get instanceColor() {

		return this._instances.colors;

	}

	set instanceColor( value ) {

		this._instances.colors = value;

	}

	/** How many of the instances are drawn; at most the count given to the constructor. */
	get count() {

		return this._count;

	}

	set count( value ) {

		this._count = Math.max( 0, Math.min( this._capacity, value ) );
		w().setInstances( this.handle, this._token.instances, this._count );

	}

	setMatrixAt( index, matrix ) {

		matrix.toArray( this._instances.matrices.array, index * 16 );

	}

	/** The local box around every drawn instance; kept in `boundingBox`. */
	computeBoundingBox() {

		const geometry = this.geometry, matrices = this._instances.matrices.array;
		if ( geometry.boundingBox === null ) geometry.computeBoundingBox();
		const box = boxNumbers( geometry.boundingBox ), bounds = [ Infinity, Infinity, Infinity, - Infinity, - Infinity, - Infinity ];
		for ( let i = 0; i < this._count; i ++ ) expandByTransformedBox( bounds, box, matrices, i * 16 );
		setBox( this.boundingBox ??= new Box3(), bounds );

	}

	/** The local sphere around every drawn instance; kept in `boundingSphere`. */
	computeBoundingSphere() {

		this.computeBoundingBox();
		( this.boundingSphere ??= new Sphere() ).set( this.boundingBox.getCenter( this.boundingSphere.center ), 0 );
		const geometry = this.geometry, matrices = this._instances.matrices.array;
		if ( geometry.boundingSphere === null ) geometry.computeBoundingSphere();
		const sphere = new Sphere();
		let radius = 0;

		for ( let i = 0; i < this._count; i ++ ) {

			sphere.copy( geometry.boundingSphere ).applyMatrix4( this.getMatrixAt( i, _matrix ) );
			radius = Math.max( radius, this.boundingSphere.center.distanceTo( sphere.center ) + sphere.radius );

		}

		this.boundingSphere.radius = radius;

	}

	/**
	 * Stores the morph weights of one instance (`object.morphTargetInfluences` of a mesh
	 * with the same geometry) in `morphTexture`. The renderer blends the targets once per
	 * mesh, with this mesh's `morphTargetInfluences`.
	 */
	setMorphAt( index, object ) {

		const influences = object.morphTargetInfluences, length = influences.length + 1;

		if ( this.morphTexture === null ) {

			this.morphTexture = new DataTexture( new Float32Array( length * this._capacity ), length, this._capacity, RedFormat, FloatType );

		}

		const array = this.morphTexture.image.data;
		let sum = 0;
		for ( const value of influences ) sum += value;
		const base = this.geometry.morphTargetsRelative ? 1 : 1 - sum;
		array[ length * index ] = base;
		array.set( influences, length * index + 1 );
		this.morphTexture.needsUpdate = true;

	}

	getMorphAt( index, object ) {

		const influences = object.morphTargetInfluences, array = this.morphTexture.image.data, length = influences.length + 1;
		for ( let i = 0; i < influences.length; i ++ ) influences[ i ] = array[ length * index + 1 + i ];

	}

	getMatrixAt( index, matrix ) {

		return matrix.fromArray( this._instances.matrices.array, index * 16 );

	}

	setColorAt( index, color ) {

		this._instances.colors ??= new InstancedBufferAttribute( new Float32Array( this._capacity * 3 ).fill( 1 ), 3 );
		color.toArray( this._instances.colors.array, index * 3 );

	}

	getColorAt( index, color ) {

		const array = this._instances.colors ? this._instances.colors.array : null;
		return array ? color.setRGB( array[ index * 3 ], array[ index * 3 + 1 ], array[ index * 3 + 2 ] ) : color.setRGB( 1, 1, 1 );

	}

	copy( source, recursive ) {

		super.copy( source, recursive );
		this._instances.matrices.copy( source.instanceMatrix );
		if ( source.instanceColor !== null ) this._instances.colors = source.instanceColor.clone();
		if ( source.morphTexture !== null ) this.morphTexture = source.morphTexture.clone();
		this._capacity = source._capacity;
		this.count = source.count;
		if ( source.boundingBox !== null ) this.boundingBox = source.boundingBox.clone();
		if ( source.boundingSphere !== null ) this.boundingSphere = source.boundingSphere.clone();
		return this;

	}

	dispose() {

		instanceSources.delete( this._token.instances );
		this.dispatchEvent( { type: 'dispose' } );
		super.dispose();

	}

}

InstancedMesh.prototype.isInstancedMesh = true;

/** One connected line through the geometry's vertices. Lines are one pixel wide. */
