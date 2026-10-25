// Bones, skeletons and skinned meshes.
//
// Part of the drawable objects; `../objects.js` gathers them.

import { w } from '../engine.js';
import { Box3, Matrix4, Quaternion, Sphere, Vector3, Vector4 } from '../math.js';
import * as W from '../../index.js';
import { Object3D } from '../object3d.js';
import { Mesh } from './mesh.js';

/** A joint of a skeleton: an ordinary object that skinned vertices follow. */
export class Bone extends Object3D {

	constructor() {

		super();
		this.type = 'Bone';

	}

}

Bone.prototype.isBone = true;

const _offset = new Matrix4(), _inverse = new Matrix4(), _position = new Vector3(), _quaternion = new Quaternion(), _scale = new Vector3();

/**
 * Bones and their inverse bind matrices. The engine computes the skinning matrices
 * from the bones' world matrices each frame; `update()` fills `boneMatrices` for code
 * that reads them.
 */
export class Skeleton {

	constructor( bones = [], boneInverses = [] ) {

		this.uuid = W.MathUtils.generateUUID();
		this.bones = bones.slice( 0 );
		this.boneInverses = boneInverses;
		this.boneMatrices = null;
		this.boneTexture = null;
		this._version = 0;
		this.init();

	}

	init() {

		const bones = this.bones, inverses = this.boneInverses;
		this.boneMatrices = new Float32Array( bones.length * 16 );

		if ( inverses.length === 0 ) {

			this.calculateInverses();

		} else if ( bones.length !== inverses.length ) {

			console.warn( 'WAMR3D: Skeleton: the number of inverse bone matrices does not match the number of bones.' );
			this.boneInverses = bones.map( () => new Matrix4() );

		}

		this._version ++;

	}

	/** The inverse bind matrices from the bones' current world matrices. */
	calculateInverses() {

		this.boneInverses = this.bones.map( ( bone ) => ( bone ? bone.matrixWorld.clone().invert() : new Matrix4() ) );
		this._version ++;

	}

	/** Moves the bones back to the bind pose. */
	pose() {

		const worlds = new Map();
		this.bones.forEach( ( bone, i ) => {

			if ( bone ) worlds.set( bone, this.boneInverses[ i ].clone().invert() );

		} );

		for ( const bone of this.bones ) {

			if ( ! bone ) continue;
			const parent = bone.parent, world = worlds.get( bone );
			const local = parent ? _offset.copy( worlds.get( parent ) ?? parent.matrixWorld ).invert().multiply( world ) : _offset.copy( world );
			local.decompose( _position, _quaternion, _scale );
			bone.position.copy( _position );
			bone.quaternion.copy( _quaternion );
			bone.scale.copy( _scale );

		}

	}

	/** Fills `boneMatrices` with each bone's world matrix times its inverse bind matrix. */
	update() {

		const matrices = this.boneMatrices;

		this.bones.forEach( ( bone, i ) => {

			_offset.multiplyMatrices( bone ? bone.matrixWorld : _inverse.identity(), this.boneInverses[ i ] );
			_offset.toArray( matrices, i * 16 );

		} );

	}

	clone() {

		return new Skeleton( this.bones, this.boneInverses );

	}

	/** The bones are drawn from a texture the renderer keeps; there is nothing to make. */
	computeBoneTexture() {

		return this;

	}

	getBoneByName( name ) {

		return this.bones.find( ( bone ) => bone.name === name );

	}

	dispose() {}

}

/**
 * A mesh deformed by the bones of its `skeleton`, weighted by the geometry's `skinIndex`
 * and `skinWeight` attributes. Skinned on the GPU; raycasts test the posed triangles.
 * Skinned meshes are not frustum-culled.
 */
export class SkinnedMesh extends Mesh {

	constructor( geometry, material ) {

		super( geometry, material );
		this.type = 'SkinnedMesh';
		this.bindMode = 'attached';
		this.bindMatrix = new Matrix4();
		this.bindMatrixInverse = new Matrix4();
		this.skeleton = undefined;
		this.boundingBox = null;
		this.boundingSphere = null;
		this._skinKey = '';

	}

	bind( skeleton, bindMatrix ) {

		this.skeleton = skeleton;

		if ( bindMatrix === undefined ) {

			this.updateMatrixWorld( true );
			this.skeleton.calculateInverses();
			bindMatrix = this.matrixWorld;

		}

		this.bindMatrix.copy( bindMatrix );
		this.bindMatrixInverse.copy( bindMatrix ).invert();
		this._skinKey = '';

	}

	pose() {

		this.skeleton.pose();

	}

	/** Scales each vertex's weights to sum to 1. */
	normalizeSkinWeights() {

		const weights = this.geometry.attributes.skinWeight, vector = new Vector4();

		for ( let i = 0; i < weights.count; i ++ ) {

			vector.fromBufferAttribute( weights, i );
			const scale = 1.0 / vector.manhattanLength();
			if ( scale !== Infinity ) vector.multiplyScalar( scale );
			else vector.set( 1, 0, 0, 0 );
			weights.setXYZW( i, vector.x, vector.y, vector.z, vector.w );

		}

		weights.needsUpdate = true;

	}

	// Sends the skeleton when it or the bind matrices changed. Called by the renderer.
	_syncSkin() {

		const skeleton = this.skeleton;
		if ( ! skeleton ) return;
		const bindMatrix = this.bindMatrix.elements;
		const key = skeleton.uuid + ':' + skeleton._version + ':' + skeleton.bones.length + ':' + this.bindMode + ':' + bindMatrix.join( ',' );
		if ( key === this._skinKey ) return;
		this._skinKey = key;
		const bones = BigUint64Array.from( skeleton.bones, ( bone ) => ( bone ? bone.handle : 0n ) );
		const inverses = new Float32Array( skeleton.bones.length * 16 );
		skeleton.boneInverses.forEach( ( matrix, i ) => matrix.toArray( inverses, i * 16 ) );
		w().setSkin( this.handle, bones, inverses, Float32Array.from( bindMatrix ), Float32Array.from( this.bindMatrixInverse.elements ), this.bindMode === 'detached' );

	}

	/** The position of vertex `index` in the current pose, in the mesh's local space (three.js `applyBoneTransform`). */
	applyBoneTransform( index, target ) {

		this._syncSkin();
		const matrices = w().skinMatrices( this.handle ), geometry = this.geometry;
		const skinIndex = geometry.attributes.skinIndex, skinWeight = geometry.attributes.skinWeight;
		const x = target.x, y = target.y, z = target.z;
		let px = 0, py = 0, pz = 0;

		for ( let k = 0; k < 4; k ++ ) {

			const weight = skinWeight.getComponent( index, k );
			if ( weight === 0 ) continue;
			const o = skinIndex.getComponent( index, k ) * 16;
			px += weight * ( matrices[ o ] * x + matrices[ o + 4 ] * y + matrices[ o + 8 ] * z + matrices[ o + 12 ] );
			py += weight * ( matrices[ o + 1 ] * x + matrices[ o + 5 ] * y + matrices[ o + 9 ] * z + matrices[ o + 13 ] );
			pz += weight * ( matrices[ o + 2 ] * x + matrices[ o + 6 ] * y + matrices[ o + 10 ] * z + matrices[ o + 14 ] );

		}

		return target.set( px, py, pz );

	}

	/** The local box around the posed vertices. */
	computeBoundingBox() {

		const position = this.geometry.attributes.position, point = new Vector3();
		this.boundingBox ??= new Box3();
		this.boundingBox.makeEmpty();

		for ( let i = 0; i < position.count; i ++ ) {

			point.fromBufferAttribute( position, i );
			this.boundingBox.expandByPoint( this.applyBoneTransform( i, point ) );

		}

	}

	/** The local sphere around the posed vertices. */
	computeBoundingSphere() {

		const position = this.geometry.attributes.position, point = new Vector3();
		this.computeBoundingBox();
		this.boundingSphere ??= new Sphere();
		this.boundingBox.getCenter( this.boundingSphere.center );
		let radius = 0;

		for ( let i = 0; i < position.count; i ++ ) {

			point.fromBufferAttribute( position, i );
			radius = Math.max( radius, this.boundingSphere.center.distanceToSquared( this.applyBoneTransform( i, point ) ) );

		}

		this.boundingSphere.radius = Math.sqrt( radius );

	}

	copy( source, recursive ) {

		super.copy( source, recursive );
		this.bindMode = source.bindMode;
		this.bindMatrix.copy( source.bindMatrix );
		this.bindMatrixInverse.copy( source.bindMatrixInverse );
		this.skeleton = source.skeleton;
		this._skinKey = '';
		return this;

	}

	dispose() {

		w().clearSkin( this.handle );
		super.dispose();

	}

}

SkinnedMesh.prototype.isSkinnedMesh = true;
