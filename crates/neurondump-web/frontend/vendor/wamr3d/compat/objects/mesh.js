// Mesh: a geometry drawn with one material or one per group, with morph targets.
//
// Part of the drawable objects; `../objects.js` gathers them.

import { w, world } from '../engine.js';
import { BufferGeometry } from '../geometry.js';
import { MeshBasicMaterial } from '../materials.js';
import { Object3D } from '../object3d.js';

export class Mesh extends Object3D {

	constructor( geometry = new BufferGeometry(), material = new MeshBasicMaterial() ) {

		super();
		this.type = 'Mesh';
		this._geometry = geometry;
		this._material = material;
		this._morphed = 0;
		this._link();
		this.updateMorphTargets();

	}

	get geometry() {

		return this._geometry;

	}

	set geometry( value ) {

		this._geometry = value;
		this._link();
		this.updateMorphTargets();

	}

	get material() {

		return this._material;

	}

	set material( value ) {

		this._material = value;
		this._link();

	}

	/** Sets `morphTargetInfluences` and `morphTargetDictionary` from the geometry's morph targets, as three.js does. */
	updateMorphTargets() {

		const morphAttributes = this._geometry.morphAttributes ?? {}, keys = Object.keys( morphAttributes );
		if ( keys.length === 0 ) return;
		const targets = morphAttributes[ keys[ 0 ] ];
		if ( targets === undefined ) return;
		this.morphTargetInfluences = [];
		this.morphTargetDictionary = {};

		for ( let m = 0; m < targets.length; m ++ ) {

			const name = targets[ m ].name || String( m );
			this.morphTargetInfluences.push( 0 );
			this.morphTargetDictionary[ name ] = m;

		}

	}

	// Blends the morph targets into the mesh's own copy of the geometry when the weights or
	// targets changed. Called by the renderer before drawing.
	_syncMorph() {

		const geometry = this._geometry, influences = this.morphTargetInfluences;
		const active = influences !== undefined && influences.length > 0 && geometry._hasMorphs?.();

		if ( ! active ) {

			if ( this._morphed !== 0 ) {

				w().disposeGeometry( this._morphed );
				this._morphed = this._token.morphed = 0;
				this._link();

			}

			return;

		}

		geometry._syncMorphAttributes();
		let sent = this._sentInfluences;

		if ( this._morphed === 0 ) {

			this._morphed = this._token.morphed = w().createGeometry();
			sent = null;
			this._link();

		}

		let changed = ! sent || sent.length !== influences.length || this._morphStamp !== geometry._morphStamp || this._morphRelative !== geometry.morphTargetsRelative;
		for ( let i = 0; ! changed && i < influences.length; i ++ ) changed = sent[ i ] !== influences[ i ];
		if ( ! changed ) return;
		this._sentInfluences = Float32Array.from( influences );
		this._morphStamp = geometry._morphStamp;
		this._morphRelative = geometry.morphTargetsRelative;
		w().morphGeometry( geometry.handle, this._morphed, this._sentInfluences, geometry.morphTargetsRelative );

	}

	// An array of materials draws each geometry group with the material at its
	// `materialIndex`. Assign a new array to change it; edits in place are not seen.
	_link() {

		const material = this._material, list = Array.isArray( material );
		w().setRenderable( this.handle, this._morphed !== 0 && this._morphed !== undefined ? this._morphed : this._geometry.handle, ( list ? material[ 0 ] : material ).handle );
		if ( list || this._listed ) world.setMaterials( this.handle, list ? BigUint64Array.from( material, ( m ) => m.handle ) : new BigUint64Array( 0 ) );
		this._listed = list;

	}

	copy( source, recursive ) {

		super.copy( source, recursive );
		this._geometry = source.geometry;
		this._material = source.material;
		this._link();
		return this;

	}

}

Mesh.prototype.isMesh = true;

/** Per-instance numbers of an `InstancedMesh`. Set `needsUpdate = true` after changing `array`. */
