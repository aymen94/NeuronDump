// Object3D: transforms, hierarchy, visibility and the engine object behind it.
//
// Part of the scene graph; `../object3d.js` gathers it.

import { ALL_PARTS, OBJECT, POSITION, QUATERNION, ROTATION, SCALE, UP, finalizer, lazyUUID, objects, peek, readFloats, readStamp, touched, w, world } from '../engine.js';
import { Matrix4, Quaternion, Vector3, engineVector } from '../math.js';
import { withEvents } from '../events.js';
import { BoundEuler, BoundLayers, BoundQuaternion, BoundVector3 } from '../transform.js';

const AXES = {};

function axis( name ) {

	return AXES[ name ] ??= new Vector3( name === 'x' ? 1 : 0, name === 'y' ? 1 : 0, name === 'z' ? 1 : 0 );

}

let nextId = 1;
const _q = new Quaternion();
const readout = [ 0, 0, 0, 0 ];

// A world-space value of `object` (0 position, 1 quaternion, 2 scale, 3 direction) into
// `target`, read from the engine without allocating.
function worldValue( object, what, target ) {

	if ( ! readFloats( w().worldValuePtr( object.handle, what ), readout, what === 1 ? 4 : 3 ) ) throw new Error( 'WAMR3D: object handle is not alive' );
	return target.fromArray( readout );

}


export class Object3D {

	constructor() {

		const handle = w().createObject();
		this.handle = handle;
		this._t = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0 ];
		this._dirty = 0;
		this._stale = 0;
		this._matrixAutoUpdate = true;
		this._matrix = null;
		this._matrixWorld = null;
		this._matrixWorldStamp = - 1;
		this._matrixTouched = false;
		this.matrixWorldNeedsUpdate = false;
		this.id = nextId ++;
		this.type = 'Object3D';
		this.parent = null;
		this.children = [];
		this.userData = {};
		this.matrixWorldAutoUpdate = true;
		this.position = new BoundVector3( this, 0, POSITION );
		this.rotation = new BoundEuler( this, 3, ROTATION, QUATERNION );
		this.quaternion = new BoundQuaternion( this, 6, QUATERNION, ROTATION );
		this.scale = new BoundVector3( this, 10, SCALE );
		this.up = new BoundVector3( this, 13, UP );
		this.layers = new BoundLayers( this );
		this.animations = [];
		this.customDepthMaterial = undefined;
		this.customDistanceMaterial = undefined;
		this.static = false;
		this._name = '';
		this._visible = true;
		this._renderOrder = 0;
		this._frustumCulled = true;
		this._castShadow = false;
		this._receiveShadow = false;
		objects.set( handle, new WeakRef( this ) );
		finalizer.register( this, this._token = { kind: OBJECT, handle, instances: 0 }, this );
		this._pivot = null;
		const up = Object3D.DEFAULT_UP;
		if ( up.x !== 0 || up.y !== 1 || up.z !== 0 ) this.up.copy( up );
		if ( Object3D.DEFAULT_MATRIX_AUTO_UPDATE === false ) this.matrixAutoUpdate = false;
		if ( Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE === false ) this.matrixWorldAutoUpdate = false;

	}

	/**
	 * The point rotation and scale turn about, in local space (three.js `pivot`), or null
	 * for the origin. Assign a new vector, or the same one again, after changing it.
	 */
	get pivot() {

		return this._pivot;

	}

	set pivot( value ) {

		this._pivot = value ?? null;
		const p = this._pivot;
		w().setPivot( this.handle, p !== null, p?.x ?? 0, p?.y ?? 0, p?.z ?? 0 );

	}

	get name() {

		return this._name;

	}

	set name( value ) {

		w().setName( this.handle, this._name = String( value ) );

	}

	get visible() {

		return this._visible;

	}

	set visible( value ) {

		w().setVisible( this.handle, this._visible = !! value );

	}

	get matrixAutoUpdate() {

		return this._matrixAutoUpdate;

	}

	/**
	 * With `false` the local matrix is yours: change it through `object.matrix` and its
	 * methods. Position, rotation and scale are then only applied by `updateMatrix()`.
	 */
	set matrixAutoUpdate( value ) {

		value = !! value;
		if ( value === this._matrixAutoUpdate ) return;
		if ( ! value ) readFloats( w().matrixPtr( this.handle ), ( this._matrix ??= new Matrix4() ).elements, 16 );
		w().setMatrixAutoUpdate( this.handle, value );
		this._matrixAutoUpdate = value;

	}

	get renderOrder() {

		return this._renderOrder;

	}

	set renderOrder( value ) {

		w().setRenderOrder( this.handle, this._renderOrder = value );

	}

	get castShadow() {

		return this._castShadow;

	}

	set castShadow( value ) {

		w().setShadowFlags( this.handle, this._castShadow = !! value, this._receiveShadow );

	}

	get receiveShadow() {

		return this._receiveShadow;

	}

	set receiveShadow( value ) {

		w().setShadowFlags( this.handle, this._castShadow, this._receiveShadow = !! value );

	}

	get frustumCulled() {

		return this._frustumCulled;

	}

	set frustumCulled( value ) {

		w().setFrustumCulled( this.handle, this._frustumCulled = !! value );

	}

	/**
	 * The local matrix. While `matrixAutoUpdate` is true it is a copy, refreshed on every
	 * access; with `matrixAutoUpdate = false` it is the matrix the object uses.
	 */
	get matrix() {

		const matrix = this._matrix ??= new Matrix4();

		if ( this._matrixAutoUpdate ) {

			readFloats( w().matrixPtr( this.handle ), matrix.elements, 16 );

		} else if ( ! this._matrixTouched ) {

			this._matrixTouched = true;
			touched.push( this );

		}

		return matrix;

	}

	set matrix( value ) {

		this.matrix.copy( value );

	}

	/** The world matrix: one `Matrix4` per object, refreshed from the engine on each access. */
	get matrixWorld() {

		const matrix = this._matrixWorld ??= new Matrix4(), stamp = readStamp();

		// read again only when the engine may have changed it since the last read; world
		// matrices change only through engine calls, not through pending transform writes
		if ( stamp === 0 || stamp !== this._matrixWorldStamp ) {

			readFloats( peek().matrixWorldPtr( this.handle ), matrix.elements, 16 );
			this._matrixWorldStamp = stamp;

		}

		return matrix;

	}

	add( ...children ) {

		for ( const child of children ) {

			if ( child === this || ! w().add( this.handle, child.handle ) ) continue;
			child._detach();
			child.parent = this;
			this.children.push( child );

		}

		return this;

	}

	remove( ...children ) {

		for ( const child of children ) {

			if ( child.parent !== this ) continue;
			w().remove( this.handle, child.handle );
			child._detach();

		}

		return this;

	}

	removeFromParent() {

		if ( this.parent !== null ) this.parent.remove( this );
		return this;

	}

	clear() {

		return this.remove( ...this.children );

	}

	/** Adds `child` while keeping its world transform. */
	attach( child ) {

		if ( w().attach( this.handle, child.handle ) ) {

			child._stale = ALL_PARTS;
			child._detach();
			child.parent = this;
			this.children.push( child );

		}

		return this;

	}

	_detach() {

		if ( this.parent === null ) return;
		const siblings = this.parent.children, index = siblings.indexOf( this );
		if ( index !== - 1 ) siblings.splice( index, 1 );
		this.parent = null;

	}

	applyMatrix4( matrix ) {

		if ( this._matrixAutoUpdate ) this.updateMatrix();
		const local = this.matrix.clone().premultiply( matrix );
		local.decompose( this.position, this.quaternion, this.scale );
		if ( ! this._matrixAutoUpdate ) this.matrix.copy( local );

	}

	applyQuaternion( q ) {

		this.quaternion.premultiply( q );
		return this;

	}

	setRotationFromAxisAngle( axis_, angle ) {

		this.quaternion.setFromAxisAngle( axis_, angle );

	}

	setRotationFromEuler( euler ) {

		this.quaternion.setFromEuler( euler, true );

	}

	setRotationFromMatrix( m ) {

		this.quaternion.setFromRotationMatrix( m );

	}

	setRotationFromQuaternion( q ) {

		this.quaternion.copy( q );

	}

	/** Rotates about an axis in world space (normalized); assumes no rotated parent, as three.js. */
	rotateOnWorldAxis( axis_, angle ) {

		this.quaternion.premultiply( _q.setFromAxisAngle( axis_, angle ) );
		return this;

	}

	/** Called by the renderer before drawing the object (once per frame and pass it is drawn in). */
	onBeforeRender( /* renderer, scene, camera, geometry, material, group */ ) {}

	onAfterRender( /* renderer, scene, camera, geometry, material, group */ ) {}

	onBeforeShadow( /* renderer, object, camera, shadowCamera, geometry, depthMaterial, group */ ) {}

	onAfterShadow( /* renderer, object, camera, shadowCamera, geometry, depthMaterial, group */ ) {}

	/** Adds the object's intersections with `raycaster`'s ray to `intersects`; nothing for a plain object. */
	raycast( /* raycaster, intersects */ ) {}

	intersectsFrustum( /* frustum */ ) {}

	lookAt( x, y, z ) {

		if ( typeof x === 'object' ) w().lookAt( this.handle, x.x, x.y, x.z );
		else w().lookAt( this.handle, x, y, z );
		this._stale = ALL_PARTS;

	}

	rotateOnAxis( axis_, angle ) {

		w().rotateOnAxis( this.handle, engineVector( axis_ ), angle );
		this._stale = ALL_PARTS;
		return this;

	}

	rotateX( angle ) {

		return this.rotateOnAxis( axis( 'x' ), angle );

	}

	rotateY( angle ) {

		return this.rotateOnAxis( axis( 'y' ), angle );

	}

	rotateZ( angle ) {

		return this.rotateOnAxis( axis( 'z' ), angle );

	}

	translateOnAxis( axis_, distance ) {

		w().translateOnAxis( this.handle, engineVector( axis_ ), distance );
		this._stale = ALL_PARTS;
		return this;

	}

	translateX( distance ) {

		return this.translateOnAxis( axis( 'x' ), distance );

	}

	translateY( distance ) {

		return this.translateOnAxis( axis( 'y' ), distance );

	}

	translateZ( distance ) {

		return this.translateOnAxis( axis( 'z' ), distance );

	}

	updateMatrix() {

		w().updateMatrix( this.handle );
		if ( ! this._matrixAutoUpdate ) this._matrix.fromArray( world.matrix( this.handle ) );

	}

	updateMatrixWorld( force = false ) {

		w().updateMatrixWorld( this.handle, force );

	}

	updateWorldMatrix( updateParents ) {

		if ( updateParents && this.parent !== null ) this.parent.updateWorldMatrix( true );
		w().updateMatrixWorld( this.handle, false );

	}

	getWorldPosition( target = new Vector3() ) {

		return worldValue( this, 0, target );

	}

	getWorldQuaternion( target = new Quaternion() ) {

		return worldValue( this, 1, target );

	}

	getWorldScale( target = new Vector3() ) {

		return worldValue( this, 2, target );

	}

	getWorldDirection( target = new Vector3() ) {

		return worldValue( this, 3, target );

	}

	localToWorld( v ) {

		return v.fromArray( w().localToWorld( this.handle, v.x, v.y, v.z ) );

	}

	worldToLocal( v ) {

		return v.fromArray( w().worldToLocal( this.handle, v.x, v.y, v.z ) );

	}

	traverse( callback ) {

		callback( this );
		for ( const child of this.children ) child.traverse( callback );

	}

	traverseVisible( callback ) {

		if ( ! this._visible ) return;
		callback( this );
		for ( const child of this.children ) child.traverseVisible( callback );

	}

	traverseAncestors( callback ) {

		if ( this.parent === null ) return;
		callback( this.parent );
		this.parent.traverseAncestors( callback );

	}

	getObjectByProperty( name, value ) {

		if ( this[ name ] === value ) return this;

		for ( const child of this.children ) {

			const found = child.getObjectByProperty( name, value );
			if ( found !== undefined ) return found;

		}

		return undefined;

	}

	getObjectsByProperty( name, value, result = [] ) {

		if ( this[ name ] === value ) result.push( this );
		for ( const child of this.children ) child.getObjectsByProperty( name, value, result );
		return result;

	}

	getObjectByName( name ) {

		return this.getObjectByProperty( 'name', name );

	}

	getObjectById( id ) {

		return this.getObjectByProperty( 'id', id );

	}

	clone( recursive = true ) {

		return new this.constructor().copy( this, recursive );

	}

	copy( source, recursive = true ) {

		this.name = source.name;
		this.up.copy( source.up );
		this.position.copy( source.position );
		this.rotation.order = source.rotation.order;
		this.quaternion.copy( source.quaternion );
		this.scale.copy( source.scale );
		this.visible = source.visible;
		this.renderOrder = source.renderOrder;
		this.frustumCulled = source.frustumCulled;
		this.castShadow = source.castShadow;
		this.receiveShadow = source.receiveShadow;
		this.layers.mask = source.layers.mask;
		this.matrixWorldAutoUpdate = source.matrixWorldAutoUpdate;
		this.static = source.static;
		this.pivot = source.pivot !== null ? source.pivot.clone() : null;
		this.animations = source.animations.slice();

		if ( ! source.matrixAutoUpdate ) {

			this.matrixAutoUpdate = false;
			this.matrix.copy( source.matrix );

		}

		this.userData = JSON.parse( JSON.stringify( source.userData ) );
		if ( recursive ) for ( const child of source.children ) this.add( child.clone() );
		return this;

	}

	/** Releases the object in the engine now instead of at garbage collection. */
	dispose() {

		this.removeFromParent();
		this.clear();
		finalizer.unregister( this );
		objects.delete( this.handle );
		w().destroy( this.handle );

	}

}

Object3D.DEFAULT_UP = new Vector3( 0, 1, 0 );
Object3D.DEFAULT_MATRIX_AUTO_UPDATE = true;
Object3D.DEFAULT_MATRIX_WORLD_AUTO_UPDATE = true;
Object3D.prototype.isObject3D = true;
withEvents( Object3D );
lazyUUID( Object3D );
