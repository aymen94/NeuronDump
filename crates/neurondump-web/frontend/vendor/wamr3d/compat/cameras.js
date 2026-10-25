// Cameras.

import { w } from './engine.js';
import { Matrix4 } from './math.js';
import { Object3D } from './object3d.js';

// ---------------------------------------------------------------- cameras

export class Camera extends Object3D {

	constructor() {

		super();
		this.type = 'Camera';
		this._projectionMatrix = null;
		this._projectionMatrixInverse = null;
		this._matrixWorldInverse = null;

	}

	// Like `matrixWorld`: one persistent matrix each, refreshed on access.
	get projectionMatrix() {

		return ( this._projectionMatrix ??= new Matrix4() ).fromArray( w().projectionMatrix( this.handle ) );

	}

	get projectionMatrixInverse() {

		return ( this._projectionMatrixInverse ??= new Matrix4() ).fromArray( w().projectionMatrixInverse( this.handle ) );

	}

	get matrixWorldInverse() {

		return ( this._matrixWorldInverse ??= new Matrix4() ).fromArray( w().matrixWorldInverse( this.handle ) );

	}

	setViewOffset( fullWidth, fullHeight, x, y, width, height ) {

		this.view = { enabled: true, fullWidth, fullHeight, offsetX: x, offsetY: y, width, height };
		this.updateProjectionMatrix();

	}

	clearViewOffset() {

		if ( this.view ) this.view.enabled = false;
		this.updateProjectionMatrix();

	}

	_applyZoomAndView() {

		if ( this.zoom !== 1 ) w().setCameraZoom( this.handle, this.zoom );
		const v = this.view;
		if ( v && v.enabled ) w().setCameraViewOffset( this.handle, v.fullWidth, v.fullHeight, v.offsetX, v.offsetY, v.width, v.height );

	}

}

Camera.prototype.isCamera = true;

export class PerspectiveCamera extends Camera {

	constructor( fov = 50, aspect = 1, near = 0.1, far = 2000 ) {

		super();
		this.type = 'PerspectiveCamera';
		this.fov = fov;
		this.aspect = aspect;
		this.near = near;
		this.far = far;
		this.zoom = 1;
		this.view = null;
		this.updateProjectionMatrix();

	}

	updateProjectionMatrix() {

		w().setPerspectiveCamera( this.handle, this.fov, this.aspect, this.near, this.far, false );
		this._applyZoomAndView();

	}

	copy( source, recursive ) {

		super.copy( source, recursive );
		Object.assign( this, { fov: source.fov, aspect: source.aspect, near: source.near, far: source.far, zoom: source.zoom, view: source.view && { ...source.view } } );
		this.updateProjectionMatrix();
		return this;

	}

}

PerspectiveCamera.prototype.isPerspectiveCamera = true;

export class OrthographicCamera extends Camera {

	constructor( left = - 1, right = 1, top = 1, bottom = - 1, near = 0.1, far = 2000 ) {

		super();
		this.type = 'OrthographicCamera';
		this.left = left;
		this.right = right;
		this.top = top;
		this.bottom = bottom;
		this.near = near;
		this.far = far;
		this.zoom = 1;
		this.view = null;
		this.updateProjectionMatrix();

	}

	updateProjectionMatrix() {

		w().setOrthographicCamera( this.handle, this.left, this.right, this.top, this.bottom, this.near, this.far, false );
		this._applyZoomAndView();

	}

	copy( source, recursive ) {

		super.copy( source, recursive );
		Object.assign( this, { left: source.left, right: source.right, top: source.top, bottom: source.bottom, near: source.near, far: source.far, zoom: source.zoom, view: source.view && { ...source.view } } );
		this.updateProjectionMatrix();
		return this;

	}

}

OrthographicCamera.prototype.isOrthographicCamera = true;
