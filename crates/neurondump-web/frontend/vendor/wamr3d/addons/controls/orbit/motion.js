// Rotation, panning and dolly applied to the camera.
//
// Part of the OrbitControls add-on (`../OrbitControls.js`).

import { Vector3 } from '../../../compat.js';
import { _twoPI, _v } from './common.js';

/** Methods of `OrbitControls`, copied onto its prototype. */
export class OrbitMotion {

	_getAutoRotationAngle( deltaTime ) {

		if ( deltaTime !== null ) return ( _twoPI / 60 * this.autoRotateSpeed ) * deltaTime;
		return _twoPI / 60 / 60 * this.autoRotateSpeed;

	}

	_getZoomScale( delta ) {

		const normalizedDelta = Math.abs( delta * 0.01 );
		return Math.pow( 0.95, this.zoomSpeed * normalizedDelta );

	}

	_rotateLeft( angle ) {

		this._sphericalDelta.theta -= angle;

	}

	_rotateUp( angle ) {

		this._sphericalDelta.phi -= angle;

	}

	_panLeft( distance, objectMatrix ) {

		_v.setFromMatrixColumn( objectMatrix, 0 );
		_v.multiplyScalar( - distance );
		this._panOffset.add( _v );

	}

	_panUp( distance, objectMatrix ) {

		if ( this.screenSpacePanning === true ) {

			_v.setFromMatrixColumn( objectMatrix, 1 );

		} else {

			_v.setFromMatrixColumn( objectMatrix, 0 );
			_v.crossVectors( this.object.up, _v );

		}

		_v.multiplyScalar( distance );
		this._panOffset.add( _v );

	}

	// deltaX and deltaY are in pixels; right and down are positive
	_pan( deltaX, deltaY ) {

		const element = this.domElement;

		if ( this.object.isPerspectiveCamera ) {

			// perspective
			const position = this.object.position;
			const offset = new Vector3().copy( position ).sub( this.target );
			let targetDistance = offset.length();
			// half of the fov is center to top of screen
			targetDistance *= Math.tan( ( this.object.fov / 2 ) * Math.PI / 180.0 );
			// we use only clientHeight here so aspect ratio does not distort speed
			const matrix = this.object.matrix;
			this._panLeft( 2 * deltaX * targetDistance / element.clientHeight, matrix );
			this._panUp( 2 * deltaY * targetDistance / element.clientHeight, matrix );

		} else if ( this.object.isOrthographicCamera ) {

			// orthographic
			const matrix = this.object.matrix;
			this._panLeft( deltaX * ( this.object.right - this.object.left ) / this.object.zoom / element.clientWidth, matrix );
			this._panUp( deltaY * ( this.object.top - this.object.bottom ) / this.object.zoom / element.clientHeight, matrix );

		} else {

			// camera neither orthographic nor perspective
			console.warn( 'WAMR3D: OrbitControls encountered an unknown camera type - pan disabled.' );
			this.enablePan = false;

		}

	}

	_dollyOut( dollyScale ) {

		if ( this.object.isPerspectiveCamera || this.object.isOrthographicCamera ) {

			this._scale /= dollyScale;

		} else {

			console.warn( 'WAMR3D: OrbitControls encountered an unknown camera type - dolly/zoom disabled.' );
			this.enableZoom = false;

		}

	}

	_dollyIn( dollyScale ) {

		if ( this.object.isPerspectiveCamera || this.object.isOrthographicCamera ) {

			this._scale *= dollyScale;

		} else {

			console.warn( 'WAMR3D: OrbitControls encountered an unknown camera type - dolly/zoom disabled.' );
			this.enableZoom = false;

		}

	}

	_updateZoomParameters( x, y ) {

		if ( ! this.zoomToCursor ) return;
		this._performCursorZoom = true;
		const rect = this.domElement.getBoundingClientRect();
		const dx = x - rect.left, dy = y - rect.top, w = rect.width, h = rect.height;
		this._mouse.x = ( dx / w ) * 2 - 1;
		this._mouse.y = - ( dy / h ) * 2 + 1;
		this._dollyDirection.set( this._mouse.x, this._mouse.y, 1 ).unproject( this.object ).sub( this.object.position ).normalize();

	}

	_clampDistance( dist ) {

		return Math.max( this.minDistance, Math.min( this.maxDistance, dist ) );

	}

}
