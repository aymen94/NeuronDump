// three.js add-on paths: `three/addons/controls/OrbitControls.js` (OrbitControls) and
// `three/addons/controls/MapControls.js` (MapControls).
//
// Orbits the camera around `target`: left drag rotates, wheel or middle drag dollies,
// right drag (or shift/ctrl/meta with left) pans; one finger rotates, two dolly and pan.

import { Controls, MOUSE, Quaternion, TOUCH, Vector2, Vector3 } from '../../compat.js';
import { EPS, STATE, Spherical, _TILT_LIMIT, _changeEvent, _plane, _ray, _twoPI, _v } from './orbit/common.js';
import { OrbitMotion } from './orbit/motion.js';
import { OrbitMouse } from './orbit/mouse.js';
import { OrbitTouch } from './orbit/touch.js';
import { OrbitPointers } from './orbit/pointers.js';

export class OrbitControls extends Controls {

	constructor( object, domElement = null ) {

		super( object, domElement );
		this.state = STATE.NONE;
		this.target = new Vector3();
		this.cursor = new Vector3();
		this.minDistance = 0;
		this.maxDistance = Infinity;
		this.minZoom = 0;
		this.maxZoom = Infinity;
		this.minTargetRadius = 0;
		this.maxTargetRadius = Infinity;
		this.minPolarAngle = 0;
		this.maxPolarAngle = Math.PI;
		this.minAzimuthAngle = - Infinity;
		this.maxAzimuthAngle = Infinity;
		this.enableDamping = false;
		this.dampingFactor = 0.05;
		this.enableZoom = true;
		this.zoomSpeed = 1.0;
		this.enableRotate = true;
		this.rotateSpeed = 1.0;
		this.keyRotateSpeed = 1.0;
		this.enablePan = true;
		this.panSpeed = 1.0;
		this.screenSpacePanning = true;
		this.keyPanSpeed = 7.0;
		this.zoomToCursor = false;
		this.autoRotate = false;
		this.autoRotateSpeed = 2.0;
		this.keys = { LEFT: 'ArrowLeft', UP: 'ArrowUp', RIGHT: 'ArrowRight', BOTTOM: 'ArrowDown' };
		this.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };
		this.touches = { ONE: TOUCH.ROTATE, TWO: TOUCH.DOLLY_PAN };
		this.target0 = this.target.clone();
		this.position0 = this.object.position.clone();
		this.zoom0 = this.object.zoom;
		this._domElementKeyEvents = null;

		this._lastPosition = new Vector3();
		this._lastQuaternion = new Quaternion();
		this._lastTargetPosition = new Vector3();
		this._quat = new Quaternion().setFromUnitVectors( new Vector3( object.up.x, object.up.y, object.up.z ), new Vector3( 0, 1, 0 ) );
		this._quatInverse = this._quat.clone().invert();
		this._spherical = new Spherical();
		this._sphericalDelta = new Spherical();
		this._scale = 1;
		this._panOffset = new Vector3();
		this._rotateStart = new Vector2();
		this._rotateEnd = new Vector2();
		this._rotateDelta = new Vector2();
		this._panStart = new Vector2();
		this._panEnd = new Vector2();
		this._panDelta = new Vector2();
		this._dollyStart = new Vector2();
		this._dollyEnd = new Vector2();
		this._dollyDelta = new Vector2();
		this._dollyDirection = new Vector3();
		this._mouse = new Vector2();
		this._performCursorZoom = false;
		this._pointers = [];
		this._pointerPositions = {};
		this._controlActive = false;

		this._onPointerMove = this._handlePointerMove.bind( this );
		this._onPointerDown = this._handlePointerDown.bind( this );
		this._onPointerUp = this._handlePointerUp.bind( this );
		this._onContextMenu = ( event ) => {

			if ( this.enabled ) event.preventDefault();

		};

		this._onMouseWheel = this._handleMouseWheel.bind( this );
		this._onKeyDown = this._handleKeyDown.bind( this );
		this._onTouchStart = this._handleTouchStart.bind( this );
		this._onTouchMove = this._handleTouchMove.bind( this );
		this._onMouseDown = this._handleMouseDown.bind( this );
		this._onMouseMove = this._handleMouseMove.bind( this );
		this._interceptControlDown = this._handleControlDown.bind( this );
		this._interceptControlUp = this._handleControlUp.bind( this );

		if ( this.domElement !== null ) this.connect( this.domElement );
		this.update();

	}

	connect( element ) {

		super.connect( element );
		this.domElement.addEventListener( 'pointerdown', this._onPointerDown );
		this.domElement.addEventListener( 'pointercancel', this._onPointerUp );
		this.domElement.addEventListener( 'contextmenu', this._onContextMenu );
		this.domElement.addEventListener( 'wheel', this._onMouseWheel, { passive: false } );
		const document = this.domElement.getRootNode();
		document.addEventListener( 'keydown', this._interceptControlDown, { passive: true, capture: true } );
		this.domElement.style.touchAction = 'none';

	}

	disconnect() {

		if ( this.domElement === null ) return;
		this.domElement.removeEventListener( 'pointerdown', this._onPointerDown );
		this.domElement.ownerDocument.removeEventListener( 'pointermove', this._onPointerMove );
		this.domElement.ownerDocument.removeEventListener( 'pointerup', this._onPointerUp );
		this.domElement.removeEventListener( 'pointercancel', this._onPointerUp );
		this.domElement.removeEventListener( 'wheel', this._onMouseWheel );
		this.domElement.removeEventListener( 'contextmenu', this._onContextMenu );
		this.stopListenToKeyEvents();
		const document = this.domElement.getRootNode();
		document.removeEventListener( 'keydown', this._interceptControlDown, { capture: true } );
		this.domElement.style.touchAction = 'auto';

	}

	dispose() {

		this.disconnect();

	}

	getPolarAngle() {

		return this._spherical.phi;

	}

	getAzimuthalAngle() {

		return this._spherical.theta;

	}

	getDistance() {

		return this.object.position.distanceTo( this.target );

	}

	listenToKeyEvents( domElement ) {

		domElement.addEventListener( 'keydown', this._onKeyDown );
		this._domElementKeyEvents = domElement;

	}

	stopListenToKeyEvents() {

		if ( this._domElementKeyEvents !== null ) {

			this._domElementKeyEvents.removeEventListener( 'keydown', this._onKeyDown );
			this._domElementKeyEvents = null;

		}

	}

	saveState() {

		this.target0.copy( this.target );
		this.position0.copy( this.object.position );
		this.zoom0 = this.object.zoom;

	}

	reset() {

		this.target.copy( this.target0 );
		this.object.position.copy( this.position0 );
		this.object.zoom = this.zoom0;
		this.object.updateProjectionMatrix();
		this.dispatchEvent( _changeEvent );
		this.update();
		this.state = STATE.NONE;

	}

	pan( deltaX, deltaY ) {

		this._pan( deltaX, deltaY );
		this.update();

	}

	dollyIn( dollyScale ) {

		this._dollyIn( dollyScale );
		this.update();

	}

	dollyOut( dollyScale ) {

		this._dollyOut( dollyScale );
		this.update();

	}

	rotateLeft( angle ) {

		this._rotateLeft( angle );
		this.update();

	}

	rotateUp( angle ) {

		this._rotateUp( angle );
		this.update();

	}

	update( deltaTime = null ) {

		const position = this.object.position;
		const offset = _v.copy( position ).sub( this.target );
		// rotate offset to "y-axis-is-up" space
		offset.applyQuaternion( this._quat );
		this._spherical.setFromVector3( offset );
		if ( this.autoRotate && this.state === STATE.NONE ) this._rotateLeft( this._getAutoRotationAngle( deltaTime ) );

		if ( this.enableDamping ) {

			this._spherical.theta += this._sphericalDelta.theta * this.dampingFactor;
			this._spherical.phi += this._sphericalDelta.phi * this.dampingFactor;

		} else {

			this._spherical.theta += this._sphericalDelta.theta;
			this._spherical.phi += this._sphericalDelta.phi;

		}

		// restrict theta to be between desired limits
		let min = this.minAzimuthAngle, max = this.maxAzimuthAngle;

		if ( isFinite( min ) && isFinite( max ) ) {

			if ( min < - Math.PI ) min += _twoPI; else if ( min > Math.PI ) min -= _twoPI;
			if ( max < - Math.PI ) max += _twoPI; else if ( max > Math.PI ) max -= _twoPI;

			if ( min <= max ) {

				this._spherical.theta = Math.max( min, Math.min( max, this._spherical.theta ) );

			} else {

				this._spherical.theta = ( this._spherical.theta > ( min + max ) / 2 ) ? Math.max( min, this._spherical.theta ) : Math.min( max, this._spherical.theta );

			}

		}

		// restrict phi to be between desired limits
		this._spherical.phi = Math.max( this.minPolarAngle, Math.min( this.maxPolarAngle, this._spherical.phi ) );
		this._spherical.makeSafe();

		// move target to panned location
		if ( this.enableDamping === true ) this.target.addScaledVector( this._panOffset, this.dampingFactor );
		else this.target.add( this._panOffset );

		// limit the target distance from the cursor to create a sphere around the center of interest
		this.target.sub( this.cursor );
		this.target.clampLength( this.minTargetRadius, this.maxTargetRadius );
		this.target.add( this.cursor );

		let zoomChanged = false;

		// adjust the camera position based on zoom only if we're not zooming to the cursor or if it's an ortho camera
		if ( this.zoomToCursor && this._performCursorZoom || this.object.isOrthographicCamera ) {

			this._spherical.radius = this._clampDistance( this._spherical.radius );

		} else {

			const prevRadius = this._spherical.radius;
			this._spherical.radius = this._clampDistance( this._spherical.radius * this._scale );
			zoomChanged = prevRadius != this._spherical.radius;

		}

		{

			const { radius, phi, theta } = this._spherical, sinPhiRadius = Math.sin( phi ) * radius;
			offset.set( sinPhiRadius * Math.sin( theta ), Math.cos( phi ) * radius, sinPhiRadius * Math.cos( theta ) );

		}
		// rotate offset back to "camera-up-vector-is-up" space
		offset.applyQuaternion( this._quatInverse );
		position.copy( this.target ).add( offset );
		this.object.lookAt( this.target );

		if ( this.enableDamping === true ) {

			this._sphericalDelta.theta *= ( 1 - this.dampingFactor );
			this._sphericalDelta.phi *= ( 1 - this.dampingFactor );
			this._panOffset.multiplyScalar( 1 - this.dampingFactor );

		} else {

			this._sphericalDelta.set( 0, 0, 0 );
			this._panOffset.set( 0, 0, 0 );

		}

		// adjust camera position
		if ( this.zoomToCursor && this._performCursorZoom ) {

			let newRadius = null;

			if ( this.object.isPerspectiveCamera ) {

				// move the camera down the pointer ray; this method avoids numerical instability
				const prevRadius = offset.length();
				newRadius = this._clampDistance( prevRadius * this._scale );
				const radiusDelta = prevRadius - newRadius;
				this.object.position.addScaledVector( this._dollyDirection, radiusDelta );
				this.object.updateMatrixWorld();
				zoomChanged = !! radiusDelta;

			} else if ( this.object.isOrthographicCamera ) {

				// adjust the ortho camera position based on zoom changes
				const mouseBefore = new Vector3( this._mouse.x, this._mouse.y, 0 );
				mouseBefore.unproject( this.object );
				const prevZoom = this.object.zoom;
				this.object.zoom = Math.max( this.minZoom, Math.min( this.maxZoom, this.object.zoom / this._scale ) );
				this.object.updateProjectionMatrix();
				zoomChanged = prevZoom !== this.object.zoom;
				const mouseAfter = new Vector3( this._mouse.x, this._mouse.y, 0 );
				mouseAfter.unproject( this.object );
				this.object.position.sub( mouseAfter ).add( mouseBefore );
				this.object.updateMatrixWorld();
				newRadius = offset.length();

			} else {

				console.warn( 'WAMR3D: OrbitControls encountered an unknown camera type - zoom to cursor disabled.' );
				this.zoomToCursor = false;

			}

			// handle the placement of the target
			if ( newRadius !== null ) {

				if ( this.screenSpacePanning ) {

					// position the orbit target in front of the new camera position
					this.target.set( 0, 0, - 1 ).transformDirection( this.object.matrix ).multiplyScalar( newRadius ).add( this.object.position );

				} else {

					// get the ray and translation plane to compute target
					_ray.origin.copy( this.object.position );
					_ray.direction.set( 0, 0, - 1 ).transformDirection( this.object.matrix );

					// if the camera is 20 degrees above the horizon then don't adjust the focus target to avoid
					// extremely large values
					if ( Math.abs( this.object.up.dot( _ray.direction ) ) < _TILT_LIMIT ) {

						this.object.lookAt( this.target );

					} else {

						_plane.setFromNormalAndCoplanarPoint( this.object.up, this.target );
						_ray.intersectPlane( _plane, this.target );

					}

				}

			}

		} else if ( this.object.isOrthographicCamera ) {

			const prevZoom = this.object.zoom;
			this.object.zoom = Math.max( this.minZoom, Math.min( this.maxZoom, this.object.zoom / this._scale ) );

			if ( prevZoom !== this.object.zoom ) {

				this.object.updateProjectionMatrix();
				zoomChanged = true;

			}

		}

		this._scale = 1;
		this._performCursorZoom = false;

		// update condition is:
		// min(camera displacement, camera rotation in radians)^2 > EPS
		// using small-angle approximation cos(x/2) = 1 - x^2 / 8
		if ( zoomChanged || this._lastPosition.distanceToSquared( this.object.position ) > EPS || 8 * ( 1 - this._lastQuaternion.dot( this.object.quaternion ) ) > EPS || this._lastTargetPosition.distanceToSquared( this.target ) > EPS ) {

			this.dispatchEvent( _changeEvent );
			this._lastPosition.copy( this.object.position );
			this._lastQuaternion.copy( this.object.quaternion );
			this._lastTargetPosition.copy( this.target );
			return true;

		}

		return false;

	}

}

// The methods in `orbit/` complete the class.
for ( const part of [ OrbitMotion, OrbitMouse, OrbitTouch, OrbitPointers ] ) {

	for ( const name of Object.getOwnPropertyNames( part.prototype ) ) {

		if ( name !== 'constructor' ) Object.defineProperty( OrbitControls.prototype, name, Object.getOwnPropertyDescriptor( part.prototype, name ) );

	}

}

/** OrbitControls set up for maps: left drag pans, right drag rotates, panning in the ground plane. */
export class MapControls extends OrbitControls {

	constructor( object, domElement ) {

		super( object, domElement );
		this.screenSpacePanning = false;
		this.mouseButtons = { LEFT: MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.ROTATE };
		this.touches = { ONE: TOUCH.PAN, TWO: TOUCH.DOLLY_ROTATE };

	}

}
