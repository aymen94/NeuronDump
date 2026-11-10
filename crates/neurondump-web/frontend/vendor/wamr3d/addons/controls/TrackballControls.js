// three.js add-on path: `three/addons/controls/TrackballControls.js` (TrackballControls).
//
// Rotates the camera freely around `target`, like turning a ball under the hand: no up
// direction is kept, so the camera can go over the poles. Left drag rotates, wheel or middle
// drag zooms, right drag pans; one finger rotates, two zoom and pan. Holding A / S / D makes
// a left drag rotate / zoom / pan. Call `update()` every frame and `handleResize()` when the
// element changes size.
//
// Beyond three.js: `rotateLeft( angle )`, `rotateUp( angle )` and `pan( dx, dy )` (pixels)
// move the camera from code, with the same signatures as OrbitControls; `update()` returns
// whether the camera moved.

import { Controls, MOUSE, Quaternion, Vector2, Vector3 } from '../../compat.js';

const _changeEvent = { type: 'change' }, _startEvent = { type: 'start' }, _endEvent = { type: 'end' };
const STATE = { NONE: - 1, ROTATE: 0, ZOOM: 1, PAN: 2, TOUCH_ROTATE: 3, TOUCH_ZOOM_PAN: 4 };
const EPS = 0.000001;

const _v2 = new Vector2();
const _axis = new Vector3(), _quaternion = new Quaternion(), _eyeDirection = new Vector3(),
	_upDirection = new Vector3(), _sidewaysDirection = new Vector3(), _moveDirection = new Vector3(),
	_pan = new Vector3(), _up = new Vector3();

export class TrackballControls extends Controls {

	constructor( object, domElement = null ) {

		super( object, domElement );
		this.enabled = true;
		this.screen = { left: 0, top: 0, width: 0, height: 0 };
		this.rotateSpeed = 1.0;
		this.zoomSpeed = 1.2;
		this.panSpeed = 0.3;
		this.noRotate = false;
		this.noZoom = false;
		this.noPan = false;
		this.staticMoving = false;
		this.dynamicDampingFactor = 0.2;
		this.minDistance = 0;
		this.maxDistance = Infinity;
		this.minZoom = 0;
		this.maxZoom = Infinity;
		this.keys = [ 'KeyA', 'KeyS', 'KeyD' ];
		this.mouseButtons = { LEFT: MOUSE.ROTATE, MIDDLE: MOUSE.DOLLY, RIGHT: MOUSE.PAN };
		this.target = new Vector3();

		this.state = STATE.NONE;
		this.keyState = STATE.NONE;
		this._lastPosition = new Vector3();
		this._lastZoom = 1;
		this._touchZoomDistanceStart = 0;
		this._touchZoomDistanceEnd = 0;
		this._lastAngle = 0;
		this._eye = new Vector3();
		this._movePrev = new Vector2();
		this._moveCurr = new Vector2();
		this._lastAxis = new Vector3();
		this._zoomStart = new Vector2();
		this._zoomEnd = new Vector2();
		this._panStart = new Vector2();
		this._panEnd = new Vector2();
		this._pointers = [];
		this._pointerPositions = {};
		// Rotation asked from code, in radians around the view axes, applied by update().
		this._codeRotate = new Vector2();
		this._moved = false;

		this.target0 = this.target.clone();
		this.position0 = this.object.position.clone();
		this.up0 = new Vector3( object.up.x, object.up.y, object.up.z );
		this.zoom0 = this.object.zoom;

		this._onPointerMove = this._handlePointerMove.bind( this );
		this._onPointerDown = this._handlePointerDown.bind( this );
		this._onPointerUp = this._handlePointerUp.bind( this );
		this._onPointerCancel = this._handlePointerUp.bind( this );
		this._onContextMenu = ( event ) => {

			if ( this.enabled ) event.preventDefault();

		};

		this._onMouseWheel = this._handleMouseWheel.bind( this );
		this._onKeyDown = this._handleKeyDown.bind( this );
		this._onKeyUp = this._handleKeyUp.bind( this );

		if ( domElement !== null ) {

			this.connect( domElement );
			this.handleResize();

		}

		this.update();

	}

	connect( element ) {

		super.connect( element );
		window.addEventListener( 'keydown', this._onKeyDown );
		window.addEventListener( 'keyup', this._onKeyUp );
		this.domElement.addEventListener( 'pointerdown', this._onPointerDown );
		this.domElement.addEventListener( 'pointercancel', this._onPointerCancel );
		this.domElement.addEventListener( 'wheel', this._onMouseWheel, { passive: false } );
		this.domElement.addEventListener( 'contextmenu', this._onContextMenu );
		this.domElement.style.touchAction = 'none';

	}

	disconnect() {

		if ( this.domElement === null ) return;
		window.removeEventListener( 'keydown', this._onKeyDown );
		window.removeEventListener( 'keyup', this._onKeyUp );
		this.domElement.removeEventListener( 'pointerdown', this._onPointerDown );
		this.domElement.ownerDocument.removeEventListener( 'pointermove', this._onPointerMove );
		this.domElement.ownerDocument.removeEventListener( 'pointerup', this._onPointerUp );
		this.domElement.removeEventListener( 'pointercancel', this._onPointerCancel );
		this.domElement.removeEventListener( 'wheel', this._onMouseWheel );
		this.domElement.removeEventListener( 'contextmenu', this._onContextMenu );
		this.domElement.style.touchAction = 'auto';

	}

	dispose() {

		this.disconnect();

	}

	/** Reads the element's place on the page; call it when the element is resized. */
	handleResize() {

		const box = this.domElement.getBoundingClientRect();
		const d = this.domElement.ownerDocument.documentElement;
		this.screen.left = box.left + window.pageXOffset - d.clientLeft;
		this.screen.top = box.top + window.pageYOffset - d.clientTop;
		this.screen.width = box.width;
		this.screen.height = box.height;

	}

	/** Turns the camera around the target's vertical screen axis (positive: the scene turns right). */
	rotateLeft( angle ) {

		this._codeRotate.x += angle;

	}

	/** Turns the camera around the horizontal screen axis (positive: the camera goes up). */
	rotateUp( angle ) {

		this._codeRotate.y += angle;

	}

	/** Moves camera and target across the screen by `dx`, `dy` pixels, like a drag. */
	pan( deltaX, deltaY ) {

		const h = this.screen.height || this.domElement?.clientHeight || 1;
		const distance = this.object.position.distanceTo( this.target );
		const scale = this.object.isPerspectiveCamera
			? 2 * distance * Math.tan( ( this.object.fov / 2 ) * Math.PI / 180 ) / h
			: ( this.object.top - this.object.bottom ) / this.object.zoom / h;
		_up.set( this.object.up.x, this.object.up.y, this.object.up.z );
		_eyeDirection.copy( this.object.position ).sub( this.target );
		_sidewaysDirection.copy( _eyeDirection ).cross( _up ).setLength( deltaX * scale );
		_upDirection.copy( _up ).setLength( deltaY * scale );
		_pan.copy( _sidewaysDirection ).add( _upDirection );
		this.object.position.add( _pan );
		this.target.add( _pan );
		this.update();

	}

	_getMouseOnScreen( pageX, pageY ) {

		return _v2.set(
			( pageX - this.screen.left ) / this.screen.width,
			( pageY - this.screen.top ) / this.screen.height
		);

	}

	_getMouseOnCircle( pageX, pageY ) {

		return _v2.set(
			( ( pageX - this.screen.width * 0.5 - this.screen.left ) / ( this.screen.width * 0.5 ) ),
			( ( this.screen.height + 2 * ( this.screen.top - pageY ) ) / this.screen.width ) // screen.width intentional
		);

	}

	_rotateCamera() {

		_moveDirection.set( this._moveCurr.x - this._movePrev.x, this._moveCurr.y - this._movePrev.y, 0 );
		let angle = _moveDirection.length();
		_up.set( this.object.up.x, this.object.up.y, this.object.up.z );

		if ( angle ) {

			this._eye.copy( this.object.position ).sub( this.target );
			_eyeDirection.copy( this._eye ).normalize();
			_upDirection.copy( _up ).normalize();
			_sidewaysDirection.crossVectors( _upDirection, _eyeDirection ).normalize();
			_upDirection.setLength( this._moveCurr.y - this._movePrev.y );
			_sidewaysDirection.setLength( this._moveCurr.x - this._movePrev.x );
			_moveDirection.copy( _upDirection.add( _sidewaysDirection ) );
			_axis.crossVectors( _moveDirection, this._eye ).normalize();
			angle *= this.rotateSpeed;
			_quaternion.setFromAxisAngle( _axis, angle );
			this._eye.applyQuaternion( _quaternion );
			_up.applyQuaternion( _quaternion );
			this.object.up.copy( _up );
			this._lastAxis.copy( _axis );
			this._lastAngle = angle;

		} else if ( ! this.staticMoving && this._lastAngle ) {

			this._lastAngle *= Math.sqrt( 1.0 - this.dynamicDampingFactor );
			if ( this._lastAngle < EPS ) this._lastAngle = 0;
			this._eye.copy( this.object.position ).sub( this.target );
			_quaternion.setFromAxisAngle( this._lastAxis, this._lastAngle );
			this._eye.applyQuaternion( _quaternion );
			_up.applyQuaternion( _quaternion );
			this.object.up.copy( _up );

		}

		this._movePrev.copy( this._moveCurr );

	}

	// Rotation asked by rotateLeft / rotateUp: around the screen's up, then its sideways axis.
	_rotateFromCode() {

		const { x, y } = this._codeRotate;
		if ( ! x && ! y ) return;
		this._codeRotate.set( 0, 0 );
		_up.set( this.object.up.x, this.object.up.y, this.object.up.z ).normalize();
		if ( x ) {

			_quaternion.setFromAxisAngle( _up, - x );
			this._eye.applyQuaternion( _quaternion );

		}

		if ( y ) {

			_axis.crossVectors( _up, this._eye ).normalize();
			_quaternion.setFromAxisAngle( _axis, - y );
			this._eye.applyQuaternion( _quaternion );
			_up.applyQuaternion( _quaternion );
			this.object.up.copy( _up );

		}

	}

	_zoomCamera() {

		let factor;

		if ( this.state === STATE.TOUCH_ZOOM_PAN ) {

			factor = this._touchZoomDistanceStart / this._touchZoomDistanceEnd;
			this._touchZoomDistanceStart = this._touchZoomDistanceEnd;

		} else {

			factor = 1.0 + ( this._zoomEnd.y - this._zoomStart.y ) * this.zoomSpeed;

		}

		if ( factor !== 1.0 && factor > 0.0 ) {

			if ( this.object.isPerspectiveCamera ) {

				this._eye.multiplyScalar( factor );

			} else if ( this.object.isOrthographicCamera ) {

				this.object.zoom = Math.max( this.minZoom, Math.min( this.maxZoom, this.object.zoom / factor ) );
				if ( this._lastZoom !== this.object.zoom ) this.object.updateProjectionMatrix();

			}

		}

		if ( this.state === STATE.TOUCH_ZOOM_PAN ) return;
		if ( this.staticMoving ) this._zoomStart.copy( this._zoomEnd );
		else this._zoomStart.y += ( this._zoomEnd.y - this._zoomStart.y ) * this.dynamicDampingFactor;

	}

	_panCamera() {

		_moveDirection.set( this._panEnd.x - this._panStart.x, this._panEnd.y - this._panStart.y, 0 );
		let length = _moveDirection.length();
		if ( ! length ) return;

		if ( this.object.isOrthographicCamera ) {

			const scaleX = ( this.object.right - this.object.left ) / this.object.zoom / this.domElement.clientWidth;
			const scaleY = ( this.object.top - this.object.bottom ) / this.object.zoom / this.domElement.clientWidth;
			_moveDirection.x *= scaleX;
			_moveDirection.y *= scaleY;
			length = _moveDirection.length();

		}

		const mx = _moveDirection.x, my = _moveDirection.y;
		length *= this._eye.length() * this.panSpeed;
		_up.set( this.object.up.x, this.object.up.y, this.object.up.z );
		_pan.copy( this._eye ).cross( _up ).setLength( mx );
		_pan.add( _upDirection.copy( _up ).setLength( my ) );
		_pan.setLength( length );
		this.object.position.add( _pan );
		this.target.add( _pan );
		if ( this.staticMoving ) {

			this._panStart.copy( this._panEnd );

		} else {

			this._panStart.add( _v2.subVectors( this._panEnd, this._panStart ).multiplyScalar( this.dynamicDampingFactor ) );

		}

	}

	_checkDistances() {

		if ( this.noZoom && this.noPan ) return;
		const d = this._eye.length();
		if ( d > this.maxDistance ) {

			this._eye.setLength( this.maxDistance );
			this.object.position.copy( this.target ).add( this._eye );
			this._zoomStart.copy( this._zoomEnd );

		}

		if ( d < this.minDistance ) {

			this._eye.setLength( this.minDistance );
			this.object.position.copy( this.target ).add( this._eye );
			this._zoomStart.copy( this._zoomEnd );

		}

	}

	/** Applies this frame's motion; returns true when the camera moved. */
	update() {

		this._eye.copy( this.object.position ).sub( this.target );
		if ( ! this.noRotate ) {

			this._rotateCamera();
			this._rotateFromCode();

		}

		if ( ! this.noZoom ) this._zoomCamera();
		if ( ! this.noPan ) this._panCamera();
		this.object.position.copy( this.target ).add( this._eye );
		if ( this.object.isPerspectiveCamera ) this._checkDistances();
		this.object.lookAt( this.target );

		const moved = this._lastPosition.distanceToSquared( this.object.position ) > EPS
			|| ( this.object.isOrthographicCamera === true && this._lastZoom !== this.object.zoom );
		if ( moved ) {

			this.dispatchEvent( _changeEvent );
			this._lastPosition.copy( this.object.position );
			this._lastZoom = this.object.zoom;

		}

		return moved;

	}

	reset() {

		this.state = STATE.NONE;
		this.keyState = STATE.NONE;
		this.target.copy( this.target0 );
		this.object.position.copy( this.position0 );
		this.object.up.copy( this.up0 );
		this.object.zoom = this.zoom0;
		this.object.updateProjectionMatrix();
		this._eye.copy( this.object.position ).sub( this.target );
		this._lastAngle = 0;
		this.object.lookAt( this.target );
		this.dispatchEvent( _changeEvent );
		this._lastPosition.copy( this.object.position );
		this._lastZoom = this.object.zoom;

	}

	/** Stops any motion still running from inertia. */
	stop() {

		this._lastAngle = 0;
		this._movePrev.copy( this._moveCurr );
		this._zoomStart.copy( this._zoomEnd );
		this._panStart.copy( this._panEnd );
		this._codeRotate.set( 0, 0 );

	}

	// pointer input

	_handlePointerDown( event ) {

		if ( this.enabled === false ) return;
		if ( this._pointers.length === 0 ) {

			this.domElement.setPointerCapture( event.pointerId );
			this.domElement.ownerDocument.addEventListener( 'pointermove', this._onPointerMove );
			this.domElement.ownerDocument.addEventListener( 'pointerup', this._onPointerUp );

		}

		this._pointers.push( event );
		this._trackPointer( event );
		if ( event.pointerType === 'touch' ) this._onTouchStart( event );
		else this._onMouseDown( event );

	}

	_handlePointerMove( event ) {

		if ( this.enabled === false ) return;
		if ( event.pointerType === 'touch' ) this._onTouchMove( event );
		else this._onMouseMove( event );

	}

	_handlePointerUp( event ) {

		if ( this.enabled === false ) return;
		if ( event.pointerType === 'touch' ) this._onTouchEnd( event );
		else this._onMouseUp();
		this._removePointer( event );
		if ( this._pointers.length === 0 ) {

			this.domElement.releasePointerCapture?.( event.pointerId );
			this.domElement.ownerDocument.removeEventListener( 'pointermove', this._onPointerMove );
			this.domElement.ownerDocument.removeEventListener( 'pointerup', this._onPointerUp );

		}

	}

	_handleKeyDown( event ) {

		if ( this.enabled === false ) return;
		if ( this.keyState !== STATE.NONE ) return;
		if ( event.code === this.keys[ STATE.ROTATE ] && ! this.noRotate ) this.keyState = STATE.ROTATE;
		else if ( event.code === this.keys[ STATE.ZOOM ] && ! this.noZoom ) this.keyState = STATE.ZOOM;
		else if ( event.code === this.keys[ STATE.PAN ] && ! this.noPan ) this.keyState = STATE.PAN;

	}

	_handleKeyUp() {

		if ( this.enabled === false ) return;
		this.keyState = STATE.NONE;

	}

	_onMouseDown( event ) {

		let mouseAction;
		switch ( event.button ) {

			case 0: mouseAction = this.mouseButtons.LEFT; break;
			case 1: mouseAction = this.mouseButtons.MIDDLE; break;
			case 2: mouseAction = this.mouseButtons.RIGHT; break;
			default: mouseAction = - 1;

		}

		switch ( mouseAction ) {

			case MOUSE.DOLLY: this.state = STATE.ZOOM; break;
			case MOUSE.ROTATE: this.state = STATE.ROTATE; break;
			case MOUSE.PAN: this.state = STATE.PAN; break;
			default: this.state = STATE.NONE;

		}

		const state = ( this.keyState !== STATE.NONE ) ? this.keyState : this.state;
		if ( state === STATE.ROTATE && ! this.noRotate ) {

			this._moveCurr.copy( this._getMouseOnCircle( event.pageX, event.pageY ) );
			this._movePrev.copy( this._moveCurr );

		} else if ( state === STATE.ZOOM && ! this.noZoom ) {

			this._zoomStart.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );
			this._zoomEnd.copy( this._zoomStart );

		} else if ( state === STATE.PAN && ! this.noPan ) {

			this._panStart.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );
			this._panEnd.copy( this._panStart );

		}

		this.dispatchEvent( _startEvent );

	}

	_onMouseMove( event ) {

		const state = ( this.keyState !== STATE.NONE ) ? this.keyState : this.state;
		if ( state === STATE.ROTATE && ! this.noRotate ) {

			this._movePrev.copy( this._moveCurr );
			this._moveCurr.copy( this._getMouseOnCircle( event.pageX, event.pageY ) );

		} else if ( state === STATE.ZOOM && ! this.noZoom ) {

			this._zoomEnd.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );

		} else if ( state === STATE.PAN && ! this.noPan ) {

			this._panEnd.copy( this._getMouseOnScreen( event.pageX, event.pageY ) );

		}

	}

	_onMouseUp() {

		this.state = STATE.NONE;
		this.dispatchEvent( _endEvent );

	}

	_handleMouseWheel( event ) {

		if ( this.enabled === false ) return;
		if ( this.noZoom === true ) return;
		event.preventDefault();
		switch ( event.deltaMode ) {

			case 2: this._zoomStart.y -= event.deltaY * 0.025; break; // pages
			case 1: this._zoomStart.y -= event.deltaY * 0.01; break; // lines
			default: this._zoomStart.y -= event.deltaY * 0.00025; // pixels

		}

		this.dispatchEvent( _startEvent );
		this.dispatchEvent( _endEvent );

	}

	_onTouchStart( event ) {

		this._trackPointer( event );
		switch ( this._pointers.length ) {

			case 1:
				this.state = STATE.TOUCH_ROTATE;
				this._moveCurr.copy( this._getMouseOnCircle( this._pointers[ 0 ].pageX, this._pointers[ 0 ].pageY ) );
				this._movePrev.copy( this._moveCurr );
				break;

			default: { // 2 or more

				this.state = STATE.TOUCH_ZOOM_PAN;
				const dx = this._pointers[ 0 ].pageX - this._pointers[ 1 ].pageX;
				const dy = this._pointers[ 0 ].pageY - this._pointers[ 1 ].pageY;
				this._touchZoomDistanceEnd = this._touchZoomDistanceStart = Math.sqrt( dx * dx + dy * dy );
				const x = ( this._pointers[ 0 ].pageX + this._pointers[ 1 ].pageX ) / 2;
				const y = ( this._pointers[ 0 ].pageY + this._pointers[ 1 ].pageY ) / 2;
				this._panStart.copy( this._getMouseOnScreen( x, y ) );
				this._panEnd.copy( this._panStart );
				break;

			}

		}

		this.dispatchEvent( _startEvent );

	}

	_onTouchMove( event ) {

		this._trackPointer( event );
		switch ( this._pointers.length ) {

			case 1:
				this._movePrev.copy( this._moveCurr );
				this._moveCurr.copy( this._getMouseOnCircle( event.pageX, event.pageY ) );
				break;

			default: { // 2 or more

				const position = this._getSecondPointerPosition( event );
				const dx = event.pageX - position.x;
				const dy = event.pageY - position.y;
				this._touchZoomDistanceEnd = Math.sqrt( dx * dx + dy * dy );
				const x = ( event.pageX + position.x ) / 2;
				const y = ( event.pageY + position.y ) / 2;
				this._panEnd.copy( this._getMouseOnScreen( x, y ) );
				break;

			}

		}

	}

	_onTouchEnd( event ) {

		switch ( this._pointers.length ) {

			case 0: this.state = STATE.NONE; break;
			case 1:
				this.state = STATE.TOUCH_ROTATE;
				this._moveCurr.copy( this._getMouseOnCircle( event.pageX, event.pageY ) );
				this._movePrev.copy( this._moveCurr );
				break;
			case 2: {

				this.state = STATE.TOUCH_ZOOM_PAN;
				for ( let i = 0; i < this._pointers.length; i ++ ) {

					if ( this._pointers[ i ].pointerId !== event.pointerId ) {

						const position = this._pointerPositions[ this._pointers[ i ].pointerId ];
						this._moveCurr.copy( this._getMouseOnCircle( position.x, position.y ) );
						this._movePrev.copy( this._moveCurr );
						break;

					}

				}

				break;

			}

		}

		this.dispatchEvent( _endEvent );

	}

	_removePointer( event ) {

		delete this._pointerPositions[ event.pointerId ];
		for ( let i = 0; i < this._pointers.length; i ++ ) {

			if ( this._pointers[ i ].pointerId === event.pointerId ) {

				this._pointers.splice( i, 1 );
				return;

			}

		}

	}

	_trackPointer( event ) {

		let position = this._pointerPositions[ event.pointerId ];
		if ( position === undefined ) {

			position = new Vector2();
			this._pointerPositions[ event.pointerId ] = position;

		}

		position.set( event.pageX, event.pageY );

	}

	_getSecondPointerPosition( event ) {

		const pointer = ( event.pointerId === this._pointers[ 0 ].pointerId ) ? this._pointers[ 1 ] : this._pointers[ 0 ];
		return this._pointerPositions[ pointer.pointerId ];

	}

}
