// Mouse and wheel input.
//
// Part of the OrbitControls add-on (`../OrbitControls.js`).

import { MOUSE } from '../../../compat.js';
import { STATE, _endEvent, _startEvent, _twoPI } from './common.js';

/** Methods of `OrbitControls`, copied onto its prototype. */
export class OrbitMouse {

	// event callbacks - update the object state

	_handleMouseDownRotate( event ) {

		this._rotateStart.set( event.clientX, event.clientY );

	}

	_handleMouseDownDolly( event ) {

		this._updateZoomParameters( event.clientX, event.clientX );
		this._dollyStart.set( event.clientX, event.clientY );

	}

	_handleMouseDownPan( event ) {

		this._panStart.set( event.clientX, event.clientY );

	}

	_handleMouseMoveRotate( event ) {

		this._rotateEnd.set( event.clientX, event.clientY );
		this._rotateDelta.subVectors( this._rotateEnd, this._rotateStart ).multiplyScalar( this.rotateSpeed );
		const element = this.domElement;
		this._rotateLeft( _twoPI * this._rotateDelta.x / element.clientHeight ); // yes, height
		this._rotateUp( _twoPI * this._rotateDelta.y / element.clientHeight );
		this._rotateStart.copy( this._rotateEnd );
		this.update();

	}

	_handleMouseMoveDolly( event ) {

		this._dollyEnd.set( event.clientX, event.clientY );
		this._dollyDelta.subVectors( this._dollyEnd, this._dollyStart );
		if ( this._dollyDelta.y > 0 ) this._dollyOut( this._getZoomScale( this._dollyDelta.y ) );
		else if ( this._dollyDelta.y < 0 ) this._dollyIn( this._getZoomScale( this._dollyDelta.y ) );
		this._dollyStart.copy( this._dollyEnd );
		this.update();

	}

	_handleMouseMovePan( event ) {

		this._panEnd.set( event.clientX, event.clientY );
		this._panDelta.subVectors( this._panEnd, this._panStart ).multiplyScalar( this.panSpeed );
		this._pan( this._panDelta.x, this._panDelta.y );
		this._panStart.copy( this._panEnd );
		this.update();

	}

	_handleMouseWheel( event ) {

		if ( this.enabled === false || this.enableZoom === false || this.state !== STATE.NONE ) return;
		event.preventDefault();
		this.dispatchEvent( _startEvent );
		this._handleWheelZoom( this._customWheelEvent( event ) );
		this.dispatchEvent( _endEvent );

	}

	_handleWheelZoom( event ) {

		this._updateZoomParameters( event.clientX, event.clientY );
		if ( event.deltaY < 0 ) this._dollyIn( this._getZoomScale( event.deltaY ) );
		else if ( event.deltaY > 0 ) this._dollyOut( this._getZoomScale( event.deltaY ) );
		this.update();

	}

	_handleMouseDown( event ) {

		let mouseAction;

		switch ( event.button ) {

			case 0: mouseAction = this.mouseButtons.LEFT; break;
			case 1: mouseAction = this.mouseButtons.MIDDLE; break;
			case 2: mouseAction = this.mouseButtons.RIGHT; break;
			default: mouseAction = - 1;

		}

		switch ( mouseAction ) {

			case MOUSE.DOLLY:
				if ( this.enableZoom === false ) return;
				this._handleMouseDownDolly( event );
				this.state = STATE.DOLLY;
				break;

			case MOUSE.ROTATE:
				if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

					if ( this.enablePan === false ) return;
					this._handleMouseDownPan( event );
					this.state = STATE.PAN;

				} else {

					if ( this.enableRotate === false ) return;
					this._handleMouseDownRotate( event );
					this.state = STATE.ROTATE;

				}

				break;

			case MOUSE.PAN:
				if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

					if ( this.enableRotate === false ) return;
					this._handleMouseDownRotate( event );
					this.state = STATE.ROTATE;

				} else {

					if ( this.enablePan === false ) return;
					this._handleMouseDownPan( event );
					this.state = STATE.PAN;

				}

				break;

			default:
				this.state = STATE.NONE;

		}

		if ( this.state !== STATE.NONE ) this.dispatchEvent( _startEvent );

	}

	_handleMouseMove( event ) {

		switch ( this.state ) {

			case STATE.ROTATE:
				if ( this.enableRotate === false ) return;
				this._handleMouseMoveRotate( event );
				break;

			case STATE.DOLLY:
				if ( this.enableZoom === false ) return;
				this._handleMouseMoveDolly( event );
				break;

			case STATE.PAN:
				if ( this.enablePan === false ) return;
				this._handleMouseMovePan( event );
				break;

		}

	}

}
