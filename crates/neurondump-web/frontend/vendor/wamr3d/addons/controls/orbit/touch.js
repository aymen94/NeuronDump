// Touch input: one finger rotates, two dolly and pan.
//
// Part of the OrbitControls add-on (`../OrbitControls.js`).

import { TOUCH } from '../../../compat.js';
import { STATE, _startEvent, _twoPI } from './common.js';

/** Methods of `OrbitControls`, copied onto its prototype. */
export class OrbitTouch {

	_handleTouchStartRotate( event ) {

		if ( this._pointers.length === 1 ) {

			this._rotateStart.set( event.pageX, event.pageY );

		} else {

			const position = this._getSecondPointerPosition( event );
			this._rotateStart.set( 0.5 * ( event.pageX + position.x ), 0.5 * ( event.pageY + position.y ) );

		}

	}

	_handleTouchStartPan( event ) {

		if ( this._pointers.length === 1 ) {

			this._panStart.set( event.pageX, event.pageY );

		} else {

			const position = this._getSecondPointerPosition( event );
			this._panStart.set( 0.5 * ( event.pageX + position.x ), 0.5 * ( event.pageY + position.y ) );

		}

	}

	_handleTouchStartDolly( event ) {

		const position = this._getSecondPointerPosition( event );
		const dx = event.pageX - position.x, dy = event.pageY - position.y;
		this._dollyStart.set( 0, Math.sqrt( dx * dx + dy * dy ) );

	}

	_handleTouchStartDollyPan( event ) {

		if ( this.enableZoom ) this._handleTouchStartDolly( event );
		if ( this.enablePan ) this._handleTouchStartPan( event );

	}

	_handleTouchStartDollyRotate( event ) {

		if ( this.enableZoom ) this._handleTouchStartDolly( event );
		if ( this.enableRotate ) this._handleTouchStartRotate( event );

	}

	_handleTouchMoveRotate( event ) {

		if ( this._pointers.length == 1 ) {

			this._rotateEnd.set( event.pageX, event.pageY );

		} else {

			const position = this._getSecondPointerPosition( event );
			this._rotateEnd.set( 0.5 * ( event.pageX + position.x ), 0.5 * ( event.pageY + position.y ) );

		}

		this._rotateDelta.subVectors( this._rotateEnd, this._rotateStart ).multiplyScalar( this.rotateSpeed );
		const element = this.domElement;
		this._rotateLeft( _twoPI * this._rotateDelta.x / element.clientHeight ); // yes, height
		this._rotateUp( _twoPI * this._rotateDelta.y / element.clientHeight );
		this._rotateStart.copy( this._rotateEnd );

	}

	_handleTouchMovePan( event ) {

		if ( this._pointers.length === 1 ) {

			this._panEnd.set( event.pageX, event.pageY );

		} else {

			const position = this._getSecondPointerPosition( event );
			this._panEnd.set( 0.5 * ( event.pageX + position.x ), 0.5 * ( event.pageY + position.y ) );

		}

		this._panDelta.subVectors( this._panEnd, this._panStart ).multiplyScalar( this.panSpeed );
		this._pan( this._panDelta.x, this._panDelta.y );
		this._panStart.copy( this._panEnd );

	}

	_handleTouchMoveDolly( event ) {

		const position = this._getSecondPointerPosition( event );
		const dx = event.pageX - position.x, dy = event.pageY - position.y;
		const distance = Math.sqrt( dx * dx + dy * dy );
		this._dollyEnd.set( 0, distance );
		this._dollyDelta.set( 0, Math.pow( this._dollyEnd.y / this._dollyStart.y, this.zoomSpeed ) );
		this._dollyOut( this._dollyDelta.y );
		this._dollyStart.copy( this._dollyEnd );
		const centerX = ( event.pageX + position.x ) * 0.5, centerY = ( event.pageY + position.y ) * 0.5;
		this._updateZoomParameters( centerX, centerY );

	}

	_handleTouchMoveDollyPan( event ) {

		if ( this.enableZoom ) this._handleTouchMoveDolly( event );
		if ( this.enablePan ) this._handleTouchMovePan( event );

	}

	_handleTouchMoveDollyRotate( event ) {

		if ( this.enableZoom ) this._handleTouchMoveDolly( event );
		if ( this.enableRotate ) this._handleTouchMoveRotate( event );

	}

	_handleTouchStart( event ) {

		this._trackPointer( event );

		switch ( this._pointers.length ) {

			case 1:
				switch ( this.touches.ONE ) {

					case TOUCH.ROTATE:
						if ( this.enableRotate === false ) return;
						this._handleTouchStartRotate( event );
						this.state = STATE.TOUCH_ROTATE;
						break;

					case TOUCH.PAN:
						if ( this.enablePan === false ) return;
						this._handleTouchStartPan( event );
						this.state = STATE.TOUCH_PAN;
						break;

					default:
						this.state = STATE.NONE;

				}

				break;

			case 2:
				switch ( this.touches.TWO ) {

					case TOUCH.DOLLY_PAN:
						if ( this.enableZoom === false && this.enablePan === false ) return;
						this._handleTouchStartDollyPan( event );
						this.state = STATE.TOUCH_DOLLY_PAN;
						break;

					case TOUCH.DOLLY_ROTATE:
						if ( this.enableZoom === false && this.enableRotate === false ) return;
						this._handleTouchStartDollyRotate( event );
						this.state = STATE.TOUCH_DOLLY_ROTATE;
						break;

					default:
						this.state = STATE.NONE;

				}

				break;

			default:
				this.state = STATE.NONE;

		}

		if ( this.state !== STATE.NONE ) this.dispatchEvent( _startEvent );

	}

	_handleTouchMove( event ) {

		this._trackPointer( event );

		switch ( this.state ) {

			case STATE.TOUCH_ROTATE:
				if ( this.enableRotate === false ) return;
				this._handleTouchMoveRotate( event );
				this.update();
				break;

			case STATE.TOUCH_PAN:
				if ( this.enablePan === false ) return;
				this._handleTouchMovePan( event );
				this.update();
				break;

			case STATE.TOUCH_DOLLY_PAN:
				if ( this.enableZoom === false && this.enablePan === false ) return;
				this._handleTouchMoveDollyPan( event );
				this.update();
				break;

			case STATE.TOUCH_DOLLY_ROTATE:
				if ( this.enableZoom === false && this.enableRotate === false ) return;
				this._handleTouchMoveDollyRotate( event );
				this.update();
				break;

			default:
				this.state = STATE.NONE;

		}

	}

}
