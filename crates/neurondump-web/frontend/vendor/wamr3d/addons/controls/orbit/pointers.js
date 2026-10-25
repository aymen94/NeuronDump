// Keyboard input, pointer tracking and the pointer and control-key listeners.
//
// Part of the OrbitControls add-on (`../OrbitControls.js`).

import { Vector2 } from '../../../compat.js';
import { STATE, _endEvent, _twoPI } from './common.js';

/** Methods of `OrbitControls`, copied onto its prototype. */
export class OrbitPointers {

	_handleKeyDown( event ) {

		if ( this.enabled === false ) return;
		let needsUpdate = false;
		const element = this.domElement;

		switch ( event.code ) {

			case this.keys.UP:
				if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

					if ( this.enableRotate ) this._rotateUp( _twoPI * this.keyRotateSpeed / element.clientHeight );

				} else if ( this.enablePan ) {

					this._pan( 0, this.keyPanSpeed );

				}

				needsUpdate = true;
				break;

			case this.keys.BOTTOM:
				if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

					if ( this.enableRotate ) this._rotateUp( - _twoPI * this.keyRotateSpeed / element.clientHeight );

				} else if ( this.enablePan ) {

					this._pan( 0, - this.keyPanSpeed );

				}

				needsUpdate = true;
				break;

			case this.keys.LEFT:
				if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

					if ( this.enableRotate ) this._rotateLeft( _twoPI * this.keyRotateSpeed / element.clientHeight );

				} else if ( this.enablePan ) {

					this._pan( this.keyPanSpeed, 0 );

				}

				needsUpdate = true;
				break;

			case this.keys.RIGHT:
				if ( event.ctrlKey || event.metaKey || event.shiftKey ) {

					if ( this.enableRotate ) this._rotateLeft( - _twoPI * this.keyRotateSpeed / element.clientHeight );

				} else if ( this.enablePan ) {

					this._pan( - this.keyPanSpeed, 0 );

				}

				needsUpdate = true;
				break;

		}

		if ( needsUpdate ) {

			// prevent the browser from scrolling on cursor keys
			event.preventDefault();
			this.update();

		}

	}

	// pointers

	_addPointer( event ) {

		this._pointers.push( event.pointerId );

	}

	_removePointer( event ) {

		delete this._pointerPositions[ event.pointerId ];
		const index = this._pointers.indexOf( event.pointerId );
		if ( index !== - 1 ) this._pointers.splice( index, 1 );

	}

	_isTrackingPointer( event ) {

		return this._pointers.includes( event.pointerId );

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

		const pointerId = ( event.pointerId === this._pointers[ 0 ] ) ? this._pointers[ 1 ] : this._pointers[ 0 ];
		return this._pointerPositions[ pointerId ];

	}

	_customWheelEvent( event ) {

		const mode = event.deltaMode;
		// minimal wheel event altered to meet delta-zoom demand
		const newEvent = { clientX: event.clientX, clientY: event.clientY, deltaY: event.deltaY };

		switch ( mode ) {

			case 1: // LINE_MODE
				newEvent.deltaY *= 16;
				break;

			case 2: // PAGE_MODE
				newEvent.deltaY *= 100;
				break;

		}

		// detect if event was triggered by pinching
		if ( event.ctrlKey && ! this._controlActive ) newEvent.deltaY *= 10;
		return newEvent;

	}

	_handlePointerDown( event ) {

		if ( this.enabled === false ) return;

		if ( this._pointers.length === 0 ) {

			this.domElement.setPointerCapture( event.pointerId );
			this.domElement.ownerDocument.addEventListener( 'pointermove', this._onPointerMove );
			this.domElement.ownerDocument.addEventListener( 'pointerup', this._onPointerUp );

		}

		if ( this._isTrackingPointer( event ) ) return;
		this._addPointer( event );
		if ( event.pointerType === 'touch' ) this._onTouchStart( event );
		else this._onMouseDown( event );

	}

	_handlePointerMove( event ) {

		if ( this.enabled === false ) return;
		if ( event.pointerType === 'touch' ) this._onTouchMove( event );
		else this._onMouseMove( event );

	}

	_handlePointerUp( event ) {

		this._removePointer( event );

		switch ( this._pointers.length ) {

			case 0:
				this.domElement.releasePointerCapture( event.pointerId );
				this.domElement.ownerDocument.removeEventListener( 'pointermove', this._onPointerMove );
				this.domElement.ownerDocument.removeEventListener( 'pointerup', this._onPointerUp );
				this.dispatchEvent( _endEvent );
				this.state = STATE.NONE;
				break;

			case 1: {

				const pointerId = this._pointers[ 0 ];
				const position = this._pointerPositions[ pointerId ];
				// minimal placeholder event - allows state correction on pointer-up
				this._onTouchStart( { pointerId, pageX: position.x, pageY: position.y } );
				break;

			}

		}

	}

	_handleControlDown( event ) {

		if ( event.key === 'Control' ) {

			this._controlActive = true;
			const document = this.domElement.getRootNode();
			document.addEventListener( 'keyup', this._interceptControlUp, { passive: true, capture: true } );

		}

	}

	_handleControlUp( event ) {

		if ( event.key === 'Control' ) {

			this._controlActive = false;
			const document = this.domElement.getRootNode();
			document.removeEventListener( 'keyup', this._interceptControlUp, { passive: true, capture: true } );

		}

	}

}
