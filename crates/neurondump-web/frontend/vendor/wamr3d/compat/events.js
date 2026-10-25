// Event listeners on objects, materials, geometries, textures, mixers and controls.

export class EventDispatcher {

	addEventListener( type, listener ) {

		if ( this._listeners === undefined ) this._listeners = {};
		const listeners = this._listeners[ type ] ??= [];
		if ( listeners.indexOf( listener ) === - 1 ) listeners.push( listener );

	}

	hasEventListener( type, listener ) {

		const listeners = this._listeners?.[ type ];
		return listeners !== undefined && listeners.indexOf( listener ) !== - 1;

	}

	removeEventListener( type, listener ) {

		const listeners = this._listeners?.[ type ];
		if ( listeners === undefined ) return;
		const index = listeners.indexOf( listener );
		if ( index !== - 1 ) listeners.splice( index, 1 );

	}

	dispatchEvent( event ) {

		const listeners = this._listeners?.[ event.type ];
		if ( listeners === undefined ) return;
		event.target = this;
		// a copy, so listeners may remove themselves while called
		const array = listeners.slice( 0 );
		for ( let i = 0, l = array.length; i < l; i ++ ) array[ i ].call( this, event );
		event.target = null;

	}

}

/** Gives a class that cannot extend `EventDispatcher` its methods. */
export function withEvents( Class ) {

	for ( const key of [ 'addEventListener', 'hasEventListener', 'removeEventListener', 'dispatchEvent' ] ) {

		if ( ! Object.prototype.hasOwnProperty.call( Class.prototype, key ) ) Class.prototype[ key ] = EventDispatcher.prototype[ key ];

	}

}

/**
 * three.js `Controls`: the base of camera controls. `connect( element )` starts listening
 * to the element's input, `disconnect()` stops, `update( delta )` applies motion.
 */
export class Controls extends EventDispatcher {

	constructor( object, domElement = null ) {

		super();
		this.object = object;
		this.domElement = domElement;
		this.enabled = true;
		this.state = - 1;
		this.keys = {};
		this.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: null };
		this.touches = { ONE: null, TWO: null };

	}

	connect( element ) {

		if ( element === undefined ) {

			console.warn( 'WAMR3D: Controls: connect() now requires an element.' );
			return;

		}

		if ( this.domElement !== null ) this.disconnect();
		this.domElement = element;

	}

	disconnect() {}

	dispose() {}

	update() {}

}
