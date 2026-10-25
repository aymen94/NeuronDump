// Spherical coordinates, events and constants shared by the orbit controls.
//
// Part of the OrbitControls add-on (`../OrbitControls.js`).

import { Vector3, Plane, Ray } from '../../../compat.js';

// Spherical coordinates in JavaScript: the controls change them on every pointer move.
export class Spherical {

	constructor( radius = 1, phi = 0, theta = 0 ) {

		this.radius = radius;
		this.phi = phi;
		this.theta = theta;

	}

	set( radius, phi, theta ) {

		this.radius = radius;
		this.phi = phi;
		this.theta = theta;
		return this;

	}

	// keeps phi a little away from the poles
	makeSafe() {

		const EPS = 0.000001;
		this.phi = Math.max( EPS, Math.min( Math.PI - EPS, this.phi ) );
		return this;

	}

	setFromVector3( v ) {

		this.radius = Math.sqrt( v.x * v.x + v.y * v.y + v.z * v.z );

		if ( this.radius === 0 ) {

			this.theta = 0;
			this.phi = 0;

		} else {

			this.theta = Math.atan2( v.x, v.z );
			this.phi = Math.acos( Math.max( - 1, Math.min( 1, v.y / this.radius ) ) );

		}

		return this;

	}

}

export const _changeEvent = { type: 'change' }, _startEvent = { type: 'start' }, _endEvent = { type: 'end' };
export const _ray = new Ray(), _plane = new Plane(), _TILT_LIMIT = Math.cos( 70 * Math.PI / 180 );
export const _v = new Vector3(), _twoPI = 2 * Math.PI;
export const STATE = { NONE: - 1, ROTATE: 0, DOLLY: 1, PAN: 2, TOUCH_ROTATE: 3, TOUCH_PAN: 4, TOUCH_DOLLY_PAN: 5, TOUCH_DOLLY_ROTATE: 6 };
export const EPS = 0.000001;
