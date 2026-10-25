// Group, Scene and fog.
//
// Part of the scene graph; `../object3d.js` gathers it.

import { Color, Euler } from '../math.js';
import { Object3D } from './object3d.js';

export class Group extends Object3D {

	constructor() {

		super();
		this.type = 'Group';

	}

}

Group.prototype.isGroup = true;

export class Scene extends Object3D {

	constructor() {

		super();
		this.type = 'Scene';
		this.background = null;
		this.fog = null;
		// An equirectangular texture that lights standard materials without an `envMap`.
		this.environment = null;
		this.environmentIntensity = 1;
		this.environmentRotation = new Euler();
		this.backgroundBlurriness = 0;
		this.backgroundIntensity = 1;
		this.backgroundRotation = new Euler();
		// drawn in place of every material that allows it
		this.overrideMaterial = null;

	}

	copy( source, recursive ) {

		super.copy( source, recursive );
		this.background = source.background && ( source.background.isColor ? source.background.clone() : source.background );
		this.environment = source.environment;
		this.fog = source.fog && source.fog.clone();
		this.environmentIntensity = source.environmentIntensity;
		this.environmentRotation.copy( source.environmentRotation );
		this.backgroundBlurriness = source.backgroundBlurriness;
		this.backgroundIntensity = source.backgroundIntensity;
		this.backgroundRotation.copy( source.backgroundRotation );
		return this;

	}

}

Scene.prototype.isScene = true;

/** Fades objects towards `color` between the distances `near` and `far` from the camera. */
export class Fog {

	constructor( color, near = 1, far = 1000 ) {

		this.name = '';
		this.color = new Color( color );
		this.near = near;
		this.far = far;

	}

	clone() {

		return new Fog( this.color, this.near, this.far );

	}

}

Fog.prototype.isFog = true;

/** Fog that thickens with the square of the distance from the camera. */
export class FogExp2 {

	constructor( color, density = 0.00025 ) {

		this.name = '';
		this.color = new Color( color );
		this.density = density;

	}

	clone() {

		return new FogExp2( this.color, this.density );

	}

}

FogExp2.prototype.isFogExp2 = true;
