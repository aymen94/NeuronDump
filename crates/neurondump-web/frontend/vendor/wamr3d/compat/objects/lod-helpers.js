// Levels of detail, and the grid and axes helpers.
//
// Part of the drawable objects; `../objects.js` gathers them.

import { w, world } from '../engine.js';
import { BufferAttribute, BufferGeometry } from '../geometry.js';
import { LineBasicMaterial } from '../materials.js';
import { Color } from '../math.js';
import { Object3D } from '../object3d.js';
import { LineSegments } from './lines-points.js';

/**
 * Shows one of its levels by distance from the camera. With `autoUpdate` (the default)
 * the renderer calls `update( camera )` before drawing.
 */
export class LOD extends Object3D {

	constructor() {

		super();
		this.type = 'LOD';
		this._currentLevel = 0;
		Object.defineProperty( this, 'levels', { enumerable: true, value: [] } );
		this.autoUpdate = true;

	}

	copy( source ) {

		super.copy( source, false );
		for ( const level of source.levels ) this.addLevel( level.object.clone(), level.distance, level.hysteresis );
		this.autoUpdate = source.autoUpdate;
		return this;

	}

	addLevel( object, distance = 0, hysteresis = 0 ) {

		distance = Math.abs( distance );
		const levels = this.levels;
		let l;
		for ( l = 0; l < levels.length; l ++ ) if ( distance < levels[ l ].distance ) break;
		levels.splice( l, 0, { distance, hysteresis, object } );
		this.add( object );
		return this;

	}

	removeLevel( distance ) {

		const levels = this.levels;

		for ( let i = 0; i < levels.length; i ++ ) {

			if ( levels[ i ].distance === distance ) {

				const [ removed ] = levels.splice( i, 1 );
				this.remove( removed.object );
				return true;

			}

		}

		return false;

	}

	getCurrentLevel() {

		return this._currentLevel;

	}

	getObjectForDistance( distance ) {

		const levels = this.levels;
		if ( levels.length === 0 ) return null;
		let i;

		for ( i = 1; i < levels.length; i ++ ) {

			let levelDistance = levels[ i ].distance;
			if ( levels[ i ].object.visible ) levelDistance -= levelDistance * levels[ i ].hysteresis;
			if ( distance < levelDistance ) break;

		}

		return levels[ i - 1 ].object;

	}

	update( camera ) {

		const levels = this.levels;
		if ( levels.length <= 1 ) return;

		const a = w().matrixWorld( camera.handle ), b = world.matrixWorld( this.handle );
		const distance = Math.hypot( a[ 12 ] - b[ 12 ], a[ 13 ] - b[ 13 ], a[ 14 ] - b[ 14 ] ) / camera.zoom;
		levels[ 0 ].object.visible = true;
		let i;

		for ( i = 1; i < levels.length; i ++ ) {

			let levelDistance = levels[ i ].distance;
			if ( levels[ i ].object.visible ) levelDistance -= levelDistance * levels[ i ].hysteresis;
			if ( distance < levelDistance ) break;
			levels[ i - 1 ].object.visible = false;
			levels[ i ].object.visible = true;

		}

		this._currentLevel = i - 1;
		for ( ; i < levels.length; i ++ ) levels[ i ].object.visible = false;

	}

}

LOD.prototype.isLOD = true;

/** A square grid of lines in the XZ plane; the two centre lines use `color1`. */
export class GridHelper extends LineSegments {

	constructor( size = 10, divisions = 10, color1 = 0x444444, color2 = 0x888888 ) {

		color1 = new Color( color1 );
		color2 = new Color( color2 );
		const center = divisions / 2, step = size / divisions, halfSize = size / 2;
		const vertices = [], colors = [];

		for ( let i = 0, k = - halfSize; i <= divisions; i ++, k += step ) {

			vertices.push( - halfSize, 0, k, halfSize, 0, k, k, 0, - halfSize, k, 0, halfSize );
			const color = i === center ? color1 : color2;
			for ( let j = 0; j < 4; j ++ ) colors.push( color.r, color.g, color.b );

		}

		const geometry = new BufferGeometry();
		geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( vertices ), 3 ) );
		geometry.setAttribute( 'color', new BufferAttribute( new Float32Array( colors ), 3 ) );
		super( geometry, new LineBasicMaterial( { vertexColors: true, toneMapped: false } ) );
		this.type = 'GridHelper';

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}

/** The three axes as lines from the origin: X red, Y green, Z blue. */
export class AxesHelper extends LineSegments {

	constructor( size = 1 ) {

		const geometry = new BufferGeometry();
		geometry.setAttribute( 'position', new BufferAttribute( new Float32Array( [ 0, 0, 0, size, 0, 0, 0, 0, 0, 0, size, 0, 0, 0, 0, 0, 0, size ] ), 3 ) );
		geometry.setAttribute( 'color', new BufferAttribute( new Float32Array( [ 1, 0, 0, 1, 0.6, 0, 0, 1, 0, 0.6, 1, 0, 0, 0, 1, 0, 0.6, 1 ] ), 3 ) );
		super( geometry, new LineBasicMaterial( { vertexColors: true, toneMapped: false } ) );
		this.type = 'AxesHelper';

	}

	setColors( xAxisColor, yAxisColor, zAxisColor ) {

		const color = new Color(), attribute = this.geometry.attributes.color, array = attribute.array;

		[ xAxisColor, yAxisColor, zAxisColor ].forEach( ( value, i ) => {

			color.set( value );
			for ( let j = 0; j < 2; j ++ ) array.set( [ color.r, color.g, color.b ], i * 6 + j * 3 );

		} );

		attribute.needsUpdate = true;
		return this;

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}
