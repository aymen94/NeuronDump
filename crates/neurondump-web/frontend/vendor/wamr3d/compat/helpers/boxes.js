// Arrow, box and plane helpers, and the line geometry the helpers share.
//
// Part of the helpers; `../helpers.js` gathers them.

import { BufferGeometry, ConeGeometry, Float32BufferAttribute } from '../geometry.js';
import { LineBasicMaterial, MeshBasicMaterial } from '../materials.js';
import { Box3, Matrix4, Quaternion, Vector3 } from '../math.js';
import { Object3D } from '../object3d.js';
import { Line, LineSegments, Mesh } from '../objects.js';

export const _vector = new Vector3(), _matrix = new Matrix4(), _box = new Box3();

export function lineGeometry( positions, colors ) {

	const geometry = new BufferGeometry();
	geometry.setAttribute( 'position', new Float32BufferAttribute( positions, 3 ) );
	if ( colors ) geometry.setAttribute( 'color', new Float32BufferAttribute( colors, 3 ) );
	return geometry;

}

// ---------------------------------------------------------------- ArrowHelper

let lineArrow = null, coneArrow = null;

/** An arrow from `origin` along `dir` (normalized): a line and a cone head. */
export class ArrowHelper extends Object3D {

	constructor( dir = new Vector3( 0, 0, 1 ), origin = new Vector3( 0, 0, 0 ), length = 1, color = 0xffff00, headLength = length * 0.2, headWidth = headLength * 0.2 ) {

		super();
		this.type = 'ArrowHelper';
		lineArrow ??= lineGeometry( [ 0, 0, 0, 0, 1, 0 ] );
		coneArrow ??= new ConeGeometry( 0.5, 1, 5, 1 ).translate( 0, - 0.5, 0 );
		this.position.copy( origin );
		this.line = new Line( lineArrow, new LineBasicMaterial( { color, toneMapped: false } ) );
		this.line.matrixAutoUpdate = false;
		this.add( this.line );
		this.cone = new Mesh( coneArrow, new MeshBasicMaterial( { color, toneMapped: false } ) );
		this.cone.matrixAutoUpdate = false;
		this.add( this.cone );
		this.setDirection( dir );
		this.setLength( length, headLength, headWidth );

	}

	setDirection( dir ) {

		if ( dir.y > 0.99999 ) this.quaternion.set( 0, 0, 0, 1 );
		else if ( dir.y < - 0.99999 ) this.quaternion.set( 1, 0, 0, 0 );
		else this.quaternion.copy( new Quaternion().setFromAxisAngle( _vector.set( dir.z, 0, - dir.x ).normalize(), Math.acos( dir.y ) ) );

	}

	setLength( length, headLength = length * 0.2, headWidth = headLength * 0.2 ) {

		this.line.matrix.makeScale( 1, Math.max( 0.0001, length - headLength ), 1 );
		this.cone.matrix.compose( _vector.set( 0, length, 0 ), new Quaternion(), new Vector3( headWidth, headLength, headWidth ) );

	}

	setColor( color ) {

		this.line.material.color.set( color );
		this.cone.material.color.set( color );

	}

	copy( source ) {

		super.copy( source, false );
		this.line.copy( source.line );
		this.cone.copy( source.cone );
		return this;

	}

	dispose() {

		this.line.material.dispose();
		this.cone.material.dispose();
		super.dispose();

	}

}

// ---------------------------------------------------------------- boxes

const BOX_EDGES = [ 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7 ];

/** The world-axis box around `object` and its descendants, refreshed by `update()` and each frame. */
export class BoxHelper extends LineSegments {

	constructor( object, color = 0xffff00 ) {

		const geometry = new BufferGeometry();
		geometry.setIndex( BOX_EDGES );
		geometry.setAttribute( 'position', new Float32BufferAttribute( new Float32Array( 24 ), 3 ) );
		super( geometry, new LineBasicMaterial( { color, toneMapped: false } ) );
		this.object = object;
		this.type = 'BoxHelper';
		this.matrixAutoUpdate = false;
		this.update();

	}

	update() {

		if ( this.object !== undefined ) _box.setFromObject( this.object );
		if ( _box.isEmpty() ) return;
		const { min, max } = _box, position = this.geometry.attributes.position, a = position.array;
		a.set( [ max.x, max.y, max.z, min.x, max.y, max.z, min.x, min.y, max.z, max.x, min.y, max.z, max.x, max.y, min.z, min.x, max.y, min.z, min.x, min.y, min.z, max.x, min.y, min.z ] );
		position.needsUpdate = true;
		this.geometry.computeBoundingSphere();

	}

	_onFrame() {

		this.update();

	}

	setFromObject( object ) {

		this.object = object;
		this.update();
		return this;

	}

	copy( source, recursive ) {

		super.copy( source, recursive );
		this.object = source.object;
		return this;

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}

/** A `Box3` shown as lines, following the box's current extent. */
export class Box3Helper extends LineSegments {

	constructor( box, color = 0xffff00 ) {

		const geometry = new BufferGeometry();
		geometry.setIndex( BOX_EDGES );
		geometry.setAttribute( 'position', new Float32BufferAttribute( [ 1, 1, 1, - 1, 1, 1, - 1, - 1, 1, 1, - 1, 1, 1, 1, - 1, - 1, 1, - 1, - 1, - 1, - 1, 1, - 1, - 1 ], 3 ) );
		super( geometry, new LineBasicMaterial( { color, toneMapped: false } ) );
		this.box = box;
		this.type = 'Box3Helper';
		this.geometry.computeBoundingSphere();

	}

	_onFrame() {

		const box = this.box;
		if ( box.isEmpty() ) return;
		box.getCenter( this.position );
		box.getSize( this.scale );
		this.scale.multiplyScalar( 0.5 );

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}

/** A `Plane` shown as a square outline of `size` with a translucent face. */
export class PlaneHelper extends Line {

	constructor( plane, size = 1, hex = 0xffff00 ) {

		super( lineGeometry( [ 1, - 1, 0, - 1, 1, 0, - 1, - 1, 0, 1, 1, 0, - 1, 1, 0, - 1, - 1, 0, 1, - 1, 0, 1, 1, 0 ] ), new LineBasicMaterial( { color: hex, toneMapped: false } ) );
		this.type = 'PlaneHelper';
		this.plane = plane;
		this.size = size;
		const face = new BufferGeometry();
		face.setAttribute( 'position', new Float32BufferAttribute( [ 1, 1, 0, - 1, 1, 0, - 1, - 1, 0, 1, 1, 0, - 1, - 1, 0, 1, - 1, 0 ], 3 ) );
		face.computeBoundingSphere();
		this.add( new Mesh( face, new MeshBasicMaterial( { color: hex, opacity: 0.2, transparent: true, depthWrite: false, toneMapped: false } ) ) );

	}

	_onFrame() {

		const normal = this.plane.normal;
		this.position.set( 0, 0, 0 );
		this.scale.set( 0.5 * this.size, 0.5 * this.size, 1 );
		this.lookAt( normal.x, normal.y, normal.z );
		this.translateZ( - this.plane.constant );

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		this.children[ 0 ].geometry.dispose();
		this.children[ 0 ].material.dispose();
		super.dispose();

	}

}
