// Camera and light helpers.
//
// Part of the helpers; `../helpers.js` gathers them.

import { Float32BufferAttribute, OctahedronGeometry, SphereGeometry } from '../geometry.js';
import { LineBasicMaterial, MeshBasicMaterial } from '../materials.js';
import { Color, Vector3 } from '../math.js';
import { Object3D } from '../object3d.js';
import { Line, LineSegments, Mesh } from '../objects.js';
import { _matrix, _vector, lineGeometry } from './boxes.js';

/** The frustum of `camera` as lines, in three.js's colours, following the camera. */
export class CameraHelper extends LineSegments {

	constructor( camera ) {

		const vertices = [], colors = [], pointMap = {};
		const add = ( a, b, color ) => {

			for ( const id of [ a, b ] ) {

				vertices.push( 0, 0, 0 );
				colors.push( color.r, color.g, color.b );
				( pointMap[ id ] ??= [] ).push( vertices.length / 3 - 1 );

			}

		};

		const frustum = new Color( 0xffaa00 ), cone = new Color( 0xff0000 ), up = new Color( 0x00aaff ), target = new Color( 0xffffff ), cross = new Color( 0x333333 );
		add( 'n1', 'n2', frustum ); add( 'n2', 'n4', frustum ); add( 'n4', 'n3', frustum ); add( 'n3', 'n1', frustum );
		add( 'f1', 'f2', frustum ); add( 'f2', 'f4', frustum ); add( 'f4', 'f3', frustum ); add( 'f3', 'f1', frustum );
		add( 'n1', 'f1', frustum ); add( 'n2', 'f2', frustum ); add( 'n3', 'f3', frustum ); add( 'n4', 'f4', frustum );
		add( 'p', 'n1', cone ); add( 'p', 'n2', cone ); add( 'p', 'n3', cone ); add( 'p', 'n4', cone );
		add( 'u1', 'u2', up ); add( 'u2', 'u3', up ); add( 'u3', 'u1', up );
		add( 'c', 't', target ); add( 'p', 'c', cross );
		add( 'cn1', 'cn2', cross ); add( 'cn3', 'cn4', cross ); add( 'cf1', 'cf2', cross ); add( 'cf3', 'cf4', cross );
		super( lineGeometry( vertices, colors ), new LineBasicMaterial( { color: 0xffffff, vertexColors: true, toneMapped: false } ) );
		this.type = 'CameraHelper';
		this.camera = camera;
		this.pointMap = pointMap;
		this.matrixAutoUpdate = false;
		this.update();

	}

	setColors( frustum, cone, up, target, cross ) {

		const colors = this.geometry.attributes.color, set = ( range, color ) => {

			for ( const i of range ) colors.setXYZ( i, color.r, color.g, color.b );

		};

		set( Array.from( { length: 24 }, ( _, i ) => i ), frustum );
		set( [ 24, 25, 26, 27, 28, 29, 30, 31 ], cone );
		set( [ 32, 33, 34, 35, 36, 37 ], up );
		set( [ 38, 39 ], target );
		set( [ 40, 41, 42, 43, 44, 45, 46, 47, 48, 49 ], cross );
		colors.needsUpdate = true;
		return this;

	}

	// Corners of the frustum in normalized device coordinates, unprojected by the camera.
	update() {

		const camera = this.camera, inverse = camera.projectionMatrixInverse, position = this.geometry.attributes.position;
		const set = ( id, x, y, z ) => {

			_vector.set( x, y, z ).applyMatrix4( inverse );
			for ( const i of this.pointMap[ id ] ?? [] ) position.setXYZ( i, _vector.x, _vector.y, _vector.z );

		};

		const w = 1, h = 1, nearZ = - 1, farZ = 1;
		set( 'c', 0, 0, nearZ ); set( 't', 0, 0, farZ );
		set( 'n1', - w, - h, nearZ ); set( 'n2', w, - h, nearZ ); set( 'n3', - w, h, nearZ ); set( 'n4', w, h, nearZ );
		set( 'f1', - w, - h, farZ ); set( 'f2', w, - h, farZ ); set( 'f3', - w, h, farZ ); set( 'f4', w, h, farZ );
		set( 'u1', w * 0.7, h * 1.1, nearZ ); set( 'u2', - w * 0.7, h * 1.1, nearZ ); set( 'u3', 0, h * 2, nearZ );
		set( 'cf1', - w, 0, farZ ); set( 'cf2', w, 0, farZ ); set( 'cf3', 0, - h, farZ ); set( 'cf4', 0, h, farZ );
		set( 'cn1', - w, 0, nearZ ); set( 'cn2', w, 0, nearZ ); set( 'cn3', 0, - h, nearZ ); set( 'cn4', 0, h, nearZ );
		position.needsUpdate = true;

	}

	_onFrame() {

		this.matrix.copy( this.camera.matrixWorld );
		this.update();

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}

// ---------------------------------------------------------------- lights

/** A directional light: a square facing its target and a line to it. */
export class DirectionalLightHelper extends Object3D {

	constructor( light, size = 1, color ) {

		super();
		this.light = light;
		this.matrixAutoUpdate = false;
		this.color = color;
		this.type = 'DirectionalLightHelper';
		this.lightPlane = new Line( lineGeometry( [ - size, size, 0, size, size, 0, size, - size, 0, - size, - size, 0, - size, size, 0 ] ), new LineBasicMaterial( { fog: false, toneMapped: false } ) );
		this.add( this.lightPlane );
		this.targetLine = new Line( lineGeometry( [ 0, 0, 0, 0, 0, 1 ] ), new LineBasicMaterial( { fog: false, toneMapped: false } ) );
		this.add( this.targetLine );
		this.update();

	}

	update() {

		this.matrix.copy( this.light.matrixWorld );
		const from = new Vector3().setFromMatrixPosition( this.light.matrixWorld ), to = new Vector3().setFromMatrixPosition( this.light.target.matrixWorld );
		const direction = to.sub( from );
		const color = this.color !== undefined ? this.color : this.light.color;
		this.lightPlane.material.color.set( color );
		this.targetLine.material.color.set( color );
		this.lightPlane.lookAt( to.x + 0, to.y, to.z );
		this.targetLine.lookAt( to.x, to.y, to.z );
		this.targetLine.scale.z = direction.length();

	}

	_onFrame() {

		this.update();

	}

	dispose() {

		this.lightPlane.geometry.dispose();
		this.lightPlane.material.dispose();
		this.targetLine.geometry.dispose();
		this.targetLine.material.dispose();
		super.dispose();

	}

}

/** A point light: a small wireframe sphere at it. */
export class PointLightHelper extends Mesh {

	constructor( light, sphereSize, color ) {

		super( new SphereGeometry( sphereSize, 4, 2 ), new MeshBasicMaterial( { wireframe: true, fog: false, toneMapped: false } ) );
		this.light = light;
		this.color = color;
		this.type = 'PointLightHelper';
		this.matrixAutoUpdate = false;
		this.update();

	}

	update() {

		this.matrix.copy( this.light.matrixWorld );
		this.material.color.set( this.color !== undefined ? this.color : this.light.color );

	}

	_onFrame() {

		this.update();

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}

/** A spot light: its cone as lines towards the target. */
export class SpotLightHelper extends Object3D {

	constructor( light, color ) {

		super();
		this.light = light;
		this.matrixAutoUpdate = false;
		this.color = color;
		this.type = 'SpotLightHelper';
		const positions = [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, - 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, - 1, 1 ];

		for ( let i = 0, j = 1, l = 32; i < l; i ++, j ++ ) {

			const p1 = ( i / l ) * Math.PI * 2, p2 = ( j / l ) * Math.PI * 2;
			positions.push( Math.cos( p1 ), Math.sin( p1 ), 1, Math.cos( p2 ), Math.sin( p2 ), 1 );

		}

		this.cone = new LineSegments( lineGeometry( positions ), new LineBasicMaterial( { fog: false, toneMapped: false } ) );
		this.add( this.cone );
		this.update();

	}

	update() {

		this.matrix.copy( this.light.matrixWorld );
		const coneLength = this.light.distance ? this.light.distance : 1000, coneWidth = coneLength * Math.tan( this.light.angle );
		this.cone.scale.set( coneWidth, coneWidth, coneLength );
		const target = new Vector3().setFromMatrixPosition( this.light.target.matrixWorld );
		_matrix.copy( this.light.matrixWorld ).invert();
		target.applyMatrix4( _matrix );
		this.cone.lookAt( target.x, target.y, target.z );
		this.cone.material.color.set( this.color !== undefined ? this.color : this.light.color );

	}

	_onFrame() {

		this.update();

	}

	dispose() {

		this.cone.geometry.dispose();
		this.cone.material.dispose();
		super.dispose();

	}

}

/** A hemisphere light: an octahedron shaded from sky to ground colour. */
export class HemisphereLightHelper extends Object3D {

	constructor( light, size, color ) {

		super();
		this.light = light;
		this.matrixAutoUpdate = false;
		this.color = color;
		this.type = 'HemisphereLightHelper';
		const geometry = new OctahedronGeometry( size ).rotateY( Math.PI * 0.5 );
		this.material = new MeshBasicMaterial( { wireframe: true, fog: false, toneMapped: false } );
		if ( this.color === undefined ) this.material.vertexColors = true;
		const count = geometry.attributes.position.count;
		geometry.setAttribute( 'color', new Float32BufferAttribute( new Float32Array( count * 3 ), 3 ) );
		this.add( new Mesh( geometry, this.material ) );
		this.update();

	}

	update() {

		const mesh = this.children[ 0 ];

		if ( this.color !== undefined ) {

			this.material.color.set( this.color );

		} else {

			const colors = mesh.geometry.attributes.color, position = mesh.geometry.attributes.position;
			const sky = this.light.color, ground = this.light.groundColor, color = new Color();

			for ( let i = 0; i < colors.count; i ++ ) {

				color.copy( position.getY( i ) < 0 ? ground : sky );
				colors.setXYZ( i, color.r, color.g, color.b );

			}

			colors.needsUpdate = true;

		}

		this.matrix.copy( this.light.matrixWorld );

	}

	_onFrame() {

		this.update();

	}

	dispose() {

		this.children[ 0 ].geometry.dispose();
		this.children[ 0 ].material.dispose();
		super.dispose();

	}

}
