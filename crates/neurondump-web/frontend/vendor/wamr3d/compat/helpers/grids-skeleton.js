// The polar grid and skeleton helpers.
//
// Part of the helpers; `../helpers.js` gathers them.

import { LineBasicMaterial } from '../materials.js';
import { Color, Matrix4 } from '../math.js';
import { LineSegments } from '../objects.js';
import { _matrix, _vector, lineGeometry } from './boxes.js';

/** A polar grid: `sectors` radial lines and `rings` circles of `divisions` segments. */
export class PolarGridHelper extends LineSegments {

	constructor( radius = 10, sectors = 16, rings = 8, divisions = 64, color1 = 0x444444, color2 = 0x888888 ) {

		const c1 = new Color( color1 ), c2 = new Color( color2 ), vertices = [], colors = [];

		if ( sectors > 1 ) {

			for ( let i = 0; i < sectors; i ++ ) {

				const v = ( i / sectors ) * ( Math.PI * 2 ), x = Math.sin( v ) * radius, z = Math.cos( v ) * radius;
				vertices.push( 0, 0, 0, x, 0, z );
				const color = i & 1 ? c1 : c2;
				colors.push( color.r, color.g, color.b, color.r, color.g, color.b );

			}

		}

		for ( let i = 0; i < rings; i ++ ) {

			const color = i & 1 ? c1 : c2, r = radius - ( radius / rings * i );

			for ( let j = 0; j < divisions; j ++ ) {

				let v = ( j / divisions ) * ( Math.PI * 2 );
				vertices.push( Math.sin( v ) * r, 0, Math.cos( v ) * r );
				colors.push( color.r, color.g, color.b );
				v = ( ( j + 1 ) / divisions ) * ( Math.PI * 2 );
				vertices.push( Math.sin( v ) * r, 0, Math.cos( v ) * r );
				colors.push( color.r, color.g, color.b );

			}

		}

		super( lineGeometry( vertices, colors ), new LineBasicMaterial( { vertexColors: true, toneMapped: false } ) );
		this.type = 'PolarGridHelper';

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}

function bonesOf( object ) {

	const list = [];
	if ( object.isBone === true ) list.push( object );
	for ( const child of object.children ) list.push( ...bonesOf( child ) );
	return list;

}

/** Lines from each bone below `object` to its parent bone, following the pose. */
export class SkeletonHelper extends LineSegments {

	constructor( object ) {

		const bones = bonesOf( object ), vertices = [], colors = [];
		const color1 = new Color( 0, 0, 1 ), color2 = new Color( 0, 1, 0 );

		for ( const bone of bones ) {

			if ( bone.parent && bone.parent.isBone ) {

				vertices.push( 0, 0, 0, 0, 0, 0 );
				colors.push( color1.r, color1.g, color1.b, color2.r, color2.g, color2.b );

			}

		}

		super( lineGeometry( vertices, colors ), new LineBasicMaterial( { vertexColors: true, depthTest: false, depthWrite: false, toneMapped: false, transparent: true } ) );
		this.isSkeletonHelper = true;
		this.type = 'SkeletonHelper';
		this.root = object;
		this.bones = bones;
		this.matrixAutoUpdate = false;

	}

	_onFrame() {

		const position = this.geometry.attributes.position, inverse = _matrix.copy( this.root.matrixWorld ).invert();
		this.matrix.copy( this.root.matrixWorld );
		let j = 0;

		for ( const bone of this.bones ) {

			if ( ! bone.parent || ! bone.parent.isBone ) continue;
			_vector.setFromMatrixPosition( new Matrix4().multiplyMatrices( inverse, bone.matrixWorld ) );
			position.setXYZ( j, _vector.x, _vector.y, _vector.z );
			_vector.setFromMatrixPosition( new Matrix4().multiplyMatrices( inverse, bone.parent.matrixWorld ) );
			position.setXYZ( j + 1, _vector.x, _vector.y, _vector.z );
			j += 2;

		}

		position.needsUpdate = true;

	}

	setColors( color1, color2 ) {

		const colors = this.geometry.attributes.color;

		for ( let i = 0; i < colors.count; i += 2 ) {

			colors.setXYZ( i, color1.r, color1.g, color1.b );
			colors.setXYZ( i + 1, color2.r, color2.g, color2.b );

		}

		colors.needsUpdate = true;
		return this;

	}

	dispose() {

		this.geometry.dispose();
		this.material.dispose();
		super.dispose();

	}

}
