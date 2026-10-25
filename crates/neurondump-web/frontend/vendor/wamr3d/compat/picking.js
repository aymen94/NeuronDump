// Raycasting: engine hits, and instances and sprites picked here.

import { hosted, objects, w, world } from './engine.js';
import { Ray, Vector2, Vector3, engineVector } from './math.js';
import { Layers } from './transform.js';

// ---------------------------------------------------------------- picking

export class Raycaster {

	constructor( origin = new Vector3(), direction = new Vector3( 0, 0, - 1 ), near = 0, far = Infinity ) {

		this.ray = new Ray( origin, direction );
		this.near = near;
		this.far = far;
		this.camera = null;
		this.layers = new Layers();
		this.params = { Mesh: {}, Line: { threshold: 1 }, LOD: {}, Points: { threshold: 1 }, Sprite: {} };

	}

	set( origin, direction ) {

		this.ray.set( origin, direction );

	}

	/** `coords` are normalized device coordinates: x and y from -1 to 1, y up. */
	setFromCamera( coords, camera ) {

		const origin = new Vector3(), direction = new Vector3(), matrixWorld = camera.matrixWorld;

		if ( camera.isPerspectiveCamera ) {

			origin.setFromMatrixPosition( matrixWorld );
			direction.set( coords.x, coords.y, 0.5 ).unproject( camera ).sub( origin ).normalize();

		} else {

			origin.set( coords.x, coords.y, ( camera.near + camera.far ) / ( camera.near - camera.far ) ).unproject( camera );
			direction.set( 0, 0, - 1 ).transformDirection( matrixWorld );

		}

		this.ray.set( origin, direction );
		this.camera = camera;

	}

	intersectObject( object, recursive = true, intersects = [] ) {

		const origin = engineVector( this.ray.origin, 0 ), direction = engineVector( this.ray.direction, 1 );
		const hits = w().raycast( object.handle, recursive, origin, direction, this.near, this.far, this.params.Line.threshold, this.params.Points.threshold, this.layers.mask );
		collectHits( hits, intersects );
		hits.free();

		// Instance matrices and the camera a sprite faces are kept here, not in the engine.
		if ( hosted.count > 0 ) intersectHosted( this, object, recursive, origin, direction, intersects );

		return intersects.sort( ( a, b ) => a.distance - b.distance );

	}

	intersectObjects( list, recursive = true, intersects = [] ) {

		for ( const object of list ) this.intersectObject( object, recursive, intersects );
		return intersects.sort( ( a, b ) => a.distance - b.distance );

	}

}

// Objects picked on the JavaScript side: instanced meshes (one engine call with every
// matrix) and sprites. Counts every one ever made, to skip the walk when there are none.


function collectHits( hits, intersects, instanced = null ) {

	for ( let i = 0; i < hits.length; i ++ ) {

		const segment = hits.lineIndex( i );

		if ( segment !== undefined ) {

			intersects.push( { distance: hits.distance( i ), point: new Vector3().fromArray( hits.point( i ) ), index: segment, face: null, faceIndex: null, barycoord: null, object: objects.get( hits.object( i ) )?.deref() } );
			continue;

		}

		const face = hits.faceVertices( i ), uv = hits.uv( i ), normal = hits.normal( i );
		const hit = {
			distance: hits.distance( i ),
			point: new Vector3().fromArray( hits.point( i ) ),
			object: instanced ?? objects.get( hits.object( i ) )?.deref(),
			faceIndex: hits.faceIndex( i ),
			face: { a: face[ 0 ], b: face[ 1 ], c: face[ 2 ], normal: new Vector3().fromArray( hits.faceNormal( i ) ), materialIndex: 0 },
			barycoord: new Vector3().fromArray( hits.barycoord( i ) )
		};
		if ( uv ) hit.uv = new Vector2().fromArray( uv );
		if ( normal ) hit.normal = new Vector3().fromArray( normal );
		if ( instanced !== null ) hit.instanceId = hits.instanceId( i );
		intersects.push( hit );

	}

}

function intersectHosted( raycaster, object, recursive, origin, direction, intersects ) {

	// the engine's raycaster tests layer 0, as three.js's does by default
	if ( ( object.layers.mask & 1 ) !== 0 ) {

		if ( object.isInstancedMesh ) {

			const count = object._count;

			if ( count > 0 ) {

				const hits = w().raycastInstances( object.handle, object._instances.matrices.array.subarray( 0, count * 16 ), origin, direction, raycaster.near, raycaster.far );
				collectHits( hits, intersects, object );
				hits.free();

			}

		} else if ( object.isSprite ) {

			intersectSprite( raycaster, object, intersects );

		}

	}

	if ( recursive ) for ( const child of object.children ) intersectHosted( raycaster, child, true, origin, direction, intersects );

}

// The ray against the sprite's quad as it faces `raycaster.camera`, as three.js does.
function intersectSprite( raycaster, sprite, intersects ) {

	const camera = raycaster.camera;

	if ( camera === null ) {

		console.error( 'WAMR3D: Raycaster.camera needs to be set (setFromCamera) to raycast against sprites.' );
		return;

	}

	const m = w().matrixWorld( sprite.handle ), view = world.matrixWorldInverse( camera.handle ), toWorld = world.matrixWorld( camera.handle );
	let sx = Math.hypot( m[ 0 ], m[ 1 ], m[ 2 ] ), sy = Math.hypot( m[ 4 ], m[ 5 ], m[ 6 ] );
	const mx = view[ 0 ] * m[ 12 ] + view[ 4 ] * m[ 13 ] + view[ 8 ] * m[ 14 ] + view[ 12 ];
	const my = view[ 1 ] * m[ 12 ] + view[ 5 ] * m[ 13 ] + view[ 9 ] * m[ 14 ] + view[ 13 ];
	const mz = view[ 2 ] * m[ 12 ] + view[ 6 ] * m[ 13 ] + view[ 10 ] * m[ 14 ] + view[ 14 ];

	if ( camera.isPerspectiveCamera && sprite.material.sizeAttenuation === false ) {

		sx *= - mz;
		sy *= - mz;

	}

	const rotation = sprite.material.rotation ?? 0, cos = Math.cos( rotation ), sin = Math.sin( rotation );
	const cx = sprite.center.x, cy = sprite.center.y;

	// A corner of the quad (-0.5..0.5) in world space.
	const corner = ( x, y ) => {

		const ax = ( x - cx + 0.5 ) * sx, ay = ( y - cy + 0.5 ) * sy;
		const vx = mx + ( rotation !== 0 ? cos * ax - sin * ay : ax ), vy = my + ( rotation !== 0 ? sin * ax + cos * ay : ay ), vz = mz;
		return [
			toWorld[ 0 ] * vx + toWorld[ 4 ] * vy + toWorld[ 8 ] * vz + toWorld[ 12 ],
			toWorld[ 1 ] * vx + toWorld[ 5 ] * vy + toWorld[ 9 ] * vz + toWorld[ 13 ],
			toWorld[ 2 ] * vx + toWorld[ 6 ] * vy + toWorld[ 10 ] * vz + toWorld[ 14 ]
		];

	};

	const ray = raycaster.ray, o = ray.origin, d = ray.direction;
	const origin = [ o.x, o.y, o.z ], direction = [ d.x, d.y, d.z ];

	const a = corner( - 0.5, - 0.5 ), c = corner( 0.5, 0.5 );
	let b = corner( 0.5, - 0.5 ), uvB = [ 1, 0 ];
	let t = rayTriangle( origin, direction, a, b, c );

	if ( t === null ) {

		b = corner( - 0.5, 0.5 );
		uvB = [ 0, 1 ];
		t = rayTriangle( origin, direction, a, c, b );
		if ( t === null ) return;

	}

	const point = [ origin[ 0 ] + direction[ 0 ] * t, origin[ 1 ] + direction[ 1 ] * t, origin[ 2 ] + direction[ 2 ] * t ];
	const distance = Math.hypot( point[ 0 ] - origin[ 0 ], point[ 1 ] - origin[ 1 ], point[ 2 ] - origin[ 2 ] );
	if ( distance < raycaster.near || distance > raycaster.far ) return;

	const weights = barycoord( point, a, b, c ) ?? [ 0, 0, 0 ];
	const uv = new Vector2( weights[ 1 ] * uvB[ 0 ] + weights[ 2 ], weights[ 1 ] * uvB[ 1 ] + weights[ 2 ] );
	intersects.push( { distance, point: new Vector3( point[ 0 ], point[ 1 ], point[ 2 ] ), uv, face: null, object: sprite } );

}

const sub3 = ( a, b ) => [ a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] ];
const dot3 = ( a, b ) => a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ];
const cross3 = ( a, b ) => [ a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ], a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ], a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ] ];

// Distance along the ray to the triangle, both faces counting, or null (three.js Ray.intersectTriangle).
function rayTriangle( origin, direction, a, b, c ) {

	const edge1 = sub3( b, a ), edge2 = sub3( c, a ), normal = cross3( edge1, edge2 );
	let DdN = dot3( direction, normal ), sign;

	if ( DdN > 0 ) sign = 1;
	else if ( DdN < 0 ) {

		sign = - 1;
		DdN = - DdN;

	} else return null;

	const diff = sub3( origin, a );
	const DdQxE2 = sign * dot3( direction, cross3( diff, edge2 ) );
	if ( DdQxE2 < 0 ) return null;
	const DdE1xQ = sign * dot3( direction, cross3( edge1, diff ) );
	if ( DdE1xQ < 0 || DdQxE2 + DdE1xQ > DdN ) return null;
	const QdN = - sign * dot3( diff, normal );
	if ( QdN < 0 ) return null;
	return QdN / DdN;

}

// Barycentric coordinates of `p` in the triangle (three.js Triangle.getBarycoord), or null.
function barycoord( p, a, b, c ) {

	const v0 = sub3( c, a ), v1 = sub3( b, a ), v2 = sub3( p, a );
	const dot00 = dot3( v0, v0 ), dot01 = dot3( v0, v1 ), dot02 = dot3( v0, v2 ), dot11 = dot3( v1, v1 ), dot12 = dot3( v1, v2 );
	const denom = dot00 * dot11 - dot01 * dot01;
	if ( denom === 0 ) return null;
	const u = ( dot11 * dot02 - dot01 * dot12 ) / denom, v = ( dot00 * dot12 - dot01 * dot02 ) / denom;
	return [ 1 - u - v, v, u ];

}
