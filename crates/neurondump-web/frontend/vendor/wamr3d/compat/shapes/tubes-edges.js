// TubeGeometry, EdgesGeometry and WireframeGeometry.
//
// Part of the shape geometries; `../shapes.js` gathers them.

import { Curves, QuadraticBezierCurve3 } from '../curves.js';
import { BufferGeometry, Float32BufferAttribute } from '../geometry.js';
import { Vector3 } from '../values.js';
import { DEG2RAD } from './extrude.js';

/** A tube of `radius` around `path` (a 3D curve), with `radialSegments` around and `tubularSegments` along. */
export class TubeGeometry extends BufferGeometry {

	constructor( path = new QuadraticBezierCurve3( new Vector3( - 1, - 1, 0 ), new Vector3( - 1, 1, 0 ), new Vector3( 1, 1, 0 ) ), tubularSegments = 64, radius = 1, radialSegments = 8, closed = false ) {

		super();
		this.type = 'TubeGeometry';
		this.parameters = { path, tubularSegments, radius, radialSegments, closed };

		const frames = path.computeFrenetFrames( tubularSegments, closed );
		this.tangents = frames.tangents;
		this.normals = frames.normals;
		this.binormals = frames.binormals;

		const rings = tubularSegments + 1, ring = radialSegments + 1;
		const vertices = new Float32Array( rings * ring * 3 ), normals = new Float32Array( rings * ring * 3 ), uvs = new Float32Array( rings * ring * 2 );
		const P = new Vector3();

		for ( let i = 0; i < rings; i ++ ) {

			// a closed tube ends on its first ring
			const segment = i === tubularSegments && closed ? 0 : i;
			path.getPointAt( segment / tubularSegments, P );
			const N = frames.normals[ segment ], B = frames.binormals[ segment ];

			for ( let j = 0; j <= radialSegments; j ++ ) {

				const angle = j / radialSegments * Math.PI * 2, sin = Math.sin( angle ), cos = - Math.cos( angle );
				let nx = cos * N.x + sin * B.x, ny = cos * N.y + sin * B.y, nz = cos * N.z + sin * B.z;
				const length = Math.hypot( nx, ny, nz ) || 1;
				nx /= length;
				ny /= length;
				nz /= length;
				const o = ( i * ring + j ) * 3;
				normals[ o ] = nx;
				normals[ o + 1 ] = ny;
				normals[ o + 2 ] = nz;
				vertices[ o ] = P.x + radius * nx;
				vertices[ o + 1 ] = P.y + radius * ny;
				vertices[ o + 2 ] = P.z + radius * nz;
				uvs[ ( i * ring + j ) * 2 ] = i / tubularSegments;
				uvs[ ( i * ring + j ) * 2 + 1 ] = j / radialSegments;

			}

		}

		const indices = new Uint32Array( tubularSegments * radialSegments * 6 );

		for ( let j = 1, o = 0; j <= tubularSegments; j ++ ) {

			for ( let i = 1; i <= radialSegments; i ++ ) {

				const a = ring * ( j - 1 ) + ( i - 1 ), b = ring * j + ( i - 1 ), c = ring * j + i, d = ring * ( j - 1 ) + i;
				indices[ o ++ ] = a;
				indices[ o ++ ] = b;
				indices[ o ++ ] = d;
				indices[ o ++ ] = b;
				indices[ o ++ ] = c;
				indices[ o ++ ] = d;

			}

		}

		this.setIndex( indices );
		this.setAttribute( 'position', new Float32BufferAttribute( vertices, 3 ) );
		this.setAttribute( 'normal', new Float32BufferAttribute( normals, 3 ) );
		this.setAttribute( 'uv', new Float32BufferAttribute( uvs, 2 ) );

	}

}

// Each triangle's vertex indices, from the index or in order.
function triangleIndex( geometry ) {

	const index = geometry.getIndex(), position = geometry.getAttribute( 'position' );
	if ( index ) return Array.from( { length: index.count }, ( _, i ) => index.getX( i ) );
	return Array.from( { length: position.count }, ( _, i ) => i );

}

/**
 * Line segments along the edges of `geometry` where its faces meet at more than
 * `thresholdAngle` degrees, and along open edges. For `LineSegments`.
 */
export class EdgesGeometry extends BufferGeometry {

	constructor( geometry = null, thresholdAngle = 1 ) {

		super();
		this.type = 'EdgesGeometry';
		this.parameters = { geometry, thresholdAngle };
		if ( geometry === null ) return;

		const precision = Math.pow( 10, 4 ), thresholdDot = Math.cos( DEG2RAD * thresholdAngle );
		const position = geometry.getAttribute( 'position' ), indices = triangleIndex( geometry );
		const points = [ new Vector3(), new Vector3(), new Vector3() ], hashes = [ '', '', '' ], corner = [ 0, 0, 0 ];
		const edgeData = new Map(), vertices = [];
		const hashOf = ( p ) => `${Math.round( p.x * precision )},${Math.round( p.y * precision )},${Math.round( p.z * precision )}`;
		const ab = new Vector3(), cb = new Vector3();

		for ( let i = 0; i + 2 < indices.length; i += 3 ) {

			for ( let j = 0; j < 3; j ++ ) {

				corner[ j ] = indices[ i + j ];
				points[ j ].fromBufferAttribute( position, corner[ j ] );
				hashes[ j ] = hashOf( points[ j ] );

			}

			// degenerate triangles have no normal
			if ( hashes[ 0 ] === hashes[ 1 ] || hashes[ 1 ] === hashes[ 2 ] || hashes[ 2 ] === hashes[ 0 ] ) continue;
			const normal = cb.subVectors( points[ 2 ], points[ 1 ] ).cross( ab.subVectors( points[ 0 ], points[ 1 ] ) ).clone();
			const lengthSq = normal.lengthSq();
			if ( lengthSq > 0 ) normal.multiplyScalar( 1 / Math.sqrt( lengthSq ) );
			else normal.set( 0, 0, 0 );

			for ( let j = 0; j < 3; j ++ ) {

				const next = ( j + 1 ) % 3, hash = `${hashes[ j ]}_${hashes[ next ]}`, reverse = `${hashes[ next ]}_${hashes[ j ]}`;
				const shared = edgeData.get( reverse );

				if ( shared ) {

					// the second face of an edge: kept if the faces meet sharply enough
					if ( normal.dot( shared.normal ) <= thresholdDot ) {

						vertices.push( points[ j ].x, points[ j ].y, points[ j ].z, points[ next ].x, points[ next ].y, points[ next ].z );

					}

					edgeData.set( reverse, null );

				} else if ( ! edgeData.has( hash ) ) {

					edgeData.set( hash, { index0: corner[ j ], index1: corner[ next ], normal } );

				}

			}

		}

		// edges with a single face
		for ( const data of edgeData.values() ) {

			if ( ! data ) continue;
			points[ 0 ].fromBufferAttribute( position, data.index0 );
			points[ 1 ].fromBufferAttribute( position, data.index1 );
			vertices.push( points[ 0 ].x, points[ 0 ].y, points[ 0 ].z, points[ 1 ].x, points[ 1 ].y, points[ 1 ].z );

		}

		this.setAttribute( 'position', new Float32BufferAttribute( vertices, 3 ) );

	}

}

/** Line segments along every distinct triangle edge of `geometry` (per group when indexed). For `LineSegments`. */
export class WireframeGeometry extends BufferGeometry {

	constructor( geometry = null ) {

		super();
		this.type = 'WireframeGeometry';
		this.parameters = { geometry };
		if ( geometry === null ) return;

		const vertices = [], edges = new Set(), start = new Vector3(), end = new Vector3();
		const position = geometry.attributes.position, index = geometry.index;

		const edge = ( a, b ) => {

			start.fromBufferAttribute( position, a );
			end.fromBufferAttribute( position, b );
			const forward = `${start.x},${start.y},${start.z}-${end.x},${end.y},${end.z}`;
			if ( edges.has( forward ) ) return;
			edges.add( forward );
			edges.add( `${end.x},${end.y},${end.z}-${start.x},${start.y},${start.z}` );
			vertices.push( start.x, start.y, start.z, end.x, end.y, end.z );

		};

		if ( index !== null ) {

			const groups = geometry.groups.length > 0 ? geometry.groups : [ { start: 0, count: index.count, materialIndex: 0 } ];

			for ( const group of groups ) {

				for ( let i = group.start, l = group.start + group.count; i < l; i += 3 ) {

					for ( let j = 0; j < 3; j ++ ) edge( index.getX( i + j ), index.getX( i + ( j + 1 ) % 3 ) );

				}

			}

		} else {

			for ( let i = 0, l = position.count / 3; i < l; i ++ ) {

				for ( let j = 0; j < 3; j ++ ) edge( 3 * i + j, 3 * i + ( j + 1 ) % 3 );

			}

		}

		this.setAttribute( 'position', new Float32BufferAttribute( vertices, 3 ) );

	}

}

TubeGeometry.prototype.toJSON = function () {

	const data = BufferGeometry.prototype.toJSON.call( this );
	data.path = this.parameters.path.toJSON();
	return data;

};

TubeGeometry.fromJSON = ( data ) => new TubeGeometry( new Curves[ data.path.type ]().fromJSON( data.path ), data.tubularSegments, data.radius, data.radialSegments, data.closed );
