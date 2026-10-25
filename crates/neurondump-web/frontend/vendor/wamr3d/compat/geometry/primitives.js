// The built-in geometries: boxes, planes, spheres, cylinders, polyhedra, lathes and the rest.
//
// Part of the geometry classes; `../geometry.js` gathers them.

import { w } from '../engine.js';
import { Vector2 } from '../math.js';
import { ADOPT, BufferGeometry } from './buffer-geometry.js';

export class BoxGeometry extends BufferGeometry {

	constructor( width = 1, height = 1, depth = 1, widthSegments = 1, heightSegments = 1, depthSegments = 1 ) {

		super( ADOPT, w().createBoxGeometry( width, height, depth, widthSegments, heightSegments, depthSegments ) );
		this.type = 'BoxGeometry';
		this.parameters = { width, height, depth, widthSegments, heightSegments, depthSegments };

	}

}

export class PlaneGeometry extends BufferGeometry {

	constructor( width = 1, height = 1, widthSegments = 1, heightSegments = 1 ) {

		super( ADOPT, w().createPlaneGeometry( width, height, widthSegments, heightSegments ) );
		this.type = 'PlaneGeometry';
		this.parameters = { width, height, widthSegments, heightSegments };

	}

}

export class SphereGeometry extends BufferGeometry {

	constructor( radius = 1, widthSegments = 32, heightSegments = 16, phiStart = 0, phiLength = Math.PI * 2, thetaStart = 0, thetaLength = Math.PI ) {

		super( ADOPT, w().createSphereGeometry( radius, widthSegments, heightSegments, phiStart, phiLength, thetaStart, thetaLength ) );
		this.type = 'SphereGeometry';
		this.parameters = { radius, widthSegments, heightSegments, phiStart, phiLength, thetaStart, thetaLength };

	}

}

export class CylinderGeometry extends BufferGeometry {

	constructor( radiusTop = 1, radiusBottom = 1, height = 1, radialSegments = 32, heightSegments = 1, openEnded = false, thetaStart = 0, thetaLength = Math.PI * 2 ) {

		super( ADOPT, w().createCylinderGeometry( radiusTop, radiusBottom, height, Math.floor( radialSegments ), Math.floor( heightSegments ), openEnded, thetaStart, thetaLength ) );
		this.type = 'CylinderGeometry';
		this.parameters = { radiusTop, radiusBottom, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength };

	}

}

export class ConeGeometry extends CylinderGeometry {

	constructor( radius = 1, height = 1, radialSegments = 32, heightSegments = 1, openEnded = false, thetaStart = 0, thetaLength = Math.PI * 2 ) {

		super( 0, radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength );
		this.type = 'ConeGeometry';
		this.parameters = { radius, height, radialSegments, heightSegments, openEnded, thetaStart, thetaLength };

	}

}

export class CircleGeometry extends BufferGeometry {

	constructor( radius = 1, segments = 32, thetaStart = 0, thetaLength = Math.PI * 2 ) {

		super( ADOPT, w().createCircleGeometry( radius, segments, thetaStart, thetaLength ) );
		this.type = 'CircleGeometry';
		this.parameters = { radius, segments, thetaStart, thetaLength };

	}

}

export class RingGeometry extends BufferGeometry {

	constructor( innerRadius = 0.5, outerRadius = 1, thetaSegments = 32, phiSegments = 1, thetaStart = 0, thetaLength = Math.PI * 2 ) {

		super( ADOPT, w().createRingGeometry( innerRadius, outerRadius, thetaSegments, phiSegments, thetaStart, thetaLength ) );
		this.type = 'RingGeometry';
		this.parameters = { innerRadius, outerRadius, thetaSegments, phiSegments, thetaStart, thetaLength };

	}

}

export class TorusGeometry extends BufferGeometry {

	constructor( radius = 1, tube = 0.4, radialSegments = 12, tubularSegments = 48, arc = Math.PI * 2, thetaStart = 0, thetaLength = Math.PI * 2 ) {

		super( ADOPT, w().createTorusGeometry( radius, tube, Math.floor( radialSegments ), Math.floor( tubularSegments ), arc, thetaStart, thetaLength ) );
		this.type = 'TorusGeometry';
		this.parameters = { radius, tube, radialSegments, tubularSegments, arc, thetaStart, thetaLength };

	}

}

/** Triangles `indices` over the xyz `vertices`, subdivided `detail` times and pushed out to `radius`. Not indexed. */
export class PolyhedronGeometry extends BufferGeometry {

	constructor( vertices = [], indices = [], radius = 1, detail = 0 ) {

		super( ADOPT, w().createPolyhedronGeometry( Float32Array.from( vertices ), Uint32Array.from( indices ), radius, Math.floor( detail ) ) );
		this.type = 'PolyhedronGeometry';
		this.parameters = { vertices, indices, radius, detail };

	}

}

// The platonic solids, built in the engine from three.js vertex tables.
function platonic( type, faces ) {

	return class extends BufferGeometry {

		constructor( radius = 1, detail = 0 ) {

			super( ADOPT, w().createPlatonicGeometry( faces, radius, Math.floor( detail ) ) );
			this.type = type;
			this.parameters = { radius, detail };

		}

	};

}

export const TetrahedronGeometry = platonic( 'TetrahedronGeometry', 4 );
export const OctahedronGeometry = platonic( 'OctahedronGeometry', 8 );
export const DodecahedronGeometry = platonic( 'DodecahedronGeometry', 12 );
export const IcosahedronGeometry = platonic( 'IcosahedronGeometry', 20 );

export class TorusKnotGeometry extends BufferGeometry {

	constructor( radius = 1, tube = 0.4, tubularSegments = 64, radialSegments = 8, p = 2, q = 3 ) {

		super( ADOPT, w().createTorusKnotGeometry( radius, tube, Math.floor( tubularSegments ), Math.floor( radialSegments ), p, q ) );
		this.type = 'TorusKnotGeometry';
		this.parameters = { radius, tube, tubularSegments, radialSegments, p, q };

	}

}

/** Turns the profile `points` (Vector2: x from the axis, y along it) around the y-axis. */
export class LatheGeometry extends BufferGeometry {

	constructor( points = [ new Vector2( 0, - 0.5 ), new Vector2( 0.5, 0 ), new Vector2( 0, 0.5 ) ], segments = 12, phiStart = 0, phiLength = Math.PI * 2 ) {

		const profile = new Float32Array( points.length * 2 );
		points.forEach( ( point, i ) => {

			profile[ 2 * i ] = point.x;
			profile[ 2 * i + 1 ] = point.y;

		} );
		super( ADOPT, w().createLatheGeometry( profile, Math.floor( segments ), phiStart, phiLength ) );
		this.type = 'LatheGeometry';
		this.parameters = { points, segments, phiStart, phiLength };

	}

}

export class CapsuleGeometry extends BufferGeometry {

	constructor( radius = 1, height = 1, capSegments = 4, radialSegments = 8, heightSegments = 1 ) {

		super( ADOPT, w().createCapsuleGeometry( radius, height, Math.floor( capSegments ), Math.floor( radialSegments ), Math.floor( heightSegments ) ) );
		this.type = 'CapsuleGeometry';
		this.parameters = { radius, height, capSegments, radialSegments, heightSegments };

	}

}

// `fromJSON( data )` of each built-in shape: its constructor called with the parameters
// `toJSON` wrote.
for ( const [ Class, names ] of [
	[ BoxGeometry, [ 'width', 'height', 'depth', 'widthSegments', 'heightSegments', 'depthSegments' ] ],
	[ PlaneGeometry, [ 'width', 'height', 'widthSegments', 'heightSegments' ] ],
	[ SphereGeometry, [ 'radius', 'widthSegments', 'heightSegments', 'phiStart', 'phiLength', 'thetaStart', 'thetaLength' ] ],
	[ CylinderGeometry, [ 'radiusTop', 'radiusBottom', 'height', 'radialSegments', 'heightSegments', 'openEnded', 'thetaStart', 'thetaLength' ] ],
	[ ConeGeometry, [ 'radius', 'height', 'radialSegments', 'heightSegments', 'openEnded', 'thetaStart', 'thetaLength' ] ],
	[ CircleGeometry, [ 'radius', 'segments', 'thetaStart', 'thetaLength' ] ],
	[ RingGeometry, [ 'innerRadius', 'outerRadius', 'thetaSegments', 'phiSegments', 'thetaStart', 'thetaLength' ] ],
	[ TorusGeometry, [ 'radius', 'tube', 'radialSegments', 'tubularSegments', 'arc', 'thetaStart', 'thetaLength' ] ],
	[ PolyhedronGeometry, [ 'vertices', 'indices', 'radius', 'detail' ] ],
	[ TetrahedronGeometry, [ 'radius', 'detail' ] ],
	[ OctahedronGeometry, [ 'radius', 'detail' ] ],
	[ DodecahedronGeometry, [ 'radius', 'detail' ] ],
	[ IcosahedronGeometry, [ 'radius', 'detail' ] ],
	[ TorusKnotGeometry, [ 'radius', 'tube', 'tubularSegments', 'radialSegments', 'p', 'q' ] ],
	[ CapsuleGeometry, [ 'radius', 'height', 'capSegments', 'radialSegments', 'heightSegments' ] ]
] ) {

	Class.fromJSON = ( data ) => new Class( ...names.map( ( name ) => data[ name ] ) );

}

LatheGeometry.fromJSON = ( data ) => new LatheGeometry( data.points.map( ( p ) => new Vector2( p.x, p.y ) ), data.segments, data.phiStart, data.phiLength );
