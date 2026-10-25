// ShapePath: SVG-style paths turned into shapes with holes.
//
// Part of the curves, paths and shapes; `../curves.js` gathers them.

import { Color } from '../math.js';
import { Vector2 } from '../values.js';
import { Box2 } from './box2.js';
import { Path, Shape } from './paths.js';
import { ShapeUtils } from './shape-utils.js';

function pointInPolygon( p, polygon ) {

	let inside = false;

	for ( let i = 0, n = polygon.length, j = n - 1; i < n; j = i ++ ) {

		const a = polygon[ i ], b = polygon[ j ];
		if ( ( a.y > p.y ) !== ( b.y > p.y ) && p.x < ( b.x - a.x ) * ( p.y - a.y ) / ( b.y - a.y ) + a.x ) inside = ! inside;

	}

	return inside;

}

// A point inside the polygon: the box centre, else the middle of the first span across it.
function interiorPoint( polygon, box ) {

	const point = box.getCenter( new Vector2() );
	if ( pointInPolygon( point, polygon ) ) return point;
	const y = point.y, intercepts = [];

	for ( let i = 0, n = polygon.length; i < n; i ++ ) {

		const a = polygon[ i ], b = polygon[ ( i + 1 ) % n ];
		if ( ( a.y > y ) !== ( b.y > y ) ) intercepts.push( a.x + ( y - a.y ) * ( b.x - a.x ) / ( b.y - a.y ) );

	}

	if ( intercepts.length > 1 ) {

		intercepts.sort( ( a, b ) => a - b );
		point.x = ( intercepts[ 0 ] + intercepts[ 1 ] ) / 2;

	}

	return point;

}

/** Sub-paths, as an SVG path gives them; `toShapes()` sorts them into outlines and holes by the fill rule. */
export class ShapePath {

	constructor() {

		this.type = 'ShapePath';
		this.color = new Color();
		this.subPaths = [];
		this.currentPath = null;
		this.userData = {};

	}

	moveTo( x, y ) {

		this.currentPath = new Path();
		this.subPaths.push( this.currentPath );
		this.currentPath.moveTo( x, y );
		return this;

	}

	lineTo( x, y ) {

		this.currentPath.lineTo( x, y );
		return this;

	}

	quadraticCurveTo( aCPx, aCPy, aX, aY ) {

		this.currentPath.quadraticCurveTo( aCPx, aCPy, aX, aY );
		return this;

	}

	bezierCurveTo( aCP1x, aCP1y, aCP2x, aCP2y, aX, aY ) {

		this.currentPath.bezierCurveTo( aCP1x, aCP1y, aCP2x, aCP2y, aX, aY );
		return this;

	}

	splineThru( pts ) {

		this.currentPath.splineThru( pts );
		return this;

	}

	// Each sub-path is nested in the smallest larger one containing it; with the winding
	// numbers so obtained, the fill rule ('nonzero' or 'evenodd', from
	// `userData.style.fillRule`) decides which are outlines and which are holes.
	toShapes() {

		let fillRule = ( this.userData.style && this.userData.style.fillRule ) || 'nonzero';

		if ( fillRule !== 'nonzero' && fillRule !== 'evenodd' ) {

			console.warn( `WAMR3D: Fill-rule "${fillRule}" is not supported, falling back to "nonzero".` );
			fillRule = 'nonzero';

		}

		const isInside = fillRule === 'nonzero' ? ( w ) => w !== 0 : ( w ) => ( w & 1 ) !== 0;
		const entries = [];

		for ( const subPath of this.subPaths ) {

			const points = subPath.getPoints();
			if ( points.length < 3 ) continue;
			const area = ShapeUtils.area( points );
			if ( area === 0 ) continue;
			const box = new Box2().setFromPoints( points );
			entries.push( { subPath, points, box, interior: interiorPoint( points, box ), absArea: Math.abs( area ), winding: area < 0 ? - 1 : 1, container: null, exclude: false, role: null } );

		}

		entries.sort( ( a, b ) => b.absArea - a.absArea );

		for ( let i = 0; i < entries.length; i ++ ) {

			const entry = entries[ i ];
			let containerWinding = 0;

			for ( let j = i - 1; j >= 0; j -- ) {

				const candidate = entries[ j ];
				if ( ! candidate.box.containsBox( entry.box ) || ! pointInPolygon( entry.interior, candidate.points ) ) continue;
				entry.container = candidate.exclude ? candidate.container : candidate;
				containerWinding = candidate.winding;
				entry.winding += containerWinding;
				break;

			}

			// a sub-path that does not change insideness draws nothing
			if ( isInside( entry.winding ) === isInside( containerWinding ) ) entry.exclude = true;

		}

		for ( const entry of entries ) {

			if ( ! entry.exclude ) entry.role = entry.container === null || entry.container.role === 'hole' ? 'outer' : 'hole';

		}

		const shapes = [], shapeOf = new Map();

		for ( const entry of entries ) {

			if ( entry.exclude || entry.role !== 'outer' ) continue;
			const shape = new Shape();
			shape.curves = entry.subPath.curves;
			shapes.push( shape );
			shapeOf.set( entry, shape );

		}

		for ( const entry of entries ) {

			if ( entry.exclude || entry.role !== 'hole' ) continue;
			const shape = shapeOf.get( entry.container );
			if ( ! shape ) continue;
			const hole = new Path();
			hole.curves = entry.subPath.curves;
			shape.holes.push( hole );

		}

		return shapes;

	}

}
