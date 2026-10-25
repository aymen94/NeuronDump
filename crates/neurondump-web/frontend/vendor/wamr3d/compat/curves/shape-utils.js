// ShapeUtils and Earcut: winding, area and triangulation of shapes with holes.
//
// Part of the curves, paths and shapes; `../curves.js` gathers them.

import earcut from '../lib/earcut.js';

function removeDupEndPts( points ) {

	const l = points.length;
	if ( l > 2 && points[ l - 1 ].equals( points[ 0 ] ) ) points.pop();

}

export const Earcut = {

	triangulate( data, holeIndices, dim = 2 ) {

		return earcut( data, holeIndices, dim );

	}

};

export class ShapeUtils {

	/** Signed area of a contour; negative when clockwise. */
	static area( contour ) {

		const n = contour.length;
		let a = 0;
		for ( let p = n - 1, q = 0; q < n; p = q ++ ) a += contour[ p ].x * contour[ q ].y - contour[ q ].x * contour[ p ].y;
		return a * 0.5;

	}

	static isClockWise( pts ) {

		return ShapeUtils.area( pts ) < 0;

	}

	/** Triangles over the contour followed by the holes' points, as index triples. Removes duplicated end points in place. */
	static triangulateShape( contour, holes ) {

		const vertices = [], holeIndices = [], faces = [];
		removeDupEndPts( contour );
		for ( const point of contour ) vertices.push( point.x, point.y );
		let holeIndex = contour.length;
		holes.forEach( removeDupEndPts );

		for ( const hole of holes ) {

			holeIndices.push( holeIndex );
			holeIndex += hole.length;
			for ( const point of hole ) vertices.push( point.x, point.y );

		}

		const triangles = earcut( vertices, holeIndices );
		for ( let i = 0; i < triangles.length; i += 3 ) faces.push( triangles.slice( i, i + 3 ) );
		return faces;

	}

}
