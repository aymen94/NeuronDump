// Vertex, index and instance buffers, quads and derived indices.
//
// Part of the WebGPU host; `../webgpu.js` creates the device and draws the frames.

import { memory } from '../index.js';
import { auxRecord } from './formats.js';
import { ATTRIBUTES, AUX_ATTRIBUTES, AUX_FLOATS, AUX_OFFSETS, AUX_SIZES, AUX_SLOT, VERTEX_COLOR_SLOT, wireframeCount } from './constants.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installBuffers( ctx ) {

	const { auxZeros, device, geometries, info, instanced } = ctx;

	function upload( data, existing, usage ) {

		let buffer = existing;

		if ( ! buffer || buffer.size < data.byteLength ) {

			existing?.destroy();
			buffer = device.createBuffer( { size: Math.max( 16, data.byteLength ), usage: usage | GPUBufferUsage.COPY_DST } );

		}

		device.queue.writeBuffer( buffer, 0, data );
		info.uploads ++;
		return buffer;

	}

	// GPU buffers follow the engine's data: an attribute or the index is uploaded again
	// only when its version changes. `state[ o ... ]` is the geometry's record from
	// `Renderer.geometryState`. Entries are keyed by the handle's slot; a new generation
	// in the slot means the geometry was disposed and the slot reused.
	function geometry( world, state, o ) {

		const slot = state[ o ], generation = state[ o + 1 ], vertices = state[ o + 2 ];
		let entry = geometries.get( slot );

		if ( entry && entry.generation !== generation ) {

			release( entry );
			entry = undefined;

		}

		if ( ! entry ) geometries.set( slot, entry = { generation, handle: BigInt( generation ) << 32n | BigInt( slot ), vertices: 0, buffers: [ null, null, null, null ], versions: [ - 1, - 1, - 1, - 1 ], sizes: [ 0, 0, 0, 0 ], aux: null, auxVersions: '', index: null, indexVersion: - 1, loops: null, quads: null, checked: 0 } );
		entry.checked = ctx.frameNumber;
		entry.vertices = vertices;
		const records = ATTRIBUTES.length + AUX_ATTRIBUTES;
		const colorSize = state[ o + 4 + records * 2 ];

		for ( let i = 0, p = o + 3; i < ATTRIBUTES.length; i ++, p += 2 ) {

			const version = state[ p ], address = state[ p + 1 ], size = i === VERTEX_COLOR_SLOT ? colorSize : ATTRIBUTES[ i ][ 1 ];

			if ( address === 0 ) {

				entry.buffers[ i ]?.destroy();
				entry.buffers[ i ] = null;
				entry.versions[ i ] = - 1;
				ctx.zeroBytes = Math.max( ctx.zeroBytes, vertices * 16 );
				continue;

			}

			if ( version === entry.versions[ i ] && size === entry.sizes[ i ] ) continue;
			// Created per upload: the index read below may grow the module's memory.
			entry.buffers[ i ] = upload( new Float32Array( memory().buffer, address, vertices * size ), entry.buffers[ i ], GPUBufferUsage.VERTEX );
			entry.versions[ i ] = version;
			entry.sizes[ i ] = size;

		}

		// tangent, uv1, skinIndex and skinWeight, interleaved when any of them changed
		const auxKey = auxRecord( state, o + 3 + ATTRIBUTES.length * 2, AUX_ATTRIBUTES );

		if ( auxKey !== entry.auxVersions ) {

			entry.auxVersions = auxKey;

			if ( auxKey === '' ) {

				entry.aux?.destroy();
				entry.aux = null;

			} else {

				const data = new Float32Array( vertices * AUX_FLOATS ), view = new Float32Array( memory().buffer );
				const offsets = AUX_OFFSETS, sizes = AUX_SIZES;

				for ( let a = 0, p = o + 3 + ATTRIBUTES.length * 2; a < AUX_ATTRIBUTES; a ++, p += 2 ) {

					const address = state[ p + 1 ];
					if ( address === 0 ) continue;
					const base = address >>> 2, size = sizes[ a ], offset = offsets[ a ];
					for ( let v = 0; v < vertices; v ++ ) for ( let k = 0; k < size; k ++ ) data[ v * AUX_FLOATS + offset + k ] = view[ base + v * size + k ];

				}

				entry.aux = upload( data, entry.aux, GPUBufferUsage.VERTEX );

			}

		}

		const indexVersion = state[ o + 3 + records * 2 ];

		if ( indexVersion !== entry.indexVersion ) {

			const index = world.getIndex( entry.handle );

			if ( index.length === 0 ) {

				entry.index?.destroy();
				entry.index = null;

			} else {

				entry.index = upload( index, entry.index, GPUBufferUsage.INDEX );

			}

			entry.indexVersion = indexVersion;

		}

		return entry;

	}

	function bindGeometry( pass, entry ) {

		for ( let slot = 0; slot < ATTRIBUTES.length; slot ++ ) pass.setVertexBuffer( slot, entry.buffers[ slot ] ?? ctx.zeros );
		pass.setVertexBuffer( AUX_SLOT, entry.aux ?? auxZeros );
		if ( entry.index ) pass.setIndexBuffer( entry.index, 'uint32' );

	}

	// Instance matrices and colours live in typed arrays on the JavaScript side; they are
	// uploaded when their version changes.
	function instances( id, count, sources ) {

		const source = id === 0 ? null : sources?.get( id );
		if ( ! source ) return null;

		let entry = instanced.get( id );
		if ( ! entry ) instanced.set( id, entry = { matrix: null, color: null, matrixVersion: - 1, colorVersion: - 1, checked: 0 } );
		entry.checked = ctx.frameNumber;

		if ( entry.matrixVersion !== source.matrixVersion ) {

			entry.matrix = upload( source.matrix, entry.matrix, GPUBufferUsage.VERTEX );
			entry.matrixVersion = source.matrixVersion;

		}

		if ( source.color === null ) {

			entry.color?.destroy();
			entry.color = null;
			entry.colorVersion = - 1;
			ctx.oneCount = Math.max( ctx.oneCount, count );

		} else if ( entry.colorVersion !== source.colorVersion ) {

			entry.color = upload( source.color, entry.color, GPUBufferUsage.VERTEX );
			entry.colorVersion = source.colorVersion;

		}

		return entry;

	}

	// The vertices of an indexed geometry in index order, for quads (which read vertices
	// per instance and so cannot follow an index), kept until the index or data changes.
	// A line loop gets the `count` vertices from `start` and then the first again, so that
	// its closing segment is drawn too; the result then starts at 0.
	function quadSource( world, entry, start = 0, count = 0, loop = false ) {

		const dashed = ( ',' + entry.auxVersions ).includes( ',4:' );

		if ( ! entry.index && ! loop ) {

			// line distances in a buffer of their own, read per instance like the positions
			if ( entry.distanceKey !== entry.auxVersions ) {

				entry.distance?.destroy();
				entry.distance = dashed ? upload( new Float32Array( world.getAttribute( entry.handle, 'lineDistance' ) ), null, GPUBufferUsage.VERTEX ) : null;
				entry.distanceKey = entry.auxVersions;

			}

			return entry;

		}

		const key = entry.indexVersion + ':' + entry.versions[ 0 ] + ':' + entry.versions[ 3 ] + ':' + entry.auxVersions + ( loop ? ':' + start + ':' + count : '' );
		if ( entry.quads && entry.quads.key === key ) return entry.quads;
		entry.quads?.position?.destroy();
		entry.quads?.color?.destroy();
		entry.quads?.distance?.destroy();
		const position = world.getAttribute( entry.handle, 'position' );
		let index = entry.index ? world.getIndex( entry.handle ) : null;

		if ( loop ) {

			const order = new Uint32Array( count + 1 );
			for ( let i = 0; i < count; i ++ ) order[ i ] = index ? index[ start + i ] : start + i;
			order[ count ] = order[ 0 ];
			index = order;

		}

		const unrolled = ( data, size ) => {

			const out = new Float32Array( index.length * size );
			for ( let i = 0; i < index.length; i ++ ) for ( let k = 0; k < size; k ++ ) out[ i * size + k ] = data[ index[ i ] * size + k ];
			return out;

		};

		const colorSize = entry.sizes[ VERTEX_COLOR_SLOT ];
		const color = entry.buffers[ VERTEX_COLOR_SLOT ] ? upload( unrolled( world.getAttribute( entry.handle, 'color' ), colorSize ), null, GPUBufferUsage.VERTEX ) : null;
		ctx.zeroBytes = Math.max( ctx.zeroBytes, index.length * 16 );
		const distance = dashed ? upload( unrolled( world.getAttribute( entry.handle, 'lineDistance' ), 1 ), null, GPUBufferUsage.VERTEX ) : null;
		entry.quads = { key, position: upload( unrolled( position, 3 ), null, GPUBufferUsage.VERTEX ), color, distance, index: null };
		entry.quads.buffers = [ entry.quads.position, null, null, color ];
		return entry.quads;

	}

	// Indices derived from the `count` vertices from `start` (through the geometry's index,
	// if any): for a line loop those vertices then the first one again, for a wireframe
	// the three edges of each triangle (as three.js builds them, shared edges twice).
	// Kept per geometry until its index changes.
	function derivedIndex( world, entry, start, count, wireframe ) {

		if ( entry.loops === null || entry.loopsVersion !== entry.indexVersion ) {

			if ( entry.loops ) for ( const buffer of entry.loops.values() ) buffer.destroy();
			entry.loops = new Map();
			entry.loopsVersion = entry.indexVersion;

		}

		const key = ( wireframe ? 'w' : 'l' ) + start + ':' + count;
		let buffer = entry.loops.get( key );
		if ( buffer ) return buffer;

		const vertices = new Uint32Array( count );

		if ( entry.index ) {

			const index = world.getIndex( entry.handle );
			for ( let i = 0; i < count; i ++ ) vertices[ i ] = index[ start + i ];

		} else {

			for ( let i = 0; i < count; i ++ ) vertices[ i ] = start + i;

		}

		let data;

		if ( wireframe ) {

			data = new Uint32Array( wireframeCount( count ) );

			for ( let i = 0, o = 0; o < data.length; i += 3, o += 6 ) {

				const a = vertices[ i ], b = vertices[ i + 1 ], c = vertices[ i + 2 ];
				data[ o ] = a; data[ o + 1 ] = b; data[ o + 2 ] = b; data[ o + 3 ] = c; data[ o + 4 ] = c; data[ o + 5 ] = a;

			}

		} else {

			data = new Uint32Array( count + 1 );
			data.set( vertices );
			data[ count ] = vertices[ 0 ];

		}

		buffer = upload( data, null, GPUBufferUsage.INDEX );
		entry.loops.set( key, buffer );
		return buffer;

	}

	function release( entry ) {

		if ( entry.release ) return entry.release();
		for ( const buffer of entry.buffers ) buffer?.destroy();
		entry.aux?.destroy();
		entry.index?.destroy();
		entry.quads?.position?.destroy();
		entry.quads?.color?.destroy();
		entry.quads?.distance?.destroy();
		entry.distance?.destroy();
		if ( entry.loops ) for ( const buffer of entry.loops.values() ) buffer.destroy();

	}

	Object.assign( ctx, { bindGeometry, derivedIndex, geometry, instances, quadSource, release } );

}
