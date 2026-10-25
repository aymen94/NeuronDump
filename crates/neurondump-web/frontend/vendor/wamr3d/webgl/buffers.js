// Vertex, index and instance buffers, pipeline state, draws and uniform-block bindings.
//
// Part of the WebGL 2 host; `../webgl.js` creates the context and draws the frames.

import { memory } from '../index.js';
import { auxRecord } from '../webgpu/formats.js';
import { ALPHA_TO_COVERAGE_KEY, ATTRIBUTES, AUX_ATTRIBUTES, BLEND_COLOR_FLOAT, COLOR_WRITE_OFF_KEY, DEPTH_FUNC_SHIFT, POLYGON_OFFSET_FLOAT, STENCIL_FLOAT, AUX_FLOATS, AUX_LOCATIONS, AUX_OFFSETS, AUX_SIZES, AUX_STRIDE, INSTANCE_COLOR_SLOT, INSTANCE_SLOT, LOOP_KEY, MISSING, PREMULTIPLIED_KEY, QUAD_B, QUAD_COLOR_B, QUAD_KEY, VERTEX_COLOR_SLOT, wireframeCount } from './constants.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installBuffers( ctx ) {

	const { FACTORS, TEXTURE_SLOTS, gl, ranges } = ctx;

	// ---------------------------------------------------------------- buffers

	const info = { calls: 0, shadowCalls: 0, triangles: 0, candidates: 0, geometries: 0, textures: 0, pipelines: 0, uploads: 0 };

	function upload( target, data, existing ) {

		const buffer = existing ?? gl.createBuffer();
		gl.bindBuffer( target, buffer );
		gl.bufferData( target, data, gl.DYNAMIC_DRAW );
		info.uploads ++;
		return buffer;

	}

	const geometries = new Map(), instanced = new Map();
	const entries = [], perInstance = [], distinct = [], setTextures = [];

	// GPU buffers follow the engine's data, re-uploaded when an attribute's version changes
	// (see `Renderer.geometryState`).
	function geometry( world, state, o ) {

		const slot = state[ o ], generation = state[ o + 1 ], vertices = state[ o + 2 ];
		let entry = geometries.get( slot );

		if ( entry && entry.generation !== generation ) {

			release( entry );
			entry = undefined;

		}

		if ( ! entry ) geometries.set( slot, entry = { generation, handle: BigInt( generation ) << 32n | BigInt( slot ), buffers: [ null, null, null, null ], versions: [ - 1, - 1, - 1, - 1 ], sizes: [ 0, 0, 0, 0 ], aux: null, auxVersions: '', index: null, indexVersion: - 1, quads: null, checked: 0 } );
		entry.checked = ctx.frameNumber;
		const records = ATTRIBUTES.length + AUX_ATTRIBUTES;
		const colorSize = state[ o + 4 + records * 2 ];

		for ( let i = 0, p = o + 3; i < ATTRIBUTES.length; i ++, p += 2 ) {

			const version = state[ p ], address = state[ p + 1 ], size = i === VERTEX_COLOR_SLOT ? colorSize : ATTRIBUTES[ i ][ 1 ];

			if ( address === 0 ) {

				if ( entry.buffers[ i ] ) gl.deleteBuffer( entry.buffers[ i ] );
				entry.buffers[ i ] = null;
				entry.versions[ i ] = - 1;
				continue;

			}

			if ( version === entry.versions[ i ] && size === entry.sizes[ i ] ) continue;
			entry.buffers[ i ] = upload( gl.ARRAY_BUFFER, new Float32Array( memory().buffer, address, vertices * size ), entry.buffers[ i ] );
			entry.versions[ i ] = version;
			entry.sizes[ i ] = size;

		}

		const auxKey = auxRecord( state, o + 3 + ATTRIBUTES.length * 2, AUX_ATTRIBUTES );

		if ( auxKey !== entry.auxVersions ) {

			entry.auxVersions = auxKey;

			if ( auxKey === '' ) {

				if ( entry.aux ) gl.deleteBuffer( entry.aux );
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

				entry.aux = upload( gl.ARRAY_BUFFER, data, entry.aux );

			}

		}

		const indexVersion = state[ o + 3 + records * 2 ];

		if ( indexVersion !== entry.indexVersion ) {

			const index = world.getIndex( entry.handle );

			if ( index.length === 0 ) {

				if ( entry.index ) gl.deleteBuffer( entry.index );
				entry.index = null;

			} else {

				entry.index = upload( gl.ELEMENT_ARRAY_BUFFER, index, entry.index );

			}

			entry.indexVersion = indexVersion;

		}

		return entry;

	}

	function instances( id, sources ) {

		const source = id === 0 ? null : sources?.get( id );
		if ( ! source ) return null;
		let entry = instanced.get( id );
		if ( ! entry ) instanced.set( id, entry = { matrix: null, color: null, matrixVersion: - 1, colorVersion: - 1, checked: 0 } );
		entry.checked = ctx.frameNumber;

		if ( entry.matrixVersion !== source.matrixVersion ) {

			entry.matrix = upload( gl.ARRAY_BUFFER, source.matrix, entry.matrix );
			entry.matrixVersion = source.matrixVersion;

		}

		if ( source.color === null ) {

			if ( entry.color ) gl.deleteBuffer( entry.color );
			entry.color = null;
			entry.colorVersion = - 1;

		} else if ( entry.colorVersion !== source.colorVersion ) {

			entry.color = upload( gl.ARRAY_BUFFER, source.color, entry.color );
			entry.colorVersion = source.colorVersion;

		}

		return entry;

	}

	function release( entry ) {

		for ( const buffer of entry.buffers ) if ( buffer ) gl.deleteBuffer( buffer );
		if ( entry.aux ) gl.deleteBuffer( entry.aux );
		if ( entry.index ) gl.deleteBuffer( entry.index );
		if ( entry.edges ) for ( const edges of entry.edges.values() ) gl.deleteBuffer( edges );
		if ( entry.quads ) for ( const buffer of [ entry.quads.position, entry.quads.color, entry.quads.distance ] ) if ( buffer ) gl.deleteBuffer( buffer );
		if ( entry.distances?.buffer ) gl.deleteBuffer( entry.distances.buffer );
		if ( entry.custom ) for ( const cached of entry.custom.values() ) gl.deleteBuffer( cached.buffer );

	}

	// The index of a wireframe over the `count` vertices from `start` (through the
	// geometry's index, if any): the three edges of each triangle, as three.js builds
	// them. Kept per geometry until its index changes.
	function wireframeIndex( world, entry, start, count ) {

		if ( ! entry.edges || entry.edgesVersion !== entry.indexVersion ) {

			if ( entry.edges ) for ( const edges of entry.edges.values() ) gl.deleteBuffer( edges );
			entry.edges = new Map();
			entry.edgesVersion = entry.indexVersion;

		}

		const key = start + ':' + count;
		let buffer = entry.edges.get( key );
		if ( buffer ) return buffer;

		const index = entry.index ? world.getIndex( entry.handle ) : null;
		const data = new Uint32Array( wireframeCount( count ) );

		for ( let i = start, o = 0; o < data.length; i += 3, o += 6 ) {

			const a = index ? index[ i ] : i, b = index ? index[ i + 1 ] : i + 1, c = index ? index[ i + 2 ] : i + 2;
			data[ o ] = a; data[ o + 1 ] = b; data[ o + 2 ] = b; data[ o + 3 ] = c; data[ o + 4 ] = c; data[ o + 5 ] = a;

		}

		buffer = upload( gl.ELEMENT_ARRAY_BUFFER, data );
		entry.edges.set( key, buffer );
		return buffer;

	}

	// What each vertex input reads, so that a draw makes GL calls only for the inputs that
	// differ from the previous draw's (geometries of one kind share all but their buffers).
	// Code that changes an input without these helpers calls `forgetInput` for it.
	const inputs = [];
	const input = ( location ) => ( inputs[ location ] ??= { array: undefined, buffer: null, size: 0, stride: 0, offset: 0, divisor: undefined, x: NaN, y: NaN, z: NaN, w: NaN } );

	// A float array input read from `buffer`, per vertex (divisor 0) or per instance (1).
	function arrayInput( location, buffer, size, stride, offset, divisor ) {

		const s = input( location );

		if ( s.array !== true ) {

			gl.enableVertexAttribArray( location );
			s.array = true;

		}

		if ( s.divisor !== divisor ) gl.vertexAttribDivisor( location, s.divisor = divisor );

		if ( s.buffer !== buffer || s.size !== size || s.stride !== stride || s.offset !== offset ) {

			gl.bindBuffer( gl.ARRAY_BUFFER, buffer );
			gl.vertexAttribPointer( location, size, gl.FLOAT, false, stride, offset );
			s.buffer = buffer;
			s.size = size;
			s.stride = stride;
			s.offset = offset;

		}

	}

	// An input without an array, holding one value for every vertex.
	function constantInput( location, x, y, z, w ) {

		const s = input( location );

		if ( s.array !== false ) {

			gl.disableVertexAttribArray( location );
			s.array = false;

		}

		if ( s.x !== x || s.y !== y || s.z !== z || s.w !== w ) {

			gl.vertexAttrib4f( location, x, y, z, w );
			s.x = x;
			s.y = y;
			s.z = z;
			s.w = w;

		}

	}

	function forgetInput( location ) {

		inputs[ location ] = undefined;

	}

	function bindGeometry( entry ) {

		for ( let slot = 0; slot < ATTRIBUTES.length; slot ++ ) {

			const buffer = entry.buffers[ slot ];
			if ( buffer ) arrayInput( slot, buffer, entry.sizes[ slot ], 0, 0, 0 );
			else constantInput( slot, ...MISSING[ slot ] );

		}

		for ( let a = 0; a < AUX_ATTRIBUTES; a ++ ) {

			const location = AUX_LOCATIONS[ a ];
			if ( entry.aux ) arrayInput( location, entry.aux, AUX_SIZES[ a ], AUX_STRIDE, AUX_OFFSETS[ a ] * 4, 0 );
			else constantInput( location, 0, 0, 0, 0 );

		}

		gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, entry.index );

	}

	// The vertices of an indexed geometry in index order, for quads, and a line loop's
	// range with its first vertex again (see the WebGPU host).
	function quadSource( world, entry, start = 0, count = 0, loop = false ) {

		if ( ! entry.index && ! loop ) {

			// line distances of their own buffer, read per instance like the positions
			if ( entry.distances?.key !== entry.auxVersions ) {

				if ( entry.distances?.buffer ) gl.deleteBuffer( entry.distances.buffer );
				const distances = ( ',' + entry.auxVersions ).includes( ',4:' ) ? world.getAttribute( entry.handle, 'lineDistance' ) : null;
				entry.distances = { key: entry.auxVersions, buffer: distances ? upload( gl.ARRAY_BUFFER, new Float32Array( distances ) ) : null };

			}

			return { position: entry.buffers[ 0 ], color: entry.buffers[ VERTEX_COLOR_SLOT ], colorSize: entry.sizes[ VERTEX_COLOR_SLOT ], distance: entry.distances.buffer };

		}

		const key = entry.indexVersion + ':' + entry.versions[ 0 ] + ':' + entry.versions[ VERTEX_COLOR_SLOT ] + ':' + entry.auxVersions + ( loop ? ':' + start + ':' + count : '' );
		if ( entry.quads && entry.quads.key === key ) return entry.quads;
		if ( entry.quads ) for ( const buffer of [ entry.quads.position, entry.quads.color, entry.quads.distance ] ) if ( buffer ) gl.deleteBuffer( buffer );
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
		entry.quads = {
			key, colorSize,
			position: upload( gl.ARRAY_BUFFER, unrolled( world.getAttribute( entry.handle, 'position' ), 3 ) ),
			color: entry.buffers[ VERTEX_COLOR_SLOT ] ? upload( gl.ARRAY_BUFFER, unrolled( world.getAttribute( entry.handle, 'color' ), colorSize ) ) : null,
			distance: ( ',' + entry.auxVersions ).includes( ',4:' ) ? upload( gl.ARRAY_BUFFER, unrolled( world.getAttribute( entry.handle, 'lineDistance' ), 1 ) ) : null
		};
		return entry.quads;

	}

	// Points or segments read per instance, from the first instance on.
	function bindQuads( source, topology, first ) {

		const pairs = topology === 2, step = pairs ? 2 : 1;

		arrayInput( 0, source.position, 3, step * 12, first * step * 12, 1 );
		arrayInput( QUAD_B, source.position, 3, step * 12, first * step * 12 + ( topology === 3 ? 0 : 12 ), 1 );

		if ( source.color ) {

			const c = source.colorSize;
			arrayInput( VERTEX_COLOR_SLOT, source.color, c, step * c * 4, first * step * c * 4, 1 );
			arrayInput( QUAD_COLOR_B, source.color, c, step * c * 4, first * step * c * 4 + ( topology === 3 ? 0 : c * 4 ), 1 );

		} else {

			for ( const location of [ VERTEX_COLOR_SLOT, QUAD_COLOR_B ] ) constantInput( location, ...MISSING[ VERTEX_COLOR_SLOT ] );

		}

		// the line distances of both ends, for dashes
		if ( source.distance ) {

			arrayInput( 1, source.distance, 1, step * 4, first * step * 4, 1 );
			arrayInput( 2, source.distance, 1, step * 4, first * step * 4 + ( topology === 3 ? 0 : 4 ), 1 );

		} else {

			for ( const location of [ 1, 2 ] ) constantInput( location, 0, 0, 0, 1 );

		}

	}

	function bindInstances( instance ) {

		// instance inputs advance per instance; without instances they hold an identity matrix
		for ( let column = 0; column < 4; column ++ ) {

			const location = INSTANCE_SLOT + column;
			if ( instance ) arrayInput( location, instance.matrix, 4, 64, column * 16, 1 );
			else constantInput( location, column === 0 ? 1 : 0, column === 1 ? 1 : 0, column === 2 ? 1 : 0, column === 3 ? 1 : 0 );

		}

		if ( instance?.color ) arrayInput( INSTANCE_COLOR_SLOT, instance.color, 3, 0, 0, 1 );
		else constantInput( INSTANCE_COLOR_SLOT, 1, 1, 1, 1 );

	}

	// Passes that draw a full-screen triangle read no vertex arrays.
	function unbindArrays() {

		for ( let location = 0; location <= QUAD_COLOR_B; location ++ ) {

			const s = input( location );

			if ( s.array !== false ) {

				gl.disableVertexAttribArray( location );
				s.array = false;

			}

		}

	}

	// ---------------------------------------------------------------- state

	// Blend factors by three.js blending mode (see `BLEND_FACTORS`).
	const BLENDS = [
		[ 'src-alpha', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha' ], [ 'src-alpha', 'one', 'one', 'one' ], [ 'zero', 'one-minus-src', 'zero', 'one' ], [ 'dst', 'one-minus-src-alpha', 'zero', 'one' ],
		[ 'one', 'one-minus-src-alpha', 'one', 'one-minus-src-alpha' ], [ 'one', 'one', 'one', 'one' ], [ 'zero', 'one-minus-src', 'zero', 'one' ], [ 'dst', 'one-minus-src-alpha', 'zero', 'one' ]
	].map( ( factors ) => factors.map( ( name ) => FACTORS[ name ] ) );

	// three.js's blend factors (from ZeroFactor, 200) and equations (from AddEquation, 100).
	const CUSTOM_FACTORS = [ gl.ZERO, gl.ONE, gl.SRC_COLOR, gl.ONE_MINUS_SRC_COLOR, gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.DST_ALPHA, gl.ONE_MINUS_DST_ALPHA, gl.DST_COLOR, gl.ONE_MINUS_DST_COLOR, gl.SRC_ALPHA_SATURATE, gl.CONSTANT_COLOR, gl.ONE_MINUS_CONSTANT_COLOR, gl.CONSTANT_ALPHA, gl.ONE_MINUS_CONSTANT_ALPHA ];
	const CUSTOM_EQUATIONS = [ gl.FUNC_ADD, gl.FUNC_SUBTRACT, gl.FUNC_REVERSE_SUBTRACT, gl.MIN, gl.MAX ];

	// three.js `depthFunc` (0 never ... 7 not equal), stencil functions (from 512) and stencil
	// operations as the engine numbers them.
	const DEPTH_FUNCTIONS = [ gl.NEVER, gl.ALWAYS, gl.LESS, gl.LEQUAL, gl.EQUAL, gl.GEQUAL, gl.GREATER, gl.NOTEQUAL ];
	const STENCIL_FUNCTIONS = [ gl.NEVER, gl.LESS, gl.EQUAL, gl.LEQUAL, gl.GREATER, gl.NOTEQUAL, gl.GEQUAL, gl.ALWAYS ];
	const STENCIL_OPERATIONS = [ gl.ZERO, gl.KEEP, gl.REPLACE, gl.INCR, gl.DECR, gl.INCR_WRAP, gl.DECR_WRAP, gl.INVERT ];

	// Whether polygon offset and the stencil test may be on. Only this file turns them on
	// or off, so a material that uses neither (nearly all) costs no call to keep them off.
	let offsetOn = true, stencilOn = true;

	// The state of the material at float `at` of `blocks` that the key does not carry:
	// polygon offset, stencil test and the constant colour of custom blending.
	function applyMaterialState( key, blocks, at ) {

		const factor = blocks[ at + POLYGON_OFFSET_FLOAT ], units = blocks[ at + POLYGON_OFFSET_FLOAT + 1 ];

		if ( factor !== 0 || units !== 0 ) {

			gl.enable( gl.POLYGON_OFFSET_FILL );
			gl.polygonOffset( factor, units );
			offsetOn = true;

		} else if ( offsetOn ) {

			gl.disable( gl.POLYGON_OFFSET_FILL );
			offsetOn = false;

		}

		const bits = blocks[ at + STENCIL_FLOAT ];

		if ( bits & 1 ) {

			gl.enable( gl.STENCIL_TEST );
			gl.stencilFunc( STENCIL_FUNCTIONS[ bits >> 1 & 7 ], blocks[ at + STENCIL_FLOAT + 1 ], blocks[ at + STENCIL_FLOAT + 2 ] );
			gl.stencilOp( STENCIL_OPERATIONS[ bits >> 4 & 7 ], STENCIL_OPERATIONS[ bits >> 7 & 7 ], STENCIL_OPERATIONS[ bits >> 10 & 7 ] );
			gl.stencilMask( blocks[ at + STENCIL_FLOAT + 3 ] );
			stencilOn = true;

		} else if ( stencilOn ) {

			gl.disable( gl.STENCIL_TEST );
			stencilOn = false;

		}

		if ( key & 4 && ( key >> 11 & 7 ) === 5 ) gl.blendColor( blocks[ at + BLEND_COLOR_FLOAT ], blocks[ at + BLEND_COLOR_FLOAT + 1 ], blocks[ at + BLEND_COLOR_FLOAT + 2 ], blocks[ at + BLEND_COLOR_FLOAT + 3 ] );

	}

	// What the passes after the scene expect: no stencil test or offset, every channel
	// written, the default blend equation.
	function resetState() {

		gl.disable( gl.STENCIL_TEST );
		gl.disable( gl.POLYGON_OFFSET_FILL );
		stencilOn = offsetOn = false;
		gl.disable( gl.SAMPLE_ALPHA_TO_COVERAGE );
		gl.colorMask( true, true, true, true );
		gl.stencilMask( 0xff );
		gl.blendEquation( gl.FUNC_ADD );

	}

	// `custom` is the material's packed custom blending, read only with CustomBlending.
	function applyState( key, custom = 0 ) {

		const side = key & 3;

		if ( side === 2 || key & QUAD_KEY ) gl.disable( gl.CULL_FACE );
		else {

			gl.enable( gl.CULL_FACE );
			gl.cullFace( side === 0 ? gl.BACK : gl.FRONT );

		}

		if ( key & 4 && ( key >> 11 & 7 ) === 5 ) {

			const factor = ( shift ) => CUSTOM_FACTORS[ custom >> shift & 15 ] ?? gl.ONE, equation = ( shift ) => CUSTOM_EQUATIONS[ custom >> shift & 7 ] ?? gl.FUNC_ADD;
			gl.enable( gl.BLEND );
			gl.blendEquationSeparate( equation( 8 ), equation( 19 ) );
			gl.blendFuncSeparate( factor( 0 ), factor( 4 ), factor( 11 ), factor( 15 ) );

		} else if ( key & 4 ) {

			const mode = Math.min( 4, Math.max( 1, key >> 11 & 7 ) ) - 1, factors = BLENDS[ ( key & PREMULTIPLIED_KEY ? 4 : 0 ) + mode ];
			gl.enable( gl.BLEND );
			gl.blendFuncSeparate( factors[ 0 ], factors[ 1 ], factors[ 2 ], factors[ 3 ] );

		} else gl.disable( gl.BLEND );

		// the other passes blend with the default equation
		if ( ( key >> 11 & 7 ) !== 5 ) gl.blendEquation( gl.FUNC_ADD );
		gl.depthMask( ( key & 8 ) !== 0 );
		gl.depthFunc( key & 16 ? DEPTH_FUNCTIONS[ key >> DEPTH_FUNC_SHIFT & 7 ] : gl.ALWAYS );
		const write = ( key & COLOR_WRITE_OFF_KEY ) === 0;
		gl.colorMask( write, write, write, write );
		if ( key & ALPHA_TO_COVERAGE_KEY ) gl.enable( gl.SAMPLE_ALPHA_TO_COVERAGE );
		else gl.disable( gl.SAMPLE_ALPHA_TO_COVERAGE );

	}

	function mode( key ) {

		const topology = key >> 5 & 3;
		if ( topology === 0 ) return gl.TRIANGLES;
		if ( topology === 1 ) return key & LOOP_KEY ? gl.LINE_LOOP : gl.LINE_STRIP;
		return topology === 2 ? gl.LINES : gl.POINTS;

	}

	function draw( key, entry, start, vertices, copies ) {

		if ( entry.index ) gl.drawElementsInstanced( mode( key ), vertices, gl.UNSIGNED_INT, start * 4, copies );
		else gl.drawArraysInstanced( mode( key ), start, vertices, copies );

	}

	// Binds the wireframe index in place of the geometry's and draws it; the caller binds
	// the geometry again before its next draw.
	function drawWireframe( world, entry, start, vertices, copies ) {

		const indices = wireframeCount( vertices );
		if ( indices === 0 ) return;
		gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, wireframeIndex( world, entry, start, vertices ) );
		gl.drawElementsInstanced( gl.LINES, indices, gl.UNSIGNED_INT, 0, copies );

	}

	const frameBuffer = gl.createBuffer(), objectBuffer = gl.createBuffer(), materialBuffer = gl.createBuffer(), shadowObjectBuffer = gl.createBuffer(), backgroundBuffer = gl.createBuffer(), scratchBuffer = gl.createBuffer();
	ctx.objectCapacity = 0;
	ctx.materialCapacity = 0;

	// Binds block `binding` to element `i` (of `size` bytes) of `buffer`, or uploads it from
	// `data` into a scratch buffer when ranges are not supported.
	function bindRange( binding, buffer, i, size, data ) {

		if ( ranges ) gl.bindBufferRange( gl.UNIFORM_BUFFER, binding, buffer, i * size, size );
		else {

			const target = binding === 3 ? scratchBuffer : buffer;
			gl.bindBuffer( gl.UNIFORM_BUFFER, target );
			gl.bufferData( gl.UNIFORM_BUFFER, data.subarray( i * size / 4, ( i + 1 ) * size / 4 ), gl.DYNAMIC_DRAW );
			gl.bindBufferBase( gl.UNIFORM_BUFFER, binding, target );

		}

	}

	function bindMaps( maps ) {

		for ( let slot = 0; slot < TEXTURE_SLOTS; slot ++ ) {

			gl.activeTexture( gl.TEXTURE0 + slot );
			gl.bindTexture( gl.TEXTURE_2D, maps[ slot ] );

		}

	}

	Object.assign( ctx, { applyMaterialState, resetState, applyState, arrayInput, forgetInput, backgroundBuffer, bindGeometry, bindInstances, bindMaps, bindQuads, bindRange, distinct, draw, drawWireframe, entries, frameBuffer, geometries, geometry, info, instanced, instances, materialBuffer, objectBuffer, perInstance, quadSource, release, scratchBuffer, setTextures, shadowObjectBuffer, unbindArrays, upload } );

}
