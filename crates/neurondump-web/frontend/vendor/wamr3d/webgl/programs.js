// GLSL programs: the engine's passes, effects and three.js shader materials.
//
// Part of the WebGL 2 host; `../webgl.js` creates the context and draws the frames.

import { Renderer } from '../index.js';
import { BONE_UNIT, ENVIRONMENT_UNIT, SHADOW_UNIT, SLOTS, TRANSMISSION_UNIT, VSM_UNIT } from './constants.js';
import { FEATURE_NAMES } from '../webgpu/formats.js';

/** Adds this part's functions and state to the host context `ctx`. */
export function installPrograms( ctx ) {

	const { gl, warn } = ctx;

	// ---------------------------------------------------------------- programs

	const programs = new Map();

	function compile( type, source, label ) {

		const shader = gl.createShader( type );
		gl.shaderSource( shader, source );
		gl.compileShader( shader );
		if ( ! gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) throw new Error( `WAMR3D: GLSL ${label}: ${gl.getShaderInfoLog( shader )}` );
		return shader;

	}

	// Compiles and links without asking for the result: a status query waits for the
	// driver, so the programs a frame needs are all started before the first is queried,
	// letting drivers that compile on worker threads build them side by side.
	function startProgram( key, vertex, fragment, defines = '' ) {

		let value = programs.get( key );
		if ( value ) return value;
		const source = ( name ) => {

			const text = Renderer.glslSource( name );
			return defines ? text.replace( '\n', '\n' + defines + '\n' ) : text;

		};

		const id = gl.createProgram();
		const stages = [ [ gl.VERTEX_SHADER, vertex ], [ gl.FRAGMENT_SHADER, fragment ] ].map( ( [ type, name ] ) => {

			const shader = gl.createShader( type );
			gl.shaderSource( shader, source( name ) );
			gl.compileShader( shader );
			gl.attachShader( id, shader );
			return [ shader, name ];

		} );
		gl.linkProgram( id );
		value = { id, key, stages, uniform: ( name ) => gl.getUniformLocation( id, name ) };
		programs.set( key, value );
		return value;

	}

	// `defines` go after the `#version` line.
	const program = ( key, vertex, fragment, defines = '' ) => finishProgram( startProgram( key, vertex, fragment, defines ) );

	// Waits for a started program, then binds its blocks and texture units.
	function finishProgram( value ) {

		if ( ! value.stages ) return value;
		const { id, key, stages } = value;

		if ( ! gl.getProgramParameter( id, gl.LINK_STATUS ) ) {

			programs.delete( key );
			for ( const [ shader, name ] of stages ) if ( ! gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) throw new Error( `WAMR3D: GLSL ${name}: ${gl.getShaderInfoLog( shader )}` );
			throw new Error( `WAMR3D: GLSL ${key}: ${gl.getProgramInfoLog( id )}` );

		}

		value.stages = null;
		gl.useProgram( id );

		const block = ( name, binding ) => {

			const index = gl.getUniformBlockIndex( id, name );
			if ( index !== gl.INVALID_INDEX ) gl.uniformBlockBinding( id, index, binding );

		};

		block( 'FrameBlock', 0 );
		block( 'ObjectBlock', 1 );
		block( 'BackgroundBlock', 2 );
		block( 'MaterialBlock', 3 );
		const unit = ( name, value_ ) => {

			const location = gl.getUniformLocation( id, name );
			if ( location !== null ) gl.uniform1i( location, value_ );

		};

		for ( let i = 0; i < SLOTS; i ++ ) unit( 'slot_' + i, i );
		unit( 'shadow_map', SHADOW_UNIT );
		unit( 'environment_map', ENVIRONMENT_UNIT );
		unit( 'transmission_map', TRANSMISSION_UNIT );
		unit( 'vsm_map', VSM_UNIT );
		unit( 'bone_map', BONE_UNIT );
		unit( 'background_map', 0 );
		unit( 'source', 0 );
		unit( 'depth_layers', 0 );
		unit( 'moments', 1 );
		const flip = gl.getUniformLocation( id, 'flip_y' );
		if ( flip !== null ) gl.uniform1f( flip, 1 );
		return value;

	}

	const define = ( srgb ) => ( srgb ? '#define OUTPUT_SRGB 1' : '' );
	const effectProgram = ( custom, srgb ) => {

		const name = Renderer.effectEntryPoint( custom );
		return program( 'effect:' + name + ( srgb ? ':srgb' : '' ), 'effect.vert', 'effect.' + name, define( srgb ) );

	};

	// ---------------------------------------------------------------- shader materials

	// three.js's prefixes of `ShaderMaterial` code: GLSL 1 style source compiled as GLSL ES
	// 3.00, the built-in uniforms and attributes, and the common `#include`s. The user's
	// `main` is renamed so that the vertex stage can flip render targets as the engine does.
	const GLSL1_VERTEX = '#define attribute in\n#define varying out\n#define texture2D texture\n';
	const GLSL1_FRAGMENT = '#define varying in\nlayout(location = 0) out highp vec4 pc_fragColor;\n#define gl_FragColor pc_fragColor\n#define gl_FragDepthEXT gl_FragDepth\n#define texture2D texture\n#define textureCube texture\n#define texture2DProj textureProj\n#define texture2DLodEXT textureLod\n#define texture2DProjLodEXT textureProjLod\n#define textureCubeLodEXT textureLod\n#define texture2DGradEXT textureGrad\n#define texture2DProjGradEXT textureProjGrad\n#define textureCubeGradEXT textureGrad\n';
	const INCLUDES = {
		common: '#define PI 3.141592653589793\n#define PI2 6.283185307179586\n#define HALF_PI 1.5707963267948966\n#define RECIPROCAL_PI 0.3183098861837907\n#define EPSILON 1e-6\n#define saturate( a ) clamp( a, 0.0, 1.0 )\nfloat pow2( const in float x ) { return x * x; }\nvec3 pow2( const in vec3 x ) { return x * x; }\nfloat luminance( const in vec3 rgb ) { return dot( rgb, vec3( 0.2126, 0.7152, 0.0722 ) ); }\nfloat pow3( const in float x ) { return x * x * x; }\nfloat pow4( const in float x ) { float x2 = x * x; return x2 * x2; }\nstruct IncidentLight { vec3 color; vec3 direction; bool visible; };\nstruct ReflectedLight { vec3 directDiffuse; vec3 directSpecular; vec3 indirectDiffuse; vec3 indirectSpecular; };\nvec3 transformDirection( in vec3 dir, in mat4 matrix ) { return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz ); }\nvec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 matrix ) { return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz ); }\n#define inverseTransformDirection transformDirectionByInverseViewMatrix\nvec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) { return normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz ); }\n',
		colorspace_fragment: '',
		tonemapping_fragment: '',
		fog_pars_vertex: '#ifdef USE_FOG\nvarying float vFogDepth;\n#endif\n',
		fog_vertex: '#ifdef USE_FOG\nvFogDepth = - mvPosition.z;\n#endif\n',
		fog_pars_fragment: '#ifdef USE_FOG\nuniform vec3 fogColor;\nvarying float vFogDepth;\n#ifdef FOG_EXP2\nuniform float fogDensity;\n#else\nuniform float fogNear;\nuniform float fogFar;\n#endif\n#endif\n',
		fog_fragment: '#ifdef USE_FOG\n#ifdef FOG_EXP2\nfloat fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );\n#else\nfloat fogFactor = smoothstep( fogNear, fogFar, vFogDepth );\n#endif\ngl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );\n#endif\n',
		lights_pars_begin: 'uniform bool receiveShadow;\nuniform vec3 ambientLightColor;\n#if defined( USE_LIGHT_PROBES )\nuniform vec3 lightProbe[ 9 ];\n#endif\nvec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {\n\tfloat x = normal.x, y = normal.y, z = normal.z;\n\tvec3 result = shCoefficients[ 0 ] * 0.886227;\n\tresult += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;\n\tresult += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;\n\tresult += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;\n\tresult += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;\n\tresult += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;\n\tresult += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );\n\tresult += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;\n\tresult += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );\n\treturn result;\n}\nvec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {\n\treturn shGetIrradianceAt( transformNormalByInverseViewMatrix( normal, viewMatrix ), lightProbe );\n}\nvec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {\n\treturn ambientLightColor;\n}\nfloat getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {\n\tfloat distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );\n\tif ( cutoffDistance > 0.0 ) distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );\n\treturn distanceFalloff;\n}\nfloat getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {\n\treturn smoothstep( coneCosine, penumbraCosine, angleCosine );\n}\n#if NUM_DIR_LIGHTS > 0\nstruct DirectionalLight { vec3 direction; vec3 color; };\nuniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];\nvoid getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {\n\tlight.color = directionalLight.color;\n\tlight.direction = directionalLight.direction;\n\tlight.visible = true;\n}\n#endif\n#if NUM_POINT_LIGHTS > 0\nstruct PointLight { vec3 position; vec3 color; float distance; float decay; };\nuniform PointLight pointLights[ NUM_POINT_LIGHTS ];\nvoid getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {\n\tvec3 lVector = pointLight.position - geometryPosition;\n\tlight.direction = normalize( lVector );\n\tlight.color = pointLight.color * getDistanceAttenuation( length( lVector ), pointLight.distance, pointLight.decay );\n\tlight.visible = ( light.color != vec3( 0.0 ) );\n}\n#endif\n#if NUM_SPOT_LIGHTS > 0\nstruct SpotLight { vec3 position; vec3 direction; vec3 color; float distance; float decay; float coneCos; float penumbraCos; };\nuniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];\nvoid getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {\n\tvec3 lVector = spotLight.position - geometryPosition;\n\tlight.direction = normalize( lVector );\n\tfloat spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, dot( light.direction, spotLight.direction ) );\n\tlight.color = spotAttenuation > 0.0 ? spotLight.color * spotAttenuation * getDistanceAttenuation( length( lVector ), spotLight.distance, spotLight.decay ) : vec3( 0.0 );\n\tlight.visible = ( light.color != vec3( 0.0 ) );\n}\n#endif\n#if NUM_HEMI_LIGHTS > 0\nstruct HemisphereLight { vec3 direction; vec3 skyColor; vec3 groundColor; };\nuniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];\nvec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {\n\tfloat hemiDiffuseWeight = 0.5 * dot( normal, hemiLight.direction ) + 0.5;\n\treturn mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );\n}\n#endif\n',
		logdepthbuf_pars_vertex: '', logdepthbuf_vertex: '', logdepthbuf_pars_fragment: '', logdepthbuf_fragment: '',
		premultiplied_alpha_fragment: '#ifdef PREMULTIPLIED_ALPHA\ngl_FragColor.rgb *= gl_FragColor.a;\n#endif\n'
	};
	const resolveIncludes = ( source ) => source.replace( /^[ \t]*#include +<([\w\d./]+)>/gm, ( _, name ) => {

		if ( INCLUDES[ name ] === undefined ) warn( `#include <${name}> is not available in shader materials` );
		return INCLUDES[ name ] ?? '';

	} );
	const STANDARD_ATTRIBUTES = [ [ 'position', 0 ], [ 'normal', 1 ], [ 'uv', 2 ], [ 'color', 3 ], [ 'instanceMatrix', 4 ], [ 'instanceColor', 8 ], [ 'tangent', 9 ], [ 'uv1', 10 ], [ 'skinIndex', 11 ], [ 'skinWeight', 12 ] ];

	// `lights` and `fog` are the frame's (see the compat renderer's extras), or null when the
	// material does not use them; three.js defines the light counts in every program.
	function shaderProgram( material, instanced, srgb, lights = null, fog = null ) {

		const key = 'shader:' + material.id + ':' + material.version + ':' + instanced + ':' + srgb + ':' + ( lights?.key ?? '' ) + ':' + ( fog ? ( fog.exp2 ? 2 : 1 ) : 0 );
		let value = programs.get( key );
		if ( value ) return value;
		const defines = Object.entries( material.defines ?? {} ).map( ( [ name, v ] ) => ( v === false ? '' : `#define ${name} ${v === true ? '' : v}` ) ).join( '\n' ) + '\n';
		const [ directional, point, spot, hemisphere ] = lights?.counts ?? [ 0, 0, 0, 0 ];
		const lighting = `#define NUM_DIR_LIGHTS ${directional}\n#define NUM_POINT_LIGHTS ${point}\n#define NUM_SPOT_LIGHTS ${spot}\n#define NUM_RECT_AREA_LIGHTS 0\n#define NUM_HEMI_LIGHTS ${hemisphere}\n#define NUM_DIR_LIGHT_SHADOWS 0\n#define NUM_POINT_LIGHT_SHADOWS 0\n#define NUM_SPOT_LIGHT_SHADOWS 0\n#define NUM_SPOT_LIGHT_MAPS 0\n#define NUM_SPOT_LIGHT_COORDS 0\n` + ( lights?.probes ? '#define USE_LIGHT_PROBES\n' : '' ) + ( fog ? '#define USE_FOG\n' + ( fog.exp2 ? '#define FOG_EXP2\n' : '' ) : '' );
		const flags = lighting + ( instanced ? '#define USE_INSTANCING\n' : '' ) + ( material.vertexColors ? '#define USE_COLOR\n' : '' ) + ( material.premultipliedAlpha ? '#define PREMULTIPLIED_ALPHA\n' : '' );
		const raw = material.isRawShaderMaterial === true, glsl3 = material.glslVersion === '300 es';
		const strip = ( source ) => source.replace( /^\s*#version[^\n]*\n/, '' );
		let vertex, fragment;

		if ( raw ) {

			vertex = material.vertexShader;
			fragment = material.fragmentShader;

		} else {

			const header = '#version 300 es\nprecision highp float;\nprecision highp int;\n' + defines + flags;
			const uniforms = 'uniform mat4 modelMatrix;\nuniform mat4 modelViewMatrix;\nuniform mat4 projectionMatrix;\nuniform mat4 viewMatrix;\nuniform mat3 normalMatrix;\nuniform vec3 cameraPosition;\nuniform bool isOrthographic;\n';
			const attributes = ( glsl3 ? '' : GLSL1_VERTEX ) + 'in vec3 position;\nin vec3 normal;\nin vec2 uv;\n#ifdef USE_INSTANCING\nin mat4 instanceMatrix;\n#endif\n#ifdef USE_COLOR\nin vec3 color;\n#endif\n';
			// the user's main, then the engine's flip of render targets
			vertex = header + uniforms + attributes + 'uniform float wamr_flip_y;\n#define main wamr_user_main\n' + resolveIncludes( strip( material.vertexShader ) ) + '\n#undef main\nvoid main() {\n\twamr_user_main();\n\tgl_Position.y *= wamr_flip_y;\n}\n';
			fragment = header + 'uniform mat4 viewMatrix;\nuniform vec3 cameraPosition;\nuniform bool isOrthographic;\n' + ( glsl3 ? '' : GLSL1_FRAGMENT ) + resolveIncludes( strip( material.fragmentShader ) );

		}

		const id = gl.createProgram();
		gl.attachShader( id, compile( gl.VERTEX_SHADER, vertex, 'ShaderMaterial vertex' ) );
		gl.attachShader( id, compile( gl.FRAGMENT_SHADER, fragment, 'ShaderMaterial fragment' ) );
		for ( const [ name, location ] of STANDARD_ATTRIBUTES ) gl.bindAttribLocation( id, location, name );
		gl.linkProgram( id );
		// other attributes take free locations from 13 on, and the program is linked again
		const custom = [];

		for ( let i = 0, n = gl.getProgramParameter( id, gl.ACTIVE_ATTRIBUTES ); i < n; i ++ ) {

			const attribute = gl.getActiveAttrib( id, i );
			if ( ! STANDARD_ATTRIBUTES.some( ( [ name ] ) => name === attribute.name ) && ! attribute.name.startsWith( 'gl_' ) ) custom.push( attribute );

		}

		if ( custom.length > 0 ) {

			let location = 13;
			for ( const attribute of custom ) {

				attribute.location = location;
				gl.bindAttribLocation( id, location, attribute.name );
				location += attribute.type === gl.FLOAT_MAT4 ? 4 : attribute.type === gl.FLOAT_MAT3 ? 3 : 1;

			}

			gl.linkProgram( id );

		}

		if ( ! gl.getProgramParameter( id, gl.LINK_STATUS ) ) throw new Error( `WAMR3D: ShaderMaterial: ${gl.getProgramInfoLog( id )}` );
		const uniforms = [];

		for ( let i = 0, n = gl.getProgramParameter( id, gl.ACTIVE_UNIFORMS ); i < n; i ++ ) {

			const uniform = gl.getActiveUniform( id, i ), name = uniform.name.replace( /\[0\]$/, '' );
			uniforms.push( { name, type: uniform.type, size: uniform.size, location: gl.getUniformLocation( id, uniform.name ) } );

		}

		const INTEGER = new Map( [ [ gl.INT, 1 ], [ gl.INT_VEC2, 2 ], [ gl.INT_VEC3, 3 ], [ gl.INT_VEC4, 4 ], [ gl.UNSIGNED_INT, 1 ], [ gl.UNSIGNED_INT_VEC2, 2 ], [ gl.UNSIGNED_INT_VEC3, 3 ], [ gl.UNSIGNED_INT_VEC4, 4 ] ] );
		for ( const attribute of custom ) attribute.integer = INTEGER.get( attribute.type ) ?? 0;
		value = { id, uniforms, custom, flip: gl.getUniformLocation( id, 'wamr_flip_y' ) };
		programs.set( key, value );
		return value;

	}

	const SAMPLERS = new Set( [ gl.SAMPLER_2D, gl.SAMPLER_CUBE, gl.SAMPLER_2D_ARRAY, gl.SAMPLER_3D, gl.INT_SAMPLER_2D, gl.UNSIGNED_INT_SAMPLER_2D, gl.SAMPLER_2D_SHADOW ] );

	// Flattens a uniform value into floats: numbers, vectors, colours, matrices, arrays of them.
	function flatten( value, out = [] ) {

		if ( typeof value === 'number' || typeof value === 'boolean' ) out.push( + value );
		else if ( value.isColor ) out.push( value.r, value.g, value.b );
		else if ( value.isMatrix3 || value.isMatrix4 ) out.push( ...value.elements );
		else if ( value.isVector2 ) out.push( value.x, value.y );
		else if ( value.isVector3 ) out.push( value.x, value.y, value.z );
		else if ( value.isVector4 || value.isQuaternion ) out.push( value.x, value.y, value.z, value.w );
		else if ( ArrayBuffer.isView( value ) || Array.isArray( value ) ) for ( const each of value ) flatten( each, out );
		return out;

	}

	// The material's uniforms by name; textures go to units from 0 on.
	function setUniforms( shader, material, builtins, sources ) {

		let unit = 0;

		for ( const uniform of shader.uniforms ) {

			const own = material.uniforms?.[ uniform.name ];
			const value = builtins[ uniform.name ] ?? own?.value;
			if ( value === undefined || value === null ) continue;
			const location = uniform.location;

			if ( SAMPLERS.has( uniform.type ) ) {

				gl.activeTexture( gl.TEXTURE0 + unit );
				gl.bindTexture( gl.TEXTURE_2D, value.isTexture ? ctx.texture( value.id, sources ) : ctx.white );
				gl.uniform1i( location, unit ++ );
				continue;

			}

			const data = value.length === 16 && ! Array.isArray( value ) && value instanceof Float32Array ? value : flatten( value );

			switch ( uniform.type ) {

				case gl.FLOAT: gl.uniform1fv( location, data ); break;
				case gl.FLOAT_VEC2: gl.uniform2fv( location, data ); break;
				case gl.FLOAT_VEC3: gl.uniform3fv( location, data ); break;
				case gl.FLOAT_VEC4: gl.uniform4fv( location, data ); break;
				case gl.FLOAT_MAT3: gl.uniformMatrix3fv( location, false, data ); break;
				case gl.FLOAT_MAT4: gl.uniformMatrix4fv( location, false, data ); break;
				case gl.INT: case gl.BOOL: gl.uniform1iv( location, data.map( Math.round ) ); break;
				case gl.INT_VEC2: case gl.BOOL_VEC2: gl.uniform2iv( location, data.map( Math.round ) ); break;
				case gl.INT_VEC3: case gl.BOOL_VEC3: gl.uniform3iv( location, data.map( Math.round ) ); break;
				case gl.INT_VEC4: case gl.BOOL_VEC4: gl.uniform4iv( location, data.map( Math.round ) ); break;
				case gl.UNSIGNED_INT: gl.uniform1uiv( location, data.map( Math.round ) ); break;
				default: break;

			}

		}

	}

	// Uploads a shader material's own attributes (those that are not built-in vertex
	// inputs) when their version changes; integer inputs get integer data.
	function bindCustomAttributes( world, entry, shader ) {

		entry.custom ??= new Map();

		for ( const attribute of shader.custom ) {

			if ( ! world.hasAttribute( entry.handle, attribute.name ) ) {

				gl.disableVertexAttribArray( attribute.location );
				ctx.forgetInput( attribute.location );
				continue;

			}

			const version = world.attributeVersion( entry.handle, attribute.name );
			let cached = entry.custom.get( attribute.name );

			if ( ! cached || cached.version !== version || cached.integer !== attribute.integer ) {

				const floats = world.getAttribute( entry.handle, attribute.name ), size = floats.length / Math.max( 1, world.attributeCount( entry.handle, attribute.name ) );
				const data = attribute.integer ? Int32Array.from( floats, Math.round ) : floats;
				cached = { buffer: ctx.upload( gl.ARRAY_BUFFER, data, cached?.buffer ), version, size, integer: attribute.integer };
				entry.custom.set( attribute.name, cached );

			}

			gl.enableVertexAttribArray( attribute.location );
			gl.bindBuffer( gl.ARRAY_BUFFER, cached.buffer );
			gl.vertexAttribDivisor( attribute.location, 0 );
			if ( attribute.integer ) gl.vertexAttribIPointer( attribute.location, cached.size, gl.INT, 0, 0 );
			else gl.vertexAttribPointer( attribute.location, cached.size, gl.FLOAT, false, 0, 0 );
			ctx.forgetInput( attribute.location );

		}

	}

	// 4x4 products for the built-in uniforms, column-major.
	function multiply( a, b, out ) {

		for ( let c = 0; c < 4; c ++ ) for ( let r = 0; r < 4; r ++ ) out[ c * 4 + r ] = a[ r ] * b[ c * 4 ] + a[ 4 + r ] * b[ c * 4 + 1 ] + a[ 8 + r ] * b[ c * 4 + 2 ] + a[ 12 + r ] * b[ c * 4 + 3 ];
		return out;

	}

	function normalMatrix( m, out ) {

		const a = m[ 0 ], b = m[ 1 ], c = m[ 2 ], d = m[ 4 ], e = m[ 5 ], f = m[ 6 ], g = m[ 8 ], h = m[ 9 ], i = m[ 10 ];
		const A = e * i - f * h, B = f * g - d * i, C = d * h - e * g, det = a * A + b * B + c * C, k = det === 0 ? 0 : 1 / det;
		// the inverse transpose of the upper 3x3
		out[ 0 ] = A * k; out[ 1 ] = B * k; out[ 2 ] = C * k;
		out[ 3 ] = ( c * h - b * i ) * k; out[ 4 ] = ( a * i - c * g ) * k; out[ 5 ] = ( b * g - a * h ) * k;
		out[ 6 ] = ( b * f - c * e ) * k; out[ 7 ] = ( c * d - a * f ) * k; out[ 8 ] = ( a * e - b * d ) * k;
		return out;

	}
	// `features` is a `meshFeatures` key; with `start` the program is only begun.
	const meshProgram = ( srgb, quad, features, start = false ) => {

		const key = ( quad ? 'quad:' : 'mesh:' ) + ( srgb ? 'srgb:' : '' ) + features;
		const value = programs.get( key );
		if ( value && ! value.stages ) return value;
		const defines = define( srgb ) + features.split( ',' ).map( ( v, i ) => `\n#define ${FEATURE_NAMES[ i ]} ${v}u` ).join( '' );
		return ( start ? startProgram : program )( key, quad ? 'mesh_quad.vert' : 'mesh.vert', 'mesh.frag', defines );

	};
	const backgroundProgram = ( srgb ) => program( srgb ? 'background:srgb' : 'background', 'fullscreen.vert', 'background.frag', define( srgb ) );

	// KHR_parallel_shader_compile tells when a program is built without waiting for it.
	const parallel = gl.getExtension( 'KHR_parallel_shader_compile' );

	// Resolves once the driver has built `started` (programs from `startProgram`) and they
	// are finished. Without the extension they are finished at once, waiting as `program`.
	function finishPrograms( started ) {

		return new Promise( ( resolve, reject ) => {

			const poll = () => {

				try {

					if ( parallel && ! gl.isContextLost() && started.some( ( value ) => value.stages && ! gl.getProgramParameter( value.id, parallel.COMPLETION_STATUS_KHR ) ) ) {

						setTimeout( poll, 8 );
						return;

					}

					for ( const value of started ) finishProgram( value );
					resolve();

				} catch ( error ) {

					reject( error );

				}

			};

			poll();

		} );

	}

	Object.assign( ctx, { backgroundProgram, bindCustomAttributes, effectProgram, finishPrograms, meshProgram, startProgram, multiply, normalMatrix, program, programs, setUniforms, shaderProgram } );

}
