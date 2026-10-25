// ShaderMaterial, RawShaderMaterial, UniformsUtils and the post-processing effect material.
//
// Part of the materials; `../materials.js` gathers them.

import { w } from '../engine.js';
import { Material } from './material.js';

/** Shader materials by their custom id, for the WebGL 2 host. */
export const shaderMaterials = new Map();
let nextShaderId = 16;

/**
 * three.js `ShaderMaterial`: GLSL `vertexShader` and `fragmentShader` with `uniforms`
 * (`{ name: { value } }`) and `defines`, compiled with three.js's built-in uniforms
 * (`modelMatrix`, `modelViewMatrix`, `projectionMatrix`, `viewMatrix`, `normalMatrix`,
 * `cameraPosition`) and attributes (`position`, `normal`, `uv`, and any other attribute of
 * the geometry). Drawn on WebGL 2 only, as three.js's WebGPURenderer draws no GLSL
 * either; on WebGPU it draws unlit in `color`. Lights and fog are not provided to it.
 * Set `needsUpdate = true` after changing the shaders or defines.
 */
export class ShaderMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'ShaderMaterial';
		this.defines = {};
		this.uniforms = {};
		this.uniformsGroups = [];
		this.vertexShader = 'void main() {\n\tgl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );\n}';
		this.fragmentShader = 'void main() {\n\tgl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );\n}';
		this.linewidth = 1;
		this.wireframe = false;
		this.wireframeLinewidth = 1;
		this.fog = false;
		this.lights = false;
		this.clipping = false;
		this.extensions = { clipCullDistance: false, multiDraw: false };
		this.defaultAttributeValues = { color: [ 1, 1, 1 ], uv: [ 0, 0 ], uv1: [ 0, 0 ] };
		this.index0AttributeName = undefined;
		this.uniformsNeedUpdate = false;
		this.glslVersion = null;
		this.id = nextShaderId ++;
		this.version = 0;
		shaderMaterials.set( this.id, this );
		if ( parameters ) this.setValues( parameters );

	}

	/** Set to true after changing the shaders or defines: the program is compiled again. */
	set needsUpdate( value ) {

		if ( value ) this.version ++;

	}

	setValues( parameters ) {

		for ( const [ key, value ] of Object.entries( parameters ) ) {

			if ( value !== undefined && ( key === 'uniforms' || key === 'defines' || key === 'vertexShader' || key === 'fragmentShader' || key === 'glslVersion' || key === 'lights' || key === 'fog' || key === 'extensions' || key === 'wireframe' ) ) this[ key ] = value;

		}

		const rest = Object.fromEntries( Object.entries( parameters ).filter( ( [ key ] ) => ! [ 'uniforms', 'defines', 'vertexShader', 'fragmentShader', 'glslVersion', 'lights', 'fog', 'extensions', 'wireframe' ].includes( key ) ) );
		super.setValues( rest );

	}

	_sync( environment ) {

		if ( this._sent[ 79 ] !== this.id ) w().setMaterialCustom( this.handle, this._sent[ 79 ] = this.id, new Float32Array( 0 ) );
		super._sync( environment );

	}

	copy( source ) {

		super.copy( source );
		this.fragmentShader = source.fragmentShader;
		this.vertexShader = source.vertexShader;
		this.uniforms = cloneUniforms( source.uniforms );
		this.defines = Object.assign( {}, source.defines );
		this.glslVersion = source.glslVersion;
		this.lights = source.lights;
		this.fog = source.fog;
		this.extensions = Object.assign( {}, source.extensions );
		return this;

	}

	dispose() {

		shaderMaterials.delete( this.id );
		super.dispose();

	}

}

ShaderMaterial.prototype._shading = 1;
ShaderMaterial.prototype.isShaderMaterial = true;

/** three.js `RawShaderMaterial`: the shaders are compiled as given, without built-in declarations. */
export class RawShaderMaterial extends ShaderMaterial {

	constructor( parameters ) {

		super( parameters );
		this.type = 'RawShaderMaterial';

	}

}

RawShaderMaterial.prototype.isRawShaderMaterial = true;

/** A copy of uniforms, values cloned where they can be (three.js `UniformsUtils.clone`). */
export function cloneUniforms( source ) {

	const out = {};

	for ( const [ name, uniform ] of Object.entries( source ?? {} ) ) {

		const value = uniform.value;
		out[ name ] = { value: value && ( value.isColor || value.isMatrix3 || value.isMatrix4 || value.isVector2 || value.isVector3 || value.isVector4 || value.isQuaternion ) ? value.clone() : Array.isArray( value ) ? value.slice() : value };

	}

	return out;

}

/** Merges uniform sets, later ones winning (three.js `UniformsUtils.merge`). */
export function mergeUniforms( sets ) {

	return Object.assign( {}, ...sets.map( cloneUniforms ) );

}

export const UniformsUtils = { clone: cloneUniforms, merge: mergeUniforms };

/**
 * A full-screen effect of the renderer, for post-processing: `effect` is its id (1 copy,
 * 2 luminosity high pass, 3 blur, 4 bloom composite, 5 FXAA, 6 output), `params` its 16
 * settings, and `map`, `normalMap`, ... its inputs in slot order.
 */
export class EffectMaterial extends Material {

	constructor( effect ) {

		super();
		this.type = 'EffectMaterial';
		this.effect = effect;
		this.params = new Float32Array( 16 );
		this.depthTest = false;
		this.depthWrite = false;
		this.normalMap = this.roughnessMap = this.metalnessMap = this.emissiveMap = null;
		this._sentParams = new Float32Array( 16 ).fill( NaN );

	}

	_sync( environment ) {

		let changed = this._sent[ 79 ] !== this.effect;
		for ( let i = 0; ! changed && i < 16; i ++ ) changed = this.params[ i ] !== this._sentParams[ i ];

		if ( changed ) {

			this._sentParams.set( this.params );
			w().setMaterialCustom( this.handle, this._sent[ 79 ] = this.effect, this.params );

		}

		super._sync( environment );

	}

}

EffectMaterial.prototype._shading = 1;
