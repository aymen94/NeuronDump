// Loaders of three.js's JSON formats for animations, buffer geometries and materials.
//
// Part of the loaders; `../loaders.js` gathers them.

import { AnimationClip } from '../animation.js';
import * as Geometries from '../geometry.js';
import * as Materials from '../materials.js';
import * as Shapes from '../shapes.js';
import { Color } from '../math.js';
import { Loader } from './manager.js';
import { FileLoader } from './files.js';

export const TYPED_ARRAYS = { Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, Int32Array, Uint32Array, Float32Array, Float64Array };

/** three.js `AnimationLoader`: clips from three.js's JSON. */
export class AnimationLoader extends Loader {

	load( url, onLoad, onProgress, onError ) {

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		loader.load( url, ( text ) => {

			try {

				onLoad( this.parse( JSON.parse( text ) ) );

			} catch ( error ) {

				if ( onError ) onError( error );
				else console.error( error );
				this.manager.itemError( url );

			}

		}, onProgress, onError );

	}

	parse( json ) {

		return json.map( ( clip ) => AnimationClip.parse( clip ) );

	}

}

// An attribute from its JSON: plain, instanced, or interleaved into a buffer of `data`
// (`data.interleavedBuffers` and `data.arrayBuffers`, shared by the geometry's attributes).
function attributeFromJSON( json, data = {}, buffers = new Map() ) {

	if ( json.isInterleavedBufferAttribute ) {

		let buffer = buffers.get( json.data );

		if ( buffer === undefined ) {

			const info = data.interleavedBuffers[ json.data ];
			const words = Uint32Array.from( data.arrayBuffers[ info.buffer ] );
			const array = new TYPED_ARRAYS[ info.type ]( words.buffer );
			buffer = info.isInstancedInterleavedBuffer ? new Geometries.InstancedInterleavedBuffer( array, info.stride, info.meshPerAttribute ) : new Geometries.InterleavedBuffer( array, info.stride );
			buffer.uuid = info.uuid;
			buffers.set( json.data, buffer );

		}

		const attribute = new Geometries.InterleavedBufferAttribute( buffer, json.itemSize, json.offset, json.normalized );
		if ( json.name !== undefined ) attribute.name = json.name;
		return attribute;

	}

	const array = new TYPED_ARRAYS[ json.type ]( json.array );
	const attribute = json.isInstancedBufferAttribute ? new Geometries.InstancedBufferAttribute( array, json.itemSize, json.normalized, json.meshPerAttribute ) : new Geometries.BufferAttribute( array, json.itemSize, json.normalized );
	if ( json.name !== undefined ) attribute.name = json.name;
	if ( json.usage !== undefined ) attribute.setUsage( json.usage );
	return attribute;

}

/** three.js `BufferGeometryLoader`: a BufferGeometry from three.js's JSON. */
export class BufferGeometryLoader extends Loader {

	load( url, onLoad, onProgress, onError ) {

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		loader.load( url, ( text ) => {

			try {

				onLoad( this.parse( JSON.parse( text ) ) );

			} catch ( error ) {

				if ( onError ) onError( error );
				else console.error( error );
				this.manager.itemError( url );

			}

		}, onProgress, onError );

	}

	parse( json ) {

		const instanced = json.isInstancedBufferGeometry === true || json.type === 'InstancedBufferGeometry';
		const geometry = instanced ? new Geometries.InstancedBufferGeometry() : new Geometries.BufferGeometry(), data = json.data;
		if ( instanced && json.instanceCount !== undefined ) geometry.instanceCount = json.instanceCount;
		const buffers = new Map();
		if ( data.index !== undefined ) geometry.setIndex( attributeFromJSON( data.index ) );
		for ( const [ name, attribute ] of Object.entries( data.attributes ) ) geometry.setAttribute( name, attributeFromJSON( attribute, data, buffers ) );

		if ( data.morphAttributes ) {

			for ( const [ name, list ] of Object.entries( data.morphAttributes ) ) geometry.morphAttributes[ name ] = list.map( ( attribute ) => attributeFromJSON( attribute, data, buffers ) );

		}

		if ( data.morphTargetsRelative ) geometry.morphTargetsRelative = true;
		for ( const group of data.groups ?? data.offsets ?? data.drawcalls ?? [] ) geometry.addGroup( group.start, group.count, group.materialIndex );

		if ( data.boundingSphere !== undefined ) {

			geometry.computeBoundingSphere();
			geometry.boundingSphere.center.fromArray( data.boundingSphere.center );
			geometry.boundingSphere.radius = data.boundingSphere.radius;

		}

		if ( json.name ) geometry.name = json.name;
		if ( json.userData ) geometry.userData = json.userData;
		return geometry;

	}

}

const MATERIALS = {
	MeshBasicMaterial: Materials.MeshBasicMaterial, MeshLambertMaterial: Materials.MeshLambertMaterial, MeshPhongMaterial: Materials.MeshPhongMaterial,
	MeshToonMaterial: Materials.MeshToonMaterial, MeshNormalMaterial: Materials.MeshNormalMaterial, MeshStandardMaterial: Materials.MeshStandardMaterial,
	MeshPhysicalMaterial: Materials.MeshPhysicalMaterial, LineBasicMaterial: Materials.LineBasicMaterial, PointsMaterial: Materials.PointsMaterial,
	SpriteMaterial: Materials.SpriteMaterial, ShaderMaterial: Materials.ShaderMaterial, RawShaderMaterial: Materials.RawShaderMaterial, Material: Materials.Material
};
const COLOR_KEYS = [ 'color', 'emissive', 'specular', 'sheenColor', 'attenuationColor', 'specularColor' ];
const PLAIN_KEYS = [ 'name', 'roughness', 'metalness', 'shininess', 'clearcoat', 'clearcoatRoughness', 'ior', 'sheen', 'sheenRoughness', 'transmission', 'thickness', 'attenuationDistance', 'iridescence', 'iridescenceIOR', 'iridescenceThicknessRange', 'anisotropy', 'anisotropyRotation', 'dispersion', 'specularIntensity', 'emissiveIntensity', 'size', 'sizeAttenuation', 'blending', 'side', 'opacity', 'transparent', 'alphaTest', 'depthTest', 'depthWrite', 'wireframe', 'linewidth', 'vertexColors', 'rotation', 'toneMapped', 'fog', 'envMapIntensity', 'lightMapIntensity', 'aoMapIntensity', 'bumpScale', 'displacementScale', 'displacementBias', 'premultipliedAlpha', 'userData', 'defines', 'vertexShader', 'fragmentShader', 'glslVersion' ];
const MAP_KEYS = [ 'map', 'normalMap', 'roughnessMap', 'metalnessMap', 'emissiveMap', 'alphaMap', 'aoMap', 'lightMap', 'bumpMap', 'displacementMap', 'specularMap', 'gradientMap', 'envMap', 'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap', 'sheenColorMap', 'sheenRoughnessMap', 'specularIntensityMap', 'specularColorMap', 'transmissionMap', 'thicknessMap', 'iridescenceMap', 'iridescenceThicknessMap', 'anisotropyMap' ];

/** three.js `MaterialLoader`: a material from three.js's JSON; `setTextures` gives the textures by uuid. */
export class MaterialLoader extends Loader {

	constructor( manager ) {

		super( manager );
		this.textures = {};

	}

	setTextures( value ) {

		this.textures = value;
		return this;

	}

	load( url, onLoad, onProgress, onError ) {

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.load( url, ( text ) => {

			try {

				onLoad( this.parse( JSON.parse( text ) ) );

			} catch ( error ) {

				if ( onError ) onError( error );
				else console.error( error );

			}

		}, onProgress, onError );

	}

	parse( json ) {

		const Type = MATERIALS[ json.type ] ?? Materials.MeshStandardMaterial;
		const material = new Type();
		if ( json.uuid !== undefined ) material.uuid = json.uuid;
		for ( const key of COLOR_KEYS ) if ( json[ key ] !== undefined && material[ key ]?.isColor ) material[ key ].setHex( json[ key ] );
		for ( const key of PLAIN_KEYS ) if ( json[ key ] !== undefined ) material[ key ] = json[ key ];
		if ( json.normalScale !== undefined && material.normalScale ) material.normalScale.fromArray( Array.isArray( json.normalScale ) ? json.normalScale : [ json.normalScale, json.normalScale ] );

		if ( json.uniforms !== undefined && material.isShaderMaterial ) {

			for ( const [ name, uniform ] of Object.entries( json.uniforms ) ) {

				const value = uniform.type === 't' ? this.textures[ uniform.value ] : uniform.type === 'c' ? new Color().setHex( uniform.value ) : uniform.value;
				material.uniforms[ name ] = { value };

			}

		}

		for ( const key of MAP_KEYS ) if ( json[ key ] !== undefined ) material[ key ] = this.textures[ json[ key ] ] ?? null;
		return material;

	}

	createMaterialFromType( type ) {

		return MaterialLoader.createMaterialFromType( type );

	}

	static createMaterialFromType( type ) {

		return new ( MATERIALS[ type ] ?? Materials.Material )();

	}

	/** Makes `parse` create `materialClass` for JSON of type `type`. */
	static registerMaterial( type, materialClass ) {

		MATERIALS[ type ] = materialClass;

	}

}

export const GEOMETRIES = { ...Geometries, ...Shapes };
