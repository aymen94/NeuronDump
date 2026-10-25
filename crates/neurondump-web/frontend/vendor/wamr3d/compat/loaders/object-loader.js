// ObjectLoader: whole scenes from three.js's JSON format.
//
// Part of the loaders; `../loaders.js` gathers them.

import { AnimationClip } from '../animation.js';
import * as Geometries from '../geometry.js';
import { Shape } from '../curves.js';
import * as Objects from '../objects.js';
import * as Lights from '../lights.js';
import { OrthographicCamera, PerspectiveCamera } from '../cameras.js';
import { Fog, FogExp2, Group, Object3D, Scene } from '../object3d.js';
import { Color, Matrix4 } from '../math.js';
import { CubeTexture, DataTexture, Texture } from '../textures.js';
import { Loader } from './manager.js';
import { FileLoader, LoaderUtils } from './files.js';
import { ImageLoader } from './images.js';
import { BufferGeometryLoader, GEOMETRIES, MaterialLoader, TYPED_ARRAYS } from './json.js';

/**
 * three.js `ObjectLoader`: scenes and objects from three.js's JSON (`object.toJSON()`):
 * built-in and buffer geometries, shapes, materials, images and textures, meshes, lines,
 * points, sprites, groups, lights, cameras, fog and animations.
 */
export class ObjectLoader extends Loader {

	load( url, onLoad, onProgress, onError ) {

		const path = this.path === '' ? LoaderUtils.extractUrlBase( url ) : this.path;
		this.resourcePath = this.resourcePath || path;
		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );
		loader.load( url, ( text ) => {

			let json;

			try {

				json = JSON.parse( text );

			} catch ( error ) {

				if ( onError ) onError( error );
				return;

			}

			this.parse( json, onLoad );

		}, onProgress, onError );

	}

	async loadAsync( url, onProgress ) {

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		const text = await loader.loadAsync( url, onProgress );
		return this.parseAsync( JSON.parse( text ) );

	}

	parse( json, onLoad ) {

		this.parseAsync( json ).then( ( object ) => {

			if ( onLoad ) onLoad( object );

		} );

	}

	async parseAsync( json ) {

		const shapes = this.parseShapes( json.shapes );
		const geometries = this.parseGeometries( json.geometries, shapes );
		const images = await this.parseImagesAsync( json.images );
		const textures = this.parseTextures( json.textures, images );
		const materials = this.parseMaterials( json.materials, textures );
		const animations = this.parseAnimations( json.animations );
		const object = this.parseObject( json.object, geometries, materials, textures, animations );
		this.parseSkeletons( json.skeletons, object );
		this.bindLightTargets( object );
		return object;

	}

	parseShapes( json = [] ) {

		return Object.fromEntries( json.map( ( data ) => [ data.uuid, this.parseShape( data ) ] ) );

	}

	parseGeometries( json = [], shapes = {} ) {

		return Object.fromEntries( json.map( ( data ) => [ data.uuid, this.parseGeometry( data, shapes ) ] ) );

	}

	parseImagesAsync( json = [] ) {

		return this.parseImages( json );

	}

	parseTextures( json = [], images = {} ) {

		return Object.fromEntries( json.map( ( data ) => [ data.uuid, this.parseTexture( data, images ) ] ) );

	}

	parseMaterials( json = [], textures = {} ) {

		const loader = new MaterialLoader().setTextures( textures );
		return Object.fromEntries( json.map( ( data ) => [ data.uuid, loader.parse( data ) ] ) );

	}

	parseAnimations( json = [] ) {

		return Object.fromEntries( json.map( ( data ) => [ data.uuid, AnimationClip.parse( data ) ] ) );

	}

	parseSkeletons( json = [], object ) {

		this.bindSkeletons( object, json );

	}

	/** Points lights' `target` at the objects their JSON names. */
	bindLightTargets( root ) {

		root.traverse( ( object ) => {

			if ( object._targetUuid === undefined ) return;
			const target = root.getObjectByProperty( 'uuid', object._targetUuid );
			if ( target ) object.target = target;

		} );

	}

	/** Makes `parse` create geometries of `type` with `geometryClass.fromJSON`. */
	static registerGeometry( type, geometryClass ) {

		GEOMETRIES[ type ] = geometryClass;

	}

	parseShape( data ) {

		return new Shape().fromJSON( data );

	}

	parseGeometry( data, shapes ) {

		let geometry;

		if ( data.type === 'BufferGeometry' || data.type === 'InstancedBufferGeometry' ) {

			geometry = new BufferGeometryLoader().parse( data );

		} else if ( GEOMETRIES[ data.type ]?.fromJSON ) {

			geometry = GEOMETRIES[ data.type ].fromJSON( data, shapes );

		} else {

			console.warn( `WAMR3D: ObjectLoader: unsupported geometry type "${data.type}"` );
			geometry = new Geometries.BufferGeometry();

		}

		geometry.uuid = data.uuid;
		if ( data.name ) geometry.name = data.name;
		if ( data.userData ) geometry.userData = data.userData;
		return geometry;

	}

	async parseImages( list ) {

		const out = {};

		await Promise.all( list.map( async ( data ) => {

			const load = async ( url ) => {

				if ( typeof url === 'object' ) return { data: new TYPED_ARRAYS[ url.type ]( url.data ), width: url.width, height: url.height };
				const full = /^(\/\/)|([a-z]+:(\/\/)?)/i.test( url ) ? url : this.resourcePath + url;
				return new Promise( ( resolve ) => new ImageLoader( this.manager ).setCrossOrigin( this.crossOrigin ).load( full, resolve, undefined, () => resolve( null ) ) );

			};

			out[ data.uuid ] = Array.isArray( data.url ) ? await Promise.all( data.url.map( load ) ) : await load( data.url );

		} ) );

		return out;

	}

	parseTexture( data, images ) {

		const image = images[ data.image ];
		const texture = Array.isArray( image ) ? new CubeTexture( image ) : image?.data ? new DataTexture( image.data, image.width, image.height ) : new Texture( image ?? null );
		texture.uuid = data.uuid;
		if ( data.name ) texture.name = data.name;
		if ( data.mapping !== undefined ) texture.mapping = data.mapping;
		if ( data.channel !== undefined ) texture.channel = data.channel;
		if ( data.offset ) texture.offset.fromArray( data.offset );
		if ( data.repeat ) texture.repeat.fromArray( data.repeat );
		if ( data.center ) texture.center.fromArray( data.center );
		if ( data.rotation !== undefined ) texture.rotation = data.rotation;
		if ( data.wrap ) [ texture.wrapS, texture.wrapT ] = data.wrap;
		for ( const key of [ 'format', 'type', 'colorSpace', 'minFilter', 'magFilter', 'anisotropy', 'flipY', 'generateMipmaps' ] ) if ( data[ key ] !== undefined ) texture[ key ] = data[ key ];
		texture.needsUpdate = true;
		return texture;

	}

	parseObject( data, geometries, materials, textures, animations ) {

		const material = ( value ) => ( Array.isArray( value ) ? value.map( ( uuid ) => materials[ uuid ] ) : materials[ value ] );
		const geometry = geometries[ data.geometry ];
		let object;

		switch ( data.type ) {

			case 'Scene':
				object = new Scene();
				if ( data.background !== undefined ) object.background = typeof data.background === 'number' ? new Color( data.background ) : textures[ data.background ] ?? null;
				if ( data.environment !== undefined ) object.environment = textures[ data.environment ] ?? null;
				if ( data.fog !== undefined ) object.fog = data.fog.type === 'Fog' ? new Fog( data.fog.color, data.fog.near, data.fog.far ) : new FogExp2( data.fog.color, data.fog.density );
				if ( data.backgroundBlurriness !== undefined ) object.backgroundBlurriness = data.backgroundBlurriness;
				if ( data.backgroundIntensity !== undefined ) object.backgroundIntensity = data.backgroundIntensity;
				break;
			case 'PerspectiveCamera':
				object = new PerspectiveCamera( data.fov, data.aspect, data.near, data.far );
				if ( data.zoom !== undefined ) object.zoom = data.zoom;
				break;
			case 'OrthographicCamera':
				object = new OrthographicCamera( data.left, data.right, data.top, data.bottom, data.near, data.far );
				if ( data.zoom !== undefined ) object.zoom = data.zoom;
				break;
			case 'AmbientLight': object = new Lights.AmbientLight( data.color, data.intensity ); break;
			case 'DirectionalLight': object = new Lights.DirectionalLight( data.color, data.intensity ); break;
			case 'PointLight': object = new Lights.PointLight( data.color, data.intensity, data.distance, data.decay ); break;
			case 'RectAreaLight': object = new Lights.RectAreaLight( data.color, data.intensity, data.width, data.height ); break;
			case 'SpotLight': object = new Lights.SpotLight( data.color, data.intensity, data.distance, data.angle, data.penumbra, data.decay ); break;
			case 'HemisphereLight': object = new Lights.HemisphereLight( data.color, data.groundColor, data.intensity ); break;
			case 'LightProbe': object = new Lights.LightProbe().fromJSON( data ); break;
			case 'SkinnedMesh': object = new Objects.SkinnedMesh( geometry, material( data.material ) ); object._skeletonUuid = data.skeleton; object._bindMatrix = data.bindMatrix; object.bindMode = data.bindMode ?? 'attached'; break;
			case 'Mesh': object = new Objects.Mesh( geometry, material( data.material ) ); break;
			case 'InstancedMesh': {

				object = new Objects.InstancedMesh( geometry, material( data.material ), data.count );
				object.instanceMatrix.array.set( data.instanceMatrix.array );
				object.instanceMatrix.needsUpdate = true;
				if ( data.instanceColor ) {

					object.instanceColor = new Objects.InstancedBufferAttribute( new Float32Array( data.instanceColor.array ), data.instanceColor.itemSize );

				}

				break;

			}

			case 'LOD': object = new Objects.LOD(); break;
			case 'Line': object = new Objects.Line( geometry, material( data.material ) ); break;
			case 'LineLoop': object = new Objects.LineLoop( geometry, material( data.material ) ); break;
			case 'LineSegments': object = new Objects.LineSegments( geometry, material( data.material ) ); break;
			case 'Points': object = new Objects.Points( geometry, material( data.material ) ); break;
			case 'Sprite': object = new Objects.Sprite( material( data.material ) ); break;
			case 'Group': object = new Group(); break;
			case 'Bone': object = new Objects.Bone(); break;
			default: object = new Object3D();

		}

		object.uuid = data.uuid;
		if ( data.name !== undefined ) object.name = data.name;

		if ( data.matrix !== undefined ) {

			const matrix = new Matrix4().fromArray( data.matrix );
			if ( data.matrixAutoUpdate === false ) {

				object.matrixAutoUpdate = false;
				object.matrix.copy( matrix );

			} else {

				matrix.decompose( object.position, object.quaternion, object.scale );

			}

		} else {

			if ( data.position !== undefined ) object.position.fromArray( data.position );
			if ( data.rotation !== undefined ) object.rotation.fromArray( data.rotation );
			if ( data.quaternion !== undefined ) object.quaternion.fromArray( data.quaternion );
			if ( data.scale !== undefined ) object.scale.fromArray( data.scale );

		}

		if ( data.up !== undefined ) object.up.fromArray( data.up );
		if ( data.castShadow !== undefined ) object.castShadow = data.castShadow;
		if ( data.receiveShadow !== undefined ) object.receiveShadow = data.receiveShadow;

		if ( data.shadow && object.shadow ) {

			const shadow = object.shadow;
			if ( data.shadow.intensity !== undefined ) shadow.intensity = data.shadow.intensity;
			if ( data.shadow.bias !== undefined ) shadow.bias = data.shadow.bias;
			if ( data.shadow.normalBias !== undefined ) shadow.normalBias = data.shadow.normalBias;
			if ( data.shadow.radius !== undefined ) shadow.radius = data.shadow.radius;
			if ( data.shadow.mapSize !== undefined ) shadow.mapSize.fromArray( data.shadow.mapSize );

		}

		if ( data.visible !== undefined ) object.visible = data.visible;
		if ( data.frustumCulled !== undefined ) object.frustumCulled = data.frustumCulled;
		if ( data.renderOrder !== undefined ) object.renderOrder = data.renderOrder;
		if ( data.userData !== undefined ) object.userData = data.userData;
		if ( data.layers !== undefined ) object.layers.mask = data.layers;
		for ( const child of data.children ?? [] ) object.add( this.parseObject( child, geometries, materials, textures, animations ) );
		if ( data.animations ) object.animations = data.animations.map( ( uuid ) => animations[ uuid ] ).filter( Boolean );

		if ( data.type === 'LOD' ) {

			if ( data.autoUpdate !== undefined ) object.autoUpdate = data.autoUpdate;

			for ( const level of data.levels ?? [] ) {

				const child = object.getObjectByProperty( 'uuid', level.object );
				if ( child !== undefined ) object.addLevel( child, level.distance, level.hysteresis );

			}

		}

		if ( data.target !== undefined && object.target ) object._targetUuid = data.target;
		return object;

	}

	bindSkeletons( root, skeletons ) {

		const bones = new Map();
		root.traverse( ( object ) => {

			if ( object.isBone ) bones.set( object.uuid, object );

		} );
		const byUuid = Object.fromEntries( skeletons.map( ( data ) => [ data.uuid, new Objects.Skeleton( data.bones.map( ( uuid ) => bones.get( uuid ) ?? new Objects.Bone() ), data.boneInverses.map( ( m ) => new Matrix4().fromArray( m ) ) ) ] ) );

		root.traverse( ( object ) => {

			if ( object.isSkinnedMesh && object._skeletonUuid !== undefined && byUuid[ object._skeletonUuid ] ) object.bind( byUuid[ object._skeletonUuid ], object._bindMatrix ? new Matrix4().fromArray( object._bindMatrix ) : undefined );

		} );

	}

}
