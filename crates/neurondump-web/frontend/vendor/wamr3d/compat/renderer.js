// The WebGPU renderer of the object API.

import * as W from '../index.js';
import { createBackend } from '../webgpu.js';
import { createWebGLBackend } from '../webgl.js';
import { BasicShadowMap, CustomToneMapping, EquirectangularReflectionMapping, EquirectangularRefractionMapping, NoToneMapping, PCFShadowMap, PCFSoftShadowMap, UnsignedByteType, VSMShadowMap } from './constants.js';
import { ignored, instanceSources, w } from './engine.js';
import { FrameEnvironment, environmentReady, environmentWidth, shaderMaterials } from './materials.js';
import { Color, Matrix4 } from './math.js';
import { backends, textureSources } from './textures.js';

// ---------------------------------------------------------------- renderer

let frameStamp = 0;
const IDENTITY3 = new Float32Array( [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ] );
const linear = ( { r, g, b } ) => ( { r, g, b, a: 1 } );
const BACKENDS = [ 'auto', 'webgpu', 'webgl' ];

// The lights of a frame in world space, for shader materials with `lights` (WebGL).
function emptyShaderLights( lights ) {

	lights.ambient = [ 0, 0, 0 ];
	lights.probe = new Array( 27 ).fill( 0 );
	lights.probes = false;
	lights.directional = [];
	lights.point = [];
	lights.spot = [];
	lights.hemisphere = [];
	return lights;

}

function collectShaderLight( lights, light ) {

	const { r, g, b } = light.color, k = light.intensity, color = [ r * k, g * k, b * k ];

	if ( light.isAmbientLight ) {

		for ( let i = 0; i < 3; i ++ ) lights.ambient[ i ] += color[ i ];
		return;

	}

	if ( light.isLightProbe ) {

		const coefficients = light.sh.coefficients;
		for ( let i = 0; i < 9; i ++ ) {

			lights.probe[ i * 3 ] += coefficients[ i ].x * k;
			lights.probe[ i * 3 + 1 ] += coefficients[ i ].y * k;
			lights.probe[ i * 3 + 2 ] += coefficients[ i ].z * k;

		}

		lights.probes = true;
		return;

	}

	const m = w().matrixWorld( light.handle ), position = [ m[ 12 ], m[ 13 ], m[ 14 ] ];
	const t = light.target ? w().matrixWorld( light.target.handle ) : null, towards = t ? [ m[ 12 ] - t[ 12 ], m[ 13 ] - t[ 13 ], m[ 14 ] - t[ 14 ] ] : null;
	if ( light.isHemisphereLight ) lights.hemisphere.push( { direction: position, skyColor: color, groundColor: [ light.groundColor.r * k, light.groundColor.g * k, light.groundColor.b * k ] } );
	else if ( light.isDirectionalLight ) lights.directional.push( { direction: towards, color } );
	else if ( light.isPointLight ) lights.point.push( { position, color, distance: light.distance, decay: light.decay } );
	else if ( light.isSpotLight ) lights.spot.push( { position, direction: towards, color, distance: light.distance, decay: light.decay, coneCos: Math.cos( light.angle ), penumbraCos: Math.cos( light.angle * ( 1 - light.penumbra ) ) } );

}

// three.js's light uniforms of shader materials, in view space as three.js gives them.
function shaderLightUniforms( lights, view ) {

	const point = ( p ) => [ view[ 0 ] * p[ 0 ] + view[ 4 ] * p[ 1 ] + view[ 8 ] * p[ 2 ] + view[ 12 ], view[ 1 ] * p[ 0 ] + view[ 5 ] * p[ 1 ] + view[ 9 ] * p[ 2 ] + view[ 13 ], view[ 2 ] * p[ 0 ] + view[ 6 ] * p[ 1 ] + view[ 10 ] * p[ 2 ] + view[ 14 ] ];
	const direction = ( d ) => {

		const x = view[ 0 ] * d[ 0 ] + view[ 4 ] * d[ 1 ] + view[ 8 ] * d[ 2 ], y = view[ 1 ] * d[ 0 ] + view[ 5 ] * d[ 1 ] + view[ 9 ] * d[ 2 ], z = view[ 2 ] * d[ 0 ] + view[ 6 ] * d[ 1 ] + view[ 10 ] * d[ 2 ];
		const length = Math.hypot( x, y, z ) || 1;
		return [ x / length, y / length, z / length ];

	};

	const uniforms = { ambientLightColor: lights.ambient, lightProbe: lights.probe, receiveShadow: false };
	const add = ( name, list, fields ) => list.forEach( ( light, i ) => {

		for ( const [ field, convert ] of fields ) uniforms[ `${name}[${i}].${field}` ] = convert ? convert( light[ field ] ) : light[ field ];

	} );
	add( 'directionalLights', lights.directional, [ [ 'direction', direction ], [ 'color' ] ] );
	add( 'pointLights', lights.point, [ [ 'position', point ], [ 'color' ], [ 'distance' ], [ 'decay' ] ] );
	add( 'spotLights', lights.spot, [ [ 'position', point ], [ 'direction', direction ], [ 'color' ], [ 'distance' ], [ 'decay' ], [ 'coneCos' ], [ 'penumbraCos' ] ] );
	add( 'hemisphereLights', lights.hemisphere, [ [ 'direction', direction ], [ 'skyColor' ], [ 'groundColor' ] ] );
	const counts = [ lights.directional.length, lights.point.length, lights.spot.length, lights.hemisphere.length ];
	return { uniforms, counts, probes: lights.probes, key: counts.join( ',' ) + ( lights.probes ? ',p' : '' ) };

}

/**
 * The drawing backend for a canvas. `'webgpu'` and `'webgl'` use that API or fail;
 * `'auto'` tries WebGPU on a GPU, then WebGL 2, then the browser's software WebGPU
 * adapter, and reports a fallback once on the console.
 */
async function createRenderBackend( canvas, { backend, antialias, powerPreference } ) {

	if ( backend === 'webgpu' ) return createBackend( canvas, { antialias, powerPreference } );
	if ( backend === 'webgl' ) return createWebGLBackend( canvas, { antialias, powerPreference } );
	const failures = [];

	for ( const attempt of [
		() => createBackend( canvas, { antialias, powerPreference } ),
		() => createWebGLBackend( canvas, { antialias, powerPreference } ),
		() => createBackend( canvas, { antialias, fallbackAdapter: true } )
	] ) {

		try {

			const created = await attempt();
			if ( failures.length > 0 ) console.info( `${failures.join( '; ' )}: drawing with ${created.name === 'webgl' ? 'WebGL 2' : 'software WebGPU'}` );
			return created;

		} catch ( error ) {

			failures.push( error.message );

		}

	}

	throw new Error( `WAMR3D: no way to draw in this browser (${failures.join( '; ' )})` );

}

/**
 * Draws a scene with WebGPU, or with WebGL 2 where WebGPU is missing. `await
 * renderer.init()` before the first `render`.
 *
 * `backend`: `'auto'` (the default) picks WebGPU and falls back to WebGL 2; `'webgpu'` or
 * `'webgl'` use only that API. `forceWebGL: true`, as in three.js, is `backend: 'webgl'`.
 * After `init()`, `renderer.backend.isWebGPUBackend` or `isWebGLBackend` says which runs.
 */
export class WebGPURenderer {

	constructor( { canvas, antialias = false, powerPreference, backend = 'auto', forceWebGL = false } = {} ) {

		if ( ! BACKENDS.includes( backend ) ) throw new Error( `WAMR3D: unknown backend "${backend}" (use ${BACKENDS.join( ', ' )})` );
		this.domElement = canvas ?? document.createElement( 'canvas' );
		this.info = { render: { calls: 0, triangles: 0 }, memory: { geometries: 0 } };
		this._options = { antialias, powerPreference, backend: forceWebGL ? 'webgl' : backend };
		/** The backend drawing the frames once `init()` has finished, otherwise null. */
		this.backend = null;
		this._pixelRatio = 1;
		this._width = this.domElement.width;
		this._height = this.domElement.height;
		this._clearColor = new Color( 0x000000 );
		this.toneMapping = NoToneMapping;
		// false keeps what the target holds and draws over it
		this.autoClear = true;
		this.toneMappingExposure = 1;
		this.shadowMap = { enabled: false, autoUpdate: true, needsUpdate: false, type: PCFShadowMap };
		// global clipping planes in world space, and whether materials' own planes apply
		this.clippingPlanes = [];
		this.localClippingEnabled = false;
		this._clipping = new Float32Array( 0 );
		this._backend = null;
		this._frame = null;
		this._renderTarget = null;
		this._initializing = null;
		this._loop = null;
		this._environment = new FrameEnvironment();
		// inverse rotations of the environment and of the background, kept while unchanged
		this._rotations = [ { key: [ NaN, NaN, NaN, '' ], matrix: new Float32Array( 9 ) }, { key: [ NaN, NaN, NaN, '' ], matrix: new Float32Array( 9 ) } ];
		this._rotationMatrix = null;
		this._probe = new Float32Array( 27 );

	}

	init() {

		return this._initializing ??= createRenderBackend( this.domElement, this._options ).then( ( backend ) => {

			this._backend = this.backend = backend;
			backends.add( backend );
			this._frame = new W.Renderer();
			return this;

		} );

	}

	getPixelRatio() {

		return this._pixelRatio;

	}

	setPixelRatio( value ) {

		this._pixelRatio = value;
		this.setSize( this._width, this._height, false );

	}

	setSize( width, height, updateStyle = true ) {

		this._width = width;
		this._height = height;
		this.domElement.width = Math.floor( width * this._pixelRatio );
		this.domElement.height = Math.floor( height * this._pixelRatio );

		if ( updateStyle ) {

			this.domElement.style.width = width + 'px';
			this.domElement.style.height = height + 'px';

		}

	}

	getSize( target ) {

		return target.set( this._width, this._height );

	}

	/** Draws into `target` until set back to null (the canvas). */
	setRenderTarget( target ) {

		this._renderTarget = target ?? null;

	}

	getRenderTarget() {

		return this._renderTarget;

	}

	setClearColor( color ) {

		this._clearColor.set( color );

	}

	getClearColor( target ) {

		return target.copy( this._clearColor );

	}

	render( scene, camera ) {

		this._renderScene( scene, camera, false );

	}

	/**
	 * Builds the shader programs or pipelines that rendering `scene` (or `targetScene`,
	 * which holds `scene`) with `camera` needs, without drawing, so that the first frame
	 * does not wait for them.
	 */
	async compileAsync( scene, camera, targetScene = null ) {

		await this.init();
		await this._renderScene( targetScene ?? scene, camera, true );

	}

	// Sends the scene to the engine, then draws it, or only builds what drawing it needs.
	_renderScene( scene, camera, compile ) {

		if ( this._backend === null ) return;

		if ( scene.matrixWorldAutoUpdate ) scene.updateMatrixWorld();
		if ( camera.parent === null && camera.matrixWorldAutoUpdate ) camera.updateMatrixWorld();

		const sky = [ 0, 0, 0 ], ground = [ 0, 0, 0 ], probe = this._probe.fill( 0 ), frame = this._frame, MAX_LIGHTS = W.Renderer.maxLights();
		let lights = 0;
		if ( this.toneMapping === CustomToneMapping ) ignored( 'CustomToneMapping' );
		frame.setToneMapping( this.toneMapping, this.toneMappingExposure );
		const shadows = this.shadowMap.enabled === true;
		frame.setShadowsEnabled( shadows );

		if ( shadows ) {

			// As three.js's WebGPU renderer, PCFSoftShadowMap draws as PCFShadowMap.
			const type = this.shadowMap.type;
			frame.setShadowType( type === BasicShadowMap || type === VSMShadowMap ? type : PCFShadowMap );

		}
		const stamp = ++ frameStamp;
		frame.clearLights();
		const shaderLights = this._backend.isWebGLBackend && shaderMaterials.size > 0 ? emptyShaderLights( this._shaderLights ??= {} ) : null;
		const environment = this._environment;
		environment.begin( scene );
		environment.localClipping = this.localClippingEnabled === true;
		this._syncClipping( frame );
		const override = scene.overrideMaterial ?? null;

		if ( override !== null ) {

			override._frame = stamp;
			override._sync( environment );

		}

		frame.setOverrideMaterial( override !== null ? override.handle : 0n );
		const stack = [ scene ];

		while ( stack.length > 0 ) {

			const object = stack.pop();
			if ( ! object._visible ) continue;
			if ( object.isLOD && object.autoUpdate ) object.update( camera );
			// audio objects follow their world transform
			if ( object._onFrame !== undefined ) object._onFrame();

			if ( object._material !== undefined ) {

				// Materials are shared by many objects; each is sent once per frame.
				const material = object._material;

				if ( Array.isArray( material ) ) {

					for ( const each of material ) if ( each._frame !== stamp ) {

						each._frame = stamp;
						each._sync( environment );

					}

				} else if ( material._frame !== stamp ) {

					material._frame = stamp;
					material._sync( environment );

				}

				if ( object.isSprite ) object._syncCenter();
				else if ( object._syncMorph !== undefined ) {

					object._syncMorph();
					if ( object.isSkinnedMesh ) object._syncSkin();

				}

			} else if ( object.isLight ) {

				const { r, g, b } = object.color, k = object.intensity;
				if ( shaderLights !== null ) collectShaderLight( shaderLights, object );

				if ( object.isRectAreaLight ) {

					if ( ++ lights > MAX_LIGHTS ) {

						ignored( `more than ${MAX_LIGHTS} directional, point, spot and area lights` );

					} else {

						// half its width along its x axis and half its height along y, of its world rotation
						const m = w().matrixWorld( object.handle ), x = Math.hypot( m[ 0 ], m[ 1 ], m[ 2 ] ) || 1, y = Math.hypot( m[ 4 ], m[ 5 ], m[ 6 ] ) || 1;
						const hw = object.width * 0.5 / x, hh = object.height * 0.5 / y;
						frame.addRectAreaLight( m[ 12 ], m[ 13 ], m[ 14 ], m[ 0 ] * hw, m[ 1 ] * hw, m[ 2 ] * hw, m[ 4 ] * hh, m[ 5 ] * hh, m[ 6 ] * hh, r, g, b, k );

					}

				} else if ( object.isLightProbe ) {

					const coefficients = object.sh.coefficients;
					for ( let i = 0; i < 9; i ++ ) {

						probe[ i * 3 ] += coefficients[ i ].x * k;
						probe[ i * 3 + 1 ] += coefficients[ i ].y * k;
						probe[ i * 3 + 2 ] += coefficients[ i ].z * k;

					}

				} else if ( object.isDirectionalLight || object.isPointLight || object.isSpotLight ) {

					if ( ++ lights > MAX_LIGHTS ) {

						ignored( `more than ${MAX_LIGHTS} directional, point, spot and area lights` );

					} else {

						const from = w().matrixWorld( object.handle ), to = object.target ? w().matrixWorld( object.target.handle ) : null;
						if ( object.isDirectionalLight ) frame.addDirectionalLight( from[ 12 ] - to[ 12 ], from[ 13 ] - to[ 13 ], from[ 14 ] - to[ 14 ], r, g, b, k );
						else if ( object.isPointLight ) frame.addPointLight( from[ 12 ], from[ 13 ], from[ 14 ], r, g, b, k, object.distance, object.decay );
						else frame.addSpotLight( from[ 12 ], from[ 13 ], from[ 14 ], to[ 12 ], to[ 13 ], to[ 14 ], r, g, b, k, object.distance, object.decay, object.angle, object.penumbra );

						if ( shadows && object.castShadow ) {

							const shadow = object.shadow, c = shadow.camera, size = shadow.mapSize;
							// a point light draws six square faces of the map width from its position
							if ( object.isPointLight ) frame.setLastLightShadow( false, 90, 1, 0, 0, c.near, object.distance || c.far, from[ 12 ], from[ 13 ], from[ 14 ], shadow.bias, shadow.normalBias, shadow.intensity, shadow.radius, size.width, size.width );
							else if ( object.isDirectionalLight ) frame.setLastLightShadow( true, c.left, c.right, c.top, c.bottom, c.near, c.far, to[ 12 ], to[ 13 ], to[ 14 ], shadow.bias, shadow.normalBias, shadow.intensity, shadow.radius, size.width, size.height );
							else frame.setLastLightShadow( false, 2 * object.angle * shadow.focus * 180 / Math.PI, size.width / size.height * shadow.aspect, 0, 0, c.near, object.distance || c.far, to[ 12 ], to[ 13 ], to[ 14 ], shadow.bias, shadow.normalBias, shadow.intensity, shadow.radius, size.width, size.height );
							frame.setLastLightShadowBlur( shadow.blurSamples ?? 8 );

						}

					}

				} else if ( object.isHemisphereLight ) {

					const c = object.groundColor;
					sky[ 0 ] += r * k; sky[ 1 ] += g * k; sky[ 2 ] += b * k;
					ground[ 0 ] += c.r * k; ground[ 1 ] += c.g * k; ground[ 2 ] += c.b * k;

				} else if ( object.isAmbientLight ) {

					sky[ 0 ] += r * k; sky[ 1 ] += g * k; sky[ 2 ] += b * k;
					ground[ 0 ] += r * k; ground[ 1 ] += g * k; ground[ 2 ] += b * k;

				}

			}

			const children = object.children;
			for ( let i = 0; i < children.length; i ++ ) stack.push( children[ i ] );

		}

		this._frame.setAmbientLight( ...sky, ...ground );
		this._frame.setLightProbe( probe );

		const map = environment.texture;
		if ( map !== null ) frame.setEnvironment( map.id, environmentWidth( map ), this._inverseRotation( environment.rotation, 0 ) );
		else frame.clearEnvironment();
		this._syncBackground( scene );

		const fog = scene.fog;
		if ( fog ) frame.setFog( fog.isFogExp2 ? 2 : 1, fog.color.r, fog.color.g, fog.color.b, fog.near ?? 0, fog.far ?? 0, fog.density ?? 0 );
		else frame.setFog( 0, 0, 0, 0, 0, 0, 0 );

		const backend = this._backend;
		const rt = this._renderTarget, texture = rt?.texture;
		const target = rt ? { id: texture.id, width: Math.max( 1, rt.width | 0 ), height: Math.max( 1, rt.height | 0 ), samples: rt.samples, type: texture.type === UnsignedByteType ? 'byte' : 'float', texture, textures: rt.textures, depthTexture: rt.depthTexture ?? null } : null;
		if ( compile ) return backend.compile( w(), this._frame, scene.handle, camera.handle, target, this._extras( camera, backend, scene, shaderLights ) );
		backend.render( w(), this._frame, scene.handle, camera.handle, linear( scene.background && scene.background.isColor ? scene.background : this._clearColor ), textureSources, instanceSources, target, this._pixelRatio, this._extras( camera, backend, scene, shaderLights ) );
		// marks what was drawn into the target, for environments made from it
		if ( rt ) for ( const each of rt.textures ) each.version ++;
		this.info.render.calls = backend.info.calls;
		this.info.render.triangles = backend.info.triangles;
		this.info.memory.geometries = backend.info.geometries;

	}

	// What the backend needs besides the frame: whether to keep what the target holds
	// (`autoClear` false), and for GLSL shader materials the camera's view and projection
	// matrices and position.
	_extras( camera, backend, scene, shaderLights ) {

		const extras = { load: this.autoClear === false };
		if ( ! backend.isWebGLBackend || shaderMaterials.size === 0 ) return extras;
		const e = camera.matrixWorld.elements, view = Float32Array.from( camera.matrixWorldInverse.elements ), fog = scene.fog;
		// what shader materials with `lights` and `fog` read
		const lights = shaderLights !== null ? shaderLightUniforms( shaderLights, view ) : null;
		const fogs = fog ? { exp2: fog.isFogExp2 === true, uniforms: { fogColor: fog.color, fogNear: fog.near ?? 1, fogFar: fog.far ?? 1000, fogDensity: fog.density ?? 0 } } : null;
		return Object.assign( extras, { shaderMaterials, view, lights, fog: fogs, projection: Float32Array.from( camera.projectionMatrix.elements ), position: [ e[ 12 ], e[ 13 ], e[ 14 ] ], orthographic: camera.isOrthographicCamera === true } );

	}

	// The inverse of an Euler rotation as a column-major 3x3 matrix, kept in `slot` while
	// the angles stay the same.
	_inverseRotation( euler, slot ) {

		const { key, matrix } = this._rotations[ slot ], x = euler.x, y = euler.y, z = euler.z, order = euler.order;
		if ( x === key[ 0 ] && y === key[ 1 ] && z === key[ 2 ] && order === key[ 3 ] ) return matrix;
		key[ 0 ] = x;
		key[ 1 ] = y;
		key[ 2 ] = z;
		key[ 3 ] = order;
		const e = ( this._rotationMatrix ??= new Matrix4() ).makeRotationFromEuler( euler ).elements;
		// a rotation's inverse is its transpose
		matrix.set( [ e[ 0 ], e[ 4 ], e[ 8 ], e[ 1 ], e[ 5 ], e[ 9 ], e[ 2 ], e[ 6 ], e[ 10 ] ] );
		return matrix;

	}

	// A texture background as three.js draws it: equirectangular and cube textures around
	// the camera (blurred from the prefiltered map when `backgroundBlurriness` is above 0),
	// other textures stretched over the frame. Colours are the clear colour instead.
	// The global clipping planes (three.js `renderer.clippingPlanes`), sent when they change.
	_syncClipping( frame ) {

		const planes = this.clippingPlanes ?? [], count = Math.min( planes.length, 8 ), sent = this._clipping;
		let same = sent.length === count * 4;

		for ( let i = 0; same && i < count; i ++ ) {

			const { normal, constant } = planes[ i ], o = i * 4;
			same = sent[ o ] === normal.x && sent[ o + 1 ] === normal.y && sent[ o + 2 ] === normal.z && sent[ o + 3 ] === constant;

		}

		if ( same ) return;
		this._clipping = new Float32Array( count * 4 );
		for ( let i = 0; i < count; i ++ ) this._clipping.set( [ planes[ i ].normal.x, planes[ i ].normal.y, planes[ i ].normal.z, planes[ i ].constant ], i * 4 );
		frame.setClippingPlanes( this._clipping );

	}

	_syncBackground( scene ) {

		const frame = this._frame, background = scene.background;

		if ( ! background || background.isColor ) {

			frame.clearBackground();
			return;

		}

		if ( ! background.isTexture ) {

			ignored( 'a scene background that is not a Color or a Texture' );
			frame.clearBackground();
			return;

		}

		if ( ! environmentReady( background ) ) {

			frame.clearBackground();
			return;

		}

		const panorama = background.isCubeTexture === true || background.mapping === EquirectangularReflectionMapping || background.mapping === EquirectangularRefractionMapping;
		const blurriness = scene.backgroundBlurriness ?? 0, kind = panorama ? ( blurriness > 0 ? 1 : 0 ) : 2;
		if ( ! panorama && ( background.offset.x !== 0 || background.offset.y !== 0 || background.repeat.x !== 1 || background.repeat.y !== 1 || background.rotation !== 0 ) ) ignored( 'offset, repeat and rotation of a background texture' );
		const rotation = panorama ? this._inverseRotation( scene.backgroundRotation, 1 ) : IDENTITY3;
		frame.setBackground( background.id, kind, environmentWidth( background ), scene.backgroundIntensity ?? 1, blurriness, rotation );

	}

	async renderAsync( scene, camera ) {

		await this.init();
		this.render( scene, camera );

	}

	/** Calls `callback( time )` every animation frame; pass null to stop. */
	setAnimationLoop( callback ) {

		this._loop = callback;
		if ( callback === null ) return;
		this.init();

		const tick = ( time ) => {

			if ( this._loop !== callback ) return;
			requestAnimationFrame( tick );
			if ( this._backend !== null ) callback( time );

		};

		requestAnimationFrame( tick );

	}

	dispose() {

		this._loop = null;
		backends.delete( this._backend );
		this._backend?.dispose();
		this._frame?.free();
		this._backend = this._frame = this.backend = null;

	}

}

WebGPURenderer.prototype.isWebGPURenderer = true;

/**
 * The same renderer drawing with WebGL 2 (`backend: 'webgl'`), for code written for
 * three.js's `WebGLRenderer`. It starts `init()` itself; frames rendered before it has
 * finished are skipped, and `setAnimationLoop` waits for it.
 */
export class WebGLRenderer extends WebGPURenderer {

	constructor( parameters = {} ) {

		super( { ...parameters, backend: 'webgl' } );
		this.init().catch( ( error ) => console.error( error ) );

	}

}

WebGLRenderer.prototype.isWebGPURenderer = false;
WebGLRenderer.prototype.isWebGLRenderer = true;
