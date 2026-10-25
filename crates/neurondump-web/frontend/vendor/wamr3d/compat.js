// WAMR3D object API.
//
//   import * as WAMR3D from 'wamr3d/compat';
//   await WAMR3D.init();
//   const scene = new WAMR3D.Scene();
//   const mesh = new WAMR3D.Mesh( new WAMR3D.BoxGeometry(), new WAMR3D.MeshLambertMaterial( { color: 0x4f8cff } ) );
//   scene.add( mesh );
//   mesh.rotation.y = 0.5;
//
// Classes, properties and chaining over the engine. All state lives in one World inside
// the WebAssembly module; the objects here are thin handles onto it.


import * as W from './index.js';

export * from './compat/constants.js';
export { init, getWorld } from './compat/engine.js';
export { Vector2, Vector3, Vector4, Quaternion, Euler, Matrix3, Matrix4, ColorManagement, Color, Box3, Sphere, Plane, Ray, Frustum, Line3, Triangle, Spherical, Cylindrical } from './compat/math.js';
export { Object3D, Group, Scene, Fog, FogExp2 } from './compat/object3d.js';
export { Layers } from './compat/transform.js';
export { Mesh, InstancedBufferAttribute, InstancedMesh, Line, Points, Sprite, LineSegments, LineLoop, LOD, GridHelper, AxesHelper, Bone, Skeleton, SkinnedMesh } from './compat/objects.js';
export { Camera, PerspectiveCamera, OrthographicCamera } from './compat/cameras.js';
export { Light, AmbientLight, HemisphereLight, DirectionalLight, LightShadow, DirectionalLightShadow, SpotLightShadow, PointLightShadow, PointLight, SpotLight, RectAreaLight, LightProbe } from './compat/lights.js';
export { SphericalHarmonics3 } from './compat/sh.js';
export { BufferAttribute, InterleavedBuffer, InterleavedBufferAttribute, Float32BufferAttribute, Int8BufferAttribute, Uint8BufferAttribute, Uint8ClampedBufferAttribute, Int16BufferAttribute, Uint16BufferAttribute, Int32BufferAttribute, Uint32BufferAttribute, InstancedInterleavedBuffer, BufferGeometry, InstancedBufferGeometry, BoxGeometry, PlaneGeometry, SphereGeometry, CylinderGeometry, ConeGeometry, CircleGeometry, RingGeometry, TorusGeometry, PolyhedronGeometry, TetrahedronGeometry, OctahedronGeometry, DodecahedronGeometry, IcosahedronGeometry, TorusKnotGeometry, LatheGeometry, CapsuleGeometry } from './compat/geometry.js';
export { Texture, RenderTarget, WebGLRenderTarget, DataTexture, CanvasTexture, DepthTexture, CubeTexture } from './compat/textures.js';
export { DataUtils } from './compat/data-utils.js';
export { Cache, LoadingManager, DefaultLoadingManager, Loader, LoaderUtils, FileLoader, ImageLoader, ImageBitmapLoader, TextureLoader, CubeTextureLoader, DataTextureLoader, AnimationLoader, BufferGeometryLoader, MaterialLoader, ObjectLoader } from './compat/loaders.js';
export { Material, MeshBasicMaterial, LineBasicMaterial, PointsMaterial, SpriteMaterial, MeshLambertMaterial, MeshPhongMaterial, MeshToonMaterial, MeshNormalMaterial, MeshStandardMaterial, MeshPhysicalMaterial, LineDashedMaterial, MeshMatcapMaterial, MeshDepthMaterial, MeshDistanceMaterial, ShadowMaterial, ShaderMaterial, RawShaderMaterial, UniformsUtils } from './compat/materials.js';
export { Raycaster } from './compat/picking.js';
export { ArrowHelper, BoxHelper, Box3Helper, PlaneHelper, CameraHelper, DirectionalLightHelper, PointLightHelper, SpotLightHelper, HemisphereLightHelper, PolarGridHelper, SkeletonHelper } from './compat/helpers.js';
export { AudioContext, AudioListener, Audio, PositionalAudio, AudioAnalyser, AudioLoader } from './compat/audio.js';
export { EventDispatcher, Controls } from './compat/events.js';
export { Box2, Curve, CurvePath, Path, Shape, ShapePath, ShapeUtils, Earcut, Curves, ArcCurve, CatmullRomCurve3, CubicBezierCurve, CubicBezierCurve3, EllipseCurve, LineCurve, LineCurve3, QuadraticBezierCurve, QuadraticBezierCurve3, SplineCurve } from './compat/curves.js';
export { ShapeGeometry, ExtrudeGeometry, TubeGeometry, EdgesGeometry, WireframeGeometry } from './compat/shapes.js';
export { Interpolant, DiscreteInterpolant, LinearInterpolant, QuaternionLinearInterpolant, CubicInterpolant, BezierInterpolant, AnimationUtils, KeyframeTrack, BooleanKeyframeTrack, ColorKeyframeTrack, NumberKeyframeTrack, QuaternionKeyframeTrack, StringKeyframeTrack, VectorKeyframeTrack, AnimationClip, PropertyBinding, PropertyMixer, AnimationAction, AnimationMixer, AnimationObjectGroup } from './compat/animation.js';
export { WebGLRenderer, WebGPURenderer } from './compat/renderer.js';

export const MathUtils = W.MathUtils;
export const Timer = W.Timer;

// ---------------------------------------------------------------- time

export class Clock {

	constructor( autoStart = true ) {

		this.autoStart = autoStart;
		this.startTime = 0;
		this.oldTime = 0;
		this.elapsedTime = 0;
		this.running = false;

	}

	start() {

		this.startTime = this.oldTime = performance.now();
		this.elapsedTime = 0;
		this.running = true;

	}

	stop() {

		this.getElapsedTime();
		this.running = this.autoStart = false;

	}

	getElapsedTime() {

		this.getDelta();
		return this.elapsedTime;

	}

	getDelta() {

		if ( this.autoStart && ! this.running ) {

			this.start();
			return 0;

		}

		if ( ! this.running ) return 0;
		const now = performance.now(), delta = ( now - this.oldTime ) / 1000;
		this.oldTime = now;
		this.elapsedTime += delta;
		return delta;

	}

}
