// The built-in materials, from MeshBasicMaterial to MeshPhysicalMaterial.
//
// Part of the materials; `../materials.js` gathers them.

import { w } from '../engine.js';
import { Color, Euler, Vector2 } from '../math.js';
import { EMISSIVE_KEYS, ENVIRONMENT_KEYS, LIT_MAPS, MAP_KEYS, Material, WIREFRAME_KEYS, mapKeys, withMaps, withWireframe } from './material.js';

const BASIC_MAPS = [ 'alphaMap', 'aoMap', 'lightMap', 'specularMap' ];
// an envMap of their own, combined with the surface (three.js `combine`, `reflectivity`,
// `refractionRatio`), for basic, Lambert and Phong materials
const OWN_ENVIRONMENT_KEYS = [ 'envMap', 'envMapRotation', 'combine', 'reflectivity', 'refractionRatio' ];

function withOwnEnvironment( material ) {

	material.envMap = null;
	material.envMapRotation = new Euler();
	material.combine = 0;
	material.reflectivity = 1;
	material.refractionRatio = 0.98;

}

/** Shows its colour (and map) as is, without lighting; a light map adds baked light. */
export class MeshBasicMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshBasicMaterial';
		withMaps( this, BASIC_MAPS );
		withOwnEnvironment( this );
		withWireframe( this );
		if ( parameters ) this.setValues( parameters );

	}

}

MeshBasicMaterial.prototype._shading = 1;
MeshBasicMaterial.prototype._keys = new Set( [ ...mapKeys( BASIC_MAPS ), ...WIREFRAME_KEYS, ...OWN_ENVIRONMENT_KEYS ] );
MeshBasicMaterial.prototype.isMeshBasicMaterial = true;

/** For `Line`, `LineSegments` and `LineLoop`: the colour as is, without lighting. */
export class LineBasicMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'LineBasicMaterial';
		this.linewidth = 1;
		if ( parameters ) this.setValues( parameters );

	}

}

LineBasicMaterial.prototype._shading = 1;
LineBasicMaterial.prototype.isLineBasicMaterial = true;

/**
 * For lines with a `lineDistance` attribute (`line.computeLineDistances()`): dashes of
 * `dashSize` separated by gaps of `gapSize`, along the distance times `scale`.
 */
export class LineDashedMaterial extends LineBasicMaterial {

	constructor( parameters ) {

		super();
		this.type = 'LineDashedMaterial';
		this.scale = 1;
		this.dashSize = 3;
		this.gapSize = 1;
		if ( parameters ) this.setValues( parameters );

	}

}

LineDashedMaterial.prototype._keys = new Set( [ 'scale', 'dashSize', 'gapSize' ] );
LineDashedMaterial.prototype.isLineDashedMaterial = true;

/**
 * For `Points`: the colour (and vertex colours, `map` and `alphaMap`) as is, without
 * lighting. Points larger than one pixel are drawn as squares of `size` pixels (times the
 * pixel ratio, smaller with distance under `sizeAttenuation`), as three.js's
 * WebGLRenderer draws them.
 */
export class PointsMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'PointsMaterial';
		this.size = 1;
		this.sizeAttenuation = true;
		withMaps( this, [ 'alphaMap' ] );
		if ( parameters ) this.setValues( parameters );

	}

}

PointsMaterial.prototype._shading = 1;
PointsMaterial.prototype._keys = new Set( [ 'alphaMap' ] );
PointsMaterial.prototype.isPointsMaterial = true;

/** For `Sprite`: colour and map without lighting, `rotation` in radians, transparent by default. */
export class SpriteMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'SpriteMaterial';
		this.transparent = true;
		this.rotation = 0;
		this.sizeAttenuation = true;
		withMaps( this, [ 'alphaMap' ] );
		if ( parameters ) this.setValues( parameters );

	}

}

SpriteMaterial.prototype._shading = 1;
SpriteMaterial.prototype._keys = new Set( [ 'alphaMap' ] );
SpriteMaterial.prototype.isSpriteMaterial = true;

/** Diffuse (Lambert) lighting, plus `emissive` light. */
export class MeshLambertMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshLambertMaterial';
		this.emissive = new Color( 0x000000 );
		this.emissiveIntensity = 1;
		this.normalMapType = 0;
		withMaps( this, this._maps );
		withOwnEnvironment( this );
		withWireframe( this );
		if ( parameters ) this.setValues( parameters );

	}

}

MeshLambertMaterial.prototype._maps = [ ...LIT_MAPS, 'specularMap' ];
MeshLambertMaterial.prototype._keys = new Set( [ ...EMISSIVE_KEYS, ...mapKeys( MeshLambertMaterial.prototype._maps ), ...WIREFRAME_KEYS, ...OWN_ENVIRONMENT_KEYS, 'normalMapType', 'flatShading' ] );
MeshLambertMaterial.prototype.isMeshLambertMaterial = true;

/** Diffuse lighting plus a Blinn-Phong highlight (`specular`, `shininess`, scaled by `specularMap`), and `emissive` light. */
export class MeshPhongMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshPhongMaterial';
		this.specular = new Color( 0x111111 );
		this.shininess = 30;
		this.emissive = new Color( 0x000000 );
		this.emissiveIntensity = 1;
		this.normalMapType = 0;
		withMaps( this, this._maps );
		withOwnEnvironment( this );
		withWireframe( this );
		if ( parameters ) this.setValues( parameters );

	}

}

const PHONG_MAPS = [ ...LIT_MAPS, 'specularMap' ];
MeshPhongMaterial.prototype._shading = 3;
MeshPhongMaterial.prototype._maps = PHONG_MAPS;
MeshPhongMaterial.prototype._keys = new Set( [ 'specular', 'shininess', ...EMISSIVE_KEYS, ...mapKeys( PHONG_MAPS ), ...WIREFRAME_KEYS, ...OWN_ENVIRONMENT_KEYS, 'normalMapType', 'flatShading' ] );
MeshPhongMaterial.prototype.isMeshPhongMaterial = true;

/** Diffuse lighting in tones: two without a `gradientMap`, else its red channel along n.l. Plus `emissive` light. */
export class MeshToonMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshToonMaterial';
		this.emissive = new Color( 0x000000 );
		this.emissiveIntensity = 1;
		withMaps( this, this._maps );
		withWireframe( this );
		if ( parameters ) this.setValues( parameters );

	}

}

const TOON_MAPS = [ ...LIT_MAPS, 'gradientMap' ];
MeshToonMaterial.prototype._shading = 5;
MeshToonMaterial.prototype._maps = TOON_MAPS;
MeshToonMaterial.prototype._keys = new Set( [ ...EMISSIVE_KEYS, ...mapKeys( TOON_MAPS ), ...WIREFRAME_KEYS ] );
MeshToonMaterial.prototype.isMeshToonMaterial = true;

const NORMAL_MAPS = [ 'normalMap', 'bumpMap', 'displacementMap' ];

/** The surface normal in view space as a colour; no lights, colour or fog. */
export class MeshNormalMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshNormalMaterial';
		withMaps( this, NORMAL_MAPS );
		withWireframe( this );
		if ( parameters ) this.setValues( parameters );

	}

}

MeshNormalMaterial.prototype._shading = 4;
MeshNormalMaterial.prototype._keys = new Set( [ ...mapKeys( NORMAL_MAPS ), ...WIREFRAME_KEYS ] );
MeshNormalMaterial.prototype.isMeshNormalMaterial = true;

/**
 * Physically based: diffuse plus a specular highlight from `roughness` and `metalness`,
 * `emissive` light, and reflections of `envMap` or `scene.environment` (an
 * equirectangular texture).
 */
export class MeshStandardMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshStandardMaterial';
		this.roughness = 1;
		this.metalness = 0;
		this.emissive = new Color( 0x000000 );
		this.emissiveIntensity = 1;
		this.envMap = null;
		this.envMapIntensity = 1;
		this.envMapRotation = new Euler();
		withMaps( this, MAP_KEYS );
		withWireframe( this );
		if ( parameters ) this.setValues( parameters );

	}

}

MeshStandardMaterial.prototype._shading = 2;
MeshStandardMaterial.prototype._maps = MAP_KEYS;
MeshStandardMaterial.prototype._keys = new Set( [ 'roughness', 'metalness', ...EMISSIVE_KEYS, ...ENVIRONMENT_KEYS, ...mapKeys( MAP_KEYS ), ...WIREFRAME_KEYS ] );
MeshStandardMaterial.prototype.isMeshStandardMaterial = true;

const PHYSICAL_KEYS = [ 'clearcoat', 'clearcoatRoughness', 'ior', 'reflectivity', 'sheen', 'sheenColor', 'sheenRoughness', 'transmission', 'thickness', 'attenuationColor', 'attenuationDistance', 'iridescence', 'iridescenceIOR', 'iridescenceThicknessRange', 'specularIntensity', 'specularColor', 'dispersion', 'anisotropy', 'anisotropyRotation', 'clearcoatNormalScale' ];

/**
 * `MeshStandardMaterial` with three.js's clearcoat, sheen, specular layer (`ior`,
 * `specularIntensity`, `specularColor`), iridescence, anisotropy and transmission (with
 * `thickness`, attenuation and `dispersion`), and the maps of those layers.
 */
export class MeshPhysicalMaterial extends MeshStandardMaterial {

	constructor( parameters ) {

		super();
		this.type = 'MeshPhysicalMaterial';
		this.clearcoat = 0;
		this.clearcoatRoughness = 0;
		this.ior = 1.5;
		this.sheen = 0;
		this.sheenColor = new Color( 0x000000 );
		this.sheenRoughness = 1;
		this.transmission = 0;
		this.thickness = 0;
		this.attenuationColor = new Color( 1, 1, 1 );
		this.attenuationDistance = Infinity;
		this.iridescence = 0;
		this.iridescenceIOR = 1.3;
		this.specularIntensity = 1;
		this.specularColor = new Color( 1, 1, 1 );
		this.dispersion = 0;
		this.anisotropy = 0;
		this.anisotropyRotation = 0;
		this.iridescenceThicknessRange = [ 100, 400 ];
		this.clearcoatNormalScale = new Vector2( 1, 1 );
		for ( const key of PHYSICAL_MAP_KEYS ) this[ key ] = null;
		if ( parameters ) this.setValues( parameters );

	}

	// three.js derives `reflectivity` from `ior`.
	get reflectivity() {

		return Math.min( Math.max( 2.5 * ( this.ior - 1 ) / ( this.ior + 1 ), 0 ), 1 );

	}

	set reflectivity( value ) {

		this.ior = ( 1 + 0.4 * value ) / ( 1 - 0.4 * value );

	}

	copy( source ) {

		super.copy( source );
		for ( const key of PHYSICAL_KEYS ) {

			const value = source[ key ];
			if ( key === 'reflectivity' || value === undefined ) continue;
			if ( value !== null && value.isColor ) this[ key ].copy( value );
			else this[ key ] = value;

		}

		return this;

	}

	_sync( environment ) {

		const sent = this._sent, sheen = this.sheen, c = this.sheenColor, s = this.specularColor;
		const values = [ this.clearcoat, this.clearcoatRoughness, c.r * sheen, c.g * sheen, c.b * sheen, this.sheenRoughness, this.ior, this.specularIntensity, s.r, s.g, s.b ];
		let changed = false;

		for ( let i = 0; i < values.length; i ++ ) {

			if ( values[ i ] !== sent[ 49 + i ] ) changed = true;
			sent[ 49 + i ] = values[ i ];

		}

		if ( changed ) w().setMaterialPhysical( this.handle, ...values );
		const a = this.attenuationColor, range = this.iridescenceThicknessRange, n = this.clearcoatNormalScale;
		const extensions = [ this.transmission, this.thickness, a.r, a.g, a.b, this.attenuationDistance, this.dispersion, this.iridescence, this.iridescenceIOR, range[ 0 ], range[ 1 ], this.anisotropy, this.anisotropyRotation, n.x, n.y ];
		changed = false;

		for ( let i = 0; i < extensions.length; i ++ ) {

			if ( extensions[ i ] !== sent[ 70 + i ] ) changed = true;
			sent[ 70 + i ] = extensions[ i ];

		}

		if ( changed ) w().setMaterialPhysicalExtensions( this.handle, ...extensions );
		super._sync( environment );

	}

}

// Maps of the extra layers.
const PHYSICAL_MAP_KEYS = [ 'clearcoatMap', 'clearcoatRoughnessMap', 'clearcoatNormalMap', 'sheenColorMap', 'sheenRoughnessMap', 'specularIntensityMap', 'specularColorMap', 'transmissionMap', 'thicknessMap', 'iridescenceMap', 'iridescenceThicknessMap', 'anisotropyMap' ];
MeshPhysicalMaterial.prototype._keys = new Set( [ ...MeshStandardMaterial.prototype._keys, ...PHYSICAL_KEYS, ...PHYSICAL_MAP_KEYS ] );
MeshPhysicalMaterial.prototype.isMeshPhysicalMaterial = true;

const MATCAP_MAPS = [ 'normalMap', 'bumpMap', 'displacementMap', 'alphaMap' ];

/** The colour of `matcap` (a lit sphere's image) along the view-space normal, times `color`; no lights. */
export class MeshMatcapMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshMatcapMaterial';
		this.matcap = null;
		this.normalMapType = 0;
		withMaps( this, MATCAP_MAPS );
		if ( parameters ) this.setValues( parameters );

	}

}

MeshMatcapMaterial.prototype._shading = 6;
MeshMatcapMaterial.prototype._keys = new Set( [ 'matcap', 'normalMapType', ...mapKeys( MATCAP_MAPS ) ] );
MeshMatcapMaterial.prototype.isMeshMatcapMaterial = true;

const DEPTH_MAPS = [ 'alphaMap', 'displacementMap' ];

/** The fragment's depth, grey (`BasicDepthPacking`, white near) or packed into RGBA, RGB or RG. */
export class MeshDepthMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshDepthMaterial';
		this.depthPacking = 3200;
		withMaps( this, DEPTH_MAPS );
		withWireframe( this );
		if ( parameters ) this.setValues( parameters );

	}

}

MeshDepthMaterial.prototype._shading = 7;
MeshDepthMaterial.prototype._keys = new Set( [ 'depthPacking', ...mapKeys( DEPTH_MAPS ), ...WIREFRAME_KEYS ] );
MeshDepthMaterial.prototype.isMeshDepthMaterial = true;

/** The distance from the camera between its near and far planes, packed into RGBA. */
export class MeshDistanceMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'MeshDistanceMaterial';
		withMaps( this, DEPTH_MAPS );
		if ( parameters ) this.setValues( parameters );

	}

}

MeshDistanceMaterial.prototype._shading = 8;
MeshDistanceMaterial.prototype._keys = new Set( mapKeys( DEPTH_MAPS ) );
MeshDistanceMaterial.prototype.isMeshDistanceMaterial = true;

/** Only the shadows falling on it, in `color` with `opacity`; transparent elsewhere. */
export class ShadowMaterial extends Material {

	constructor( parameters ) {

		super();
		this.type = 'ShadowMaterial';
		this.color.setHex( 0x000000 );
		this.transparent = true;
		if ( parameters ) this.setValues( parameters );

	}

}

ShadowMaterial.prototype._shading = 9;
ShadowMaterial.prototype.isShadowMaterial = true;
