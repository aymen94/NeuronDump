// The environment maps a frame's standard materials are lit by.
//
// Part of the materials; `../materials.js` gathers them.

import { ignored } from '../engine.js';

// The environment a frame's standard materials are lit by: as in three.js, a material's
// own `envMap`, or else `scene.environment`. The engine stacks every distinct map of a
// frame (up to eight) in one texture.
export class FrameEnvironment {

	constructor() {

		this.scene = null;
		this.sceneIntensity = 1;
		this.texture = null;
		this.rotation = null;
		// whether materials' own clipping planes apply (renderer.localClippingEnabled)
		this.localClipping = false;

	}

	begin( scene ) {

		const environment = scene.environment ?? null;
		this.scene = environment !== null && environment.isTexture ? environment : null;
		if ( environment !== null && this.scene === null ) ignored( 'a scene environment that is not a Texture' );
		this.sceneIntensity = scene.environmentIntensity ?? 1;
		this.texture = this.scene !== null && environmentReady( this.scene ) ? this.scene : null;
		this.rotation = this.texture !== null ? scene.environmentRotation : null;

	}

	// The intensity `material` is lit with, or -1 when it has no environment.
	intensityFor( material ) {

		const own = material.envMap ?? null;
		if ( own !== null ) return environmentReady( own ) ? material.envMapIntensity : - 1;
		return this.texture !== null ? this.sceneIntensity : - 1;

	}

}

// An environment or background can be drawn once its image has arrived.
export function environmentReady( texture ) {

	// a render target's texture once something was drawn into it
	if ( texture.isRenderTargetTexture ) return texture.version > 0;

	const image = texture.image, loaded = ( each ) => each !== null && each !== undefined && ( each.data !== undefined || each.width > 0 );
	if ( texture.isCubeTexture ) return Array.isArray( image ) && image.length === 6 && image.every( ( face ) => loaded( face?.isDataTexture ? face.image : face ) );
	return loaded( image );

}

// Width of the equirectangular image an environment or background is drawn from: a cube
// texture becomes one four faces wide.
export function environmentWidth( texture ) {

	if ( texture.isRenderTargetTexture ) return texture.renderTarget.width;
	if ( ! texture.isCubeTexture ) return texture.image.width;
	const face = texture.image[ 0 ];
	return 4 * ( face.isDataTexture ? face.image.width : face.width );

}
