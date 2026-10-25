// GPU state kept between frames: targets, uniform buffers, bones, placeholders and per-frame lists.
//
// Part of the WebGPU host; `../webgpu.js` creates the device and draws the frames.

/** Adds this part's functions and state to the host context `ctx`. */
export function installState( ctx ) {

	const { SHADOW_FORMAT, device } = ctx;

	ctx.shadowTexture = device.createTexture( { size: [ 1, 1, 1 ], format: SHADOW_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT } );
	ctx.boneTexture = device.createTexture( { size: [ 4, 1 ], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST } );
	ctx.boneView = ctx.boneTexture.createView();
	ctx.boneGroup = null;
	ctx.transmission = null;
	ctx.vsmTexture = null;
	ctx.vsmView = null;
	ctx.vsmScratch = null;
	// Two versions of group 3: one reading the transmission copy, one with the
	// placeholder for the pass that draws the copy (a texture cannot be read and drawn
	// into at once).
	ctx.frameTexturesGroup = null;
	ctx.frameTexturesGroupNoCopy = null;

	ctx.frameBuffer = null;
	ctx.frameGroup = null;
	ctx.objectBuffer = null;
	ctx.objectCapacity = 0;
	ctx.materialBuffer = null;
	ctx.materialCapacity = 0;
	ctx.objectGroup = null;
	ctx.depthTexture = null;
	ctx.colorTexture = null;
	ctx.depthView = null;
	ctx.colorView = null;
	ctx.targetWidth = 0;
	ctx.targetHeight = 0;
	// The linear target of a tone-mapped frame, its multisampled companion, and the output pass.
	ctx.hdrTexture = null;
	ctx.hdrMultisample = null;
	ctx.hdrView = null;
	ctx.hdrMultisampleView = null;
	ctx.outputGroup = null;
	ctx.outputPipeline = null;
	ctx.outputSettings = null;
	// Bound in place of an attribute a geometry does not have; sized for the widest input.
	ctx.zeros = null;
	ctx.zeroBytes = 64;
	const auxZeros = device.createBuffer( { size: 64, usage: GPUBufferUsage.VERTEX } );
	const geometries = new Map(), textures = new Map(), blits = new Map(), instanced = new Map();
	// Per-frame lists, kept between frames so that drawing does not allocate them.
	const entries = [], groups = [], perInstance = [], distinct = [];

	// What an object without instances is drawn with: one identity matrix. `ones` is the
	// instance colour of everything that has none, and grows with the largest count.
	const identity = device.createBuffer( { size: 64, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST } );
	device.queue.writeBuffer( identity, 0, new Float32Array( [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ] ) );
	ctx.ones = null;
	ctx.oneCount = 1;
	ctx.frameNumber = 0;
	ctx.drawingInto = 0;
	const info = { calls: 0, shadowCalls: 0, triangles: 0, candidates: 0, geometries: 0, textures: 0, pipelines: 0, uploads: 0 };

	Object.assign( ctx, { auxZeros, blits, distinct, entries, geometries, groups, identity, info, instanced, perInstance, textures } );

}
