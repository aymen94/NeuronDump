/* @ts-self-types="./wamr3d_wasm.d.ts" */

export class Box3 {
    static __wrap(ptr) {
        const obj = Object.create(Box3.prototype);
        obj.__wbg_ptr = ptr;
        Box3Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Box3Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_box3_free(ptr, 0);
    }
    /**
     * @param {Matrix4} m
     */
    applyMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.box3_applyMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Vector3} point
     * @returns {Vector3}
     */
    clampPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.box3_clampPoint(this.__wbg_ptr, point.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Box3}
     */
    clone() {
        const ret = wasm.box3_clone(this.__wbg_ptr);
        return Box3.__wrap(ret);
    }
    /**
     * @param {Box3} other
     * @returns {boolean}
     */
    containsBox(other) {
        _assertClass(other, Box3);
        const ret = wasm.box3_containsBox(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Vector3} point
     * @returns {boolean}
     */
    containsPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.box3_containsPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Box3} source
     */
    copy(source) {
        _assertClass(source, Box3);
        wasm.box3_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Vector3} point
     * @returns {number}
     */
    distanceToPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.box3_distanceToPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Box3} other
     * @returns {boolean}
     */
    equals(other) {
        _assertClass(other, Box3);
        const ret = wasm.box3_equals(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Vector3} point
     */
    expandByPoint(point) {
        _assertClass(point, Vector3);
        wasm.box3_expandByPoint(this.__wbg_ptr, point.__wbg_ptr);
    }
    /**
     * @param {number} scalar
     */
    expandByScalar(scalar) {
        wasm.box3_expandByScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {Vector3} vector
     */
    expandByVector(vector) {
        _assertClass(vector, Vector3);
        wasm.box3_expandByVector(this.__wbg_ptr, vector.__wbg_ptr);
    }
    /**
     * @returns {Sphere}
     */
    getBoundingSphere() {
        const ret = wasm.box3_getBoundingSphere(this.__wbg_ptr);
        return Sphere.__wrap(ret);
    }
    /**
     * @returns {Vector3}
     */
    getCenter() {
        const ret = wasm.box3_getCenter(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Vector3}
     */
    getSize() {
        const ret = wasm.box3_getSize(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Box3} other
     */
    intersect(other) {
        _assertClass(other, Box3);
        wasm.box3_intersect(this.__wbg_ptr, other.__wbg_ptr);
    }
    /**
     * @param {Box3} other
     * @returns {boolean}
     */
    intersectsBox(other) {
        _assertClass(other, Box3);
        const ret = wasm.box3_intersectsBox(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Plane} plane
     * @returns {boolean}
     */
    intersectsPlane(plane) {
        _assertClass(plane, Plane);
        const ret = wasm.box3_intersectsPlane(this.__wbg_ptr, plane.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Sphere} sphere
     * @returns {boolean}
     */
    intersectsSphere(sphere) {
        _assertClass(sphere, Sphere);
        const ret = wasm.box3_intersectsSphere(this.__wbg_ptr, sphere.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    isEmpty() {
        const ret = wasm.box3_isEmpty(this.__wbg_ptr);
        return ret !== 0;
    }
    makeEmpty() {
        wasm.box3_makeEmpty(this.__wbg_ptr);
    }
    /**
     * @returns {Vector3}
     */
    get max() {
        const ret = wasm.box3_max(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Vector3}
     */
    get min() {
        const ret = wasm.box3_min(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * An empty box.
     */
    constructor() {
        const ret = wasm.box3_new();
        this.__wbg_ptr = ret;
        Box3Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * The box around a flat list of xyz positions.
     * @param {Float32Array} positions
     */
    setFromArray(positions) {
        const ptr0 = passArrayF32ToWasm0(positions, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.box3_setFromArray(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {Vector3} center
     * @param {Vector3} size
     */
    setFromCenterAndSize(center, size) {
        _assertClass(center, Vector3);
        _assertClass(size, Vector3);
        wasm.box3_setFromCenterAndSize(this.__wbg_ptr, center.__wbg_ptr, size.__wbg_ptr);
    }
    /**
     * @param {Vector3} min
     * @param {Vector3} max
     */
    set(min, max) {
        _assertClass(min, Vector3);
        _assertClass(max, Vector3);
        wasm.box3_set(this.__wbg_ptr, min.__wbg_ptr, max.__wbg_ptr);
    }
    /**
     * @param {Vector3} offset
     */
    translate(offset) {
        _assertClass(offset, Vector3);
        wasm.box3_translate(this.__wbg_ptr, offset.__wbg_ptr);
    }
    /**
     * @param {Box3} other
     */
    union(other) {
        _assertClass(other, Box3);
        wasm.box3_union(this.__wbg_ptr, other.__wbg_ptr);
    }
}
if (Symbol.dispose) Box3.prototype[Symbol.dispose] = Box3.prototype.free;

export class Color {
    static __wrap(ptr) {
        const obj = Object.create(Color.prototype);
        obj.__wbg_ptr = ptr;
        ColorFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ColorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_color_free(ptr, 0);
    }
    /**
     * @param {Color} a
     * @param {Color} b
     */
    addColors(a, b) {
        _assertClass(a, Color);
        _assertClass(b, Color);
        wasm.color_addColors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {number} s
     */
    addScalar(s) {
        wasm.color_addScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Color} color
     */
    add(color) {
        _assertClass(color, Color);
        wasm.color_add(this.__wbg_ptr, color.__wbg_ptr);
    }
    /**
     * @param {Matrix3} m
     */
    applyMatrix3(m) {
        _assertClass(m, Matrix3);
        wasm.color_applyMatrix3(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    get b() {
        const ret = wasm.color_b(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Color}
     */
    clone() {
        const ret = wasm.color_clone(this.__wbg_ptr);
        return Color.__wrap(ret);
    }
    convertLinearToSRGB() {
        wasm.color_convertLinearToSRGB(this.__wbg_ptr);
    }
    convertSRGBToLinear() {
        wasm.color_convertSRGBToLinear(this.__wbg_ptr);
    }
    /**
     * @param {Color} source
     */
    copy(source) {
        _assertClass(source, Color);
        wasm.color_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Color} c
     * @returns {boolean}
     */
    equals(c) {
        _assertClass(c, Color);
        const ret = wasm.color_equals(this.__wbg_ptr, c.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * From an sRGB hex value such as `0xff8800`.
     * @param {number} hex
     * @returns {Color}
     */
    static fromHex(hex) {
        const ret = wasm.color_fromHex(hex);
        return Color.__wrap(ret);
    }
    /**
     * From a CSS colour string: `#f80`, `rgb(255,136,0)`, `hsl(32,100%,50%)` or a name.
     * @param {string} style
     * @returns {Color}
     */
    static fromStyle(style) {
        const ptr0 = passStringToWasm0(style, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.color_fromStyle(ptr0, len0);
        return Color.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get g() {
        const ret = wasm.color_g(this.__wbg_ptr);
        return ret;
    }
    /**
     * Hue, saturation, lightness.
     * @returns {Float32Array}
     */
    getHSL() {
        const ret = wasm.color_getHSL(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {string}
     */
    getHexString() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.color_getHexString(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The sRGB hex value.
     * @returns {number}
     */
    getHex() {
        const ret = wasm.color_getHex(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * A CSS string such as `rgb(255,136,0)`.
     * @returns {string}
     */
    getStyle() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.color_getStyle(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * @param {Color} a
     * @param {Color} b
     * @param {number} alpha
     */
    lerpColors(a, b, alpha) {
        _assertClass(a, Color);
        _assertClass(b, Color);
        wasm.color_lerpColors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr, alpha);
    }
    /**
     * @param {Color} color
     * @param {number} alpha
     */
    lerpHSL(color, alpha) {
        _assertClass(color, Color);
        wasm.color_lerpHSL(this.__wbg_ptr, color.__wbg_ptr, alpha);
    }
    /**
     * @param {Color} color
     * @param {number} alpha
     */
    lerp(color, alpha) {
        _assertClass(color, Color);
        wasm.color_lerp(this.__wbg_ptr, color.__wbg_ptr, alpha);
    }
    /**
     * @param {number} s
     */
    multiplyScalar(s) {
        wasm.color_multiplyScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Color} color
     */
    multiply(color) {
        _assertClass(color, Color);
        wasm.color_multiply(this.__wbg_ptr, color.__wbg_ptr);
    }
    /**
     * Components in the linear working colour space.
     * @param {number} r
     * @param {number} g
     * @param {number} b
     */
    constructor(r, g, b) {
        const ret = wasm.color_new(r, g, b);
        this.__wbg_ptr = ret;
        ColorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} h
     * @param {number} s
     * @param {number} l
     */
    offsetHSL(h, s, l) {
        wasm.color_offsetHSL(this.__wbg_ptr, h, s, l);
    }
    /**
     * @returns {number}
     */
    get r() {
        const ret = wasm.color_r(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector3} v
     */
    setFromVector3(v) {
        _assertClass(v, Vector3);
        wasm.color_setFromVector3(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * Hue, saturation and lightness, each 0 to 1.
     * @param {number} h
     * @param {number} s
     * @param {number} l
     */
    setHSL(h, s, l) {
        wasm.color_setHSL(this.__wbg_ptr, h, s, l);
    }
    /**
     * Sets from an sRGB hex value.
     * @param {number} hex
     */
    setHex(hex) {
        wasm.color_setHex(this.__wbg_ptr, hex);
    }
    /**
     * Sets linear components.
     * @param {number} r
     * @param {number} g
     * @param {number} b
     */
    setRGB(r, g, b) {
        wasm.color_setRGB(this.__wbg_ptr, r, g, b);
    }
    /**
     * @param {number} scalar
     */
    setScalar(scalar) {
        wasm.color_setScalar(this.__wbg_ptr, scalar);
    }
    /**
     * Sets from a CSS colour string. Strings that do not parse leave the colour unchanged.
     * @param {string} style
     */
    setStyle(style) {
        const ptr0 = passStringToWasm0(style, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.color_setStyle(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * @param {number} value
     */
    set b(value) {
        wasm.color_set_b(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set g(value) {
        wasm.color_set_g(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set r(value) {
        wasm.color_set_r(this.__wbg_ptr, value);
    }
    /**
     * @param {Color} color
     */
    sub(color) {
        _assertClass(color, Color);
        wasm.color_sub(this.__wbg_ptr, color.__wbg_ptr);
    }
    /**
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.color_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
}
if (Symbol.dispose) Color.prototype[Symbol.dispose] = Color.prototype.free;

export class Cylindrical {
    static __wrap(ptr) {
        const obj = Object.create(Cylindrical.prototype);
        obj.__wbg_ptr = ptr;
        CylindricalFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        CylindricalFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_cylindrical_free(ptr, 0);
    }
    /**
     * @returns {Cylindrical}
     */
    clone() {
        const ret = wasm.cylindrical_clone(this.__wbg_ptr);
        return Cylindrical.__wrap(ret);
    }
    /**
     * @param {Cylindrical} source
     */
    copy(source) {
        _assertClass(source, Cylindrical);
        wasm.cylindrical_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * Distance from the Y axis, angle around it in radians, height.
     * @param {number} radius
     * @param {number} theta
     * @param {number} y
     */
    constructor(radius, theta, y) {
        const ret = wasm.cylindrical_new(radius, theta, y);
        this.__wbg_ptr = ret;
        CylindricalFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get radius() {
        const ret = wasm.cylindrical_radius(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    setFromCartesianCoords(x, y, z) {
        wasm.cylindrical_setFromCartesianCoords(this.__wbg_ptr, x, y, z);
    }
    /**
     * @param {Vector3} v
     */
    setFromVector3(v) {
        _assertClass(v, Vector3);
        wasm.cylindrical_setFromVector3(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {number} radius
     * @param {number} theta
     * @param {number} y
     */
    set(radius, theta, y) {
        wasm.cylindrical_set(this.__wbg_ptr, radius, theta, y);
    }
    /**
     * @param {number} value
     */
    set radius(value) {
        wasm.cylindrical_set_radius(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set theta(value) {
        wasm.cylindrical_set_theta(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set y(value) {
        wasm.cylindrical_set_y(this.__wbg_ptr, value);
    }
    /**
     * @returns {number}
     */
    get theta() {
        const ret = wasm.cylindrical_theta(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.cylindrical_y(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Cylindrical.prototype[Symbol.dispose] = Cylindrical.prototype.free;

export class Euler {
    static __wrap(ptr) {
        const obj = Object.create(Euler.prototype);
        obj.__wbg_ptr = ptr;
        EulerFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EulerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_euler_free(ptr, 0);
    }
    /**
     * @returns {Euler}
     */
    clone() {
        const ret = wasm.euler_clone(this.__wbg_ptr);
        return Euler.__wrap(ret);
    }
    /**
     * @param {Euler} source
     */
    copy(source) {
        _assertClass(source, Euler);
        wasm.euler_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Euler} e
     * @returns {boolean}
     */
    equals(e) {
        _assertClass(e, Euler);
        const ret = wasm.euler_equals(this.__wbg_ptr, e.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * `order` is one of "XYZ", "YXZ", "ZXY", "ZYX", "YZX", "XZY".
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {string} order
     */
    constructor(x, y, z, order) {
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.euler_new(x, y, z, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        EulerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {string}
     */
    get order() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.euler_order(this.__wbg_ptr);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Re-expresses the same rotation in another order.
     * @param {string} order
     */
    reorder(order) {
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.euler_reorder(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {Quaternion} q
     * @param {string} order
     */
    setFromQuaternion(q, order) {
        _assertClass(q, Quaternion);
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.euler_setFromQuaternion(this.__wbg_ptr, q.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {Matrix4} m
     * @param {string} order
     */
    setFromRotationMatrix(m, order) {
        _assertClass(m, Matrix4);
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.euler_setFromRotationMatrix(this.__wbg_ptr, m.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Changes the order label only; the angles are kept. Use `reorder` to keep the rotation.
     * @param {string} order
     */
    setOrder(order) {
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.euler_setOrder(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {string} order
     */
    set(x, y, z, order) {
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.euler_set(this.__wbg_ptr, x, y, z, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {number} value
     */
    set x(value) {
        wasm.euler_set_x(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set y(value) {
        wasm.euler_set_y(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set z(value) {
        wasm.euler_set_z(this.__wbg_ptr, value);
    }
    /**
     * The three angles.
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.euler_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.euler_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.euler_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get z() {
        const ret = wasm.euler_z(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Euler.prototype[Symbol.dispose] = Euler.prototype.free;

export class Frustum {
    static __wrap(ptr) {
        const obj = Object.create(Frustum.prototype);
        obj.__wbg_ptr = ptr;
        FrustumFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FrustumFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_frustum_free(ptr, 0);
    }
    /**
     * @returns {Frustum}
     */
    clone() {
        const ret = wasm.frustum_clone(this.__wbg_ptr);
        return Frustum.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @returns {boolean}
     */
    containsPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.frustum_containsPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Frustum} source
     */
    copy(source) {
        _assertClass(source, Frustum);
        wasm.frustum_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Box3} other
     * @returns {boolean}
     */
    intersectsBox(other) {
        _assertClass(other, Box3);
        const ret = wasm.frustum_intersectsBox(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Sphere} sphere
     * @returns {boolean}
     */
    intersectsSphere(sphere) {
        _assertClass(sphere, Sphere);
        const ret = wasm.frustum_intersectsSphere(this.__wbg_ptr, sphere.__wbg_ptr);
        return ret !== 0;
    }
    constructor() {
        const ret = wasm.frustum_new();
        this.__wbg_ptr = ret;
        FrustumFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * `m` is usually projection * view. `webgpu` selects clip-space depth 0..1.
     * @param {Matrix4} m
     * @param {boolean} webgpu
     * @param {boolean} reversed_depth
     */
    setFromProjectionMatrix(m, webgpu, reversed_depth) {
        _assertClass(m, Matrix4);
        wasm.frustum_setFromProjectionMatrix(this.__wbg_ptr, m.__wbg_ptr, webgpu, reversed_depth);
    }
}
if (Symbol.dispose) Frustum.prototype[Symbol.dispose] = Frustum.prototype.free;

export class Line3 {
    static __wrap(ptr) {
        const obj = Object.create(Line3.prototype);
        obj.__wbg_ptr = ptr;
        Line3Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Line3Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_line3_free(ptr, 0);
    }
    /**
     * @param {Matrix4} m
     */
    applyMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.line3_applyMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {number} t
     * @returns {Vector3}
     */
    at(t) {
        const ret = wasm.line3_at(this.__wbg_ptr, t);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Line3}
     */
    clone() {
        const ret = wasm.line3_clone(this.__wbg_ptr);
        return Line3.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @param {boolean} clamp_to_line
     * @returns {number}
     */
    closestPointToPointParameter(point, clamp_to_line) {
        _assertClass(point, Vector3);
        const ret = wasm.line3_closestPointToPointParameter(this.__wbg_ptr, point.__wbg_ptr, clamp_to_line);
        return ret;
    }
    /**
     * @param {Vector3} point
     * @param {boolean} clamp_to_line
     * @returns {Vector3}
     */
    closestPointToPoint(point, clamp_to_line) {
        _assertClass(point, Vector3);
        const ret = wasm.line3_closestPointToPoint(this.__wbg_ptr, point.__wbg_ptr, clamp_to_line);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Line3} source
     */
    copy(source) {
        _assertClass(source, Line3);
        wasm.line3_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @returns {Vector3}
     */
    delta() {
        const ret = wasm.line3_delta(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * Squared distance between the closest points of two segments.
     * @param {Line3} line
     * @returns {number}
     */
    distanceSqToLine3(line) {
        _assertClass(line, Line3);
        const ret = wasm.line3_distanceSqToLine3(this.__wbg_ptr, line.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    distanceSq() {
        const ret = wasm.line3_distanceSq(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    distance() {
        const ret = wasm.line3_distance(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Vector3}
     */
    get end() {
        const ret = wasm.line3_end(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Line3} line
     * @returns {boolean}
     */
    equals(line) {
        _assertClass(line, Line3);
        const ret = wasm.line3_equals(this.__wbg_ptr, line.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {Vector3}
     */
    getCenter() {
        const ret = wasm.line3_getCenter(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Vector3} start
     * @param {Vector3} end
     */
    constructor(start, end) {
        _assertClass(start, Vector3);
        _assertClass(end, Vector3);
        const ret = wasm.line3_new(start.__wbg_ptr, end.__wbg_ptr);
        this.__wbg_ptr = ret;
        Line3Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Vector3} start
     * @param {Vector3} end
     */
    set(start, end) {
        _assertClass(start, Vector3);
        _assertClass(end, Vector3);
        wasm.line3_set(this.__wbg_ptr, start.__wbg_ptr, end.__wbg_ptr);
    }
    /**
     * @returns {Vector3}
     */
    get start() {
        const ret = wasm.line3_start(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
}
if (Symbol.dispose) Line3.prototype[Symbol.dispose] = Line3.prototype.free;

export class Matrix3 {
    static __wrap(ptr) {
        const obj = Object.create(Matrix3.prototype);
        obj.__wbg_ptr = ptr;
        Matrix3Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Matrix3Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_matrix3_free(ptr, 0);
    }
    /**
     * @returns {Matrix3}
     */
    clone() {
        const ret = wasm.matrix3_clone(this.__wbg_ptr);
        return Matrix3.__wrap(ret);
    }
    /**
     * @param {Matrix3} source
     */
    copy(source) {
        _assertClass(source, Matrix3);
        wasm.matrix3_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    determinant() {
        const ret = wasm.matrix3_determinant(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Matrix3} m
     * @returns {boolean}
     */
    equals(m) {
        _assertClass(m, Matrix3);
        const ret = wasm.matrix3_equals(this.__wbg_ptr, m.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Sets the 9 elements, column-major.
     * @param {Float32Array} elements
     */
    fromArray(elements) {
        const ptr0 = passArrayF32ToWasm0(elements, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.matrix3_fromArray(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {Matrix4} m
     */
    getNormalMatrix(m) {
        _assertClass(m, Matrix4);
        wasm.matrix3_getNormalMatrix(this.__wbg_ptr, m.__wbg_ptr);
    }
    identity() {
        wasm.matrix3_identity(this.__wbg_ptr);
    }
    invert() {
        wasm.matrix3_invert(this.__wbg_ptr);
    }
    /**
     * @param {number} theta
     */
    makeRotation(theta) {
        wasm.matrix3_makeRotation(this.__wbg_ptr, theta);
    }
    /**
     * @param {number} x
     * @param {number} y
     */
    makeScale(x, y) {
        wasm.matrix3_makeScale(this.__wbg_ptr, x, y);
    }
    /**
     * @param {number} x
     * @param {number} y
     */
    makeTranslation(x, y) {
        wasm.matrix3_makeTranslation(this.__wbg_ptr, x, y);
    }
    /**
     * @param {Matrix3} a
     * @param {Matrix3} b
     */
    multiplyMatrices(a, b) {
        _assertClass(a, Matrix3);
        _assertClass(b, Matrix3);
        wasm.matrix3_multiplyMatrices(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {number} s
     */
    multiplyScalar(s) {
        wasm.matrix3_multiplyScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Matrix3} m
     */
    multiply(m) {
        _assertClass(m, Matrix3);
        wasm.matrix3_multiply(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * The identity matrix.
     */
    constructor() {
        const ret = wasm.matrix3_new();
        this.__wbg_ptr = ret;
        Matrix3Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Matrix3} m
     */
    premultiply(m) {
        _assertClass(m, Matrix3);
        wasm.matrix3_premultiply(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    setFromMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.matrix3_setFromMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {number} tx
     * @param {number} ty
     * @param {number} sx
     * @param {number} sy
     * @param {number} rotation
     * @param {number} cx
     * @param {number} cy
     */
    setUvTransform(tx, ty, sx, sy, rotation, cx, cy) {
        wasm.matrix3_setUvTransform(this.__wbg_ptr, tx, ty, sx, sy, rotation, cx, cy);
    }
    /**
     * The 9 elements, column-major.
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.matrix3_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    transpose() {
        wasm.matrix3_transpose(this.__wbg_ptr);
    }
}
if (Symbol.dispose) Matrix3.prototype[Symbol.dispose] = Matrix3.prototype.free;

export class Matrix4 {
    static __wrap(ptr) {
        const obj = Object.create(Matrix4.prototype);
        obj.__wbg_ptr = ptr;
        Matrix4Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Matrix4Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_matrix4_free(ptr, 0);
    }
    /**
     * @returns {Matrix4}
     */
    clone() {
        const ret = wasm.matrix4_clone(this.__wbg_ptr);
        return Matrix4.__wrap(ret);
    }
    /**
     * @param {Vector3} position
     * @param {Quaternion} quaternion
     * @param {Vector3} scale
     */
    compose(position, quaternion, scale) {
        _assertClass(position, Vector3);
        _assertClass(quaternion, Quaternion);
        _assertClass(scale, Vector3);
        wasm.matrix4_compose(this.__wbg_ptr, position.__wbg_ptr, quaternion.__wbg_ptr, scale.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    copyPosition(m) {
        _assertClass(m, Matrix4);
        wasm.matrix4_copyPosition(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    copy(m) {
        _assertClass(m, Matrix4);
        wasm.matrix4_copy(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * Returns ten numbers: position xyz, quaternion xyzw, scale xyz.
     * @returns {Float32Array}
     */
    decompose() {
        const ret = wasm.matrix4_decompose(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    determinantAffine() {
        const ret = wasm.matrix4_determinantAffine(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    determinant() {
        const ret = wasm.matrix4_determinant(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Matrix4} m
     * @returns {boolean}
     */
    equals(m) {
        _assertClass(m, Matrix4);
        const ret = wasm.matrix4_equals(this.__wbg_ptr, m.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Matrix4} m
     */
    extractRotation(m) {
        _assertClass(m, Matrix4);
        wasm.matrix4_extractRotation(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * Sets the 16 elements, column-major.
     * @param {Float32Array} elements
     */
    fromArray(elements) {
        const ptr0 = passArrayF32ToWasm0(elements, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.matrix4_fromArray(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @returns {number}
     */
    getMaxScaleOnAxis() {
        const ret = wasm.matrix4_getMaxScaleOnAxis(this.__wbg_ptr);
        return ret;
    }
    identity() {
        wasm.matrix4_identity(this.__wbg_ptr);
    }
    /**
     * A singular matrix becomes the zero matrix.
     */
    invert() {
        wasm.matrix4_invert(this.__wbg_ptr);
    }
    /**
     * Sets the rotation part so that -Z points from `eye` to `target`.
     * @param {Vector3} eye
     * @param {Vector3} target
     * @param {Vector3} up
     */
    lookAt(eye, target, up) {
        _assertClass(eye, Vector3);
        _assertClass(target, Vector3);
        _assertClass(up, Vector3);
        wasm.matrix4_lookAt(this.__wbg_ptr, eye.__wbg_ptr, target.__wbg_ptr, up.__wbg_ptr);
    }
    /**
     * @param {Vector3} x_axis
     * @param {Vector3} y_axis
     * @param {Vector3} z_axis
     */
    makeBasis(x_axis, y_axis, z_axis) {
        _assertClass(x_axis, Vector3);
        _assertClass(y_axis, Vector3);
        _assertClass(z_axis, Vector3);
        wasm.matrix4_makeBasis(this.__wbg_ptr, x_axis.__wbg_ptr, y_axis.__wbg_ptr, z_axis.__wbg_ptr);
    }
    /**
     * `webgpu` selects clip-space depth 0..1 instead of -1..1.
     * @param {number} left
     * @param {number} right
     * @param {number} top
     * @param {number} bottom
     * @param {number} near
     * @param {number} far
     * @param {boolean} webgpu
     * @param {boolean} reversed_depth
     */
    makeOrthographic(left, right, top, bottom, near, far, webgpu, reversed_depth) {
        wasm.matrix4_makeOrthographic(this.__wbg_ptr, left, right, top, bottom, near, far, webgpu, reversed_depth);
    }
    /**
     * `webgpu` selects clip-space depth 0..1 instead of -1..1.
     * @param {number} left
     * @param {number} right
     * @param {number} top
     * @param {number} bottom
     * @param {number} near
     * @param {number} far
     * @param {boolean} webgpu
     * @param {boolean} reversed_depth
     */
    makePerspective(left, right, top, bottom, near, far, webgpu, reversed_depth) {
        wasm.matrix4_makePerspective(this.__wbg_ptr, left, right, top, bottom, near, far, webgpu, reversed_depth);
    }
    /**
     * @param {Vector3} axis
     * @param {number} angle
     */
    makeRotationAxis(axis, angle) {
        _assertClass(axis, Vector3);
        wasm.matrix4_makeRotationAxis(this.__wbg_ptr, axis.__wbg_ptr, angle);
    }
    /**
     * Sets the rotation part from Euler angles. `order` as in `Quaternion.setFromEuler`.
     * @param {Euler} euler
     */
    makeRotationFromEuler(euler) {
        _assertClass(euler, Euler);
        wasm.matrix4_makeRotationFromEuler(this.__wbg_ptr, euler.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     */
    makeRotationFromQuaternion(q) {
        _assertClass(q, Quaternion);
        wasm.matrix4_makeRotationFromQuaternion(this.__wbg_ptr, q.__wbg_ptr);
    }
    /**
     * @param {number} theta
     */
    makeRotationX(theta) {
        wasm.matrix4_makeRotationX(this.__wbg_ptr, theta);
    }
    /**
     * @param {number} theta
     */
    makeRotationY(theta) {
        wasm.matrix4_makeRotationY(this.__wbg_ptr, theta);
    }
    /**
     * @param {number} theta
     */
    makeRotationZ(theta) {
        wasm.matrix4_makeRotationZ(this.__wbg_ptr, theta);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    makeScale(x, y, z) {
        wasm.matrix4_makeScale(this.__wbg_ptr, x, y, z);
    }
    /**
     * @param {number} xy
     * @param {number} xz
     * @param {number} yx
     * @param {number} yz
     * @param {number} zx
     * @param {number} zy
     */
    makeShear(xy, xz, yx, yz, zx, zy) {
        wasm.matrix4_makeShear(this.__wbg_ptr, xy, xz, yx, yz, zx, zy);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    makeTranslation(x, y, z) {
        wasm.matrix4_makeTranslation(this.__wbg_ptr, x, y, z);
    }
    /**
     * @param {Matrix4} a
     * @param {Matrix4} b
     */
    multiplyMatrices(a, b) {
        _assertClass(a, Matrix4);
        _assertClass(b, Matrix4);
        wasm.matrix4_multiplyMatrices(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {number} s
     */
    multiplyScalar(s) {
        wasm.matrix4_multiplyScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Matrix4} m
     */
    multiply(m) {
        _assertClass(m, Matrix4);
        wasm.matrix4_multiply(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * The identity matrix.
     */
    constructor() {
        const ret = wasm.matrix4_new();
        this.__wbg_ptr = ret;
        Matrix4Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Matrix4} m
     */
    premultiply(m) {
        _assertClass(m, Matrix4);
        wasm.matrix4_premultiply(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     */
    scale(v) {
        _assertClass(v, Vector3);
        wasm.matrix4_scale(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Matrix3} m
     */
    setFromMatrix3(m) {
        _assertClass(m, Matrix3);
        wasm.matrix4_setFromMatrix3(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    setPosition(x, y, z) {
        wasm.matrix4_setPosition(this.__wbg_ptr, x, y, z);
    }
    /**
     * The 16 elements, column-major.
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.matrix4_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    transpose() {
        wasm.matrix4_transpose(this.__wbg_ptr);
    }
}
if (Symbol.dispose) Matrix4.prototype[Symbol.dispose] = Matrix4.prototype.free;

export class Plane {
    static __wrap(ptr) {
        const obj = Object.create(Plane.prototype);
        obj.__wbg_ptr = ptr;
        PlaneFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        PlaneFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_plane_free(ptr, 0);
    }
    /**
     * Transforms the plane by an affine matrix.
     * @param {Matrix4} m
     */
    applyMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.plane_applyMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @returns {Plane}
     */
    clone() {
        const ret = wasm.plane_clone(this.__wbg_ptr);
        return Plane.__wrap(ret);
    }
    /**
     * @returns {number}
     */
    get constant() {
        const ret = wasm.plane_constant(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Vector3}
     */
    coplanarPoint() {
        const ret = wasm.plane_coplanarPoint(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Plane} source
     */
    copy(source) {
        _assertClass(source, Plane);
        wasm.plane_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Vector3} point
     * @returns {number}
     */
    distanceToPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.plane_distanceToPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Sphere} sphere
     * @returns {number}
     */
    distanceToSphere(sphere) {
        _assertClass(sphere, Sphere);
        const ret = wasm.plane_distanceToSphere(this.__wbg_ptr, sphere.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Plane} other
     * @returns {boolean}
     */
    equals(other) {
        _assertClass(other, Plane);
        const ret = wasm.plane_equals(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Box3} other
     * @returns {boolean}
     */
    intersectsBox(other) {
        _assertClass(other, Box3);
        const ret = wasm.plane_intersectsBox(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Sphere} sphere
     * @returns {boolean}
     */
    intersectsSphere(sphere) {
        _assertClass(sphere, Sphere);
        const ret = wasm.plane_intersectsSphere(this.__wbg_ptr, sphere.__wbg_ptr);
        return ret !== 0;
    }
    negate() {
        wasm.plane_negate(this.__wbg_ptr);
    }
    /**
     * The plane `normal . p + constant = 0`.
     * @param {Vector3} normal
     * @param {number} constant
     */
    constructor(normal, constant) {
        _assertClass(normal, Vector3);
        const ret = wasm.plane_new(normal.__wbg_ptr, constant);
        this.__wbg_ptr = ret;
        PlaneFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Vector3}
     */
    get normal() {
        const ret = wasm.plane_normal(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    normalize() {
        wasm.plane_normalize(this.__wbg_ptr);
    }
    /**
     * @param {Vector3} point
     * @returns {Vector3}
     */
    projectPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.plane_projectPoint(this.__wbg_ptr, point.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    setComponents(x, y, z, w) {
        wasm.plane_setComponents(this.__wbg_ptr, x, y, z, w);
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     * @param {Vector3} c
     */
    setFromCoplanarPoints(a, b, c) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        _assertClass(c, Vector3);
        wasm.plane_setFromCoplanarPoints(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr, c.__wbg_ptr);
    }
    /**
     * @param {Vector3} normal
     * @param {Vector3} point
     */
    setFromNormalAndCoplanarPoint(normal, point) {
        _assertClass(normal, Vector3);
        _assertClass(point, Vector3);
        wasm.plane_setFromNormalAndCoplanarPoint(this.__wbg_ptr, normal.__wbg_ptr, point.__wbg_ptr);
    }
    /**
     * @param {number} constant
     */
    set constant(constant) {
        wasm.plane_set_constant(this.__wbg_ptr, constant);
    }
    /**
     * @param {Vector3} offset
     */
    translate(offset) {
        _assertClass(offset, Vector3);
        wasm.plane_translate(this.__wbg_ptr, offset.__wbg_ptr);
    }
}
if (Symbol.dispose) Plane.prototype[Symbol.dispose] = Plane.prototype.free;

export class Quaternion {
    static __wrap(ptr) {
        const obj = Object.create(Quaternion.prototype);
        obj.__wbg_ptr = ptr;
        QuaternionFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        QuaternionFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_quaternion_free(ptr, 0);
    }
    /**
     * @param {Quaternion} q
     * @returns {number}
     */
    angleTo(q) {
        _assertClass(q, Quaternion);
        const ret = wasm.quaternion_angleTo(this.__wbg_ptr, q.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {Quaternion}
     */
    clone() {
        const ret = wasm.quaternion_clone(this.__wbg_ptr);
        return Quaternion.__wrap(ret);
    }
    conjugate() {
        wasm.quaternion_conjugate(this.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     */
    copy(q) {
        _assertClass(q, Quaternion);
        wasm.quaternion_copy(this.__wbg_ptr, q.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     * @returns {number}
     */
    dot(q) {
        _assertClass(q, Quaternion);
        const ret = wasm.quaternion_dot(this.__wbg_ptr, q.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Quaternion} q
     * @returns {boolean}
     */
    equals(q) {
        _assertClass(q, Quaternion);
        const ret = wasm.quaternion_equals(this.__wbg_ptr, q.__wbg_ptr);
        return ret !== 0;
    }
    identity() {
        wasm.quaternion_identity(this.__wbg_ptr);
    }
    invert() {
        wasm.quaternion_invert(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    lengthSq() {
        const ret = wasm.quaternion_lengthSq(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    length() {
        const ret = wasm.quaternion_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Quaternion} a
     * @param {Quaternion} b
     */
    multiplyQuaternions(a, b) {
        _assertClass(a, Quaternion);
        _assertClass(b, Quaternion);
        wasm.quaternion_multiplyQuaternions(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     */
    multiply(q) {
        _assertClass(q, Quaternion);
        wasm.quaternion_multiply(this.__wbg_ptr, q.__wbg_ptr);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    constructor(x, y, z, w) {
        const ret = wasm.quaternion_new(x, y, z, w);
        this.__wbg_ptr = ret;
        QuaternionFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    normalize() {
        wasm.quaternion_normalize(this.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     */
    premultiply(q) {
        _assertClass(q, Quaternion);
        wasm.quaternion_premultiply(this.__wbg_ptr, q.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     * @param {number} step
     */
    rotateTowards(q, step) {
        _assertClass(q, Quaternion);
        wasm.quaternion_rotateTowards(this.__wbg_ptr, q.__wbg_ptr, step);
    }
    /**
     * `axis` must be normalized.
     * @param {Vector3} axis
     * @param {number} angle
     */
    setFromAxisAngle(axis, angle) {
        _assertClass(axis, Vector3);
        wasm.quaternion_setFromAxisAngle(this.__wbg_ptr, axis.__wbg_ptr, angle);
    }
    /**
     * `order` is one of "XYZ", "YXZ", "ZXY", "ZYX", "YZX", "XZY".
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {string} order
     */
    setFromEuler(x, y, z, order) {
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.quaternion_setFromEuler(this.__wbg_ptr, x, y, z, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {Matrix4} m
     */
    setFromRotationMatrix(m) {
        _assertClass(m, Matrix4);
        wasm.quaternion_setFromRotationMatrix(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Vector3} from
     * @param {Vector3} to
     */
    setFromUnitVectors(from, to) {
        _assertClass(from, Vector3);
        _assertClass(to, Vector3);
        wasm.quaternion_setFromUnitVectors(this.__wbg_ptr, from.__wbg_ptr, to.__wbg_ptr);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    set(x, y, z, w) {
        wasm.quaternion_set(this.__wbg_ptr, x, y, z, w);
    }
    /**
     * @param {number} value
     */
    set w(value) {
        wasm.quaternion_set_w(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set x(value) {
        wasm.quaternion_set_x(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set y(value) {
        wasm.quaternion_set_y(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set z(value) {
        wasm.quaternion_set_z(this.__wbg_ptr, value);
    }
    /**
     * @param {Quaternion} a
     * @param {Quaternion} b
     * @param {number} t
     */
    slerpQuaternions(a, b, t) {
        _assertClass(a, Quaternion);
        _assertClass(b, Quaternion);
        wasm.quaternion_slerpQuaternions(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr, t);
    }
    /**
     * @param {Quaternion} q
     * @param {number} t
     */
    slerp(q, t) {
        _assertClass(q, Quaternion);
        wasm.quaternion_slerp(this.__wbg_ptr, q.__wbg_ptr, t);
    }
    /**
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.quaternion_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get w() {
        const ret = wasm.quaternion_w(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.quaternion_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.quaternion_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get z() {
        const ret = wasm.quaternion_z(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Quaternion.prototype[Symbol.dispose] = Quaternion.prototype.free;

export class Ray {
    static __wrap(ptr) {
        const obj = Object.create(Ray.prototype);
        obj.__wbg_ptr = ptr;
        RayFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RayFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_ray_free(ptr, 0);
    }
    /**
     * @param {Matrix4} m
     */
    applyMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.ray_applyMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {number} t
     * @returns {Vector3}
     */
    at(t) {
        const ret = wasm.ray_at(this.__wbg_ptr, t);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Ray}
     */
    clone() {
        const ret = wasm.ray_clone(this.__wbg_ptr);
        return Ray.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @returns {Vector3}
     */
    closestPointToPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.ray_closestPointToPoint(this.__wbg_ptr, point.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Ray} source
     */
    copy(source) {
        _assertClass(source, Ray);
        wasm.ray_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @returns {Vector3}
     */
    get direction() {
        const ret = wasm.ray_direction(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @returns {number}
     */
    distanceSqToPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.ray_distanceSqToPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret;
    }
    /**
     * The distance along the ray to the plane, or `undefined`.
     * @param {Plane} plane
     * @returns {number | undefined}
     */
    distanceToPlane(plane) {
        _assertClass(plane, Plane);
        const ret = wasm.ray_distanceToPlane(this.__wbg_ptr, plane.__wbg_ptr);
        return ret === Number.MAX_SAFE_INTEGER ? undefined : ret;
    }
    /**
     * @param {Vector3} point
     * @returns {number}
     */
    distanceToPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.ray_distanceToPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Ray} other
     * @returns {boolean}
     */
    equals(other) {
        _assertClass(other, Ray);
        const ret = wasm.ray_equals(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * The hit point, or `undefined`.
     * @param {Box3} other
     * @returns {Vector3 | undefined}
     */
    intersectBox(other) {
        _assertClass(other, Box3);
        const ret = wasm.ray_intersectBox(this.__wbg_ptr, other.__wbg_ptr);
        return ret === 0 ? undefined : Vector3.__wrap(ret);
    }
    /**
     * The hit point, or `undefined`.
     * @param {Plane} plane
     * @returns {Vector3 | undefined}
     */
    intersectPlane(plane) {
        _assertClass(plane, Plane);
        const ret = wasm.ray_intersectPlane(this.__wbg_ptr, plane.__wbg_ptr);
        return ret === 0 ? undefined : Vector3.__wrap(ret);
    }
    /**
     * The hit point, or `undefined`.
     * @param {Sphere} sphere
     * @returns {Vector3 | undefined}
     */
    intersectSphere(sphere) {
        _assertClass(sphere, Sphere);
        const ret = wasm.ray_intersectSphere(this.__wbg_ptr, sphere.__wbg_ptr);
        return ret === 0 ? undefined : Vector3.__wrap(ret);
    }
    /**
     * The hit point on triangle `a`, `b`, `c`, or `undefined`.
     * @param {Vector3} a
     * @param {Vector3} b
     * @param {Vector3} c
     * @param {boolean} backface_culling
     * @returns {Vector3 | undefined}
     */
    intersectTriangle(a, b, c, backface_culling) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        _assertClass(c, Vector3);
        const ret = wasm.ray_intersectTriangle(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr, c.__wbg_ptr, backface_culling);
        return ret === 0 ? undefined : Vector3.__wrap(ret);
    }
    /**
     * @param {Box3} other
     * @returns {boolean}
     */
    intersectsBox(other) {
        _assertClass(other, Box3);
        const ret = wasm.ray_intersectsBox(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Plane} plane
     * @returns {boolean}
     */
    intersectsPlane(plane) {
        _assertClass(plane, Plane);
        const ret = wasm.ray_intersectsPlane(this.__wbg_ptr, plane.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Sphere} sphere
     * @returns {boolean}
     */
    intersectsSphere(sphere) {
        _assertClass(sphere, Sphere);
        const ret = wasm.ray_intersectsSphere(this.__wbg_ptr, sphere.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Vector3} v
     */
    lookAt(v) {
        _assertClass(v, Vector3);
        wasm.ray_lookAt(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * `direction` must be normalized.
     * @param {Vector3} origin
     * @param {Vector3} direction
     */
    constructor(origin, direction) {
        _assertClass(origin, Vector3);
        _assertClass(direction, Vector3);
        const ret = wasm.ray_new(origin.__wbg_ptr, direction.__wbg_ptr);
        this.__wbg_ptr = ret;
        RayFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {Vector3}
     */
    get origin() {
        const ret = wasm.ray_origin(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {number} t
     */
    recast(t) {
        wasm.ray_recast(this.__wbg_ptr, t);
    }
    /**
     * @param {Vector3} origin
     * @param {Vector3} direction
     */
    set(origin, direction) {
        _assertClass(origin, Vector3);
        _assertClass(direction, Vector3);
        wasm.ray_set(this.__wbg_ptr, origin.__wbg_ptr, direction.__wbg_ptr);
    }
}
if (Symbol.dispose) Ray.prototype[Symbol.dispose] = Ray.prototype.free;

/**
 * The result of a raycast: hits sorted nearest first. Index 0 is the closest.
 */
export class RaycastHits {
    static __wrap(ptr) {
        const obj = Object.create(RaycastHits.prototype);
        obj.__wbg_ptr = ptr;
        RaycastHitsFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RaycastHitsFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_raycasthits_free(ptr, 0);
    }
    /**
     * @param {number} index
     * @returns {Float32Array}
     */
    barycoord(index) {
        const ret = wasm.raycasthits_barycoord(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Distance from the ray origin, in world units.
     * @param {number} index
     * @returns {number}
     */
    distance(index) {
        const ret = wasm.raycasthits_distance(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Index of the triangle within the geometry.
     * @param {number} index
     * @returns {number}
     */
    faceIndex(index) {
        const ret = wasm.raycasthits_faceIndex(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * The triangle normal in the local space of the object.
     * @param {number} index
     * @returns {Float32Array}
     */
    faceNormal(index) {
        const ret = wasm.raycasthits_faceNormal(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The three vertex indices of the triangle.
     * @param {number} index
     * @returns {Uint32Array}
     */
    faceVertices(index) {
        const ret = wasm.raycasthits_faceVertices(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The instance hit, for results of `raycastInstances`; otherwise `undefined`.
     * @param {number} index
     * @returns {number | undefined}
     */
    instanceId(index) {
        const ret = wasm.raycasthits_instanceId(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] === Number.MAX_SAFE_INTEGER ? undefined : ret[0];
    }
    /**
     * @returns {number}
     */
    get length() {
        const ret = wasm.raycasthits_length(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * For a hit on a line, where the segment starts; `undefined` for a mesh.
     * @param {number} index
     * @returns {number | undefined}
     */
    lineIndex(index) {
        const ret = wasm.raycasthits_lineIndex(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] === Number.MAX_SAFE_INTEGER ? undefined : ret[0];
    }
    /**
     * Interpolated vertex normal facing the ray, or `undefined` if the geometry has none.
     * @param {number} index
     * @returns {Float32Array | undefined}
     */
    normal(index) {
        const ret = wasm.raycasthits_normal(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
    /**
     * The object that was hit.
     * @param {number} index
     * @returns {bigint}
     */
    object(index) {
        const ret = wasm.raycasthits_object(this.__wbg_ptr, index);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return BigInt.asUintN(64, ret[0]);
    }
    /**
     * The hit point in world space.
     * @param {number} index
     * @returns {Float32Array}
     */
    point(index) {
        const ret = wasm.raycasthits_point(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Interpolated texture coordinate, or `undefined` if the geometry has none.
     * @param {number} index
     * @returns {Float32Array | undefined}
     */
    uv(index) {
        const ret = wasm.raycasthits_uv(this.__wbg_ptr, index);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        let v1;
        if (ret[0] !== 0) {
            v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
            wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        }
        return v1;
    }
}
if (Symbol.dispose) RaycastHits.prototype[Symbol.dispose] = RaycastHits.prototype.free;

/**
 * Prepares frames for the WebGPU host: culling, draw order, uniform data and shader.
 * The host (`webgpu.js` in the package) issues the GPU calls.
 */
export class Renderer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RendererFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_renderer_free(ptr, 0);
    }
    /**
     * Adds a directional light: direction towards the light, linear colour, intensity.
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} intensity
     */
    addDirectionalLight(x, y, z, r, g, b, intensity) {
        wasm.renderer_addDirectionalLight(this.__wbg_ptr, x, y, z, r, g, b, intensity);
    }
    /**
     * Adds a point light at a world position. `distance` 0 means no cutoff.
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} intensity
     * @param {number} distance
     * @param {number} decay
     */
    addPointLight(x, y, z, r, g, b, intensity, distance, decay) {
        wasm.renderer_addPointLight(this.__wbg_ptr, x, y, z, r, g, b, intensity, distance, decay);
    }
    /**
     * Adds a rectangular area light (three.js `RectAreaLight`): its centre, half its
     * width and half its height as world vectors (it shines along the cross product of
     * the two), linear colour and intensity.
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} wx
     * @param {number} wy
     * @param {number} wz
     * @param {number} hx
     * @param {number} hy
     * @param {number} hz
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} intensity
     */
    addRectAreaLight(x, y, z, wx, wy, wz, hx, hy, hz, r, g, b, intensity) {
        wasm.renderer_addRectAreaLight(this.__wbg_ptr, x, y, z, wx, wy, wz, hx, hy, hz, r, g, b, intensity);
    }
    /**
     * Adds a spot light at a world position shining at a world target. `angle` is the
     * cone's half angle in radians; `penumbra` (0..1) is the soft share of the cone.
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} target_x
     * @param {number} target_y
     * @param {number} target_z
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} intensity
     * @param {number} distance
     * @param {number} decay
     * @param {number} angle
     * @param {number} penumbra
     */
    addSpotLight(x, y, z, target_x, target_y, target_z, r, g, b, intensity, distance, decay, angle, penumbra) {
        wasm.renderer_addSpotLight(this.__wbg_ptr, x, y, z, target_x, target_y, target_z, r, g, b, intensity, distance, decay, angle, penumbra);
    }
    /**
     * Bit of the pipeline state set when the geometry has tangents, a second uv set or
     * skinning attributes.
     * @returns {number}
     */
    static auxKey() {
        const ret = wasm.renderer_auxKey();
        return ret >>> 0;
    }
    /**
     * The background of the last prepared frame: texture id, kind, and `lodMax` of the
     * prefiltered map; empty without one.
     * @returns {Uint32Array}
     */
    backgroundInfo() {
        const ret = wasm.renderer_backgroundInfo(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * WGSL source of the background pass.
     * @returns {string}
     */
    static backgroundShaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_backgroundShaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Uniforms of the background pass of the last prepared frame (see
     * `backgroundShaderSource`); empty without a background.
     * @returns {Float32Array}
     */
    backgroundUniforms() {
        const ret = wasm.renderer_backgroundUniforms(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Matrices per row of the bone texture.
     * @returns {number}
     */
    static boneTextureMatricesPerRow() {
        const ret = wasm.renderer_boneTextureMatricesPerRow();
        return ret >>> 0;
    }
    /**
     * Objects that had something to draw before culling, in the last prepared frame.
     * @returns {number}
     */
    candidateCount() {
        const ret = wasm.renderer_candidateCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    clearBackground() {
        wasm.renderer_clearBackground(this.__wbg_ptr);
    }
    clearEnvironment() {
        wasm.renderer_clearEnvironment(this.__wbg_ptr);
    }
    clearLights() {
        wasm.renderer_clearLights(this.__wbg_ptr);
    }
    /**
     * WGSL source of the pass that turns a cube texture into an equirectangular image.
     * @returns {string}
     */
    static cubeToEquirectShaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_cubeToEquirectShaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The split-sum table the mesh pipeline reads at group 0, binding 1: 512 floats
     * (a uniform `array<vec4<f32>, 128>`), the same for every frame.
     * @returns {Float32Array}
     */
    static dfgTable() {
        const ret = wasm.renderer_dfgTable();
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Slot of the displacement map in a texture set: the one texture the vertex stage
     * samples.
     * @returns {number}
     */
    static displacementSlot() {
        const ret = wasm.renderer_displacementSlot();
        return ret >>> 0;
    }
    /**
     * The geometry handle of each draw, in draw order.
     * @returns {BigUint64Array}
     */
    drawGeometries() {
        const ret = wasm.renderer_drawGeometries(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Per draw, which record of `geometryState` describes its geometry.
     * @returns {Uint32Array}
     */
    drawGeometrySlots() {
        const ret = wasm.renderer_drawGeometrySlots(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Numbers per draw in `drawInfo` and `shadowDrawInfo`.
     * @returns {number}
     */
    static drawInfoSize() {
        const ret = wasm.renderer_drawInfoSize();
        return ret >>> 0;
    }
    /**
     * Eight numbers per draw, in draw order: pipeline state, texture set (see
     * `textureSets`), first index or vertex, count, instance data id (0 for none),
     * number of instances, material block (see `materialUniforms`) and the material's
     * custom shader (0 none, 1-6 an effect, 16 and above a host shader material). Pipeline state
     * bits: 0-1 side (0 front, 1 back, 2 both), 2 blending, 3 depth write, 4 depth test,
     * 5-6 primitive (0 triangles, 1 line strip, 2 line list, 3 points), 7 line loop, 8
     * colours of four floats, 9 wireframe (a line list over the edges of the triangles
     * from first to first + count), 10 tangent, second uv set or skinning present, 11-13
     * blending mode, 14 premultiplied alpha, 15 points or lines drawn as quads.
     * @returns {Uint32Array}
     */
    drawInfo() {
        const ret = wasm.renderer_drawInfo(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The entry point (WGSL) or name after `effect.` (GLSL) of an effect id, empty for
     * other ids.
     * @param {number} id
     * @returns {string}
     */
    static effectEntryPoint(id) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_effectEntryPoint(id);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * WGSL of the full-screen effects (see `Renderer.effectEntryPoint`).
     * @returns {string}
     */
    static effectShaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_effectShaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * The environments of the last prepared frame, stacked in one texture: the `lodMax`
     * they are all prefiltered at, the width and height of one prefiltered map, then the
     * texture id of each in stacking order; empty without any.
     * @returns {Uint32Array}
     */
    environmentInfo() {
        const ret = wasm.renderer_environmentInfo(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {Float32Array}
     */
    frameUniforms() {
        const ret = wasm.renderer_frameUniforms(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Numbers per record of `geometryState`.
     * @returns {number}
     */
    static geometryStateSize() {
        const ret = wasm.renderer_geometryStateSize();
        return ret >>> 0;
    }
    /**
     * One record per distinct geometry of the last prepared frame: slot index and
     * generation of the handle, vertex count, then for position, normal, uv and colour
     * the attribute's version and the address of its floats (0 when the shader input
     * cannot use it), then the index version and the floats per colour (3 or 4).
     * Everything the host needs to decide what to upload, in one call.
     * @returns {Uint32Array}
     */
    geometryState() {
        const ret = wasm.renderer_geometryState(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * GLSL ES 3.00 source of a pass, for the WebGL 2 host (`webgl.js`): `mesh.vert`,
     * `mesh.frag`, `shadow.vert`, `shadow.frag`, `fullscreen.vert`, `background.frag`,
     * `output.frag`, `pmrem.vert`, `pmrem.equirect`, `pmrem.ggx` or `cube.equirect`; empty
     * for others.
     * @param {string} name
     * @returns {string}
     */
    static glslSource(name) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.renderer_glslSource(ptr0, len0);
            deferred2_0 = ret[0];
            deferred2_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Bytes per material in `materialUniforms`.
     * @returns {number}
     */
    static materialUniformSize() {
        const ret = wasm.renderer_materialUniformSize();
        return ret >>> 0;
    }
    /**
     * Number of floats at `materialUniformsPtr`.
     * @returns {number}
     */
    materialUniformsLength() {
        const ret = wasm.renderer_materialUniformsLength(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Address of the material blocks of the last prepared frame inside the module's
     * memory; valid until the next `prepare` or memory growth.
     * @returns {number}
     */
    materialUniformsPtr() {
        const ret = wasm.renderer_materialUniformsPtr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Lights beyond this many are ignored.
     * @returns {number}
     */
    static maxLights() {
        const ret = wasm.renderer_maxLights();
        return ret >>> 0;
    }
    /**
     * The most shadow maps per frame, layers of the shadow texture array: one per
     * directional or spot light, six per point light.
     * @returns {number}
     */
    static maxShadows() {
        const ret = wasm.renderer_maxShadows();
        return ret >>> 0;
    }
    constructor() {
        const ret = wasm.renderer_new();
        this.__wbg_ptr = ret;
        RendererFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Bytes of each object's uniforms that the shader reads.
     * @returns {number}
     */
    static objectBindingSize() {
        const ret = wasm.renderer_objectBindingSize();
        return ret >>> 0;
    }
    /**
     * Bytes per object in `objectUniforms`.
     * @returns {number}
     */
    static objectUniformSize() {
        const ret = wasm.renderer_objectUniformSize();
        return ret >>> 0;
    }
    /**
     * Number of floats at `objectUniformsPtr`.
     * @returns {number}
     */
    objectUniformsLength() {
        const ret = wasm.renderer_objectUniformsLength(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Address of the object uniform data inside the module's memory, for a view
     * without a copy. Valid until the next `prepare` or memory growth.
     * @returns {number}
     */
    objectUniformsPtr() {
        const ret = wasm.renderer_objectUniformsPtr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {Float32Array}
     */
    objectUniforms() {
        const ret = wasm.renderer_objectUniforms(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * How many of the first draws are opaque.
     * @returns {number}
     */
    opaqueCount() {
        const ret = wasm.renderer_opaqueCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * WGSL source of the output pass that tone-maps a linear frame onto the canvas.
     * @returns {string}
     */
    static outputShaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_outputShaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * WGSL source of the passes that prefilter an environment map (three.js PMREM).
     * @returns {string}
     */
    static pmremShaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_pmremShaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Builds the frame for `camera` looking at the subtree of `root` and returns the
     * number of draws. World matrices must be up to date.
     * @param {World} world
     * @param {bigint} root
     * @param {bigint} camera
     * @returns {number}
     */
    prepare(world, root, camera) {
        _assertClass(world, World);
        const ret = wasm.renderer_prepare(this.__wbg_ptr, world.__wbg_ptr, root, camera);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Bit of the pipeline state set for points and lines drawn as quads.
     * @returns {number}
     */
    static quadKey() {
        const ret = wasm.renderer_quadKey();
        return ret >>> 0;
    }
    /**
     * Ambient light from above (sky) and below (ground), linear colours.
     * @param {number} sky_r
     * @param {number} sky_g
     * @param {number} sky_b
     * @param {number} ground_r
     * @param {number} ground_g
     * @param {number} ground_b
     */
    setAmbientLight(sky_r, sky_g, sky_b, ground_r, ground_g, ground_b) {
        wasm.renderer_setAmbientLight(this.__wbg_ptr, sky_r, sky_g, sky_b, ground_r, ground_g, ground_b);
    }
    /**
     * Draws a texture behind the scene (three.js `scene.background`): `kind` 0 for an
     * equirectangular image, 1 for the same blurred by `blurriness` (from its
     * prefiltered map), 2 for an image filling the frame. `source_width` is the image
     * width in pixels; `rotation` turns view directions (column-major 3x3, already
     * inverted).
     * @param {number} texture
     * @param {number} kind
     * @param {number} source_width
     * @param {number} intensity
     * @param {number} blurriness
     * @param {Float32Array} rotation
     */
    setBackground(texture, kind, source_width, intensity, blurriness, rotation) {
        const ptr0 = passArrayF32ToWasm0(rotation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.renderer_setBackground(this.__wbg_ptr, texture, kind, source_width, intensity, blurriness, ptr0, len0);
    }
    /**
     * The renderer's global clipping planes in world space, four numbers each (normal and
     * constant); at most eight are used.
     * @param {Float32Array} planes
     */
    setClippingPlanes(planes) {
        const ptr0 = passArrayF32ToWasm0(planes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.renderer_setClippingPlanes(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Replaces all lights with one directional light.
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} intensity
     */
    setDirectionalLight(x, y, z, r, g, b, intensity) {
        wasm.renderer_setDirectionalLight(this.__wbg_ptr, x, y, z, r, g, b, intensity);
    }
    /**
     * Lights standard materials that use it with an equirectangular environment map:
     * the host's texture id, the source image width, and the rotation of sampling
     * directions as a 3x3 matrix (column-major, already inverted).
     * @param {number} texture
     * @param {number} source_width
     * @param {Float32Array} rotation
     */
    setEnvironment(texture, source_width, rotation) {
        const ptr0 = passArrayF32ToWasm0(rotation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.renderer_setEnvironment(this.__wbg_ptr, texture, source_width, ptr0, len0);
    }
    /**
     * Fog: `kind` 0 none, 1 linear between `near` and `far`, 2 exponential squared
     * with `density`. The colour is linear.
     * @param {number} kind
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} near
     * @param {number} far
     * @param {number} density
     */
    setFog(kind, r, g, b, near, far, density) {
        wasm.renderer_setFog(this.__wbg_ptr, kind, r, g, b, near, far, density);
    }
    /**
     * The VSM blur taps of the last light's shadow (three.js `shadow.blurSamples`).
     * @param {number} samples
     */
    setLastLightShadowBlur(samples) {
        wasm.renderer_setLastLightShadowBlur(this.__wbg_ptr, samples);
    }
    /**
     * Gives the light added last a shadow. `orthographic` selects the projection:
     * `a, b, c, d` are left, right, top, bottom for an orthographic camera, or the
     * field of view in degrees and the aspect for a perspective one. The shadow camera
     * looks at the target. A point light uses only `near`, `far`, the bias, intensity,
     * radius and map width: it draws six square 90 degree faces from its position.
     * @param {boolean} orthographic
     * @param {number} a
     * @param {number} b
     * @param {number} c
     * @param {number} d
     * @param {number} near
     * @param {number} far
     * @param {number} target_x
     * @param {number} target_y
     * @param {number} target_z
     * @param {number} bias
     * @param {number} normal_bias
     * @param {number} intensity
     * @param {number} radius
     * @param {number} map_width
     * @param {number} map_height
     */
    setLastLightShadow(orthographic, a, b, c, d, near, far, target_x, target_y, target_z, bias, normal_bias, intensity, radius, map_width, map_height) {
        wasm.renderer_setLastLightShadow(this.__wbg_ptr, orthographic, a, b, c, d, near, far, target_x, target_y, target_z, bias, normal_bias, intensity, radius, map_width, map_height);
    }
    /**
     * The summed spherical harmonics of the scene's light probes: 27 numbers, nine
     * linear colours already times the intensity.
     * @param {Float32Array} coefficients
     */
    setLightProbe(coefficients) {
        const ptr0 = passArrayF32ToWasm0(coefficients, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.renderer_setLightProbe(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * three.js `Scene.overrideMaterial`: a material handle, or 0 for none.
     * @param {bigint} handle
     */
    setOverrideMaterial(handle) {
        wasm.renderer_setOverrideMaterial(this.__wbg_ptr, handle);
    }
    /**
     * three.js `renderer.shadowMap.type`: 0 `BasicShadowMap`, 3 `VSMShadowMap`; any
     * other value filters like `PCFShadowMap`.
     * @param {number} kind
     */
    setShadowType(kind) {
        wasm.renderer_setShadowType(this.__wbg_ptr, kind);
    }
    /**
     * three.js `renderer.shadowMap.enabled`.
     * @param {boolean} enabled
     */
    setShadowsEnabled(enabled) {
        wasm.renderer_setShadowsEnabled(this.__wbg_ptr, enabled);
    }
    /**
     * Tone mapping of the output, as a three.js constant (0 none, 1 linear, 2 Reinhard,
     * 3 Cineon, 4 ACES filmic, 6 AgX, 7 neutral), and its exposure.
     * @param {number} kind
     * @param {number} exposure
     */
    setToneMapping(kind, exposure) {
        wasm.renderer_setToneMapping(this.__wbg_ptr, kind, exposure);
    }
    /**
     * The size of the drawing buffer in pixels and the pixel ratio; points larger than
     * a pixel, wide lines and transmission read them. Set before `prepare`.
     * @param {number} width
     * @param {number} height
     * @param {number} pixel_ratio
     */
    setViewport(width, height, pixel_ratio) {
        wasm.renderer_setViewport(this.__wbg_ptr, width, height, pixel_ratio);
    }
    /**
     * WGSL source of the mesh pipeline.
     * @returns {string}
     */
    static shaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_shaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * For a shadow pass, its VSM blur radius in texels and taps.
     * @param {number} pass
     * @returns {Float32Array}
     */
    shadowBlur(pass) {
        const ret = wasm.renderer_shadowBlur(this.__wbg_ptr, pass);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Shadow passes of the last prepared frame.
     * @returns {number}
     */
    shadowCount() {
        const ret = wasm.renderer_shadowCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Like `drawGeometrySlots`, for the casters of a shadow pass.
     * @param {number} pass
     * @returns {Uint32Array}
     */
    shadowDrawGeometrySlots(pass) {
        const ret = wasm.renderer_shadowDrawGeometrySlots(this.__wbg_ptr, pass);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Like `drawInfo`, for the casters of a shadow pass. The texture set and material
     * are indices into the frame's lists (`textureSets`, `materialUniforms`).
     * @param {number} pass
     * @returns {Uint32Array}
     */
    shadowDrawInfo(pass) {
        const ret = wasm.renderer_shadowDrawInfo(this.__wbg_ptr, pass);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The shadow filter of the last prepared frame: 0 basic, 1 PCF, 2 VSM.
     * @returns {number}
     */
    shadowFilter() {
        const ret = wasm.renderer_shadowFilter(this.__wbg_ptr);
        return ret;
    }
    /**
     * Width and height of each layer of the shadow texture array.
     * @returns {Uint32Array}
     */
    shadowMapSize() {
        const ret = wasm.renderer_shadowMapSize(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Object uniforms of a shadow pass (only the model matrix is read), same stride as
     * `objectUniforms`.
     * @param {number} pass
     * @returns {Float32Array}
     */
    shadowObjectUniforms(pass) {
        const ret = wasm.renderer_shadowObjectUniforms(this.__wbg_ptr, pass);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Layer, width and height of a pass's map, then its view-projection (16 floats).
     * @param {number} pass
     * @returns {Float32Array}
     */
    shadowPassInfo(pass) {
        const ret = wasm.renderer_shadowPassInfo(this.__wbg_ptr, pass);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * WGSL source of the shadow pipelines: entry points `vertex` and `fragment` (only
     * for casters with an alpha test).
     * @returns {string}
     */
    static shadowShaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_shadowShaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Number of floats at `skinMatricesPtr`.
     * @returns {number}
     */
    skinMatricesLength() {
        const ret = wasm.renderer_skinMatricesLength(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Address of the frame's skinning matrices (16 floats each, 256 per row of the bone
     * texture); valid until the next `prepare` or memory growth.
     * @returns {number}
     */
    skinMatricesPtr() {
        const ret = wasm.renderer_skinMatricesPtr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The texture sets of the last prepared frame, `textureSlots()` ids each (colour,
     * normal, roughness, metalness, emissive, alpha, ambient-occlusion, light, bump,
     * displacement, specular and gradient maps; 0 for none). `drawInfo` gives each
     * draw's set.
     * @returns {Uint32Array}
     */
    textureSets() {
        const ret = wasm.renderer_textureSets(this.__wbg_ptr);
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Texture ids per set in `textureSets`.
     * @returns {number}
     */
    static textureSlots() {
        const ret = wasm.renderer_textureSlots();
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    static toneMappingExposureIndex() {
        const ret = wasm.renderer_toneMappingExposureIndex();
        return ret >>> 0;
    }
    /**
     * Index in `frameUniforms` of the tone mapping (a three.js constant, 0 for none);
     * the exposure follows `TONE_MAPPING_EXPOSURE_INDEX`.
     * @returns {number}
     */
    static toneMappingIndex() {
        const ret = wasm.renderer_toneMappingIndex();
        return ret >>> 0;
    }
    /**
     * How many transmissive draws follow the opaque ones.
     * @returns {number}
     */
    transmissiveCount() {
        const ret = wasm.renderer_transmissiveCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * WGSL source of the VSM blur: entry points `vertex`, `vertical` and `horizontal`.
     * @returns {string}
     */
    static vsmShaderSource() {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.renderer_vsmShaderSource();
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) Renderer.prototype[Symbol.dispose] = Renderer.prototype.free;

export class Sphere {
    static __wrap(ptr) {
        const obj = Object.create(Sphere.prototype);
        obj.__wbg_ptr = ptr;
        SphereFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SphereFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_sphere_free(ptr, 0);
    }
    /**
     * @param {Matrix4} m
     */
    applyMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.sphere_applyMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @returns {Vector3}
     */
    get center() {
        const ret = wasm.sphere_center(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @returns {Vector3}
     */
    clampPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.sphere_clampPoint(this.__wbg_ptr, point.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Sphere}
     */
    clone() {
        const ret = wasm.sphere_clone(this.__wbg_ptr);
        return Sphere.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @returns {boolean}
     */
    containsPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.sphere_containsPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Sphere} source
     */
    copy(source) {
        _assertClass(source, Sphere);
        wasm.sphere_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Vector3} point
     * @returns {number}
     */
    distanceToPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.sphere_distanceToPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Sphere} other
     * @returns {boolean}
     */
    equals(other) {
        _assertClass(other, Sphere);
        const ret = wasm.sphere_equals(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Vector3} point
     */
    expandByPoint(point) {
        _assertClass(point, Vector3);
        wasm.sphere_expandByPoint(this.__wbg_ptr, point.__wbg_ptr);
    }
    /**
     * @returns {Box3}
     */
    getBoundingBox() {
        const ret = wasm.sphere_getBoundingBox(this.__wbg_ptr);
        return Box3.__wrap(ret);
    }
    /**
     * @param {Box3} other
     * @returns {boolean}
     */
    intersectsBox(other) {
        _assertClass(other, Box3);
        const ret = wasm.sphere_intersectsBox(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Plane} plane
     * @returns {boolean}
     */
    intersectsPlane(plane) {
        _assertClass(plane, Plane);
        const ret = wasm.sphere_intersectsPlane(this.__wbg_ptr, plane.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Sphere} other
     * @returns {boolean}
     */
    intersectsSphere(other) {
        _assertClass(other, Sphere);
        const ret = wasm.sphere_intersectsSphere(this.__wbg_ptr, other.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {boolean}
     */
    isEmpty() {
        const ret = wasm.sphere_isEmpty(this.__wbg_ptr);
        return ret !== 0;
    }
    makeEmpty() {
        wasm.sphere_makeEmpty(this.__wbg_ptr);
    }
    /**
     * @param {Vector3} center
     * @param {number} radius
     */
    constructor(center, radius) {
        _assertClass(center, Vector3);
        const ret = wasm.sphere_new(center.__wbg_ptr, radius);
        this.__wbg_ptr = ret;
        SphereFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get radius() {
        const ret = wasm.sphere_radius(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector3} center
     * @param {number} radius
     */
    set(center, radius) {
        _assertClass(center, Vector3);
        wasm.sphere_set(this.__wbg_ptr, center.__wbg_ptr, radius);
    }
    /**
     * @param {number} radius
     */
    set radius(radius) {
        wasm.sphere_set_radius(this.__wbg_ptr, radius);
    }
    /**
     * @param {Vector3} offset
     */
    translate(offset) {
        _assertClass(offset, Vector3);
        wasm.sphere_translate(this.__wbg_ptr, offset.__wbg_ptr);
    }
    /**
     * @param {Sphere} other
     */
    union(other) {
        _assertClass(other, Sphere);
        wasm.sphere_union(this.__wbg_ptr, other.__wbg_ptr);
    }
}
if (Symbol.dispose) Sphere.prototype[Symbol.dispose] = Sphere.prototype.free;

export class Spherical {
    static __wrap(ptr) {
        const obj = Object.create(Spherical.prototype);
        obj.__wbg_ptr = ptr;
        SphericalFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        SphericalFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_spherical_free(ptr, 0);
    }
    /**
     * @returns {Spherical}
     */
    clone() {
        const ret = wasm.spherical_clone(this.__wbg_ptr);
        return Spherical.__wrap(ret);
    }
    /**
     * @param {Spherical} source
     */
    copy(source) {
        _assertClass(source, Spherical);
        wasm.spherical_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    makeSafe() {
        wasm.spherical_makeSafe(this.__wbg_ptr);
    }
    /**
     * Radius, polar angle from +Y and azimuth around Y, in radians.
     * @param {number} radius
     * @param {number} phi
     * @param {number} theta
     */
    constructor(radius, phi, theta) {
        const ret = wasm.spherical_new(radius, phi, theta);
        this.__wbg_ptr = ret;
        SphericalFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    get phi() {
        const ret = wasm.spherical_phi(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get radius() {
        const ret = wasm.spherical_radius(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    setFromCartesianCoords(x, y, z) {
        wasm.spherical_setFromCartesianCoords(this.__wbg_ptr, x, y, z);
    }
    /**
     * @param {Vector3} v
     */
    setFromVector3(v) {
        _assertClass(v, Vector3);
        wasm.spherical_setFromVector3(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {number} radius
     * @param {number} phi
     * @param {number} theta
     */
    set(radius, phi, theta) {
        wasm.spherical_set(this.__wbg_ptr, radius, phi, theta);
    }
    /**
     * @param {number} value
     */
    set phi(value) {
        wasm.spherical_set_phi(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set radius(value) {
        wasm.spherical_set_radius(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set theta(value) {
        wasm.spherical_set_theta(this.__wbg_ptr, value);
    }
    /**
     * @returns {number}
     */
    get theta() {
        const ret = wasm.spherical_theta(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Spherical.prototype[Symbol.dispose] = Spherical.prototype.free;

/**
 * Frame timing. Pass every timestamp in milliseconds, for example `performance.now()`.
 */
export class Timer {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TimerFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_timer_free(ptr, 0);
    }
    /**
     * Seconds between the last two updates.
     * @returns {number}
     */
    getDelta() {
        const ret = wasm.timer_getDelta(this.__wbg_ptr);
        return ret;
    }
    /**
     * Seconds accumulated over all updates.
     * @returns {number}
     */
    getElapsed() {
        const ret = wasm.timer_getElapsed(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} now
     */
    constructor(now) {
        const ret = wasm.timer_new(now);
        this.__wbg_ptr = ret;
        TimerFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {number} now
     */
    reset(now) {
        wasm.timer_reset(this.__wbg_ptr, now);
    }
    /**
     * @param {number} timescale
     */
    setTimescale(timescale) {
        wasm.timer_setTimescale(this.__wbg_ptr, timescale);
    }
    /**
     * @param {number} timestamp
     */
    update(timestamp) {
        wasm.timer_update(this.__wbg_ptr, timestamp);
    }
}
if (Symbol.dispose) Timer.prototype[Symbol.dispose] = Timer.prototype.free;

export class Triangle {
    static __wrap(ptr) {
        const obj = Object.create(Triangle.prototype);
        obj.__wbg_ptr = ptr;
        TriangleFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        TriangleFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_triangle_free(ptr, 0);
    }
    /**
     * @returns {Vector3}
     */
    get a() {
        const ret = wasm.triangle_a(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Vector3}
     */
    get b() {
        const ret = wasm.triangle_b(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Vector3}
     */
    get c() {
        const ret = wasm.triangle_c(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Triangle}
     */
    clone() {
        const ret = wasm.triangle_clone(this.__wbg_ptr);
        return Triangle.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @returns {Vector3}
     */
    closestPointToPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.triangle_closestPointToPoint(this.__wbg_ptr, point.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Vector3} point
     * @returns {boolean}
     */
    containsPoint(point) {
        _assertClass(point, Vector3);
        const ret = wasm.triangle_containsPoint(this.__wbg_ptr, point.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Triangle} source
     */
    copy(source) {
        _assertClass(source, Triangle);
        wasm.triangle_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Triangle} triangle
     * @returns {boolean}
     */
    equals(triangle) {
        _assertClass(triangle, Triangle);
        const ret = wasm.triangle_equals(this.__wbg_ptr, triangle.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @returns {number}
     */
    getArea() {
        const ret = wasm.triangle_getArea(this.__wbg_ptr);
        return ret;
    }
    /**
     * `undefined` for a degenerate triangle.
     * @param {Vector3} point
     * @returns {Vector3 | undefined}
     */
    getBarycoord(point) {
        _assertClass(point, Vector3);
        const ret = wasm.triangle_getBarycoord(this.__wbg_ptr, point.__wbg_ptr);
        return ret === 0 ? undefined : Vector3.__wrap(ret);
    }
    /**
     * @returns {Vector3}
     */
    getMidpoint() {
        const ret = wasm.triangle_getMidpoint(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Vector3}
     */
    getNormal() {
        const ret = wasm.triangle_getNormal(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @returns {Plane}
     */
    getPlane() {
        const ret = wasm.triangle_getPlane(this.__wbg_ptr);
        return Plane.__wrap(ret);
    }
    /**
     * @param {Box3} b
     * @returns {boolean}
     */
    intersectsBox(b) {
        _assertClass(b, Box3);
        const ret = wasm.triangle_intersectsBox(this.__wbg_ptr, b.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Vector3} direction
     * @returns {boolean}
     */
    isFrontFacing(direction) {
        _assertClass(direction, Vector3);
        const ret = wasm.triangle_isFrontFacing(this.__wbg_ptr, direction.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     * @param {Vector3} c
     */
    constructor(a, b, c) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        _assertClass(c, Vector3);
        const ret = wasm.triangle_new(a.__wbg_ptr, b.__wbg_ptr, c.__wbg_ptr);
        this.__wbg_ptr = ret;
        TriangleFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     * @param {Vector3} c
     */
    set(a, b, c) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        _assertClass(c, Vector3);
        wasm.triangle_set(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr, c.__wbg_ptr);
    }
}
if (Symbol.dispose) Triangle.prototype[Symbol.dispose] = Triangle.prototype.free;

export class Vector2 {
    static __wrap(ptr) {
        const obj = Object.create(Vector2.prototype);
        obj.__wbg_ptr = ptr;
        Vector2Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Vector2Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vector2_free(ptr, 0);
    }
    /**
     * @param {number} s
     */
    addScalar(s) {
        wasm.vector2_addScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Vector2} v
     * @param {number} s
     */
    addScaledVector(v, s) {
        _assertClass(v, Vector2);
        wasm.vector2_addScaledVector(this.__wbg_ptr, v.__wbg_ptr, s);
    }
    /**
     * @param {Vector2} a
     * @param {Vector2} b
     */
    addVectors(a, b) {
        _assertClass(a, Vector2);
        _assertClass(b, Vector2);
        wasm.vector2_addVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector2} v
     */
    add(v) {
        _assertClass(v, Vector2);
        wasm.vector2_add(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector2} v
     * @returns {number}
     */
    angleTo(v) {
        _assertClass(v, Vector2);
        const ret = wasm.vector2_angleTo(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    angle() {
        const ret = wasm.vector2_angle(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Matrix3} m
     */
    applyMatrix3(m) {
        _assertClass(m, Matrix3);
        wasm.vector2_applyMatrix3(this.__wbg_ptr, m.__wbg_ptr);
    }
    ceil() {
        wasm.vector2_ceil(this.__wbg_ptr);
    }
    /**
     * @param {number} min
     * @param {number} max
     */
    clampLength(min, max) {
        wasm.vector2_clampLength(this.__wbg_ptr, min, max);
    }
    /**
     * @param {number} min
     * @param {number} max
     */
    clampScalar(min, max) {
        wasm.vector2_clampScalar(this.__wbg_ptr, min, max);
    }
    /**
     * @param {Vector2} min
     * @param {Vector2} max
     */
    clamp(min, max) {
        _assertClass(min, Vector2);
        _assertClass(max, Vector2);
        wasm.vector2_clamp(this.__wbg_ptr, min.__wbg_ptr, max.__wbg_ptr);
    }
    /**
     * @returns {Vector2}
     */
    clone() {
        const ret = wasm.vector2_clone(this.__wbg_ptr);
        return Vector2.__wrap(ret);
    }
    /**
     * @param {Vector2} source
     */
    copy(source) {
        _assertClass(source, Vector2);
        wasm.vector2_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {Vector2} v
     * @returns {number}
     */
    cross(v) {
        _assertClass(v, Vector2);
        const ret = wasm.vector2_cross(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector2} v
     * @returns {number}
     */
    distanceToSquared(v) {
        _assertClass(v, Vector2);
        const ret = wasm.vector2_distanceToSquared(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector2} v
     * @returns {number}
     */
    distanceTo(v) {
        _assertClass(v, Vector2);
        const ret = wasm.vector2_distanceTo(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} scalar
     */
    divideScalar(scalar) {
        wasm.vector2_divideScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {Vector2} v
     */
    divide(v) {
        _assertClass(v, Vector2);
        wasm.vector2_divide(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector2} v
     * @returns {number}
     */
    dot(v) {
        _assertClass(v, Vector2);
        const ret = wasm.vector2_dot(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector2} v
     * @returns {boolean}
     */
    equals(v) {
        _assertClass(v, Vector2);
        const ret = wasm.vector2_equals(this.__wbg_ptr, v.__wbg_ptr);
        return ret !== 0;
    }
    floor() {
        wasm.vector2_floor(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    lengthSq() {
        const ret = wasm.vector2_lengthSq(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    length() {
        const ret = wasm.vector2_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector2} a
     * @param {Vector2} b
     * @param {number} alpha
     */
    lerpVectors(a, b, alpha) {
        _assertClass(a, Vector2);
        _assertClass(b, Vector2);
        wasm.vector2_lerpVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr, alpha);
    }
    /**
     * @param {Vector2} v
     * @param {number} alpha
     */
    lerp(v, alpha) {
        _assertClass(v, Vector2);
        wasm.vector2_lerp(this.__wbg_ptr, v.__wbg_ptr, alpha);
    }
    /**
     * @param {Vector2} v
     * @returns {number}
     */
    manhattanDistanceTo(v) {
        _assertClass(v, Vector2);
        const ret = wasm.vector2_manhattanDistanceTo(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    manhattanLength() {
        const ret = wasm.vector2_manhattanLength(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector2} v
     */
    max(v) {
        _assertClass(v, Vector2);
        wasm.vector2_max(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector2} v
     */
    min(v) {
        _assertClass(v, Vector2);
        wasm.vector2_min(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {number} scalar
     */
    multiplyScalar(scalar) {
        wasm.vector2_multiplyScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {Vector2} v
     */
    multiply(v) {
        _assertClass(v, Vector2);
        wasm.vector2_multiply(this.__wbg_ptr, v.__wbg_ptr);
    }
    negate() {
        wasm.vector2_negate(this.__wbg_ptr);
    }
    /**
     * @param {number} x
     * @param {number} y
     */
    constructor(x, y) {
        const ret = wasm.vector2_new(x, y);
        this.__wbg_ptr = ret;
        Vector2Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    normalize() {
        wasm.vector2_normalize(this.__wbg_ptr);
    }
    /**
     * @param {Vector2} center
     * @param {number} angle
     */
    rotateAround(center, angle) {
        _assertClass(center, Vector2);
        wasm.vector2_rotateAround(this.__wbg_ptr, center.__wbg_ptr, angle);
    }
    roundToZero() {
        wasm.vector2_roundToZero(this.__wbg_ptr);
    }
    round() {
        wasm.vector2_round(this.__wbg_ptr);
    }
    /**
     * @param {number} length
     */
    setLength(length) {
        wasm.vector2_setLength(this.__wbg_ptr, length);
    }
    /**
     * @param {number} scalar
     */
    setScalar(scalar) {
        wasm.vector2_setScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {number} x
     * @param {number} y
     */
    set(x, y) {
        wasm.vector2_set(this.__wbg_ptr, x, y);
    }
    /**
     * @param {number} value
     */
    set x(value) {
        wasm.vector2_set_x(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set y(value) {
        wasm.vector2_set_y(this.__wbg_ptr, value);
    }
    /**
     * @param {number} s
     */
    subScalar(s) {
        wasm.vector2_subScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Vector2} a
     * @param {Vector2} b
     */
    subVectors(a, b) {
        _assertClass(a, Vector2);
        _assertClass(b, Vector2);
        wasm.vector2_subVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector2} v
     */
    sub(v) {
        _assertClass(v, Vector2);
        wasm.vector2_sub(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.vector2_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.vector2_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.vector2_y(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Vector2.prototype[Symbol.dispose] = Vector2.prototype.free;

export class Vector3 {
    static __wrap(ptr) {
        const obj = Object.create(Vector3.prototype);
        obj.__wbg_ptr = ptr;
        Vector3Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Vector3Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vector3_free(ptr, 0);
    }
    /**
     * @param {number} s
     */
    addScalar(s) {
        wasm.vector3_addScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Vector3} v
     * @param {number} s
     */
    addScaledVector(v, s) {
        _assertClass(v, Vector3);
        wasm.vector3_addScaledVector(this.__wbg_ptr, v.__wbg_ptr, s);
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     */
    addVectors(a, b) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        wasm.vector3_addVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     */
    add(v) {
        _assertClass(v, Vector3);
        wasm.vector3_add(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     * @returns {number}
     */
    angleTo(v) {
        _assertClass(v, Vector3);
        const ret = wasm.vector3_angleTo(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector3} axis
     * @param {number} angle
     */
    applyAxisAngle(axis, angle) {
        _assertClass(axis, Vector3);
        wasm.vector3_applyAxisAngle(this.__wbg_ptr, axis.__wbg_ptr, angle);
    }
    /**
     * `order` is one of "XYZ", "YXZ", "ZXY", "ZYX", "YZX", "XZY".
     * @param {Euler} euler
     */
    applyEuler(euler) {
        _assertClass(euler, Euler);
        wasm.vector3_applyEuler(this.__wbg_ptr, euler.__wbg_ptr);
    }
    /**
     * @param {Matrix3} m
     */
    applyMatrix3(m) {
        _assertClass(m, Matrix3);
        wasm.vector3_applyMatrix3(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    applyMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.vector3_applyMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Matrix3} m
     */
    applyNormalMatrix(m) {
        _assertClass(m, Matrix3);
        wasm.vector3_applyNormalMatrix(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     */
    applyQuaternion(q) {
        _assertClass(q, Quaternion);
        wasm.vector3_applyQuaternion(this.__wbg_ptr, q.__wbg_ptr);
    }
    ceil() {
        wasm.vector3_ceil(this.__wbg_ptr);
    }
    /**
     * @param {number} min
     * @param {number} max
     */
    clampLength(min, max) {
        wasm.vector3_clampLength(this.__wbg_ptr, min, max);
    }
    /**
     * @param {number} min
     * @param {number} max
     */
    clampScalar(min, max) {
        wasm.vector3_clampScalar(this.__wbg_ptr, min, max);
    }
    /**
     * @param {Vector3} min
     * @param {Vector3} max
     */
    clamp(min, max) {
        _assertClass(min, Vector3);
        _assertClass(max, Vector3);
        wasm.vector3_clamp(this.__wbg_ptr, min.__wbg_ptr, max.__wbg_ptr);
    }
    /**
     * @returns {Vector3}
     */
    clone() {
        const ret = wasm.vector3_clone(this.__wbg_ptr);
        return Vector3.__wrap(ret);
    }
    /**
     * @param {Vector3} v
     */
    copy(v) {
        _assertClass(v, Vector3);
        wasm.vector3_copy(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     */
    crossVectors(a, b) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        wasm.vector3_crossVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     */
    cross(v) {
        _assertClass(v, Vector3);
        wasm.vector3_cross(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     * @returns {number}
     */
    distanceToSquared(v) {
        _assertClass(v, Vector3);
        const ret = wasm.vector3_distanceToSquared(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector3} v
     * @returns {number}
     */
    distanceTo(v) {
        _assertClass(v, Vector3);
        const ret = wasm.vector3_distanceTo(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} scalar
     */
    divideScalar(scalar) {
        wasm.vector3_divideScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {Vector3} v
     */
    divide(v) {
        _assertClass(v, Vector3);
        wasm.vector3_divide(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     * @returns {number}
     */
    dot(v) {
        _assertClass(v, Vector3);
        const ret = wasm.vector3_dot(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector3} v
     * @returns {boolean}
     */
    equals(v) {
        _assertClass(v, Vector3);
        const ret = wasm.vector3_equals(this.__wbg_ptr, v.__wbg_ptr);
        return ret !== 0;
    }
    floor() {
        wasm.vector3_floor(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    lengthSq() {
        const ret = wasm.vector3_lengthSq(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    length() {
        const ret = wasm.vector3_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     * @param {number} alpha
     */
    lerpVectors(a, b, alpha) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        wasm.vector3_lerpVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr, alpha);
    }
    /**
     * @param {Vector3} v
     * @param {number} alpha
     */
    lerp(v, alpha) {
        _assertClass(v, Vector3);
        wasm.vector3_lerp(this.__wbg_ptr, v.__wbg_ptr, alpha);
    }
    /**
     * @param {Vector3} v
     * @returns {number}
     */
    manhattanDistanceTo(v) {
        _assertClass(v, Vector3);
        const ret = wasm.vector3_manhattanDistanceTo(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    manhattanLength() {
        const ret = wasm.vector3_manhattanLength(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector3} v
     */
    max(v) {
        _assertClass(v, Vector3);
        wasm.vector3_max(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     */
    min(v) {
        _assertClass(v, Vector3);
        wasm.vector3_min(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {number} scalar
     */
    multiplyScalar(scalar) {
        wasm.vector3_multiplyScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     */
    multiplyVectors(a, b) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        wasm.vector3_multiplyVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     */
    multiply(v) {
        _assertClass(v, Vector3);
        wasm.vector3_multiply(this.__wbg_ptr, v.__wbg_ptr);
    }
    negate() {
        wasm.vector3_negate(this.__wbg_ptr);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    constructor(x, y, z) {
        const ret = wasm.vector3_new(x, y, z);
        this.__wbg_ptr = ret;
        Vector3Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    normalize() {
        wasm.vector3_normalize(this.__wbg_ptr);
    }
    /**
     * @param {Vector3} normal
     */
    projectOnPlane(normal) {
        _assertClass(normal, Vector3);
        wasm.vector3_projectOnPlane(this.__wbg_ptr, normal.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     */
    projectOnVector(v) {
        _assertClass(v, Vector3);
        wasm.vector3_projectOnVector(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector3} normal
     */
    reflect(normal) {
        _assertClass(normal, Vector3);
        wasm.vector3_reflect(this.__wbg_ptr, normal.__wbg_ptr);
    }
    roundToZero() {
        wasm.vector3_roundToZero(this.__wbg_ptr);
    }
    round() {
        wasm.vector3_round(this.__wbg_ptr);
    }
    /**
     * @param {Color} c
     */
    setFromColor(c) {
        _assertClass(c, Color);
        wasm.vector3_setFromColor(this.__wbg_ptr, c.__wbg_ptr);
    }
    /**
     * @param {number} radius
     * @param {number} theta
     * @param {number} y
     */
    setFromCylindricalCoords(radius, theta, y) {
        wasm.vector3_setFromCylindricalCoords(this.__wbg_ptr, radius, theta, y);
    }
    /**
     * @param {Cylindrical} c
     */
    setFromCylindrical(c) {
        _assertClass(c, Cylindrical);
        wasm.vector3_setFromCylindrical(this.__wbg_ptr, c.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    setFromMatrixPosition(m) {
        _assertClass(m, Matrix4);
        wasm.vector3_setFromMatrixPosition(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    setFromMatrixScale(m) {
        _assertClass(m, Matrix4);
        wasm.vector3_setFromMatrixScale(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {number} radius
     * @param {number} phi
     * @param {number} theta
     */
    setFromSphericalCoords(radius, phi, theta) {
        wasm.vector3_setFromSphericalCoords(this.__wbg_ptr, radius, phi, theta);
    }
    /**
     * @param {Spherical} s
     */
    setFromSpherical(s) {
        _assertClass(s, Spherical);
        wasm.vector3_setFromSpherical(this.__wbg_ptr, s.__wbg_ptr);
    }
    /**
     * @param {number} length
     */
    setLength(length) {
        wasm.vector3_setLength(this.__wbg_ptr, length);
    }
    /**
     * @param {number} scalar
     */
    setScalar(scalar) {
        wasm.vector3_setScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    set(x, y, z) {
        wasm.vector3_set(this.__wbg_ptr, x, y, z);
    }
    /**
     * @param {number} x
     */
    set x(x) {
        wasm.vector3_set_x(this.__wbg_ptr, x);
    }
    /**
     * @param {number} y
     */
    set y(y) {
        wasm.vector3_set_y(this.__wbg_ptr, y);
    }
    /**
     * @param {number} z
     */
    set z(z) {
        wasm.vector3_set_z(this.__wbg_ptr, z);
    }
    /**
     * @param {number} s
     */
    subScalar(s) {
        wasm.vector3_subScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Vector3} a
     * @param {Vector3} b
     */
    subVectors(a, b) {
        _assertClass(a, Vector3);
        _assertClass(b, Vector3);
        wasm.vector3_subVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector3} v
     */
    sub(v) {
        _assertClass(v, Vector3);
        wasm.vector3_sub(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.vector3_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {Matrix4} m
     */
    transformDirection(m) {
        _assertClass(m, Matrix4);
        wasm.vector3_transformDirection(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.vector3_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.vector3_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get z() {
        const ret = wasm.vector3_z(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Vector3.prototype[Symbol.dispose] = Vector3.prototype.free;

export class Vector4 {
    static __wrap(ptr) {
        const obj = Object.create(Vector4.prototype);
        obj.__wbg_ptr = ptr;
        Vector4Finalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        Vector4Finalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_vector4_free(ptr, 0);
    }
    /**
     * @param {number} s
     */
    addScalar(s) {
        wasm.vector4_addScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Vector4} v
     * @param {number} s
     */
    addScaledVector(v, s) {
        _assertClass(v, Vector4);
        wasm.vector4_addScaledVector(this.__wbg_ptr, v.__wbg_ptr, s);
    }
    /**
     * @param {Vector4} a
     * @param {Vector4} b
     */
    addVectors(a, b) {
        _assertClass(a, Vector4);
        _assertClass(b, Vector4);
        wasm.vector4_addVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector4} v
     */
    add(v) {
        _assertClass(v, Vector4);
        wasm.vector4_add(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    applyMatrix4(m) {
        _assertClass(m, Matrix4);
        wasm.vector4_applyMatrix4(this.__wbg_ptr, m.__wbg_ptr);
    }
    ceil() {
        wasm.vector4_ceil(this.__wbg_ptr);
    }
    /**
     * @param {number} min
     * @param {number} max
     */
    clampScalar(min, max) {
        wasm.vector4_clampScalar(this.__wbg_ptr, min, max);
    }
    /**
     * @returns {Vector4}
     */
    clone() {
        const ret = wasm.vector4_clone(this.__wbg_ptr);
        return Vector4.__wrap(ret);
    }
    /**
     * @param {Vector4} source
     */
    copy(source) {
        _assertClass(source, Vector4);
        wasm.vector4_copy(this.__wbg_ptr, source.__wbg_ptr);
    }
    /**
     * @param {number} scalar
     */
    divideScalar(scalar) {
        wasm.vector4_divideScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {Vector4} v
     */
    divide(v) {
        _assertClass(v, Vector4);
        wasm.vector4_divide(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector4} v
     * @returns {number}
     */
    dot(v) {
        _assertClass(v, Vector4);
        const ret = wasm.vector4_dot(this.__wbg_ptr, v.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector4} v
     * @returns {boolean}
     */
    equals(v) {
        _assertClass(v, Vector4);
        const ret = wasm.vector4_equals(this.__wbg_ptr, v.__wbg_ptr);
        return ret !== 0;
    }
    floor() {
        wasm.vector4_floor(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    lengthSq() {
        const ret = wasm.vector4_lengthSq(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    length() {
        const ret = wasm.vector4_length(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector4} v
     * @param {number} alpha
     */
    lerp(v, alpha) {
        _assertClass(v, Vector4);
        wasm.vector4_lerp(this.__wbg_ptr, v.__wbg_ptr, alpha);
    }
    /**
     * @returns {number}
     */
    manhattanLength() {
        const ret = wasm.vector4_manhattanLength(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {Vector4} v
     */
    max(v) {
        _assertClass(v, Vector4);
        wasm.vector4_max(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {Vector4} v
     */
    min(v) {
        _assertClass(v, Vector4);
        wasm.vector4_min(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @param {number} scalar
     */
    multiplyScalar(scalar) {
        wasm.vector4_multiplyScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {Vector4} v
     */
    multiply(v) {
        _assertClass(v, Vector4);
        wasm.vector4_multiply(this.__wbg_ptr, v.__wbg_ptr);
    }
    negate() {
        wasm.vector4_negate(this.__wbg_ptr);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    constructor(x, y, z, w) {
        const ret = wasm.vector4_new(x, y, z, w);
        this.__wbg_ptr = ret;
        Vector4Finalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    normalize() {
        wasm.vector4_normalize(this.__wbg_ptr);
    }
    roundToZero() {
        wasm.vector4_roundToZero(this.__wbg_ptr);
    }
    round() {
        wasm.vector4_round(this.__wbg_ptr);
    }
    /**
     * @param {Quaternion} q
     */
    setAxisAngleFromQuaternion(q) {
        _assertClass(q, Quaternion);
        wasm.vector4_setAxisAngleFromQuaternion(this.__wbg_ptr, q.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    setAxisAngleFromRotationMatrix(m) {
        _assertClass(m, Matrix4);
        wasm.vector4_setAxisAngleFromRotationMatrix(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {Matrix4} m
     */
    setFromMatrixPosition(m) {
        _assertClass(m, Matrix4);
        wasm.vector4_setFromMatrixPosition(this.__wbg_ptr, m.__wbg_ptr);
    }
    /**
     * @param {number} length
     */
    setLength(length) {
        wasm.vector4_setLength(this.__wbg_ptr, length);
    }
    /**
     * @param {number} scalar
     */
    setScalar(scalar) {
        wasm.vector4_setScalar(this.__wbg_ptr, scalar);
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    set(x, y, z, w) {
        wasm.vector4_set(this.__wbg_ptr, x, y, z, w);
    }
    /**
     * @param {number} value
     */
    set w(value) {
        wasm.vector4_set_w(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set x(value) {
        wasm.vector4_set_x(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set y(value) {
        wasm.vector4_set_y(this.__wbg_ptr, value);
    }
    /**
     * @param {number} value
     */
    set z(value) {
        wasm.vector4_set_z(this.__wbg_ptr, value);
    }
    /**
     * @param {number} s
     */
    subScalar(s) {
        wasm.vector4_subScalar(this.__wbg_ptr, s);
    }
    /**
     * @param {Vector4} a
     * @param {Vector4} b
     */
    subVectors(a, b) {
        _assertClass(a, Vector4);
        _assertClass(b, Vector4);
        wasm.vector4_subVectors(this.__wbg_ptr, a.__wbg_ptr, b.__wbg_ptr);
    }
    /**
     * @param {Vector4} v
     */
    sub(v) {
        _assertClass(v, Vector4);
        wasm.vector4_sub(this.__wbg_ptr, v.__wbg_ptr);
    }
    /**
     * @returns {Float32Array}
     */
    toArray() {
        const ret = wasm.vector4_toArray(this.__wbg_ptr);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @returns {number}
     */
    get w() {
        const ret = wasm.vector4_w(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get x() {
        const ret = wasm.vector4_x(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get y() {
        const ret = wasm.vector4_y(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get z() {
        const ret = wasm.vector4_z(this.__wbg_ptr);
        return ret;
    }
}
if (Symbol.dispose) Vector4.prototype[Symbol.dispose] = Vector4.prototype.free;

/**
 * Owner of a scene: objects, their transforms and hierarchy, geometries and materials.
 */
export class World {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WorldFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_world_free(ptr, 0);
    }
    /**
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @returns {boolean}
     */
    __setPosition(object, x, y, z) {
        const ret = wasm.world___setPosition(this.__wbg_ptr, object, x, y, z);
        return ret !== 0;
    }
    /**
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     * @returns {boolean}
     */
    __setQuaternion(object, x, y, z, w) {
        const ret = wasm.world___setQuaternion(this.__wbg_ptr, object, x, y, z, w);
        return ret !== 0;
    }
    /**
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @returns {boolean}
     */
    __setScale(object, x, y, z) {
        const ret = wasm.world___setScale(this.__wbg_ptr, object, x, y, z);
        return ret !== 0;
    }
    /**
     * @param {bigint} handle
     * @param {number} start
     * @param {number} count
     * @param {number} material_index
     */
    addGroup(handle, start, count, material_index) {
        const ret = wasm.world_addGroup(this.__wbg_ptr, handle, start, count, material_index);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Parents `children[i]` to `parents[i]` for every index.
     * @param {BigUint64Array} parents
     * @param {BigUint64Array} children
     */
    addMany(parents, children) {
        const ptr0 = passArray64ToWasm0(parents, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray64ToWasm0(children, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        wasm.world_addMany(this.__wbg_ptr, ptr0, len0, ptr1, len1);
    }
    /**
     * Makes `child` a child of `parent`. Returns false if either is dead or they are
     * the same object.
     * @param {bigint} parent
     * @param {bigint} child
     * @returns {boolean}
     */
    add(parent, child) {
        const ret = wasm.world_add(this.__wbg_ptr, parent, child);
        return ret !== 0;
    }
    /**
     * Transforms the geometry's positions, normals and tangents.
     * @param {bigint} handle
     * @param {Matrix4} matrix
     */
    applyMatrix4ToGeometry(handle, matrix) {
        _assertClass(matrix, Matrix4);
        const ret = wasm.world_applyMatrix4ToGeometry(this.__wbg_ptr, handle, matrix.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Writes parts of many transforms in one call. `values` holds 17 numbers per
     * object laid out as in `transformValues`; `masks` says which parts to take:
     * 1 position, 2 Euler angles and order, 4 quaternion, 8 scale, 16 up. Objects that
     * are not alive are skipped.
     * @param {BigUint64Array} objects
     * @param {Uint8Array} masks
     * @param {Float32Array} values
     */
    applyTransforms(objects, masks, values) {
        const ptr0 = passArray64ToWasm0(objects, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(masks, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        wasm.world_applyTransforms(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
    }
    /**
     * Reparents `object` under `parent` while keeping its world transform.
     * @param {bigint} parent
     * @param {bigint} object
     * @returns {boolean}
     */
    attach(parent, object) {
        const ret = wasm.world_attach(this.__wbg_ptr, parent, object);
        return ret !== 0;
    }
    /**
     * Number of vertices in the attribute.
     * @param {bigint} handle
     * @param {string} name
     * @returns {number}
     */
    attributeCount(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_attributeCount(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Number of values (not vertices) in the attribute.
     * @param {bigint} handle
     * @param {string} name
     * @returns {number}
     */
    attributeLength(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_attributeLength(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Byte offset of a float attribute's data in the module's memory, for building a
     * zero-copy view. The offset is invalid once the attribute is replaced or the
     * geometry disposed, and any view is detached when the memory grows.
     * @param {bigint} handle
     * @param {string} name
     * @returns {number}
     */
    attributePtr(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_attributePtr(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Upload counter of an attribute; it changes whenever the data should be re-uploaded.
     * @param {bigint} handle
     * @param {string} name
     * @returns {number}
     */
    attributeVersion(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_attributeVersion(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * Moves the geometry so that its bounding box is centred on the origin.
     * @param {bigint} handle
     */
    centerGeometry(handle) {
        const ret = wasm.world_centerGeometry(this.__wbg_ptr, handle);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @returns {BigUint64Array}
     */
    children(object) {
        const ret = wasm.world_children(this.__wbg_ptr, object);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * @param {bigint} object
     */
    clearCameraViewOffset(object) {
        const ret = wasm.world_clearCameraViewOffset(this.__wbg_ptr, object);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} handle
     */
    clearGroups(handle) {
        const ret = wasm.world_clearGroups(this.__wbg_ptr, handle);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Removes a geometry's index: its vertices are then drawn in order.
     * @param {bigint} handle
     */
    clearIndex(handle) {
        const ret = wasm.world_clearIndex(this.__wbg_ptr, handle);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     */
    clearRenderable(object) {
        wasm.world_clearRenderable(this.__wbg_ptr, object);
    }
    /**
     * @param {bigint} object
     */
    clearSkin(object) {
        wasm.world_clearSkin(this.__wbg_ptr, object);
    }
    /**
     * @param {bigint} object
     */
    clear(object) {
        wasm.world_clear(this.__wbg_ptr, object);
    }
    /**
     * Copies an object (and with `recursive` its subtree). The copy has no parent and
     * shares geometries and materials with the original.
     * @param {bigint} object
     * @param {boolean} recursive
     * @returns {bigint | undefined}
     */
    cloneObject(object, recursive) {
        const ret = wasm.world_cloneObject(this.__wbg_ptr, object, recursive);
        return ret[0] === 0 ? undefined : BigInt.asUintN(64, ret[1]);
    }
    /**
     * Returns min xyz, max xyz.
     * @param {bigint} handle
     * @returns {Float32Array}
     */
    computeBoundingBox(handle) {
        const ret = wasm.world_computeBoundingBox(this.__wbg_ptr, handle);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Returns center xyz, radius.
     * @param {bigint} handle
     * @returns {Float32Array}
     */
    computeBoundingSphere(handle) {
        const ret = wasm.world_computeBoundingSphere(this.__wbg_ptr, handle);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} handle
     */
    computeTangents(handle) {
        const ret = wasm.world_computeTangents(this.__wbg_ptr, handle);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} handle
     */
    computeVertexNormals(handle) {
        const ret = wasm.world_computeVertexNormals(this.__wbg_ptr, handle);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Copies an object's world matrix into an existing matrix without allocating an
     * intermediate JavaScript typed array.
     * @param {bigint} object
     * @param {Matrix4} target
     */
    copyMatrixWorld(object, target) {
        _assertClass(target, Matrix4);
        const ret = wasm.world_copyMatrixWorld(this.__wbg_ptr, object, target.__wbg_ptr);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * A box centred on the origin, with position, normal and uv attributes.
     * @param {number} width
     * @param {number} height
     * @param {number} depth
     * @param {number} width_segments
     * @param {number} height_segments
     * @param {number} depth_segments
     * @returns {bigint}
     */
    createBoxGeometry(width, height, depth, width_segments, height_segments, depth_segments) {
        const ret = wasm.world_createBoxGeometry(this.__wbg_ptr, width, height, depth, width_segments, height_segments, depth_segments);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} radius
     * @param {number} height
     * @param {number} cap_segments
     * @param {number} radial_segments
     * @param {number} height_segments
     * @returns {bigint}
     */
    createCapsuleGeometry(radius, height, cap_segments, radial_segments, height_segments) {
        const ret = wasm.world_createCapsuleGeometry(this.__wbg_ptr, radius, height, cap_segments, radial_segments, height_segments);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} radius
     * @param {number} segments
     * @param {number} theta_start
     * @param {number} theta_length
     * @returns {bigint}
     */
    createCircleGeometry(radius, segments, theta_start, theta_length) {
        const ret = wasm.world_createCircleGeometry(this.__wbg_ptr, radius, segments, theta_start, theta_length);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} radius_top
     * @param {number} radius_bottom
     * @param {number} height
     * @param {number} radial_segments
     * @param {number} height_segments
     * @param {boolean} open_ended
     * @param {number} theta_start
     * @param {number} theta_length
     * @returns {bigint}
     */
    createCylinderGeometry(radius_top, radius_bottom, height, radial_segments, height_segments, open_ended, theta_start, theta_length) {
        const ret = wasm.world_createCylinderGeometry(this.__wbg_ptr, radius_top, radius_bottom, height, radial_segments, height_segments, open_ended, theta_start, theta_length);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {bigint}
     */
    createGeometry() {
        const ret = wasm.world_createGeometry(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * `points` holds the profile as x, y pairs.
     * @param {Float32Array} points
     * @param {number} segments
     * @param {number} phi_start
     * @param {number} phi_length
     * @returns {bigint}
     */
    createLatheGeometry(points, segments, phi_start, phi_length) {
        const ptr0 = passArrayF32ToWasm0(points, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_createLatheGeometry(this.__wbg_ptr, ptr0, len0, segments, phi_start, phi_length);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {string} name
     * @returns {bigint}
     */
    createMaterial(name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_createMaterial(this.__wbg_ptr, ptr0, len0);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @returns {bigint}
     */
    createObject() {
        const ret = wasm.world_createObject(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Creates `count` objects at once and returns their handles.
     * @param {number} count
     * @returns {BigUint64Array}
     */
    createObjects(count) {
        const ret = wasm.world_createObjects(this.__wbg_ptr, count);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * A rectangle in the XY plane facing +Z.
     * @param {number} width
     * @param {number} height
     * @param {number} width_segments
     * @param {number} height_segments
     * @returns {bigint}
     */
    createPlaneGeometry(width, height, width_segments, height_segments) {
        const ret = wasm.world_createPlaneGeometry(this.__wbg_ptr, width, height, width_segments, height_segments);
        return BigInt.asUintN(64, ret);
    }
    /**
     * A platonic solid: `faces` 4 (tetrahedron), 8 (octahedron), 12 (dodecahedron) or
     * 20 (icosahedron).
     * @param {number} faces
     * @param {number} radius
     * @param {number} detail
     * @returns {bigint}
     */
    createPlatonicGeometry(faces, radius, detail) {
        const ret = wasm.world_createPlatonicGeometry(this.__wbg_ptr, faces, radius, detail);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return BigInt.asUintN(64, ret[0]);
    }
    /**
     * A polyhedron from xyz `vertices` and triangle `indices`, subdivided `detail` times
     * and pushed out to `radius`. Not indexed.
     * @param {Float32Array} vertices
     * @param {Uint32Array} indices
     * @param {number} radius
     * @param {number} detail
     * @returns {bigint}
     */
    createPolyhedronGeometry(vertices, indices, radius, detail) {
        const ptr0 = passArrayF32ToWasm0(vertices, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray32ToWasm0(indices, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.world_createPolyhedronGeometry(this.__wbg_ptr, ptr0, len0, ptr1, len1, radius, detail);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} inner_radius
     * @param {number} outer_radius
     * @param {number} theta_segments
     * @param {number} phi_segments
     * @param {number} theta_start
     * @param {number} theta_length
     * @returns {bigint}
     */
    createRingGeometry(inner_radius, outer_radius, theta_segments, phi_segments, theta_start, theta_length) {
        const ret = wasm.world_createRingGeometry(this.__wbg_ptr, inner_radius, outer_radius, theta_segments, phi_segments, theta_start, theta_length);
        return BigInt.asUintN(64, ret);
    }
    /**
     * A sphere, or a part of one. A full sphere is `phiStart = 0, phiLength = 2 PI,
     * thetaStart = 0, thetaLength = PI`.
     * @param {number} radius
     * @param {number} width_segments
     * @param {number} height_segments
     * @param {number} phi_start
     * @param {number} phi_length
     * @param {number} theta_start
     * @param {number} theta_length
     * @returns {bigint}
     */
    createSphereGeometry(radius, width_segments, height_segments, phi_start, phi_length, theta_start, theta_length) {
        const ret = wasm.world_createSphereGeometry(this.__wbg_ptr, radius, width_segments, height_segments, phi_start, phi_length, theta_start, theta_length);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} radius
     * @param {number} tube
     * @param {number} radial_segments
     * @param {number} tubular_segments
     * @param {number} arc
     * @param {number} theta_start
     * @param {number} theta_length
     * @returns {bigint}
     */
    createTorusGeometry(radius, tube, radial_segments, tubular_segments, arc, theta_start, theta_length) {
        const ret = wasm.world_createTorusGeometry(this.__wbg_ptr, radius, tube, radial_segments, tubular_segments, arc, theta_start, theta_length);
        return BigInt.asUintN(64, ret);
    }
    /**
     * @param {number} radius
     * @param {number} tube
     * @param {number} tubular_segments
     * @param {number} radial_segments
     * @param {number} p
     * @param {number} q
     * @returns {bigint}
     */
    createTorusKnotGeometry(radius, tube, tubular_segments, radial_segments, p, q) {
        const ret = wasm.world_createTorusKnotGeometry(this.__wbg_ptr, radius, tube, tubular_segments, radial_segments, p, q);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Removes an attribute; removing one the geometry does not have does nothing.
     * @param {bigint} handle
     * @param {string} name
     */
    deleteAttribute(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_deleteAttribute(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @returns {boolean}
     */
    destroyRecursive(object) {
        const ret = wasm.world_destroyRecursive(this.__wbg_ptr, object);
        return ret !== 0;
    }
    /**
     * Destroys one object; its children stay alive as roots.
     * @param {bigint} object
     * @returns {boolean}
     */
    destroy(object) {
        const ret = wasm.world_destroy(this.__wbg_ptr, object);
        return ret !== 0;
    }
    /**
     * Frees the geometry. Its handle is dead afterwards.
     * @param {bigint} handle
     * @returns {boolean}
     */
    disposeGeometry(handle) {
        const ret = wasm.world_disposeGeometry(this.__wbg_ptr, handle);
        return ret !== 0;
    }
    /**
     * @param {bigint} handle
     * @returns {boolean}
     */
    disposeMaterial(handle) {
        const ret = wasm.world_disposeMaterial(this.__wbg_ptr, handle);
        return ret !== 0;
    }
    /**
     * Takes the queued events as triples `[kind, a, b]`. Kinds: 0 added (a = object),
     * 1 removed (a = object), 2 childAdded (a = parent, b = child), 3 childRemoved
     * (a = parent, b = child), 4 objectDisposed, 5 geometryDisposed, 6 materialDisposed,
     * 7 renderTargetDisposed, 8 interleavedBufferDisposed (a = the handle).
     * @returns {BigUint64Array}
     */
    drainEvents() {
        const ret = wasm.world_drainEvents(this.__wbg_ptr);
        var v1 = getArrayU64FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
        return v1;
    }
    /**
     * Number of objects drawing this geometry.
     * @param {bigint} handle
     * @returns {number}
     */
    geometryRefCount(handle) {
        const ret = wasm.world_geometryRefCount(this.__wbg_ptr, handle);
        return ret >>> 0;
    }
    /**
     * A copy of the attribute's values.
     * @param {bigint} handle
     * @param {string} name
     * @returns {Float32Array}
     */
    getAttribute(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_getAttribute(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * The geometry's groups as three numbers each: start, count, material index.
     * @param {bigint} handle
     * @returns {Uint32Array}
     */
    getGroups(handle) {
        const ret = wasm.world_getGroups(this.__wbg_ptr, handle);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * A copy of the index, or an empty array for a non-indexed geometry.
     * @param {bigint} handle
     * @returns {Uint32Array}
     */
    getIndex(handle) {
        const ret = wasm.world_getIndex(this.__wbg_ptr, handle);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} root
     * @param {string} name
     * @returns {bigint | undefined}
     */
    getObjectByName(root, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_getObjectByName(this.__wbg_ptr, root, ptr0, len0);
        return ret[0] === 0 ? undefined : BigInt.asUintN(64, ret[1]);
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    getWorldDirection(object) {
        const ret = wasm.world_getWorldDirection(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    getWorldPosition(object) {
        const ret = wasm.world_getWorldPosition(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    getWorldQuaternion(object) {
        const ret = wasm.world_getWorldQuaternion(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    getWorldScale(object) {
        const ret = wasm.world_getWorldScale(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} handle
     * @param {string} name
     * @returns {boolean}
     */
    hasAttribute(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_hasAttribute(this.__wbg_ptr, handle, ptr0, len0);
        return ret !== 0;
    }
    /**
     * Upload counter of the index; 0 for a non-indexed geometry.
     * @param {bigint} handle
     * @returns {number}
     */
    indexVersion(handle) {
        const ret = wasm.world_indexVersion(this.__wbg_ptr, handle);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0] >>> 0;
    }
    /**
     * @param {bigint} object
     * @returns {boolean}
     */
    isAlive(object) {
        const ret = wasm.world_isAlive(this.__wbg_ptr, object);
        return ret !== 0;
    }
    /**
     * @param {bigint} object
     * @returns {boolean}
     */
    isVisible(object) {
        const ret = wasm.world_isVisible(this.__wbg_ptr, object);
        return ret !== 0;
    }
    /**
     * The layer bit mask of the object.
     * @param {bigint} object
     * @returns {number}
     */
    layers(object) {
        const ret = wasm.world_layers(this.__wbg_ptr, object);
        return ret >>> 0;
    }
    /**
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @returns {Float32Array}
     */
    localToWorld(object, x, y, z) {
        const ret = wasm.world_localToWorld(this.__wbg_ptr, object, x, y, z);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Rotates the object to face a point in world space (a camera looks at it).
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    lookAt(object, x, y, z) {
        wasm.world_lookAt(this.__wbg_ptr, object, x, y, z);
    }
    /**
     * Tells the renderer the attribute changed, after writing through a view.
     * @param {bigint} handle
     * @param {string} name
     */
    markAttributeNeedsUpdate(handle, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_markAttributeNeedsUpdate(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Writes the world matrices of many objects, 16 numbers each, into a buffer inside
     * the module and returns its address, for a `Float32Array` view without a copy
     * (for example as the instance matrices of an instanced mesh). The view is valid
     * until the next call of this method or until the module's memory grows. Objects
     * that are not alive get an identity matrix.
     * @param {BigUint64Array} objects
     * @returns {number}
     */
    matricesWorldPtr(objects) {
        const ptr0 = passArray64ToWasm0(objects, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_matricesWorldPtr(this.__wbg_ptr, ptr0, len0);
        return ret >>> 0;
    }
    /**
     * The world matrices of many objects, 16 elements each, in one array.
     * @param {BigUint64Array} objects
     * @returns {Float32Array}
     */
    matricesWorld(objects) {
        const ptr0 = passArray64ToWasm0(objects, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_matricesWorld(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * The same for the local matrix.
     * @param {bigint} object
     * @returns {number}
     */
    matrixPtr(object) {
        const ret = wasm.world_matrixPtr(this.__wbg_ptr, object);
        return ret >>> 0;
    }
    /**
     * The view matrix of a camera, as of the last world-matrix update.
     * @param {bigint} object
     * @returns {Float32Array}
     */
    matrixWorldInverse(object) {
        const ret = wasm.world_matrixWorldInverse(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Address of an object's world matrix (16 floats, column-major) inside the module,
     * for reading without a copy; 0 for an object that is not alive. Read it at once: it
     * moves when objects are created and is detached when the memory grows.
     * @param {bigint} object
     * @returns {number}
     */
    matrixWorldPtr(object) {
        const ret = wasm.world_matrixWorldPtr(this.__wbg_ptr, object);
        return ret >>> 0;
    }
    /**
     * The world matrix, 16 elements, column-major.
     * @param {bigint} object
     * @returns {Float32Array}
     */
    matrixWorld(object) {
        const ret = wasm.world_matrixWorld(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * The local matrix, 16 elements, column-major.
     * @param {bigint} object
     * @returns {Float32Array}
     */
    matrix(object) {
        const ret = wasm.world_matrix(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Writes `base`'s morph targets blended with `influences` into `target` (see
     * `wamr_core::morph`); `relative` is three.js `morphTargetsRelative`. The targets are
     * attributes of `base` named `morph.position.<i>`, `morph.normal.<i>` and
     * `morph.color.<i>`.
     * @param {bigint} base
     * @param {bigint} target
     * @param {Float32Array} influences
     * @param {boolean} relative
     */
    morphGeometry(base, target, influences, relative) {
        const ptr0 = passArrayF32ToWasm0(influences, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_morphGeometry(this.__wbg_ptr, base, target, ptr0, len0, relative);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @returns {string}
     */
    name(object) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.world_name(this.__wbg_ptr, object);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    constructor() {
        const ret = wasm.world_new();
        this.__wbg_ptr = ret;
        WorldFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * @returns {number}
     */
    objectCount() {
        const ret = wasm.world_objectCount(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * The parent's handle, or `undefined` for a root.
     * @param {bigint} object
     * @returns {bigint | undefined}
     */
    parent(object) {
        const ret = wasm.world_parent(this.__wbg_ptr, object);
        return ret[0] === 0 ? undefined : BigInt.asUintN(64, ret[1]);
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    position(object) {
        const ret = wasm.world_position(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    projectionMatrixInverse(object) {
        const ret = wasm.world_projectionMatrixInverse(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    projectionMatrix(object) {
        const ret = wasm.world_projectionMatrix(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    quaternion(object) {
        const ret = wasm.world_quaternion(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Casts a ray from a camera through a point of its view, given in normalized device
     * coordinates (x and y from -1 to 1, y up).
     * @param {bigint} camera
     * @param {number} x
     * @param {number} y
     * @param {bigint} root
     * @param {boolean} recursive
     * @returns {RaycastHits}
     */
    raycastFromCamera(camera, x, y, root, recursive) {
        const ret = wasm.world_raycastFromCamera(this.__wbg_ptr, camera, x, y, root, recursive);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return RaycastHits.__wrap(ret[0]);
    }
    /**
     * Casts a ray against each instance of the mesh `object`: `matrices` holds 16 floats
     * (column-major) per instance, placing it within the object. `raycast` skips objects
     * with instances, whose matrices the engine does not hold.
     * @param {bigint} object
     * @param {Float32Array} matrices
     * @param {Vector3} origin
     * @param {Vector3} direction
     * @param {number} near
     * @param {number} far
     * @returns {RaycastHits}
     */
    raycastInstances(object, matrices, origin, direction, near, far) {
        const ptr0 = passArrayF32ToWasm0(matrices, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        _assertClass(origin, Vector3);
        _assertClass(direction, Vector3);
        const ret = wasm.world_raycastInstances(this.__wbg_ptr, object, ptr0, len0, origin.__wbg_ptr, direction.__wbg_ptr, near, far);
        return RaycastHits.__wrap(ret);
    }
    /**
     * Casts a ray against the meshes at and (with `recursive`) below `root`. World
     * matrices must be up to date. `direction` must be normalized; pass `Infinity` as
     * `far` for no limit. Sprites and objects with instances are skipped (see
     * `raycastInstances`).
     * `line_threshold` and `points_threshold` are how close, in world units, the ray must pass to a line or a point to
     * hit it; anything that is not a number means 1. Objects whose layers do not share a
     * bit with `layers` are skipped (three.js `raycaster.layers`).
     * @param {bigint} root
     * @param {boolean} recursive
     * @param {Vector3} origin
     * @param {Vector3} direction
     * @param {number} near
     * @param {number} far
     * @param {number} line_threshold
     * @param {number} points_threshold
     * @param {number} layers
     * @returns {RaycastHits}
     */
    raycast(root, recursive, origin, direction, near, far, line_threshold, points_threshold, layers) {
        _assertClass(origin, Vector3);
        _assertClass(direction, Vector3);
        const ret = wasm.world_raycast(this.__wbg_ptr, root, recursive, origin.__wbg_ptr, direction.__wbg_ptr, near, far, line_threshold, points_threshold, layers);
        return RaycastHits.__wrap(ret);
    }
    /**
     * @param {bigint} object
     */
    removeFromParent(object) {
        wasm.world_removeFromParent(this.__wbg_ptr, object);
    }
    /**
     * @param {bigint} parent
     * @param {bigint} child
     * @returns {boolean}
     */
    remove(parent, child) {
        const ret = wasm.world_remove(this.__wbg_ptr, parent, child);
        return ret !== 0;
    }
    /**
     * Rotates about an axis in object space. `axis` must be normalized.
     * @param {bigint} object
     * @param {Vector3} axis
     * @param {number} angle
     */
    rotateOnAxis(object, axis, angle) {
        _assertClass(axis, Vector3);
        const ret = wasm.world_rotateOnAxis(this.__wbg_ptr, object, axis.__wbg_ptr, angle);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @returns {string}
     */
    rotationOrder(object) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ret = wasm.world_rotationOrder(this.__wbg_ptr, object);
            var ptr1 = ret[0];
            var len1 = ret[1];
            if (ret[3]) {
                ptr1 = 0; len1 = 0;
                throw takeFromExternrefTable0(ret[2]);
            }
            deferred2_0 = ptr1;
            deferred2_1 = len1;
            return getStringFromWasm0(ptr1, len1);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * The Euler angles xyz, in the object's rotation order.
     * @param {bigint} object
     * @returns {Float32Array}
     */
    rotation(object) {
        const ret = wasm.world_rotation(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * @param {bigint} object
     * @returns {Float32Array}
     */
    scale(object) {
        const ret = wasm.world_scale(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Adds or replaces a float attribute, copying `data` into engine memory. The vector is
     * the copy the glue already made, so the values are copied once.
     * @param {bigint} handle
     * @param {string} name
     * @param {Float32Array} data
     * @param {number} item_size
     */
    setAttribute(handle, name, data, item_size) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.world_setAttribute(this.__wbg_ptr, handle, ptr0, len0, ptr1, len1, item_size);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Changes the aspect ratio of a perspective camera and updates its projection.
     * @param {bigint} object
     * @param {number} aspect
     */
    setCameraAspect(object, aspect) {
        const ret = wasm.world_setCameraAspect(this.__wbg_ptr, object, aspect);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * three.js `focus`, `filmGauge` and `filmOffset` of a perspective camera; updates its
     * projection.
     * @param {bigint} object
     * @param {number} focus
     * @param {number} film_gauge
     * @param {number} film_offset
     */
    setCameraFilm(object, focus, film_gauge, film_offset) {
        wasm.world_setCameraFilm(this.__wbg_ptr, object, focus, film_gauge, film_offset);
    }
    /**
     * Restricts a camera to a sub-rectangle of a larger virtual viewport.
     * @param {bigint} object
     * @param {number} full_width
     * @param {number} full_height
     * @param {number} x
     * @param {number} y
     * @param {number} width
     * @param {number} height
     */
    setCameraViewOffset(object, full_width, full_height, x, y, width, height) {
        const ret = wasm.world_setCameraViewOffset(this.__wbg_ptr, object, full_width, full_height, x, y, width, height);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Zoom factor of a camera; updates its projection.
     * @param {bigint} object
     * @param {number} zoom
     */
    setCameraZoom(object, zoom) {
        const ret = wasm.world_setCameraZoom(this.__wbg_ptr, object, zoom);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} handle
     * @param {number} start
     * @param {number} count
     */
    setDrawRange(handle, start, count) {
        const ret = wasm.world_setDrawRange(this.__wbg_ptr, handle, start, count);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @param {boolean} culled
     */
    setFrustumCulled(object, culled) {
        wasm.world_setFrustumCulled(this.__wbg_ptr, object, culled);
    }
    /**
     * @param {bigint} handle
     * @param {Uint32Array} indices
     */
    setIndex(handle, indices) {
        const ptr0 = passArray32ToWasm0(indices, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setIndex(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Draws the object `count` times with the per-instance data the renderer's host
     * keeps under `id`; id 0 goes back to a single draw.
     * @param {bigint} object
     * @param {number} id
     * @param {number} count
     */
    setInstances(object, id, count) {
        wasm.world_setInstances(this.__wbg_ptr, object, id, count);
    }
    /**
     * @param {bigint} object
     * @param {number} mask
     */
    setLayers(object, mask) {
        wasm.world_setLayers(this.__wbg_ptr, object, mask);
    }
    /**
     * Makes the object draw `geometry_handle` as a closed line: through all vertices and
     * back to the first.
     * @param {bigint} object
     * @param {bigint} geometry_handle
     * @param {bigint} material_handle
     */
    setLineLoop(object, geometry_handle, material_handle) {
        wasm.world_setLineLoop(this.__wbg_ptr, object, geometry_handle, material_handle);
    }
    /**
     * Makes the object draw `geometry_handle` as a line: one connected line through
     * all vertices, or separate segments per pair of vertices.
     * @param {bigint} object
     * @param {bigint} geometry_handle
     * @param {bigint} material_handle
     * @param {boolean} segments
     */
    setLine(object, geometry_handle, material_handle, segments) {
        wasm.world_setLine(this.__wbg_ptr, object, geometry_handle, material_handle, segments);
    }
    /**
     * three.js `blending` (0 none, 1 normal, 2 additive, 3 subtractive, 4 multiply, 5
     * custom), `premultipliedAlpha`, and the packed factors and equations of custom
     * blending (see `Material::custom_blending`).
     * @param {bigint} handle
     * @param {number} blending
     * @param {boolean} premultiplied_alpha
     * @param {number} custom_blending
     */
    setMaterialBlending(handle, blending, premultiplied_alpha, custom_blending) {
        const ret = wasm.world_setMaterialBlending(this.__wbg_ptr, handle, blending, premultiplied_alpha, custom_blending);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * A material's clipping planes in world space, four numbers each (normal and
     * constant); at most eight are used.
     * @param {bigint} handle
     * @param {Float32Array} planes
     */
    setMaterialClipPlanes(handle, planes) {
        const ptr0 = passArrayF32ToWasm0(planes, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setMaterialClipPlanes(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Base colour of a material, linear components.
     * @param {bigint} handle
     * @param {number} r
     * @param {number} g
     * @param {number} b
     */
    setMaterialColor(handle, r, g, b) {
        const ret = wasm.world_setMaterialColor(this.__wbg_ptr, handle, r, g, b);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Draws a material with a custom shader (1-6 the renderer's effects, 16 and above a
     * host's shader materials; 0 the built-in shading) and its 16 settings.
     * @param {bigint} handle
     * @param {number} custom
     * @param {Float32Array} params
     */
    setMaterialCustom(handle, custom, params) {
        const ptr0 = passArrayF32ToWasm0(params, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setMaterialCustom(this.__wbg_ptr, handle, custom, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Emitted light of a material: linear colour already multiplied by the intensity.
     * @param {bigint} handle
     * @param {number} r
     * @param {number} g
     * @param {number} b
     */
    setMaterialEmissive(handle, r, g, b) {
        const ret = wasm.world_setMaterialEmissive(this.__wbg_ptr, handle, r, g, b);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * A standard material's own environment map (three.js `envMap`): the host's texture
     * id (0 for none, the frame's environment is used), the width of its equirectangular
     * image, and the inverted rotation of sampling directions (column-major 3x3).
     * @param {bigint} handle
     * @param {number} texture
     * @param {number} width
     * @param {Float32Array} rotation
     */
    setMaterialEnvMap(handle, texture, width, rotation) {
        const ptr0 = passArrayF32ToWasm0(rotation, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setMaterialEnvMap(this.__wbg_ptr, handle, texture, width, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Whether a standard material is lit by the frame's environment map, and how
     * strongly (three.js `envMapIntensity` or the scene's `environmentIntensity`).
     * @param {bigint} handle
     * @param {boolean} uses_environment
     * @param {number} intensity
     */
    setMaterialEnvironment(handle, uses_environment, intensity) {
        const ret = wasm.world_setMaterialEnvironment(this.__wbg_ptr, handle, uses_environment, intensity);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The remaining textures of a material, as ids the host resolves (0 for none): light,
     * bump, displacement, specular and toon gradient maps.
     * @param {bigint} handle
     * @param {number} light
     * @param {number} bump
     * @param {number} displacement
     * @param {number} specular
     * @param {number} gradient
     */
    setMaterialExtraMaps(handle, light, bump, displacement, specular, gradient) {
        const ret = wasm.world_setMaterialExtraMaps(this.__wbg_ptr, handle, light, bump, displacement, specular, gradient);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * three.js `flatShading`: each triangle lit with its own face normal.
     * @param {bigint} handle
     * @param {boolean} flat_shading
     */
    setMaterialFlatShading(handle, flat_shading) {
        const ret = wasm.world_setMaterialFlatShading(this.__wbg_ptr, handle, flat_shading);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * three.js `normalScale`, `aoMapIntensity` and `alphaTest` of a material.
     * @param {bigint} handle
     * @param {number} normal_scale_x
     * @param {number} normal_scale_y
     * @param {number} ao_map_intensity
     * @param {number} alpha_test
     */
    setMaterialMapSettings(handle, normal_scale_x, normal_scale_y, ao_map_intensity, alpha_test) {
        const ret = wasm.world_setMaterialMapSettings(this.__wbg_ptr, handle, normal_scale_x, normal_scale_y, ao_map_intensity, alpha_test);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * three.js `lightMapIntensity`, `bumpScale`, `displacementScale` and
     * `displacementBias` of a material.
     * @param {bigint} handle
     * @param {number} light_map_intensity
     * @param {number} bump_scale
     * @param {number} displacement_scale
     * @param {number} displacement_bias
     */
    setMaterialMapStrengths(handle, light_map_intensity, bump_scale, displacement_scale, displacement_bias) {
        const ret = wasm.world_setMaterialMapStrengths(this.__wbg_ptr, handle, light_map_intensity, bump_scale, displacement_scale, displacement_bias);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The textures of one or more maps by table index (see `setMaterialMapUv`):
     * `ids[i]` for map `first + i`, 0 for none. The colour map is map 0.
     * @param {bigint} handle
     * @param {number} first
     * @param {Uint32Array} ids
     */
    setMaterialMapTextures(handle, first, ids) {
        const ptr0 = passArray32ToWasm0(ids, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setMaterialMapTextures(this.__wbg_ptr, handle, first, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Offset, repeat, rotation and rotation centre applied to the texture coordinates of
     * every map (see `setMaterialMapUv` for one map).
     * @param {bigint} handle
     * @param {number} offset_x
     * @param {number} offset_y
     * @param {number} repeat_x
     * @param {number} repeat_y
     * @param {number} rotation
     * @param {number} center_x
     * @param {number} center_y
     */
    setMaterialMapTransform(handle, offset_x, offset_y, repeat_x, repeat_y, rotation, center_x, center_y) {
        const ret = wasm.world_setMaterialMapTransform(this.__wbg_ptr, handle, offset_x, offset_y, repeat_x, repeat_y, rotation, center_x, center_y);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The uv transform and uv set (0 `uv`, 1 `uv1`) of one map of a material, by its
     * index in the map table: 0 colour, 1 normal, 2 roughness, 3 metalness, 4 emissive, 5
     * alpha, 6 ambient occlusion, 7 light, 8 bump, 9 displacement, 10 specular, 11
     * gradient, 12 clearcoat, 13 clearcoat roughness, 14 clearcoat normal, 15 sheen
     * colour, 16 sheen roughness, 17 specular intensity, 18 specular colour, 19
     * transmission, 20 thickness, 21 iridescence, 22 iridescence thickness, 23
     * anisotropy, 24 matcap.
     * @param {bigint} handle
     * @param {number} map
     * @param {number} offset_x
     * @param {number} offset_y
     * @param {number} repeat_x
     * @param {number} repeat_y
     * @param {number} rotation
     * @param {number} center_x
     * @param {number} center_y
     * @param {number} channel
     */
    setMaterialMapUv(handle, map, offset_x, offset_y, repeat_x, repeat_y, rotation, center_x, center_y, channel) {
        const ret = wasm.world_setMaterialMapUv(this.__wbg_ptr, handle, map, offset_x, offset_y, repeat_x, repeat_y, rotation, center_x, center_y, channel);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The colour texture of a material, as an id the WebGPU host resolves; 0 for none.
     * @param {bigint} handle
     * @param {number} texture
     */
    setMaterialMap(handle, texture) {
        const ret = wasm.world_setMaterialMap(this.__wbg_ptr, handle, texture);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The textures of a material besides `map`, as ids the WebGPU host resolves (0 for
     * none): normal, roughness, metalness, emissive, alpha and ambient-occlusion maps.
     * @param {bigint} handle
     * @param {number} normal
     * @param {number} roughness
     * @param {number} metalness
     * @param {number} emissive
     * @param {number} alpha
     * @param {number} ao
     */
    setMaterialMaps(handle, normal, roughness, metalness, emissive, alpha, ao) {
        const ret = wasm.world_setMaterialMaps(this.__wbg_ptr, handle, normal, roughness, metalness, emissive, alpha, ao);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The light-through and thin-film layers of three.js `MeshPhysicalMaterial`:
     * `transmission`, `thickness`, linear `attenuationColor`, `attenuationDistance`
     * (Infinity for none), `dispersion`, `iridescence`, `iridescenceIOR`,
     * `iridescenceThicknessRange`, `anisotropy`, `anisotropyRotation` and
     * `clearcoatNormalScale`.
     * @param {bigint} handle
     * @param {number} transmission
     * @param {number} thickness
     * @param {number} attenuation_r
     * @param {number} attenuation_g
     * @param {number} attenuation_b
     * @param {number} attenuation_distance
     * @param {number} dispersion
     * @param {number} iridescence
     * @param {number} iridescence_ior
     * @param {number} thickness_min
     * @param {number} thickness_max
     * @param {number} anisotropy
     * @param {number} anisotropy_rotation
     * @param {number} clearcoat_normal_scale_x
     * @param {number} clearcoat_normal_scale_y
     */
    setMaterialPhysicalExtensions(handle, transmission, thickness, attenuation_r, attenuation_g, attenuation_b, attenuation_distance, dispersion, iridescence, iridescence_ior, thickness_min, thickness_max, anisotropy, anisotropy_rotation, clearcoat_normal_scale_x, clearcoat_normal_scale_y) {
        const ret = wasm.world_setMaterialPhysicalExtensions(this.__wbg_ptr, handle, transmission, thickness, attenuation_r, attenuation_g, attenuation_b, attenuation_distance, dispersion, iridescence, iridescence_ior, thickness_min, thickness_max, anisotropy, anisotropy_rotation, clearcoat_normal_scale_x, clearcoat_normal_scale_y);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The layers of three.js `MeshPhysicalMaterial` that a standard material draws:
     * clearcoat and its roughness, sheen (linear `sheenColor` already multiplied by
     * `sheen`) and its roughness, `ior`, `specularIntensity` and linear `specularColor`.
     * @param {bigint} handle
     * @param {number} clearcoat
     * @param {number} clearcoat_roughness
     * @param {number} sheen_r
     * @param {number} sheen_g
     * @param {number} sheen_b
     * @param {number} sheen_roughness
     * @param {number} ior
     * @param {number} specular_intensity
     * @param {number} specular_r
     * @param {number} specular_g
     * @param {number} specular_b
     */
    setMaterialPhysical(handle, clearcoat, clearcoat_roughness, sheen_r, sheen_g, sheen_b, sheen_roughness, ior, specular_intensity, specular_r, specular_g, specular_b) {
        const ret = wasm.world_setMaterialPhysical(this.__wbg_ptr, handle, clearcoat, clearcoat_roughness, sheen_r, sheen_g, sheen_b, sheen_roughness, ior, specular_intensity, specular_r, specular_g, specular_b);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * The fixed-function state and shading switches of a material (see
     * `wamr_core::RenderState`), as numbers in this order: visible, depth function,
     * colour write, polygon offset, its factor and units, alpha to coverage, dithering,
     * alpha hash, fog, object-space normal map, blend colour r, g, b and alpha, stencil
     * write, function, reference, function mask, write mask, fail, depth fail and pass
     * operations (three.js constants), clip intersection, clip shadows, allow override,
     * environment combine (0 multiply, 1 mix, 2 add), reflectivity, refraction ratio,
     * refraction (1) or reflection, dashed, dash size, gap size, dash scale, depth
     * packing, line width in world units, shadow side (-1 for none).
     * @param {bigint} handle
     * @param {Float32Array} values
     */
    setMaterialRenderState(handle, values) {
        const ptr0 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setMaterialRenderState(this.__wbg_ptr, handle, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * How a material is lit: `shading` 0 diffuse, 1 unlit, 2 standard (which uses
     * `roughness` and `metalness`), 3 Blinn-Phong, 4 normal, 5 toon, 6 matcap, 7 depth,
     * 8 distance, 9 shadow only.
     * @param {bigint} handle
     * @param {number} shading
     * @param {number} roughness
     * @param {number} metalness
     */
    setMaterialShading(handle, shading, roughness, metalness) {
        const ret = wasm.world_setMaterialShading(this.__wbg_ptr, handle, shading, roughness, metalness);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Which faces of the material count for picking: 0 front, 1 back, 2 both.
     * @param {bigint} handle
     * @param {number} side
     */
    setMaterialSide(handle, side) {
        const ret = wasm.world_setMaterialSide(this.__wbg_ptr, handle, side);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * `PointsMaterial.size` and a line material's `linewidth`, in pixels. Above 1 they
     * are drawn as quads.
     * @param {bigint} handle
     * @param {number} point_size
     * @param {number} line_width
     */
    setMaterialSizes(handle, point_size, line_width) {
        const ret = wasm.world_setMaterialSizes(this.__wbg_ptr, handle, point_size, line_width);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Highlight of a Blinn-Phong material: linear specular colour and shininess.
     * @param {bigint} handle
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} shininess
     */
    setMaterialSpecular(handle, r, g, b, shininess) {
        const ret = wasm.world_setMaterialSpecular(this.__wbg_ptr, handle, r, g, b, shininess);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Sprite settings of a material: rotation in radians, and whether the sprite gets
     * smaller with distance.
     * @param {bigint} handle
     * @param {number} rotation
     * @param {boolean} size_attenuation
     */
    setMaterialSprite(handle, rotation, size_attenuation) {
        const ret = wasm.world_setMaterialSprite(this.__wbg_ptr, handle, rotation, size_attenuation);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Blending and depth state. `opacity` only shows when `transparent` is set.
     * @param {bigint} handle
     * @param {boolean} transparent
     * @param {number} opacity
     * @param {boolean} depth_test
     * @param {boolean} depth_write
     * @param {boolean} vertex_colors
     */
    setMaterialState(handle, transparent, opacity, depth_test, depth_write, vertex_colors) {
        const ret = wasm.world_setMaterialState(this.__wbg_ptr, handle, transparent, opacity, depth_test, depth_write, vertex_colors);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * three.js `wireframe`: meshes draw their triangles' edges as lines.
     * @param {bigint} handle
     * @param {boolean} wireframe
     */
    setMaterialWireframe(handle, wireframe) {
        const ret = wasm.world_setMaterialWireframe(this.__wbg_ptr, handle, wireframe);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * One material per geometry group, indexed by the group's material index. An empty
     * list goes back to the single material of the object.
     * @param {bigint} object
     * @param {BigUint64Array} materials
     */
    setMaterials(object, materials) {
        const ptr0 = passArray64ToWasm0(materials, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.world_setMaterials(this.__wbg_ptr, object, ptr0, len0);
    }
    /**
     * With `false` the local matrix is no longer rebuilt from position, rotation and
     * scale; set it with `setMatrix`.
     * @param {bigint} object
     * @param {boolean} value
     */
    setMatrixAutoUpdate(object, value) {
        const ret = wasm.world_setMatrixAutoUpdate(this.__wbg_ptr, object, value);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Sets the local matrix (16 numbers, column-major) and marks the world matrix as
     * out of date. It only lasts on objects whose matrix is not updated automatically.
     * @param {bigint} object
     * @param {Float32Array} elements
     */
    setMatrix(object, elements) {
        const ptr0 = passArrayF32ToWasm0(elements, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setMatrix(this.__wbg_ptr, object, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @param {string} name
     */
    setName(object, name) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.world_setName(this.__wbg_ptr, object, ptr0, len0);
    }
    /**
     * Makes the object an orthographic camera.
     * @param {bigint} object
     * @param {number} left
     * @param {number} right
     * @param {number} top
     * @param {number} bottom
     * @param {number} near
     * @param {number} far
     * @param {boolean} webgpu
     */
    setOrthographicCamera(object, left, right, top, bottom, near, far, webgpu) {
        wasm.world_setOrthographicCamera(this.__wbg_ptr, object, left, right, top, bottom, near, far, webgpu);
    }
    /**
     * Makes the object a perspective camera. `fov` is the vertical field of view in
     * degrees; `webgpu` selects clip-space depth 0..1 instead of -1..1.
     * @param {bigint} object
     * @param {number} fov
     * @param {number} aspect
     * @param {number} near
     * @param {number} far
     * @param {boolean} webgpu
     */
    setPerspectiveCamera(object, fov, aspect, near, far, webgpu) {
        wasm.world_setPerspectiveCamera(this.__wbg_ptr, object, fov, aspect, near, far, webgpu);
    }
    /**
     * The point rotation and scale turn about, in local space (three.js `pivot`);
     * `enabled` false for the origin.
     * @param {bigint} object
     * @param {boolean} enabled
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    setPivot(object, enabled, x, y, z) {
        wasm.world_setPivot(this.__wbg_ptr, object, enabled, x, y, z);
    }
    /**
     * Makes the object draw `geometry_handle` as points, one pixel per vertex.
     * @param {bigint} object
     * @param {bigint} geometry_handle
     * @param {bigint} material_handle
     */
    setPoints(object, geometry_handle, material_handle) {
        wasm.world_setPoints(this.__wbg_ptr, object, geometry_handle, material_handle);
    }
    /**
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    setPosition(object, x, y, z) {
        const ret = wasm.world_setPosition(this.__wbg_ptr, object, x, y, z);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Sets the quaternion; the Euler angles follow.
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} w
     */
    setQuaternion(object, x, y, z, w) {
        const ret = wasm.world_setQuaternion(this.__wbg_ptr, object, x, y, z, w);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @param {number} order
     */
    setRenderOrder(object, order) {
        wasm.world_setRenderOrder(this.__wbg_ptr, object, order);
    }
    /**
     * Makes the object draw `geometry_handle` with `material_handle`.
     * @param {bigint} object
     * @param {bigint} geometry_handle
     * @param {bigint} material_handle
     */
    setRenderable(object, geometry_handle, material_handle) {
        wasm.world_setRenderable(this.__wbg_ptr, object, geometry_handle, material_handle);
    }
    /**
     * @param {bigint} object
     * @param {string} order
     */
    setRotationOrder(object, order) {
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setRotationOrder(this.__wbg_ptr, object, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Sets the Euler angles; the quaternion follows. `order` as in `Quaternion.setFromEuler`.
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {string} order
     */
    setRotation(object, x, y, z, order) {
        const ptr0 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setRotation(this.__wbg_ptr, object, x, y, z, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Sets the Euler angles of many objects from one array of three numbers per object.
     * @param {BigUint64Array} objects
     * @param {Float32Array} angles
     * @param {string} order
     */
    setRotations(objects, angles, order) {
        const ptr0 = passArray64ToWasm0(objects, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(angles, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(order, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ret = wasm.world_setRotations(this.__wbg_ptr, ptr0, len0, ptr1, len1, ptr2, len2);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     */
    setScale(object, x, y, z) {
        const ret = wasm.world_setScale(this.__wbg_ptr, object, x, y, z);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * three.js `castShadow` and `receiveShadow`.
     * @param {bigint} object
     * @param {boolean} cast
     * @param {boolean} receive
     */
    setShadowFlags(object, cast, receive) {
        wasm.world_setShadowFlags(this.__wbg_ptr, object, cast, receive);
    }
    /**
     * Replaces the inverse bind matrices of a skin (three.js `Skeleton.calculateInverses`
     * or a changed `boneInverses`).
     * @param {bigint} object
     * @param {Float32Array} inverses
     */
    setSkinInverses(object, inverses) {
        const ptr0 = passArrayF32ToWasm0(inverses, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.world_setSkinInverses(this.__wbg_ptr, object, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Makes `object` a skinned mesh (three.js `SkinnedMesh.bind`): the handles of its
     * bones (0 for a missing bone), their inverse bind matrices (16 numbers each), the
     * bind matrix and its inverse, and whether the bind mode is detached.
     * @param {bigint} object
     * @param {BigUint64Array} bones
     * @param {Float32Array} inverses
     * @param {Float32Array} bind_matrix
     * @param {Float32Array} bind_matrix_inverse
     * @param {boolean} detached
     */
    setSkin(object, bones, inverses, bind_matrix, bind_matrix_inverse, detached) {
        const ptr0 = passArray64ToWasm0(bones, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(inverses, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passArrayF32ToWasm0(bind_matrix, wasm.__wbindgen_malloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passArrayF32ToWasm0(bind_matrix_inverse, wasm.__wbindgen_malloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.world_setSkin(this.__wbg_ptr, object, ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, detached);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Makes the object a sprite: `geometry_handle` (a quad) faces the camera, anchored
     * at `center_x, center_y` (0.5, 0.5 is the middle) and sized by the world scale.
     * @param {bigint} object
     * @param {bigint} geometry_handle
     * @param {bigint} material_handle
     * @param {number} center_x
     * @param {number} center_y
     */
    setSprite(object, geometry_handle, material_handle, center_x, center_y) {
        wasm.world_setSprite(this.__wbg_ptr, object, geometry_handle, material_handle, center_x, center_y);
    }
    /**
     * Sets one number of an object's transform; see `transformValue` for the slots.
     * Changing an Euler angle updates the quaternion and the other way round.
     * @param {bigint} object
     * @param {number} slot
     * @param {number} value
     */
    setTransformValue(object, slot, value) {
        const ret = wasm.world_setTransformValue(this.__wbg_ptr, object, slot, value);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Sets position (xyz), quaternion (xyzw) and scale (xyz) of many objects from one
     * array of ten numbers per object.
     * @param {BigUint64Array} objects
     * @param {Float32Array} values
     */
    setTransforms(objects, values) {
        const ptr0 = passArray64ToWasm0(objects, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(values, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.world_setTransforms(this.__wbg_ptr, ptr0, len0, ptr1, len1);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @param {boolean} visible
     */
    setVisible(object, visible) {
        wasm.world_setVisible(this.__wbg_ptr, object, visible);
    }
    /**
     * The skinning matrices of `object` at the current world matrices, 16 numbers per
     * bone (each takes a bind-pose vertex to its posed place in the mesh's local space),
     * as three.js `SkinnedMesh.applyBoneTransform` applies them.
     * @param {bigint} object
     * @returns {Float32Array}
     */
    skinMatrices(object) {
        const ret = wasm.world_skinMatrices(this.__wbg_ptr, object);
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * A new geometry with the index expanded away.
     * @param {bigint} handle
     * @returns {bigint}
     */
    toNonIndexed(handle) {
        const ret = wasm.world_toNonIndexed(this.__wbg_ptr, handle);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return BigInt.asUintN(64, ret[0]);
    }
    /**
     * One number of an object's transform. Slots: 0-2 position, 3-5 Euler angles,
     * 6-9 quaternion, 10-12 scale, 13-15 up.
     * @param {bigint} object
     * @param {number} slot
     * @returns {number}
     */
    transformValue(object, slot) {
        const ret = wasm.world_transformValue(this.__wbg_ptr, object, slot);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return ret[0];
    }
    /**
     * Position (3), Euler angles (3), quaternion (4), scale (3), up (3) and the index of
     * the Euler order (XYZ, YXZ, ZXY, ZYX, YZX, XZY) of an object: 17 numbers.
     * @param {bigint} object
     * @returns {Float32Array}
     */
    transformValues(object) {
        const ret = wasm.world_transformValues(this.__wbg_ptr, object);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Moves along an axis in object space. `axis` must be normalized.
     * @param {bigint} object
     * @param {Vector3} axis
     * @param {number} distance
     */
    translateOnAxis(object, axis, distance) {
        _assertClass(axis, Vector3);
        const ret = wasm.world_translateOnAxis(this.__wbg_ptr, object, axis.__wbg_ptr, distance);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Updates local and world matrices of `root` and everything below it.
     * @param {bigint} root
     * @param {boolean} force
     */
    updateMatrixWorld(root, force) {
        wasm.world_updateMatrixWorld(this.__wbg_ptr, root, force);
    }
    /**
     * Recomputes the local matrix of one object from position, rotation and scale.
     * @param {bigint} object
     */
    updateMatrix(object) {
        const ret = wasm.world_updateMatrix(this.__wbg_ptr, object);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * @param {bigint} object
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @returns {Float32Array}
     */
    worldToLocal(object, x, y, z) {
        const ret = wasm.world_worldToLocal(this.__wbg_ptr, object, x, y, z);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v1;
    }
    /**
     * Like `getWorldPosition`, `getWorldQuaternion`, `getWorldScale` and
     * `getWorldDirection` (`what` 0 to 3), written to a fixed buffer inside the module
     * whose address is returned (3 floats, 4 for the quaternion), so that nothing is
     * allocated per call. 0 for an object that is not alive. Read it at once: the next
     * call overwrites it.
     * @param {bigint} object
     * @param {number} what
     * @returns {number}
     */
    worldValuePtr(object, what) {
        const ret = wasm.world_worldValuePtr(this.__wbg_ptr, object, what);
        return ret >>> 0;
    }
}
if (Symbol.dispose) World.prototype[Symbol.dispose] = World.prototype.free;

/**
 * @param {number} value
 * @returns {number}
 */
export function ceilPowerOfTwo(value) {
    const ret = wasm.ceilPowerOfTwo(value);
    return ret;
}

/**
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export function clamp(value, min, max) {
    const ret = wasm.clamp(value, min, max);
    return ret;
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} lambda
 * @param {number} dt
 * @returns {number}
 */
export function damp(x, y, lambda, dt) {
    const ret = wasm.damp(x, y, lambda, dt);
    return ret;
}

/**
 * @param {number} degrees
 * @returns {number}
 */
export function degToRad(degrees) {
    const ret = wasm.degToRad(degrees);
    return ret;
}

/**
 * @param {number} n
 * @param {number} m
 * @returns {number}
 */
export function euclideanModulo(n, m) {
    const ret = wasm.euclideanModulo(n, m);
    return ret;
}

/**
 * @param {number} value
 * @returns {number}
 */
export function floorPowerOfTwo(value) {
    const ret = wasm.floorPowerOfTwo(value);
    return ret;
}

/**
 * A version 4 UUID in lower case.
 * @returns {string}
 */
export function generateUUID() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.generateUUID();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} value
 * @returns {number}
 */
export function inverseLerp(x, y, value) {
    const ret = wasm.inverseLerp(x, y, value);
    return ret;
}

/**
 * @param {number} value
 * @returns {boolean}
 */
export function isPowerOfTwo(value) {
    const ret = wasm.isPowerOfTwo(value);
    return ret !== 0;
}

/**
 * @param {number} x
 * @param {number} y
 * @param {number} t
 * @returns {number}
 */
export function lerp(x, y, t) {
    const ret = wasm.lerp(x, y, t);
    return ret;
}

/**
 * @param {number} x
 * @param {number} a1
 * @param {number} a2
 * @param {number} b1
 * @param {number} b2
 * @returns {number}
 */
export function mapLinear(x, a1, a2, b1, b2) {
    const ret = wasm.mapLinear(x, a1, a2, b1, b2);
    return ret;
}

/**
 * @param {number} x
 * @param {number} length
 * @returns {number}
 */
export function pingpong(x, length) {
    const ret = wasm.pingpong(x, length);
    return ret;
}

/**
 * @param {number} radians
 * @returns {number}
 */
export function radToDeg(radians) {
    const ret = wasm.radToDeg(radians);
    return ret;
}

/**
 * Random float in [low, high).
 * @param {number} low
 * @param {number} high
 * @returns {number}
 */
export function randFloat(low, high) {
    const ret = wasm.randFloat(low, high);
    return ret;
}

/**
 * Random float in (-range / 2, range / 2].
 * @param {number} range
 * @returns {number}
 */
export function randFloatSpread(range) {
    const ret = wasm.randFloatSpread(range);
    return ret;
}

/**
 * Random integer in [low, high], both included.
 * @param {number} low
 * @param {number} high
 * @returns {number}
 */
export function randInt(low, high) {
    const ret = wasm.randInt(low, high);
    return ret;
}

/**
 * Deterministic pseudo-random number in [0, 1). Pass a seed to restart the sequence.
 * @param {number | null} [seed]
 * @returns {number}
 */
export function seededRandom(seed) {
    const ret = wasm.seededRandom(isLikeNone(seed) ? Number.MAX_SAFE_INTEGER : (seed) >>> 0);
    return ret;
}

/**
 * Restarts the sequence behind `randInt`, `randFloat`, `randFloatSpread` and
 * `generateUUID` from a 64-bit seed given as two halves. `init()` seeds it from
 * `crypto.getRandomValues`; call this for a reproducible sequence.
 * @param {number} high
 * @param {number} low
 */
export function setRandomSeed(high, low) {
    wasm.setRandomSeed(high, low);
}

/**
 * @param {number} x
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export function smootherstep(x, min, max) {
    const ret = wasm.smootherstep(x, min, max);
    return ret;
}

/**
 * @param {number} x
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export function smoothstep(x, min, max) {
    const ret = wasm.smoothstep(x, min, max);
    return ret;
}

/**
 * The crate version.
 * @returns {string}
 */
export function version() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.version();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_30c8987f7c2ed4e2: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./wamr3d_wasm_bg.js": import0,
    };
}

const Box3Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_box3_free(ptr, 1));
const ColorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_color_free(ptr, 1));
const CylindricalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_cylindrical_free(ptr, 1));
const EulerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_euler_free(ptr, 1));
const FrustumFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_frustum_free(ptr, 1));
const Line3Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_line3_free(ptr, 1));
const Matrix3Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_matrix3_free(ptr, 1));
const Matrix4Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_matrix4_free(ptr, 1));
const PlaneFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_plane_free(ptr, 1));
const QuaternionFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_quaternion_free(ptr, 1));
const RayFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_ray_free(ptr, 1));
const RaycastHitsFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_raycasthits_free(ptr, 1));
const RendererFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_renderer_free(ptr, 1));
const SphereFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_sphere_free(ptr, 1));
const SphericalFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_spherical_free(ptr, 1));
const TimerFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_timer_free(ptr, 1));
const TriangleFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_triangle_free(ptr, 1));
const Vector2Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vector2_free(ptr, 1));
const Vector3Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vector3_free(ptr, 1));
const Vector4Finalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_vector4_free(ptr, 1));
const WorldFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_world_free(ptr, 1));

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getBigUint64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedBigUint64ArrayMemory0 = null;
function getBigUint64ArrayMemory0() {
    if (cachedBigUint64ArrayMemory0 === null || cachedBigUint64ArrayMemory0.byteLength === 0) {
        cachedBigUint64ArrayMemory0 = new BigUint64Array(wasm.memory.buffer);
    }
    return cachedBigUint64ArrayMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getBigUint64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedBigUint64ArrayMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('wamr3d_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
