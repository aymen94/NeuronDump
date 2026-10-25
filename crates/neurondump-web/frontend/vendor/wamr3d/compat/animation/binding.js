// PropertyBinding: resolving a track name to a property of an object, material or bone.
//
// Part of the keyframe animation system; `../animation.js` gathers it.

const RESERVED = '\\[\\]\\.:\\/';
const reservedRe = new RegExp( '[' + RESERVED + ']', 'g' );
const wordChar = '[^' + RESERVED + ']';
const wordCharOrDot = '[^' + RESERVED.replace( '\\.', '' ) + ']';
const directoryRe = /((?:WC+[\/:])*)/.source.replace( 'WC', wordChar );
const nodeRe = /(WCOD+)?/.source.replace( 'WCOD', wordCharOrDot );
const objectRe = /(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace( 'WC', wordChar );
const propertyRe = /\.(WC+)(?:\[(.+)\])?/.source.replace( 'WC', wordChar );
const trackRe = new RegExp( '^' + directoryRe + nodeRe + objectRe + propertyRe + '$' );
const supportedObjectNames = [ 'material', 'materials', 'bones', 'map' ];

// One path bound in every object of an AnimationObjectGroup.
class Composite {

	constructor( targetGroup, path, parsedPath ) {

		this._targetGroup = targetGroup;
		this._bindings = targetGroup.subscribe_( path, parsedPath || PropertyBinding.parseTrackName( path ) );

	}

	getValue( array, offset ) {

		this.bind();
		const binding = this._bindings[ this._targetGroup.nCachedObjects_ ];
		if ( binding !== undefined ) binding.getValue( array, offset );

	}

	setValue( array, offset ) {

		const bindings = this._bindings;
		for ( let i = this._targetGroup.nCachedObjects_; i !== bindings.length; ++ i ) bindings[ i ].setValue( array, offset );

	}

	bind() {

		const bindings = this._bindings;
		for ( let i = this._targetGroup.nCachedObjects_; i !== bindings.length; ++ i ) bindings[ i ].bind();

	}

	unbind() {

		const bindings = this._bindings;
		for ( let i = this._targetGroup.nCachedObjects_; i !== bindings.length; ++ i ) bindings[ i ].unbind();

	}

}

const BINDING_DIRECT = 0, BINDING_ARRAY = 1, BINDING_ELEMENT = 2, BINDING_FROM_TO_ARRAY = 3;
const VERSION_NONE = 0, VERSION_NEEDS_UPDATE = 1, VERSION_MATRIX_WORLD = 2;

/** Reads and writes the property a track path names, below `rootNode`. Binds lazily on first use. */
export class PropertyBinding {

	constructor( rootNode, path, parsedPath ) {

		this.path = path;
		this.parsedPath = parsedPath || PropertyBinding.parseTrackName( path );
		this.node = PropertyBinding.findNode( rootNode, this.parsedPath.nodeName );
		this.rootNode = rootNode;
		this.getValue = this._getValue_unbound;
		this.setValue = this._setValue_unbound;

	}

	static create( root, path, parsedPath ) {

		return root && root.isAnimationObjectGroup ? new PropertyBinding.Composite( root, path, parsedPath ) : new PropertyBinding( root, path, parsedPath );

	}

	/** Spaces become underscores and the path's reserved characters are removed. */
	static sanitizeNodeName( name ) {

		return name.replace( /\s/g, '_' ).replace( reservedRe, '' );

	}

	static parseTrackName( trackName ) {

		const matches = trackRe.exec( trackName );
		if ( matches === null ) throw new Error( 'WAMR3D: PropertyBinding: cannot parse trackName: ' + trackName );
		const results = { nodeName: matches[ 2 ], objectName: matches[ 3 ], objectIndex: matches[ 4 ], propertyName: matches[ 5 ], propertyIndex: matches[ 6 ] };
		const lastDot = results.nodeName && results.nodeName.lastIndexOf( '.' );

		// 'node.material.opacity': the object name was read as part of the node name
		if ( lastDot !== undefined && lastDot !== - 1 ) {

			const objectName = results.nodeName.substring( lastDot + 1 );

			if ( supportedObjectNames.indexOf( objectName ) !== - 1 ) {

				results.nodeName = results.nodeName.substring( 0, lastDot );
				results.objectName = objectName;

			}

		}

		if ( results.propertyName === null || results.propertyName.length === 0 ) throw new Error( 'WAMR3D: PropertyBinding: can not parse propertyName from trackName: ' + trackName );
		return results;

	}

	static findNode( root, nodeName ) {

		if ( nodeName === undefined || nodeName === '' || nodeName === '.' || nodeName === - 1 || nodeName === root.name || nodeName === root.uuid ) return root;

		if ( root.skeleton ) {

			const bone = root.skeleton.getBoneByName( nodeName );
			if ( bone !== undefined ) return bone;

		}

		if ( root.children ) {

			const search = ( children ) => {

				for ( const child of children ) {

					if ( child.name === nodeName || child.uuid === nodeName ) return child;
					const found = search( child.children );
					if ( found ) return found;

				}

				return null;

			};

			const found = search( root.children );
			if ( found ) return found;

		}

		return null;

	}

	_getValue_unavailable() {}

	_setValue_unavailable() {}

	_getValue_direct( buffer, offset ) {

		buffer[ offset ] = this.targetObject[ this.propertyName ];

	}

	_getValue_array( buffer, offset ) {

		const source = this.resolvedProperty;
		for ( let i = 0; i !== source.length; ++ i ) buffer[ offset ++ ] = source[ i ];

	}

	_getValue_arrayElement( buffer, offset ) {

		buffer[ offset ] = this.resolvedProperty[ this.propertyIndex ];

	}

	_getValue_toArray( buffer, offset ) {

		this.resolvedProperty.toArray( buffer, offset );

	}

	_setValue_direct( buffer, offset ) {

		this.targetObject[ this.propertyName ] = buffer[ offset ];

	}

	_setValue_direct_setNeedsUpdate( buffer, offset ) {

		this.targetObject[ this.propertyName ] = buffer[ offset ];
		this.targetObject.needsUpdate = true;

	}

	_setValue_direct_setMatrixWorldNeedsUpdate( buffer, offset ) {

		this.targetObject[ this.propertyName ] = buffer[ offset ];
		this.targetObject.matrixWorldNeedsUpdate = true;

	}

	_setValue_array( buffer, offset ) {

		const dest = this.resolvedProperty;
		for ( let i = 0; i !== dest.length; ++ i ) dest[ i ] = buffer[ offset ++ ];

	}

	_setValue_array_setNeedsUpdate( buffer, offset ) {

		this._setValue_array( buffer, offset );
		this.targetObject.needsUpdate = true;

	}

	_setValue_array_setMatrixWorldNeedsUpdate( buffer, offset ) {

		this._setValue_array( buffer, offset );
		this.targetObject.matrixWorldNeedsUpdate = true;

	}

	_setValue_arrayElement( buffer, offset ) {

		this.resolvedProperty[ this.propertyIndex ] = buffer[ offset ];

	}

	_setValue_arrayElement_setNeedsUpdate( buffer, offset ) {

		this.resolvedProperty[ this.propertyIndex ] = buffer[ offset ];
		this.targetObject.needsUpdate = true;

	}

	_setValue_arrayElement_setMatrixWorldNeedsUpdate( buffer, offset ) {

		this.resolvedProperty[ this.propertyIndex ] = buffer[ offset ];
		this.targetObject.matrixWorldNeedsUpdate = true;

	}

	_setValue_fromArray( buffer, offset ) {

		this.resolvedProperty.fromArray( buffer, offset );

	}

	_setValue_fromArray_setNeedsUpdate( buffer, offset ) {

		this.resolvedProperty.fromArray( buffer, offset );
		this.targetObject.needsUpdate = true;

	}

	_setValue_fromArray_setMatrixWorldNeedsUpdate( buffer, offset ) {

		this.resolvedProperty.fromArray( buffer, offset );
		this.targetObject.matrixWorldNeedsUpdate = true;

	}

	_getValue_unbound( targetArray, offset ) {

		this.bind();
		this.getValue( targetArray, offset );

	}

	_setValue_unbound( sourceArray, offset ) {

		this.bind();
		this.setValue( sourceArray, offset );

	}

	// Resolves the path to an object and property once, then picks the getter and setter
	// for that kind of property.
	bind() {

		let targetObject = this.node;
		const parsedPath = this.parsedPath, objectName = parsedPath.objectName, propertyName = parsedPath.propertyName;
		let propertyIndex = parsedPath.propertyIndex;

		if ( ! targetObject ) {

			targetObject = PropertyBinding.findNode( this.rootNode, parsedPath.nodeName );
			this.node = targetObject;

		}

		this.getValue = this._getValue_unavailable;
		this.setValue = this._setValue_unavailable;

		if ( ! targetObject ) {

			console.warn( 'WAMR3D: PropertyBinding: no target node found for track: ' + this.path + '.' );
			return;

		}

		if ( objectName ) {

			let objectIndex = parsedPath.objectIndex;

			switch ( objectName ) {

				case 'materials':
					if ( ! targetObject.material || ! targetObject.material.materials ) {

						console.error( 'WAMR3D: PropertyBinding: can not bind to material.materials.', this );
						return;

					}

					targetObject = targetObject.material.materials;
					break;

				case 'bones':
					if ( ! targetObject.skeleton ) {

						console.error( 'WAMR3D: PropertyBinding: can not bind to bones as node does not have a skeleton.', this );
						return;

					}

					targetObject = targetObject.skeleton.bones;

					// a bone may be named instead of numbered
					for ( let i = 0; i < targetObject.length; i ++ ) {

						if ( targetObject[ i ].name === objectIndex ) {

							objectIndex = i;
							break;

						}

					}

					break;

				case 'map':
					if ( 'map' in targetObject ) {

						targetObject = targetObject.map;
						break;

					}

					if ( ! targetObject.material || ! targetObject.material.map ) {

						console.error( 'WAMR3D: PropertyBinding: can not bind to material.map.', this );
						return;

					}

					targetObject = targetObject.material.map;
					break;

				default:
					if ( targetObject[ objectName ] === undefined ) {

						console.error( 'WAMR3D: PropertyBinding: can not bind to objectName of node undefined.', this );
						return;

					}

					targetObject = targetObject[ objectName ];

			}

			if ( objectIndex !== undefined ) {

				if ( targetObject[ objectIndex ] === undefined ) {

					console.error( 'WAMR3D: PropertyBinding: trying to bind to objectIndex of objectName, but is undefined.', this, targetObject );
					return;

				}

				targetObject = targetObject[ objectIndex ];

			}

		}

		const nodeProperty = targetObject[ propertyName ];

		if ( nodeProperty === undefined ) {

			console.error( 'WAMR3D: PropertyBinding: trying to update property for track: ' + parsedPath.nodeName + '.' + propertyName + " but it wasn't found.", targetObject );
			return;

		}

		this.targetObject = targetObject;
		const versioning = targetObject.isMaterial === true ? VERSION_NEEDS_UPDATE : targetObject.isObject3D === true ? VERSION_MATRIX_WORLD : VERSION_NONE;
		let bindingType = BINDING_DIRECT;

		if ( propertyIndex !== undefined ) {

			if ( propertyName === 'morphTargetInfluences' ) {

				if ( ! targetObject.geometry || ! targetObject.geometry.morphAttributes ) {

					console.error( 'WAMR3D: PropertyBinding: can not bind to morphTargetInfluences because node does not have a geometry with morphAttributes.', this );
					return;

				}

				if ( targetObject.morphTargetDictionary[ propertyIndex ] !== undefined ) propertyIndex = targetObject.morphTargetDictionary[ propertyIndex ];

			}

			bindingType = BINDING_ELEMENT;
			this.resolvedProperty = nodeProperty;
			this.propertyIndex = propertyIndex;

		} else if ( nodeProperty.fromArray !== undefined && nodeProperty.toArray !== undefined ) {

			bindingType = BINDING_FROM_TO_ARRAY;
			this.resolvedProperty = nodeProperty;

		} else if ( Array.isArray( nodeProperty ) ) {

			bindingType = BINDING_ARRAY;
			this.resolvedProperty = nodeProperty;

		} else {

			this.propertyName = propertyName;

		}

		this.getValue = this.GetterByBindingType[ bindingType ];
		this.setValue = this.SetterByBindingTypeAndVersioning[ bindingType ][ versioning ];

	}

	unbind() {

		this.node = null;
		this.getValue = this._getValue_unbound;
		this.setValue = this._setValue_unbound;

	}

}

PropertyBinding.Composite = Composite;
PropertyBinding.prototype.BindingType = { Direct: BINDING_DIRECT, EntireArray: BINDING_ARRAY, ArrayElement: BINDING_ELEMENT, HasFromToArray: BINDING_FROM_TO_ARRAY };
PropertyBinding.prototype.Versioning = { None: VERSION_NONE, NeedsUpdate: VERSION_NEEDS_UPDATE, MatrixWorldNeedsUpdate: VERSION_MATRIX_WORLD };
PropertyBinding.prototype.GetterByBindingType = [
	PropertyBinding.prototype._getValue_direct,
	PropertyBinding.prototype._getValue_array,
	PropertyBinding.prototype._getValue_arrayElement,
	PropertyBinding.prototype._getValue_toArray
];
PropertyBinding.prototype.SetterByBindingTypeAndVersioning = [
	[ PropertyBinding.prototype._setValue_direct, PropertyBinding.prototype._setValue_direct_setNeedsUpdate, PropertyBinding.prototype._setValue_direct_setMatrixWorldNeedsUpdate ],
	[ PropertyBinding.prototype._setValue_array, PropertyBinding.prototype._setValue_array_setNeedsUpdate, PropertyBinding.prototype._setValue_array_setMatrixWorldNeedsUpdate ],
	[ PropertyBinding.prototype._setValue_arrayElement, PropertyBinding.prototype._setValue_arrayElement_setNeedsUpdate, PropertyBinding.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate ],
	[ PropertyBinding.prototype._setValue_fromArray, PropertyBinding.prototype._setValue_fromArray_setNeedsUpdate, PropertyBinding.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate ]
];
