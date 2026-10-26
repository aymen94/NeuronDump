//! Header parsing: magic / version, metadata, tensor infos, alignment, sizes and bounds warnings.

use super::*;

impl GgufFile {
    pub(super) fn parse(storage: Storage, declared_size: Option<u64>, opts: ParseOptions) -> Result<Self> {
        let data = storage.bytes();
        // Sanity checks below compare counts against what is left of the real file.
        let total = declared_size.unwrap_or(data.len() as u64);
        let mut warnings = Vec::new();
        let mut r = Reader::new(data);

        let magic: [u8; 4] = {
            let s = r.take(4)?;
            [s[0], s[1], s[2], s[3]]
        };
        if &magic != GGUF_MAGIC {
            return Err(match &magic {
                b"lmgg" | b"ggml" => GgufError::LegacyFormat("ggml"),
                b"fmgg" | b"ggmf" => GgufError::LegacyFormat("ggmf"),
                b"tjgg" | b"ggjt" => GgufError::LegacyFormat("ggjt"),
                _ => GgufError::BadMagic(magic),
            });
        }

        let raw_version = r.u32()?;
        let (version, endianness) = if (1..=3).contains(&raw_version) {
            (raw_version, Endianness::Little)
        } else if (1..=3).contains(&raw_version.swap_bytes()) {
            r.big_endian = true;
            warnings.push("big-endian GGUF file: tensor data is not usable on little-endian hosts".into());
            (raw_version.swap_bytes(), Endianness::Big)
        } else if raw_version > 3 && raw_version < 0x1_0000 {
            warnings.push(format!("unknown GGUF version {raw_version}; parsing as v3"));
            (raw_version, Endianness::Little)
        } else {
            return Err(GgufError::UnsupportedVersion(raw_version));
        };
        r.wide = version >= 2;

        let n_tensors = r.len()?;
        let n_kv = r.len()?;

        // Each KV needs at least a key length + type id; each tensor info at least
        // name length + n_dims + type + offset.
        let min_kv = if r.wide { 12 } else { 8 };
        if n_kv.saturating_mul(min_kv) > total.saturating_sub(r.pos as u64) {
            return Err(r.malformed(format!("metadata count {n_kv} exceeds file size")));
        }

        let mut metadata = IndexMap::with_capacity(n_kv as usize);
        let mut lossy_keys = Vec::new();
        for _ in 0..n_kv {
            let mut lossy = false;
            let key = r.string(&mut lossy)?;
            let ty = r.value_type()?;
            let mut value_lossy = false;
            let value = r.value(ty, opts.load_arrays, &mut value_lossy)?;
            if lossy {
                warnings.push(format!("metadata key {key:?} is not valid UTF-8 (decoded lossily)"));
            }
            if value_lossy {
                lossy_keys.push(key.clone());
            }
            if metadata.contains_key(&key) {
                warnings.push(format!("duplicate metadata key {key:?}; keeping the first value"));
                continue;
            }
            metadata.insert(key, value);
        }
        for key in lossy_keys {
            warnings.push(format!("metadata {key:?} contains invalid UTF-8 strings (decoded lossily)"));
        }

        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(v) => match v.as_u64() {
                Some(a) if a > 0 && a.is_power_of_two() => a,
                _ => {
                    warnings.push(format!("invalid general.alignment {v}; using {DEFAULT_ALIGNMENT}"));
                    DEFAULT_ALIGNMENT
                }
            },
        };

        let min_ti = if r.wide { 8 + 4 + 4 + 8 } else { 4 + 4 + 4 + 8 };
        if n_tensors.saturating_mul(min_ti) > total.saturating_sub(r.pos as u64) {
            return Err(r.malformed(format!("tensor count {n_tensors} exceeds file size")));
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut index = HashMap::with_capacity(n_tensors as usize);
        for i in 0..n_tensors as usize {
            let mut lossy = false;
            let name = r.string(&mut lossy)?;
            if lossy {
                warnings.push(format!("tensor name {name:?} is not valid UTF-8"));
            }
            let n_dims = r.u32()?;
            if n_dims > MAX_DIMS {
                return Err(r.malformed(format!("tensor {name:?} has {n_dims} dimensions")));
            }
            if n_dims > 4 {
                warnings.push(format!("tensor {name:?} has {n_dims} dimensions (ggml supports 4)"));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.len()?);
            }
            let ggml_type = GgmlType(r.u32()?);
            let offset = r.u64()?;
            let n_elements = dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| r.malformed(format!("tensor {name:?} element count overflows")))?;
            let (n_bytes, size_estimated) = match ggml_type.bytes_for(n_elements) {
                Some(b) => (b, false),
                None => {
                    if ggml_type.is_known() {
                        warnings.push(format!(
                            "tensor {name:?}: {n_elements} elements is not a whole number of {ggml_type} blocks"
                        ));
                    } else {
                        warnings.push(format!(
                            "tensor {name:?} uses unknown ggml type id {}; size derived from offsets",
                            ggml_type.0
                        ));
                    }
                    (0, true)
                }
            };
            if offset % alignment != 0 {
                warnings.push(format!("tensor {name:?} offset {offset} is not {alignment}-byte aligned"));
            }
            if index.insert(name.clone(), i).is_some() {
                warnings.push(format!("duplicate tensor name {name:?}"));
            }
            tensors.push(TensorInfo { name, dims, ggml_type, offset, n_elements, n_bytes, size_estimated });
        }

        let header_size = r.pos as u64;
        let data_offset = header_size.div_ceil(alignment) * alignment;
        let file_size = total;
        let data_len = file_size.saturating_sub(data_offset);

        // Derive sizes we could not compute from the gap to the next tensor.
        if tensors.iter().any(|t| t.size_estimated) {
            let mut order: Vec<usize> = (0..tensors.len()).collect();
            order.sort_by_key(|&i| tensors[i].offset);
            for (pos, &i) in order.iter().enumerate() {
                if tensors[i].size_estimated {
                    let next = order.get(pos + 1).map_or(data_len, |&j| tensors[j].offset);
                    tensors[i].n_bytes = next.saturating_sub(tensors[i].offset);
                }
            }
        }

        let mut oob = 0usize;
        for t in &tensors {
            if data_offset.saturating_add(t.offset).saturating_add(t.n_bytes) > file_size {
                oob += 1;
                if oob <= 5 {
                    warnings.push(format!("tensor {:?} extends beyond the end of the file", t.name));
                }
            }
        }
        if oob > 5 {
            warnings.push(format!("{oob} tensors in total extend beyond the end of the file (truncated download?)"));
        }

        Ok(Self {
            version,
            endianness,
            alignment,
            data_offset,
            header_size,
            file_size,
            metadata,
            tensors,
            warnings,
            index,
            storage,
        })
    }

}
