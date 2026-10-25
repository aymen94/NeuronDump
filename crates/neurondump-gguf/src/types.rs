//! ggml tensor types and the llama.cpp file-type (`general.file_type`) table.

use std::borrow::Cow;
use std::fmt;

use serde::{Serialize, Serializer};

/// A ggml tensor type id. Kept as a raw id so that unknown (future) types survive
/// parsing; layout information comes from a static table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GgmlType(pub u32);

/// Static layout of a ggml type: elements per block and bytes per block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct TypeLayout {
    pub name: &'static str,
    pub block_size: u64,
    pub type_size: u64,
    pub quantized: bool,
    /// Removed from current ggml (repacked/legacy layouts); still sized for old files.
    pub legacy: bool,
}

const fn t(name: &'static str, block_size: u64, type_size: u64, quantized: bool) -> Option<TypeLayout> {
    Some(TypeLayout { name, block_size, type_size, quantized, legacy: false })
}

const fn legacy(name: &'static str, block_size: u64, type_size: u64) -> Option<TypeLayout> {
    Some(TypeLayout { name, block_size, type_size, quantized: true, legacy: true })
}

/// Indexed by type id. Sizes mirror ggml's `type_traits` (`ggml.c`) and block structs
/// (`ggml-common.h`).
const LAYOUTS: [Option<TypeLayout>; 40] = [
    t("F32", 1, 4, false),          // 0
    t("F16", 1, 2, false),          // 1
    t("Q4_0", 32, 18, true),        // 2
    t("Q4_1", 32, 20, true),        // 3
    None,                           // 4  Q4_2 (removed before GGUF existed)
    None,                           // 5  Q4_3 (removed before GGUF existed)
    t("Q5_0", 32, 22, true),        // 6
    t("Q5_1", 32, 24, true),        // 7
    t("Q8_0", 32, 34, true),        // 8
    t("Q8_1", 32, 36, true),        // 9
    t("Q2_K", 256, 84, true),       // 10
    t("Q3_K", 256, 110, true),      // 11
    t("Q4_K", 256, 144, true),      // 12
    t("Q5_K", 256, 176, true),      // 13
    t("Q6_K", 256, 210, true),      // 14
    t("Q8_K", 256, 292, true),      // 15
    t("IQ2_XXS", 256, 66, true),    // 16
    t("IQ2_XS", 256, 74, true),     // 17
    t("IQ3_XXS", 256, 98, true),    // 18
    t("IQ1_S", 256, 50, true),      // 19
    t("IQ4_NL", 32, 18, true),      // 20
    t("IQ3_S", 256, 110, true),     // 21
    t("IQ2_S", 256, 82, true),      // 22
    t("IQ4_XS", 256, 136, true),    // 23
    t("I8", 1, 1, false),           // 24
    t("I16", 1, 2, false),          // 25
    t("I32", 1, 4, false),          // 26
    t("I64", 1, 8, false),          // 27
    t("F64", 1, 8, false),          // 28
    t("IQ1_M", 256, 56, true),      // 29
    t("BF16", 1, 2, false),         // 30
    legacy("Q4_0_4_4", 32, 18),     // 31
    legacy("Q4_0_4_8", 32, 18),     // 32
    legacy("Q4_0_8_8", 32, 18),     // 33
    t("TQ1_0", 256, 54, true),      // 34
    t("TQ2_0", 256, 66, true),      // 35
    legacy("IQ4_NL_4_4", 32, 18),   // 36
    legacy("IQ4_NL_4_8", 32, 18),   // 37
    legacy("IQ4_NL_8_8", 32, 18),   // 38
    t("MXFP4", 32, 17, true),       // 39
];

impl GgmlType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q5_0: Self = Self(6);
    pub const Q5_1: Self = Self(7);
    pub const Q8_0: Self = Self(8);
    pub const Q8_1: Self = Self(9);
    pub const Q2_K: Self = Self(10);
    pub const Q3_K: Self = Self(11);
    pub const Q4_K: Self = Self(12);
    pub const Q5_K: Self = Self(13);
    pub const Q6_K: Self = Self(14);
    pub const Q8_K: Self = Self(15);
    pub const I8: Self = Self(24);
    pub const I16: Self = Self(25);
    pub const I32: Self = Self(26);
    pub const I64: Self = Self(27);
    pub const F64: Self = Self(28);
    pub const BF16: Self = Self(30);
    pub const MXFP4: Self = Self(39);

    pub fn layout(self) -> Option<&'static TypeLayout> {
        LAYOUTS.get(self.0 as usize).and_then(|l| l.as_ref())
    }

    pub fn is_known(self) -> bool {
        self.layout().is_some()
    }

    pub fn name(self) -> Cow<'static, str> {
        match self.layout() {
            Some(l) => Cow::Borrowed(l.name),
            None => Cow::Owned(format!("UNKNOWN_{}", self.0)),
        }
    }

    /// Parses a type name as printed by [`GgmlType::name`] (case-insensitive).
    pub fn from_name(name: &str) -> Option<Self> {
        LAYOUTS
            .iter()
            .position(|l| l.is_some_and(|l| l.name.eq_ignore_ascii_case(name)))
            .map(|i| Self(i as u32))
    }

    pub fn is_quantized(self) -> bool {
        self.layout().is_some_and(|l| l.quantized)
    }

    /// Bytes needed for `n_elements` values, or `None` if the type is unknown or
    /// `n_elements` is not a whole number of blocks.
    pub fn bytes_for(self, n_elements: u64) -> Option<u64> {
        let l = self.layout()?;
        if n_elements % l.block_size != 0 {
            return None;
        }
        (n_elements / l.block_size).checked_mul(l.type_size)
    }

    /// Average storage cost in bits per weight.
    pub fn bits_per_weight(self) -> Option<f64> {
        self.layout().map(|l| l.type_size as f64 * 8.0 / l.block_size as f64)
    }

    pub fn all_known() -> impl Iterator<Item = (GgmlType, &'static TypeLayout)> {
        LAYOUTS
            .iter()
            .enumerate()
            .filter_map(|(i, l)| l.as_ref().map(|l| (GgmlType(i as u32), l)))
    }
}

impl fmt::Debug for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

impl Serialize for GgmlType {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name())
    }
}

/// Name of a llama.cpp `general.file_type` (`enum llama_ftype`) value.
pub fn file_type_name(ftype: u64) -> Option<&'static str> {
    Some(match ftype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        4 => "Q4_1_SOME_F16",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        33 => "Q4_0_4_4",
        34 => "Q4_0_4_8",
        35 => "Q4_0_8_8",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        _ => return None,
    })
}
