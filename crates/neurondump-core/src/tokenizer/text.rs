//! Raw token text → readable text: GPT-2 byte maps, `<0xNN>` byte tokens, SentencePiece spaces.

use super::*;

/// GPT-2 `bytes_to_unicode` inverse: printable stand-in char -> byte.
pub(super) fn byte_decoder() -> &'static HashMap<char, u8> {
    static M: OnceLock<HashMap<char, u8>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        let mut n = 0u32;
        for b in 0u32..256 {
            let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let c = if printable {
                b
            } else {
                n += 1;
                255 + n
            };
            if let Some(ch) = char::from_u32(c) {
                m.insert(ch, b as u8);
            }
        }
        m
    })
}

pub fn byte_encoder() -> &'static HashMap<u8, char> {
    static M: OnceLock<HashMap<u8, char>> = OnceLock::new();
    M.get_or_init(|| byte_decoder().iter().map(|(c, b)| (*b, *c)).collect())
}

pub(super) fn parse_byte_token(raw: &str) -> Option<u8> {
    let hex = raw.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 {
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// Decodes a raw vocabulary entry into readable text.
pub fn decode_raw(model: TokenizerModel, raw: &str) -> String {
    match model {
        TokenizerModel::Bpe => {
            let dec = byte_decoder();
            let mut bytes = Vec::with_capacity(raw.len());
            for ch in raw.chars() {
                match dec.get(&ch) {
                    Some(b) => bytes.push(*b),
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(e) => e.as_bytes().iter().map(|b| if b.is_ascii() { (*b as char).to_string() } else { format!("<0x{b:02X}>") }).collect(),
            }
        }
        TokenizerModel::Spm | TokenizerModel::Wpm | TokenizerModel::Other => {
            if let Some(b) = parse_byte_token(raw) {
                if b.is_ascii() {
                    return (b as char).to_string();
                }
                return raw.to_string();
            }
            raw.replace('\u{2581}', " ")
        }
    }
}
