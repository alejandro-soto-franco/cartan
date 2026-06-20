//! Shared VTK XML inline-binary encoders.
use base64::Engine;

pub fn encode_f64_le(data: &[f64]) -> String {
    let nbytes = (data.len() * 8) as u64;
    let mut buf = Vec::with_capacity(8 + data.len() * 8);
    buf.extend_from_slice(&nbytes.to_le_bytes());
    for &x in data { buf.extend_from_slice(&x.to_le_bytes()); }
    base64::engine::general_purpose::STANDARD.encode(&buf)
}

pub fn encode_i64_le(data: &[i64]) -> String {
    let nbytes = (data.len() * 8) as u64;
    let mut buf = Vec::with_capacity(8 + data.len() * 8);
    buf.extend_from_slice(&nbytes.to_le_bytes());
    for &x in data { buf.extend_from_slice(&x.to_le_bytes()); }
    base64::engine::general_purpose::STANDARD.encode(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encodes_nonempty() {
        let s = encode_f64_le(&[1.0, 2.0]);
        assert!(!s.is_empty());
        let i = encode_i64_le(&[0, 1, 2]);
        assert!(!i.is_empty());
    }
}
