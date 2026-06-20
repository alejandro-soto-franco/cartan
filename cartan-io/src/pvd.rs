use std::io::Write;
use std::path::Path;

pub fn write_pvd(path: &Path, entries: &[(f64, String)]) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, r#"<?xml version="1.0"?>"#)?;
    writeln!(f, r#"<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">"#)?;
    writeln!(f, r#"  <Collection>"#)?;
    for (t, file) in entries {
        writeln!(f, r#"    <DataSet timestep="{t}" group="" part="0" file="{file}"/>"#)?;
    }
    writeln!(f, r#"  </Collection>"#)?;
    writeln!(f, r#"</VTKFile>"#)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pvd_has_entries() {
        let tmp = std::env::temp_dir().join("ff_cartan.pvd");
        write_pvd(&tmp, &[(0.0, "m_0000.vtp".into())]).unwrap();
        let body = std::fs::read_to_string(&tmp).unwrap();
        assert!(body.contains(r#"file="m_0000.vtp""#));
        assert!(body.contains("Collection"));
    }
}
