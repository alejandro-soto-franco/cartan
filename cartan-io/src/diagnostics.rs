//! CSV diagnostics writer for time series recording.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// A CSV writer for diagnostics data (time series metrics).
pub struct DiagnosticsCsv {
    writer: BufWriter<File>,
    ncols: usize,
}

impl DiagnosticsCsv {
    /// Create a new CSV file with the given column headers.
    pub fn new(path: &Path, columns: &[&str]) -> std::io::Result<DiagnosticsCsv> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let header = columns.join(",");
        writeln!(writer, "{}", header)?;
        Ok(DiagnosticsCsv {
            writer,
            ncols: columns.len(),
        })
    }

    /// Write a row of values (must have exactly `ncols` elements).
    pub fn push_row(&mut self, row: &[f64]) -> std::io::Result<()> {
        assert_eq!(
            row.len(),
            self.ncols,
            "Row has {} values but expected {}",
            row.len(),
            self.ncols
        );
        let line = row
            .iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(self.writer, "{}", line)?;
        Ok(())
    }

    /// Flush and close the CSV file.
    pub fn finish(mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_header_and_rows() {
        let dir = std::env::temp_dir().join("cartan_diag_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diagnostics.csv");
        let mut d =
            DiagnosticsCsv::new(&path, &["time", "energy", "magnetic_flux_residual"]).unwrap();
        d.push_row(&[0.0, 1.5, 0.0]).unwrap();
        d.push_row(&[0.1, 1.4, 1e-13]).unwrap();
        d.finish().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "time,energy,magnetic_flux_residual");
        assert_eq!(lines.len(), 3);
        assert!(lines[1].starts_with("0"));
    }
}
