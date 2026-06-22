from cartan_viz import io, diagnostics


def test_series_loads_three_frames_with_times(run_dir):
    s = io.Series(run_dir / "frames.pvd")
    assert len(s) == 3
    assert list(s.times) == [0.0, 0.1, 0.2]
    f = s[1]
    assert abs(f.time - 0.1) < 1e-12
    assert "B" in f.mesh.cell_data


def test_diagnostics_csv_roundtrip(run_dir):
    d = diagnostics.load(run_dir / "diagnostics.csv")
    assert "energy" in d.columns
    assert len(d.time) == 3
    assert abs(d.column("energy")[0] - 1.0) < 1e-9
