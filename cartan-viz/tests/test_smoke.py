def test_fixture_has_contract(run_dir):
    assert (run_dir / "frames.pvd").exists()
    assert (run_dir / "frame_0000.vtu").exists()
    assert (run_dir / "diagnostics.csv").exists()
    text = (run_dir / "diagnostics.csv").read_text()
    assert text.splitlines()[0] == "time,energy,magnetic_flux_residual"
