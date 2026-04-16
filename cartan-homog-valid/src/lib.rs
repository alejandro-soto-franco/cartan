//! Validation harness for `cartan-homog`: NPZ fixtures generated from ECHOES,
//! affine-invariant tolerance macros, and the capstone fractured-sandstone pipeline test.

pub mod fixture;
pub mod approx;

#[cfg(test)]
mod smoke {
    #[test]
    fn harness_compiles() {
        assert!(true);
    }

    #[test]
    fn fixture_loader_empty_dir_returns_empty() {
        let tmp = std::env::temp_dir().join("cartan-homog-valid-empty-test");
        let _ = std::fs::create_dir_all(&tmp);
        let fixtures = crate::fixture::Fixture::load_all(&tmp);
        assert!(fixtures.is_empty());
    }

    #[test]
    fn approx_spd_zero_distance_for_equal() {
        let a = nalgebra::Matrix3::<f64>::identity() * 3.0;
        let b = a;
        let d = crate::approx::ai_distance_order2(&a, &b).unwrap();
        assert!(d < 1e-12);
    }
}
