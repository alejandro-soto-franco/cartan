#[derive(Clone, Debug, PartialEq)]
pub struct IntegrationOpts {
    pub lebedev_degree: usize,
    pub eps_abs: f64,
    pub eps_rel: f64,
}

impl Default for IntegrationOpts {
    fn default() -> Self {
        // Degree 14 is what v1.3 ships; 26/50/110/194 are v1.4 when we need
        // crack-limit accuracy (small-aspect spheroids in anisotropic ref).
        Self { lebedev_degree: 14, eps_abs: 1e-10, eps_rel: 1e-10 }
    }
}
