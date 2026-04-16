#[derive(Clone, Debug, PartialEq)]
pub struct IntegrationOpts {
    pub lebedev_degree: usize,
    pub eps_abs: f64,
    pub eps_rel: f64,
}

impl Default for IntegrationOpts {
    fn default() -> Self {
        Self { lebedev_degree: 110, eps_abs: 1e-10, eps_rel: 1e-10 }
    }
}
