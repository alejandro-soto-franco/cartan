# cartan-homog Roadmap

Technical roadmap for `cartan-homog`, organised by milestone. Items marked (commercial) are specifically motivated by the path toward a commercially viable reservoir petrophysics platform.

## Current state (v1.3, 2026-04-16)

10 mean-field schemes + full-field DEC solver + stochastic ensembles + ECHOES validation. 200x median speedup over ECHOES. 598 workspace tests, CI green. Validated on Berea sandstone published parameters.

## v1.4: shape coverage + solver depth

- [ ] Higher-order Lebedev grids (degrees 26, 50, 110, 194) for crack-limit accuracy in anisotropic reference media
- [ ] Spheroid / Ellipsoid Order4 Hill via 4D surface quadrature (currently skipped in 42/126 ECHOES cases)
- [ ] Recursive multilevel AMG (current two-level stalls on high-contrast periodic problems, falling through to dense LU)
- [ ] Red-green adaptive tet refinement with hanging-node closure (current red refinement is uniform only)
- [ ] `cartan-py` Python binding for `homogenize_voxel` and `FullField::homogenize`

## v2.0: production-grade full-field solver (commercial)

The gap between the 27 GPa sphere-pore prediction and the 6.6 GPa measured Berea stiffness is the central obstacle to commercial viability. Closing it requires full-field solves on real micro-CT data at production resolution.

- [ ] **GPU-accelerated iterative solver**: target 256-cubed segmented micro-CT image (16M DOFs) in < 60 seconds on a single workstation GPU. The `cartan-gpu-sys` VkFFT work provides the FFT backend; the solver needs a GPU-resident CG with AMG V-cycle on the device. This is the critical-path item for competing with Avizo/PerGeos.
- [ ] **Parallel assembly via rayon**: the current P1-FEM stiffness assembly is sequential. With ~100M tets at 256-cubed, parallel assembly saves minutes.
- [ ] **Image segmentation integration**: accept TIFF stacks and HDF5 volumes from standard micro-CT scanners (Bruker, Zeiss Xradia, Thermo Fisher). Phase segmentation (threshold, watershed, ML-based) as a preprocessing step, not in-library.
- [ ] **Lab calibration workflow**: given measured K_eff from a core-plug experiment, invert for microstructural parameters (crack density, grain-contact compliance) that reproduce the measurement. Inverse homogenisation via cartan-optim on SPD(N).

## v2.1: multi-scale upscaling (commercial)

Reservoir simulation operates at the metre-to-kilometre scale. Core plugs are centimetre-scale. Micro-CT is micrometre-scale. The commercial value is in bridging all three.

- [ ] **Core-plug to well-log upscaling**: given a set of micro-CT-derived effective properties at core-plug locations along a well, produce a continuous property profile at the well-log scale using SPD geodesic interpolation (already available via `cartan-manifolds::Spd<N>`).
- [ ] **Geostatistical random fields on SPD**: replace pointwise-iid Wishart with spatially correlated tensor fields via Karhunen-Loeve expansion on SPD(N). Input: variogram parameters from well-log analysis. Output: N realisations of the full 3D permeability tensor field.
- [ ] **Reservoir-grid output**: export effective properties in corner-point GRDECL format for Eclipse/OPM-Flow.

## v2.2: stochastic reservoir UQ platform (commercial, target product)

The commercially differentiated offering: stochastic reservoir uncertainty quantification at the RVE level.

**Pitch**: given a mean porosity/permeability model from well logs, generate N realisations of the microstructure (via spatial KL on SPD), homogenise each one (200x faster than ECHOES makes N = 10,000 feasible in minutes), propagate through the macroscale Darcy solve, deliver posterior distributions over effective reservoir properties with confidence intervals.

- [ ] **Spatial KL expansion on SPD**: correlated random tensor fields, not pointwise-iid Wishart. This is the foundational primitive.
- [ ] **GPU-accelerated Darcy solve**: the macroscale slab solve needs to be fast enough to run 10,000 times in an ensemble. With the v2.0 GPU solver this is < 1 hour on commodity hardware.
- [ ] **Python API via cartan-py**: the tool must integrate into existing Jupyter/Petrel workflows. Expose `WishartRveEnsemble`, `FullField::homogenize_voxel`, `SlabProblem`, and the spatial KL sampler.
- [ ] **Reporting and audit trail**: JSON + Parquet output with per-realisation metadata. Regulatory / reserves-estimation workflows need provenance.

## v3.0: pressure-dependent and saturated-rock physics (commercial)

- [ ] **Pressure-dependent crack closure**: David-Zimmerman 2012 model. Effective moduli vary with confining pressure as microcracks close. The v1.3 `PennyCrack` shape with a pressure-dependent density parameter is the foundation.
- [ ] **Fluid substitution**: Gassmann's equation for saturated vs dry rock. Given dry effective moduli + fluid bulk modulus + porosity, predict saturated moduli. Standard in well-log interpretation.
- [ ] **Biot poroelasticity**: couple the homogenised stiffness with pore-pressure loading to predict the Biot coefficient tensor (concentration-tensor route already available via `SchemeOpts::store_concentration`).
- [ ] **Viscoelastic extension**: ECHOES supports aging linear viscoelasticity and complex moduli. A cartan equivalent would extend `TensorOrder` to complex-valued Kelvin-Mandel matrices.

## Non-goals

- **Full digital-rock-physics platform**: CT image acquisition, segmentation UI, core-sample tracking database. These are application-layer concerns for a 10-person company with geoscientists and a sales team. cartan-homog is a solver backend.
- **Lattice Boltzmann permeability solver**: LB is the standard for absolute permeability from voxel images (Andra et al. 2013 benchmarks). cartan's DEC/FEM approach targets the elastic-moduli and anisotropic-permeability use cases where LB is not applicable.

## References

- Andra et al. 2013. Digital rock physics benchmarks (Parts I and II). Computers and Geosciences 50.
- Arns et al. 2002. Computation of linear elastic properties from micro-CT images. Computational Geosciences 6.
- David, Zimmerman. 2012. Pore structure model for elastic wave velocities in fluid-saturated sandstones. J. Geophys. Res. 117.
- Hart, Wang. 1995. Complete poroelastic moduli of Berea and Indiana. J. Geophys. Res. 100.
- Hashin, Shtrikman. 1963. A variational approach to the elastic behaviour of multiphase materials. J. Mech. Phys. Solids 11.
- Milton. 2002. The Theory of Composites. Cambridge University Press.
- Zimmerman. 1991. Compressibility of Sandstones. Elsevier.
