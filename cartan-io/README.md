# cartan-io

VTK, ParaView and Blender export for cartan.

[![crates.io](https://img.shields.io/crates/v/cartan-io.svg)](https://crates.io/crates/cartan-io)
[![docs.rs](https://docs.rs/cartan-io/badge.svg)](https://docs.rs/cartan-io)

Part of the [cartan](https://crates.io/crates/cartan) workspace.

## Overview

`cartan-io` writes meshes and fields in formats other tools already read, so a
simulation can be inspected without a bespoke viewer.

| What | Format |
|---|---|
| Meshes and fields | VTK `UnstructuredGrid` and `PolyData` |
| Time series | ParaView `.pvd` collections |
| Vertex animation | Blender MDD |
| FEEC fields | reconstruction from cochains for export |

The writer is dimension-generic, so the same call handles a triangle mesh in
R^3 and a tetrahedral mesh in R^4.

## Example

```rust,no_run
use cartan_io::Field;

// Fields carry their own name and arity, so the writer needs no schema.
let temperature = Field::Scalar {
    name: "temperature".into(),
    values: vec![1.0, 2.0, 3.0],
};

// A director has no head or tail. Marking it nematic tells the renderer to
// draw it double-ended rather than as an arrow.
let director = Field::Vector {
    name: "director".into(),
    values: vec![1.0, 0.0, 0.0],
    nematic: true,
};
```

## Frames

Interpolated field values come out in the **ambient** frame, not in a
per-simplex local frame. VTK has no notion of a chart, so a vector must be
expressed in the embedding space to be drawn correctly. Convert after reading
if you need intrinsic components.

## Recording a run

`cartan_io::run` writes a directory of timesteps plus a `.pvd` index, which
ParaView opens as a single time series. See
`cartan-maxwell/examples/maxwell_record.rs` for a worked use.

## License

[MIT](LICENSE-MIT)
