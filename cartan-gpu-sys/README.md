# cartan-gpu-sys

Raw FFI bindings to the vendored VkFFT C++ library.

[![crates.io](https://img.shields.io/crates/v/cartan-gpu-sys.svg)](https://crates.io/crates/cartan-gpu-sys)
[![docs.rs](https://docs.rs/cartan-gpu-sys/badge.svg)](https://docs.rs/cartan-gpu-sys)

Part of the [cartan](https://crates.io/crates/cartan) workspace.
**Internal plumbing for `cartan-gpu` — downstream cartan crates should
not depend on this crate directly.**

## What this crate does

`cartan-gpu-sys` vendors VkFFT (currently pinned to
[v1.3.4](https://github.com/DTolm/VkFFT/releases/tag/v1.3.4)) as a git
submodule and exposes a minimal, FFI-friendly C wrapper around it. The
upstream VkFFT API uses pointer-to-handle conventions and `static inline`
helpers that are hostile to direct `bindgen`; the wrappers in
`wrapper.h` / `vkfft_shim.c` accept plain `uint64_t` Vulkan handles and
do all of the pointer gymnastics on the C side.

Public surface (after `bindgen`):

- `cartan_vkfft_version()` — returns VkFFT's packed runtime version,
  also exposed safely as `vkfft_runtime_version() -> u64`.
- `cartan_vkfft_plan(app, backing, …)` — compile a 1D/2D/3D plan
  against a caller-allocated `CartanVkFftBacking` struct that holds
  stable storage for VkFFT's pointer-to-handle config fields.
- `cartan_vkfft_exec(app, …, buffer, inverse)` — record VkFFTAppend
  into a private command buffer, submit, wait, free. The buffer is
  passed by handle so callers can override the plan's default buffer
  per call (used by `cartan-gpu::SharedFftBuffer` to operate on
  externally-allocated memory).
- `cartan_vkfft_delete(app)` — release VkFFT-side resources.

`build.rs` compiles `vkfft_shim.c` to a `.so` together with the static
glslang libraries and SPIRV-Tools, then runs `bindgen` over `wrapper.h`.

## Build prerequisites

- Vulkan SDK with headers (Fedora: `vulkan-headers`, `vulkan-loader-devel`)
- glslang devel package (Fedora: `glslang-devel`)
- SPIRV-Tools shared libraries (Fedora: `SPIRV-Tools`)
- A C++ toolchain capable of building VkFFT (gcc / clang)
- `pkg-config`

## Layout

```
cartan-gpu-sys/
  build.rs           # compiles vkfft_shim.c + runs bindgen
  wrapper.h          # public C surface (bindgen reads this)
  vkfft_shim.c       # the wrapper implementation
  vendor/VkFFT/      # git submodule, pinned tag v1.3.4
  src/lib.rs         # include!(concat!(env!("OUT_DIR"), "/bindings.rs"))
```

## License

[MIT](../LICENSE-MIT) for the Rust+shim code. VkFFT is MIT-licensed
upstream; see `vendor/VkFFT/LICENSE`.
