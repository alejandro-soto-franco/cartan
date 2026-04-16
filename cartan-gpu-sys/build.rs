use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=vendor/VkFFT/vkFFT/vkFFT.h");
    println!("cargo:rerun-if-changed=build.rs");

    let vkfft_include = PathBuf::from("vendor/VkFFT/vkFFT");
    let vulkan_include = pkg_config::Config::new()
        .atleast_version("1.3")
        .probe("vulkan")
        .expect("Vulkan SDK (vulkan-headers + vulkan-loader-devel) not found via pkg-config");

    // VkFFT's runtime shader compilation path depends on glslang's C interface.
    // Fedora puts the header at /usr/include/glslang/Include/glslang_c_interface.h
    // but VkFFT includes it unprefixed. Add the Include subdir to the search path.
    let glslang_include_candidates = [
        PathBuf::from("/usr/include/glslang/Include"),
        PathBuf::from("/usr/local/include/glslang/Include"),
    ];
    let glslang_include = glslang_include_candidates
        .iter()
        .find(|p| p.join("glslang_c_interface.h").exists())
        .expect("glslang_c_interface.h not found; install glslang-devel (Fedora) or equivalent");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // VkFFT is header-only; create a single-translation-unit shim so cc produces one object file.
    let shim_path = out_dir.join("vkfft_shim.c");
    std::fs::write(
        &shim_path,
        "#define VKFFT_BACKEND 0\n#include \"vkFFT/vkFFT.h\"\n",
    )
    .expect("write shim.c");

    let mut build = cc::Build::new();
    build
        .file(&shim_path)
        .include("vendor/VkFFT")
        .include(&vkfft_include)
        .include(glslang_include)
        .define("VKFFT_BACKEND", "0")
        .flag_if_supported("-std=c11")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-sign-compare")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-variable")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-implicit-function-declaration")
        .warnings(false);
    for p in &vulkan_include.include_paths {
        build.include(p);
    }
    build.compile("vkfft");

    // Bindgen with a minimal allowlist so we don't churn on VkFFT's large internal surface.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-Ivendor/VkFFT")
        .clang_arg("-Ivendor/VkFFT/vkFFT")
        .clang_arg(format!("-I{}", glslang_include.display()))
        .clang_arg("-DVKFFT_BACKEND=0")
        .clang_args(
            vulkan_include
                .include_paths
                .iter()
                .map(|p| format!("-I{}", p.display())),
        )
        .allowlist_type("VkFFTConfiguration")
        .allowlist_type("VkFFTApplication")
        .allowlist_type("VkFFTLaunchParams")
        .allowlist_type("VkFFTResult.*")
        .allowlist_function("initializeVkFFT")
        .allowlist_function("deleteVkFFT")
        .allowlist_function("VkFFTAppend")
        .allowlist_function("getVkFFTVersion")
        .layout_tests(false)
        .derive_default(true)
        .generate()
        .expect("bindgen");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings");
}
