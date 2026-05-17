use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=vkfft_shim.c");
    println!("cargo:rerun-if-changed=vendor/VkFFT/vkFFT/vkFFT.h");
    println!("cargo:rerun-if-changed=build.rs");

    let vkfft_include = PathBuf::from("vendor/VkFFT/vkFFT");
    let vulkan_include = pkg_config::Config::new()
        .atleast_version("1.3")
        .probe("vulkan")
        .expect("Vulkan SDK (vulkan-headers + vulkan-loader-devel) not found via pkg-config");

    let glslang_include_candidates = [
        PathBuf::from("/usr/include/glslang/Include"),
        PathBuf::from("/usr/local/include/glslang/Include"),
    ];
    let glslang_include = glslang_include_candidates
        .iter()
        .find(|p| p.join("glslang_c_interface.h").exists())
        .expect("glslang_c_interface.h not found; install glslang-devel (Fedora) or equivalent");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Step 1: compile vkfft_shim.c to an object file.
    let obj_path = out_dir.join("vkfft_shim.o");
    let mut include_args: Vec<String> = vec![
        "-I.".into(),
        format!("-Ivendor/VkFFT"),
        format!("-I{}", vkfft_include.display()),
        format!("-I{}", glslang_include.display()),
        "-DVKFFT_BACKEND=0".into(),
    ];
    for p in &vulkan_include.include_paths {
        include_args.push(format!("-I{}", p.display()));
    }

    let status = Command::new("cc")
        .args(&["-c", "-fPIC", "-O2", "-std=c11"])
        .args(&include_args)
        .args(&["-w"]) // suppress warnings
        .arg("vkfft_shim.c")
        .arg("-o")
        .arg(&obj_path)
        .status()
        .expect("failed to run cc");
    assert!(status.success(), "cc failed to compile vkfft_shim.c");

    // Step 2: link the .o + static glslang + dynamic SPIRV-Tools + vulkan
    // into a single shared library. This resolves all C++ symbols at .so
    // link time, avoiding ABI mismatch when the Rust binary loads.
    let lib_path = out_dir.join("libvkfft_bundle.so");
    let status = Command::new("c++")
        .arg("-shared")
        .arg("-fPIC")
        .arg(&obj_path)
        .arg("-o")
        .arg(&lib_path)
        // Static glslang libs (absorbed into the .so)
        .args(&["-Wl,--whole-archive", "-Wl,--allow-multiple-definition"])
        .args(&[
            "-lglslang",
            "-lMachineIndependent",
            "-lGenericCodeGen",
            "-lOSDependent",
            "-lSPIRV",
            "-lglslang-default-resource-limits",
        ])
        .args(&["-Wl,--no-whole-archive"])
        // Dynamic deps
        .args(&["-lSPIRV-Tools-shared", "-lSPIRV-Tools-opt", "-lvulkan"])
        .arg(format!("-L/usr/lib64"))
        .status()
        .expect("failed to run c++ for shared lib");
    assert!(status.success(), "c++ failed to create libvkfft_bundle.so");

    // Tell cargo to link our shared lib and embed rpath so the runtime linker finds it
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=dylib=vkfft_bundle");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());

    // Bindgen
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
        .allowlist_type("VkFFTApplication")
        .allowlist_type("VkFFTResult.*")
        .allowlist_function("cartan_vkfft_.*")
        .layout_tests(false)
        .derive_default(true)
        .wrap_unsafe_ops(true)
        .generate()
        .expect("bindgen");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings");
}
