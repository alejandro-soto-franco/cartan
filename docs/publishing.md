# Publishing cartan to crates.io

Internal developer reference. Not committed to the repo (see .gitignore).

---

## Pre-publish Checklist

Run through this list in order before every publish.

### 1. Code quality

- [ ] All tests pass: `cargo test --workspace`
- [ ] No clippy warnings: `cargo clippy --workspace -- -D warnings`
- [ ] No broken doc links: `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --workspace --no-deps`
- [ ] Formatting clean: `cargo fmt --check`

### 2. Metadata (each crate's Cargo.toml)

Every crate that gets published needs these fields set correctly:

```toml
[package]
name = "cartan-core"         # exact crate name
description = "..."          # one-line, shown on crates.io search results
version.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
keywords = [...]             # max 5, used for crates.io search
categories = [...]           # from crates.io category list

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

The `[package.metadata.docs.rs]` block controls how docs.rs builds the documentation.
Without it, docs.rs uses default settings which may miss feature-gated items.

### 3. docs.rs attribute guards (if features are added later)

If any item is behind a feature flag, annotate it:

```rust
#[cfg_attr(docsrs, doc(cfg(feature = "my-feature")))]
```

This makes docs.rs show a badge indicating which feature enables the item.
Not needed currently (no features defined yet), but add this before adding any.

### 4. README accuracy

- [ ] Manifolds table reflects current implementation status
- [ ] Quick start example compiles (`cargo test --doc`)
- [ ] No references to planned features as if they are done
- [ ] No broken links

### 5. Version bump

All crates in the workspace share a version (set in `[workspace.package]`).
To bump:

```toml
# In root Cargo.toml
[workspace.package]
version = "0.1.1"   # bump this
```

All member crates inherit this automatically via `version.workspace = true`.

### 6. Changelog

Update `CHANGELOG.md` (create it if it does not exist yet) with the new version:

```markdown
## [0.1.1] - 2026-MM-DD
### Added
- ...
### Fixed
- ...
### Changed
- ...
```

---

## Publish Order

Crates must be published in dependency order. Cargo will error if a dependency
is not yet on crates.io when you try to publish a dependent crate.

```
1. cartan-core          (no cartan deps)
2. cartan-nalgebra      (depends on cartan-core)
3. cartan-manifolds     (depends on cartan-core, cartan-nalgebra)
4. cartan-dec           (depends on cartan-core, cartan-manifolds)
5. cartan-optim         (depends on cartan-core, cartan-manifolds)
6. cartan-geo           (depends on cartan-core, cartan-manifolds)
7. cartan               (facade, depends on all of the above)
```

Publish each crate individually:

```bash
cd cartan-core   && cargo publish
cd ../cartan-nalgebra && cargo publish
cd ../cartan-manifolds && cargo publish
cd ../cartan-dec && cargo publish
cd ../cartan-optim && cargo publish
cd ../cartan-geo && cargo publish
cd ../cartan && cargo publish
```

Wait a few seconds between each publish. crates.io indexes new versions with a
short delay, and subsequent crates in the chain need the dependency to be
available on the index before publishing.

### Dry run first

Always dry-run before the real publish:

```bash
cargo publish --dry-run -p cartan-core
```

This catches metadata errors, missing files, and dependency resolution issues
without actually uploading anything.

---

## crates.io Token Setup

1. Log in to crates.io with your GitHub account.
2. Go to Account Settings > API Tokens > New Token.
3. Name it (e.g., "cartan-publish") and generate.
4. Store it: `cargo login <token>` or set `CARGO_REGISTRY_TOKEN` in the environment.

The token is stored in `~/.cargo/credentials.toml`. Do not commit this file.

---

## After Publishing

### docs.rs

docs.rs builds automatically within a few minutes of each publish.
The URL is `https://docs.rs/cartan/` (for the facade crate).
Per-crate: `https://docs.rs/cartan-core/`, `https://docs.rs/cartan-dec/`, etc.

If the build fails on docs.rs, check the build log at:
`https://docs.rs/crate/cartan/latest/builds`

Common failure causes:
- Missing `[package.metadata.docs.rs]` (usually just a warning, not a failure)
- Feature-gated dependencies that docs.rs cannot resolve
- Proc macros that require a nightly compiler (not an issue for cartan currently)

### cartan-docs rustdoc pipeline

The cartan-docs site fetches rustdoc at build time from the gh-pages branch of
the cartan repo. See docs/cartan-docs-alignment.md for the full pipeline.

After publishing a new cartan version, trigger a redeploy of cartan-docs on
Vercel to pick up the updated rustdoc output.

---

## Yanking a Version

If a published version has a critical bug:

```bash
cargo yank --version 0.1.0 -p cartan-core
```

This does not delete the crate but marks it as not recommended for new installs.
Existing users pinned to that version are not affected.

Yank all crates in the workspace that share the broken version. Then publish
a patched version following the normal publish order above.
