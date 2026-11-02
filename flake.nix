# flake.nix — bit-reproducible dev environment for miniserve.
#
# `nix develop` shells you into the SAME configuration CI tests:
#   - Python 3.11 (matches cibuildwheel + CI default)
#   - uv as the package manager; `uv.lock` (checked in) pins the resolved
#     dependency set, so every developer resolves the identical wheels
#     that CI installs from the same pinned pyproject.toml constraints
#   - CUDA toolkit (Linux only — CUDA attributes exist on darwin but the
#     toolkit build is linux-only; kernels/ is guarded for CPU-only hosts)
#   - pytest and ruff come from uv via the locked set, never from nixpkgs
#     (keeps the version pins in pyproject.toml authoritative)
#
# Pinned nixpkgs: nixos-24.05 (stable, June 2024 → Nov 2027 support).

{
  description = "miniserve — continuous-batching LLM serving engine (reproducible dev env)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      forAllSystems = f: builtins.listToAttrs
        (map (s: { name = s; value = f s; }) systems);
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            name = "miniserve-dev";

            packages = [
              (pkgs.python311.withPackages (_: [ ]))   # base interpreter; deps via uv
              pkgs.uv
            ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
              # CUDA toolchain for kernels/ (paged_attn_triton autotune, graphs).
              # Selected by nixpkgs' default CUDA attribute on 24.05.
              pkgs.cudaPackages.cuda-toolkit
            ];

            shellHook = ''
              echo "miniserve dev shell (python 3.11 + uv)"
              echo "  first use:   uv sync --extra dev"
              echo "  run tests:   uv run pytest -q"
              echo "  run gate:    uv run python bench/regression_gate.py --baseline bench/data/baseline.csv"
              [ ! -d .venv ] && uv sync --extra dev || true
            '';
          };
        });
    };
}
