{
  description = "Data Forecast Generator - Development Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        lib = nixpkgs.lib;
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfreePredicate =
            pkg:
            builtins.elem (lib.getName pkg) [
              "github-copilot-cli"
            ];
        };
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Keep shell lean; install project deps with uv from pyproject.toml
            python
            python312Packages.pip
            python312Packages.virtualenv

            # Build and dependency management
            uv

            # Linting and formatting
            ruff
            mypy
            python312Packages.pytest

            # Development tools
            git
            zsh
            github-cli
            github-copilot-cli

            # Utilities
            curl
            jq
          ];

          shellHook = ''
            export SHELL="${pkgs.zsh}/bin/zsh"

            echo "🚀 Data Forecast Generator - Development Environment"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version 2>/dev/null || echo 'not in PATH')"
            echo "ruff: $(ruff --version 2>/dev/null || echo 'not in PATH')"
            echo "shell: $SHELL"
            echo ""
            echo "Quick commands:"
            echo "  uv sync --extra dev --no-install-project  # Install dependencies"
            echo "  pytest                                    # Run tests"
            echo "  ruff check scripts tests                  # Lint code"
            echo "  ruff format scripts tests                 # Format code"
            echo ""
          '';
        };
      }
    );
}
