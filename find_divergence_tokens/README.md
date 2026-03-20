# Find divergence tokens
Code to find divergence tokens as defined in ["Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer" (https://arxiv.org/abs/2509.23886)](https://arxiv.org/abs/2509.23886)

## installation

```bash
git clone https://github.com/mmulet/find_divergence_tokens.git
```
or if you are already in a git repo

```bash
git submodule add https://github.com/mmulet/find_divergence_tokens.git
```

then install the package with

```bash
uv add ./find_divergence_tokens
```

## Usage
See `./tests/test_all.py` for an example of how to use the functions in this package to find divergence tokens. Along with `./test/out_test/` to see some example output.