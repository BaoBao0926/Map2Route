# Vendored LIMP Components

This directory is reserved for isolated official LIMP method-level code that can
be reused without the original robot runtime, perception stack, Open3D scene
pipeline, or notebooks.

Official-code inspection notes:

- `robotlimp/limp/language/temporal_logic/ltl_progression.py` contains a
  mostly isolated co-safe LTL progression implementation over tuple formulas.
- `robotlimp/limp/language/temporal_logic/dfa.py` wraps that progression into a
  DFA, but imports through the original `limp` package namespace.
- `robotlimp/limp/utils/gen_utils.py` handles parts of the string-LTL / Spot
  conversion path, but imports `spot`, planner modules, and LLM helpers at top
  level.

For the current SemPathBench adapter, main runs require a formal backend and
fail closed with `AUTOMATON_BACKEND_UNAVAILABLE` when it is missing. The local
`residual_fallback` backend is only for smoke tests and must be explicitly
enabled; it is not vendored official LIMP progression.
