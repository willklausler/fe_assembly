#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail if llms.txt misses a public name or an example program.

Every identifier in a PUBLIC statement of a module in src/, every procedure
bound to a type there, and every example/*.f90 (verbatim) must appear in
llms.txt.
"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parent.parent
llms = (root / "llms.txt").read_text()

names = []
for path in sorted((root / "src").glob("*.f90")):
    source = path.read_text()
    for line in source.splitlines():
        match = re.match(r"\s*public\s*::\s*(.*)", line, re.IGNORECASE)
        if match:
            names += re.split(r"\s*,\s*", re.sub(r"!.*", "", match.group(1)).strip())
    # Type-bound procedures are public without a PUBLIC statement; types may carry
    # attributes (abstract, extends) before ::.
    for body in re.findall(r"^\s*type\b[^(\n]*?::.*?^\s*end\s+type", source, re.M | re.S | re.I):
        body = re.sub(r"!.*", "", body)
        names += re.findall(r"^\s*procedure\b[^:\n]*::\s*(\w+)", body, re.M | re.I)

errors = []
for name in dict.fromkeys(names):
    if re.fullmatch(r"\w+", name) and not re.search(rf"\b{name}\b", llms):
        errors.append(f"public name not in llms.txt: {name}")

for example in sorted((root / "example").glob("*.f90")):
    if example.read_text().strip() not in llms:
        errors.append(f"example not verbatim in llms.txt: {example.name}")

print("\n".join(errors) or "llms.txt covers all public names and examples")
sys.exit(1 if errors else 0)
