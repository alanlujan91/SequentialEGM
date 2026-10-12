---
remark-version: "1.0"
remark-name: SequentialEGM
tier: 2
tags:
  - REMARK
  - Notebook
# [[[cog
# import yaml
# project = yaml.safe_load(open("myst.yml"))["project"]
# def files(toc):
#     for entry in toc:
#         if "file" in entry:
#             yield entry["file"]
#         yield from files(entry.get("children", []))
# notebooks = [f for f in files(project["toc"]) if f.startswith("code/examples/")]
# fields = {"title": project["title"], "keywords": project["keywords"], "notebooks": notebooks}
# cog.out(yaml.safe_dump(fields, sort_keys=False))
# ]]]
title: The Sequential Endogenous Grid Method
keywords:
- endogenous grid method
- dynamic programming
- interpolation
- computational economics
notebooks:
- code/examples/example_ConsumptionLabor.md
- code/examples/example_TwoAssetHousehold.md
- code/examples/example_HealthInvestment.md
- code/examples/example_ENGINEInterpolation.md
# [[[end]]]
---

# The Sequential Endogenous Grid Method

<!-- [[[cog
import pathlib
import yaml
front = pathlib.Path("content/paper/egmn_paper_only.md").read_text().split("---")[1]
cog.outl(yaml.safe_load(front)["parts"]["abstract"])
]]] -->
Heterogeneous-agent models with multiple decisions face a computational dilemma. Joint optimization over all choices is slow; strong separability restrictions speed computation but rule out economically interesting interactions. I introduce Sequential EGM (EGM$^n$), which eases this tension by decomposing the problem into sequential stages so that each stage can exploit the Endogenous Grid Method's closed-form inversion, chaining speed gains across decisions wherever separable utility or an invertible transition permits it. The resulting curvilinear grids require interpolation that respects their structure, so I develop ENGINE (ENdogenous Grid INterpolation and Extrapolation), which exploits the index structure inherited from exogenous grids, with no construction step and no triangulation. In the canonical liquid-illiquid household block, EGM$^n$ solves an order of magnitude faster than the nested method in serial and $3.7$ to $7.9$ times faster at sixteen threads. Replacing existing curvilinear interpolation with ENGINE cuts full-solve time $2.8$- to $4.5$-fold at matched accuracy.
<!-- [[[end]]] -->

## Reproduction

- `reproduce.sh` regenerates every table, figure and quoted reading of the paper and
  online appendix into `content/figures/`, one command per row of the exhibit table in
  `code/README.md`.
- `reproduce_min.sh` runs the test suite and builds the website with every example
  notebook executed, in a few minutes.

Both scripts install the locked environment with `uv sync --locked` first.

## Reference

Lujan, Alan. "The Sequential Endogenous Grid Method." Working paper. See `CITATION.cff`.
