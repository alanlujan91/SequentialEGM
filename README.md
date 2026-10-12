# The Sequential Endogenous Grid Method

[Paper (PDF)](content/public/SequentialEGMn.pdf)

---

## Author

**Alan Lujan**<sup>a,b,*</sup>

<sup>a</sup> Johns Hopkins University, Krieger School of Arts and Sciences, Washington, DC, USA

<sup>b</sup> Econ-ARK

<sup>*</sup> Corresponding author. E-mail: alujan@jhu.edu

ORCID: [0000-0002-5289-7054](https://orcid.org/0000-0002-5289-7054)

---

## Abstract

<!-- [[[cog
import pathlib
import yaml
front = pathlib.Path("content/paper/egmn_paper_only.md").read_text().split("---")[1]
cog.outl(yaml.safe_load(front)["parts"]["abstract"])
]]] -->
Heterogeneous-agent models with multiple decisions face a computational dilemma. Joint optimization over all choices is slow; strong separability restrictions speed computation but rule out economically interesting interactions. I introduce Sequential EGM (EGM$^n$), which eases this tension by decomposing the problem into sequential stages so that each stage can exploit the Endogenous Grid Method's closed-form inversion, chaining speed gains across decisions wherever separable utility or an invertible transition permits it. The resulting curvilinear grids require interpolation that respects their structure, so I develop ENGINE (ENdogenous Grid INterpolation and Extrapolation), which exploits the index structure inherited from exogenous grids, with no construction step and no triangulation. In the canonical liquid-illiquid household block, EGM$^n$ solves an order of magnitude faster than the nested method in serial and $3.7$ to $7.9$ times faster at sixteen threads. Replacing existing curvilinear interpolation with ENGINE cuts full-solve time $2.8$- to $4.5$-fold at matched accuracy.
<!-- [[[end]]] -->

## Keywords

Endogenous grid method, dynamic programming, interpolation, computational economics

## JEL Classification

C61, C63, E21

---

## Acknowledgements

<!-- [[[cog
import pathlib
import re
import yaml
front = pathlib.Path("content/paper/egmn_paper_only.md").read_text().split("---")[1]
ack = yaml.safe_load(front)["parts"]["acknowledgement"]
cog.outl(re.sub(r"\s*\{cite:[a-z]+\}`[^`]+`", "", ack))
]]] -->
I thank the editors, Thomas Lubik and Eric Young, and three anonymous referees at the *Journal of Economic Dynamics and Control*, whose reports substantially improved and streamlined the exposition of this work. I thank Chris Carroll, Matthew White, and Fedor Iskhakov for their helpful comments and suggestions. The remaining errors are my own. This paper was awarded the [Outstanding Graduate Student Paper Award](https://comp-econ.com/contest-and-prize/) at the 29th International Conference on Computing in Economics and Finance (CEF 2023, Nice, France). Presentations at the Johns Hopkins University Macro Brownbag (2024), the [International Association for Applied Econometrics annual conference](https://editorialexpress.com/conference/IAAE2024Greece/program/IAAE2024Greece.html) (2024, Thessaloniki, Greece), and the [University of Texas at Austin MA in Economics Program 10-Year Reunion Celebration](https://ma.eco.utexas.edu/10-year-reunion-celebration-invitation/) (2024) improved the exposition and applications. During the development of this work, I received support through the Econ-ARK project, funded by a grant from the [Alfred P. Sloan Foundation](https://sloan.org/grant-detail/g-2017-9832). All figures and other numerical results were produced using the [Econ-ARK HARK](https://econ-ark.org/) toolkit.
<!-- [[[end]]] -->

---

## Statements

### Declaration of Competing Interests

The author declares no competing interests.

### Funding

This work was supported by the Alfred P. Sloan Foundation through the Econ-ARK project [grant number g-2017-9832].

### Data and Code Availability

No datasets were generated or analyzed during this study. All code implementing the Sequential EGM method and ENGINE interpolation algorithm is available at https://github.com/alanlujan91/sequential_egm. The methods are also implemented in the open-source [Econ-ARK/HARK](https://github.com/econ-ark/HARK) toolkit for heterogeneous agent modeling.

### Declaration of Generative AI Use

<!-- [[[cog
import pathlib
import yaml
front = pathlib.Path("content/paper/egmn_paper_only.md").read_text().split("---")[1]
cog.outl(yaml.safe_load(front)["parts"]["declaration"])
]]] -->
During the preparation of this work, the author used Claude Code (Anthropic) to edit and proofread the manuscript and assist with coding. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.
<!-- [[[end]]] -->

---

## Key Points

<!-- [[[cog
import pathlib
import yaml
front = pathlib.Path("content/paper/egmn_paper_wp.md").read_text().split("---")[1]
for point in yaml.safe_load(front)["parts"]["keypoints"]:
    cog.outl(f"- {point}")
]]] -->
- Splitting a decision into stages lets each qualifying stage invert its first-order condition in closed form.
- On the two-asset household block, EGM$^n$ solves about thirteen times faster in serial than a matched nested-EGM implementation.
- Swapping in ENGINE for the curvilinear interpolation method cuts full-solve time by a factor of roughly three to four and a half.
<!-- [[[end]]] -->

---

## Citation

If you use this software, please cite:

```bibtex
@article{Lujan2026segm,
  title    = {The Sequential Endogenous Grid Method},
  author   = {Lujan, Alan},
  year     = 2026
}
```

---

## Requirements

- Python >= 3.12
- Key dependencies: [econ-ark](https://github.com/econ-ark/HARK), numba, numpy, scipy, matplotlib

## Installation

```bash
git clone https://github.com/alanlujan91/sequential_egm.git
cd sequential_egm
uv sync
```

## Reproducing the Results

```bash
./reproduce.sh       # every table, figure and quoted reading, into content/figures/
./reproduce_min.sh   # the test suite and the website with every notebook executed
```

`reproduce.sh` runs the exhibit table of `code/README.md` row by row. Its timing
benchmarks measure the machine they run on, so speedups vary from run to run.

---

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| [Consumption-Labor Model](code/examples/example_ConsumptionLabor.md) | Consumption-labor choice, extended with a risky portfolio share |
| [Two-Asset Household](code/examples/example_TwoAssetHousehold.md) | The liquid-illiquid household block in closed form, and one row of the NEGM comparison rerun |
| [Health Investment Model](code/examples/example_HealthInvestment.md) | HARK's baseline solution, the Sequential EGM stages with ENGINE, and their comparison |
| [ENGINE Interpolation](code/examples/example_ENGINEInterpolation.md) | The two mesh conditions, both passes by hand, and the regrid |

---

## Repository Structure

- `content/paper/` - Paper source files (MyST Markdown)
- `content/public/` - Generated PDF and figures
- `code/` - Implementation and examples
- `code/examples/` - Executable MyST Markdown notebooks demonstrating the methods
- `content/references.bib` - Bibliography

## Building the Paper

```bash
make pdf          # working paper and online appendix
make submission   # also the journal paper
```

A bare `myst build --pdf` skips the cross-document repair step, so use the make targets (see `code/README.md`).

## License

<!-- [[[cog
import pathlib
import yaml
project = yaml.safe_load(pathlib.Path("myst.yml").read_text())["project"]
cog.outl(f"- Content: {project['license']['content']}")
cog.outl(f"- Code: {project['license']['code']} (see [LICENSE](LICENSE))")
cog.outl()
cog.outl("## Copyright")
cog.outl()
cog.outl(project["copyright"])
]]] -->
- Content: CC-BY-NC-ND-4.0
- Code: MIT (see [LICENSE](LICENSE))

## Copyright

© 2026 Alan Lujan
<!-- [[[end]]] -->
