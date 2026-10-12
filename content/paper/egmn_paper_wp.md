---
title: "The Sequential Endogenous Grid Method"
short_title: "Sequential EGM"
tags:
  - C61
  - C63
  - E21
exports:
  - id: paper-wp-pdf
    format: typst
    template: https://github.com/econ-ark/econ-ark-myst/archive/c4f87653d13125c9ada8324b7fee59fd82a04632.zip
    output: ../public/SequentialEGMn_wp.pdf
    subject: Working paper
    # The two three-panel figures span the margin rail too, so their labels stay legible.
    wide_figures: fig:consumption-labor, fig:partition
  - format: cff
    output: ../../CITATION.cff
parts:
  abstract: >-
    Heterogeneous-agent models with multiple decisions face a computational
    dilemma. Joint optimization over all choices is slow; strong separability
    restrictions speed computation but rule out economically interesting
    interactions. I introduce Sequential EGM (EGM$^n$), which eases this
    tension by decomposing the problem into sequential stages so that each stage
    can exploit the Endogenous Grid Method's closed-form inversion, chaining speed
    gains across decisions wherever separable utility or an invertible
    transition permits it. The resulting curvilinear grids require
    interpolation that respects their structure, so I develop ENGINE
    (ENdogenous Grid INterpolation and Extrapolation), which exploits the
    index structure inherited from exogenous grids, with no construction step
    and no triangulation. In the canonical liquid-illiquid household block,
    EGM$^n$ solves an order of magnitude faster than the nested method in
    serial and $3.7$ to $7.9$ times faster at sixteen threads. Replacing existing
    curvilinear interpolation with ENGINE cuts full-solve time $2.8$- to
    $4.5$-fold at matched accuracy.
  keypoints:
    - Splitting a decision into stages lets each qualifying stage invert its first-order condition in closed form.
    - On the two-asset household block, EGM$^n$ solves about thirteen times faster in serial than a matched nested-EGM implementation.
    - Swapping in ENGINE for the curvilinear interpolation method cuts full-solve time by a factor of roughly three to four and a half.
  acknowledgement: >-
    I thank the editors, Thomas Lubik and Eric Young, and three anonymous referees at the
    *Journal of Economic Dynamics and Control*, whose reports substantially
    improved and streamlined the exposition of this work. I thank Chris
    Carroll, Matthew White, and Fedor Iskhakov for their helpful comments and
    suggestions. The remaining errors are my own. This paper was awarded the
    [Outstanding Graduate Student Paper Award](https://comp-econ.com/contest-and-prize/)
    at the 29th International Conference on Computing in Economics and Finance
    (CEF 2023, Nice, France). Presentations at the Johns Hopkins University
    Macro Brownbag (2024), the
    [International Association for Applied Econometrics annual conference](https://editorialexpress.com/conference/IAAE2024Greece/program/IAAE2024Greece.html)
    (2024, Thessaloniki, Greece), and the
    [University of Texas at Austin MA in Economics Program 10-Year Reunion Celebration](https://ma.eco.utexas.edu/10-year-reunion-celebration-invitation/)
    (2024) improved the exposition and applications. During the development of
    this work, I received support through the Econ-ARK project, funded
    by a grant from the
    [Alfred P. Sloan Foundation](https://sloan.org/grant-detail/g-2017-9832).
    All figures and other numerical results were produced using the
    [Econ-ARK HARK](https://econ-ark.org/) toolkit {cite:p}`Carroll2018`.
  declaration: >-
    During the preparation of this work, the author used Claude Code (Anthropic) to
    edit and proofread the manuscript and assist with coding. After
    using this tool, the author reviewed and edited the content as needed and
    takes full responsibility for the content of the published article.
---

:::{include} 1_intro.md
:::

:::{include} 2_method.md
:::

:::{include} 3_multdim_abrs.md
:::

:::{include} 4_multinterp.md
:::

:::{include} 5_conditions.md
:::

:::{include} 6_conclusion.md
:::

+++ {"part": "appendix"}

:::{include} 7_appendix.md
:::

+++
