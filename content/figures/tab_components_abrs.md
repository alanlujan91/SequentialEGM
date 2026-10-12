```{latex}
\begin{table}[htbp]
\centering
\caption{Runtime in seconds by component of the EGM$^n$ backward solve, summed over a converged one-thread solve (shares in parentheses). Stage inversions are Stage 1's consumption EGM and the Stage 0 inversion; mesh-to-grid interpolation is the ENGINE regrid; regular-grid inversion is corner selection, Euler inversion, and envelope updates.}
\label{tab-components-abrs}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Grid & Expectation & Stage inversions & Mesh-to-grid & Regular-grid inversion \\
\midrule
$(40, 60)$ & 0.01 (1\%) & 0.28 (36\%) & 0.04 (5\%) & 0.45 (58\%) \\
$(80, 120)$ & 0.03 (1\%) & 1.15 (35\%) & 0.15 (5\%) & 1.92 (59\%) \\
$(160, 240)$ & 0.10 (1\%) & 4.67 (35\%) & 0.63 (5\%) & 8.09 (60\%) \\
\bottomrule
\end{tabular}
\end{table}
```
