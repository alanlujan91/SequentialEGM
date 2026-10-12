```{latex}
\begin{table}[htbp]
\centering
\caption{Time in seconds to a converged two-asset solution, NEGM versus EGM$^n$, at matched grids, tolerance, interpolation, and thread counts. Parentheses give the speed ratio over NEGM at the same thread count.}
\label{tab-timing-negm}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Grid & \multicolumn{2}{c}{1 thread} & \multicolumn{2}{c}{16 threads} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
& NEGM & EGM$^n$ & NEGM & EGM$^n$ \\
\midrule
$(40, 60)$ & 9.81 & 0.78 (12.5x) & 0.89 & 0.24 (3.7x) \\
$(80, 120)$ & 42.22 & 3.22 (13.1x) & 3.27 & 0.54 (6.1x) \\
$(160, 240)$ & 178.59 & 13.66 (13.1x) & 13.56 & 1.73 (7.9x) \\
\bottomrule
\end{tabular}
\end{table}
```
