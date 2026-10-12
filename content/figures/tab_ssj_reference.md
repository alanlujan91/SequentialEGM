```{latex}
\begin{table}[htbp]
\centering
\caption{Time in seconds and converged-policy accuracy: the two-asset block's native solver versus EGM$^n$, at matched grids, tolerance, and linear interpolation. The native solver is serial; parentheses give its time over EGM$^n$'s at sixteen threads. Iterations are backward steps to the shared tolerance; accuracy is the off-grid robust Euler median ($\log_{10}$), EGM$^n$ first.}
\label{tab-ssj-reference}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrcc@{}}
\toprule
Grid & Native, serial & \multicolumn{2}{c}{EGM$^n$} & Iterations & Accuracy \\
\cmidrule(lr){3-4}
& & 1 thread & 16 threads & EGM$^n$/native & EGM$^n$/native \\
\midrule
$(40, 60)$ & 0.25 & 0.76 & 0.24 (1.1x) & 395/478 & $-2.86$/$-2.78$ \\
$(80, 120)$ & 0.78 & 3.22 & 0.55 (1.4x) & 410/467 & $-3.44$/$-3.58$ \\
$(160, 240)$ & 2.94 & 13.41 & 1.74 (1.7x) & 424/464 & $-4.21$/$-4.36$ \\
\bottomrule
\end{tabular}
\end{table}
```
