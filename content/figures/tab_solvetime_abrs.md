```{latex}
\begin{table}[htbp]
\centering
\caption{Full solve time and converged-policy accuracy with ENGINE or the curvilinear method of \citet{White2015} as the Stage-0 regrid of an otherwise identical EGM$^n$ solve. ``Solve (s)'' is the best of three warmed solves to tolerance $10^{-9}$. Residuals are the median, 95th percentile, and maximum of the $\log_{10}$ robust consumption-unit Euler error at off-grid midpoints below 85 percent of each grid maximum.}
\label{tab-solvetime-abrs}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Grid & Interpolator & Solve (s) & \multicolumn{3}{c}{$\log_{10}$ Euler error} \\
\cmidrule(l){4-6}
& & & Median & p95 & Max \\
\midrule
Coarse ($24 \times 20$) & ENGINE & 0.139 & $-1.79$ & $-1.10$ & $-1.08$ \\
Coarse ($24 \times 20$) & Curvilinear & 0.383 & $-1.79$ & $-1.10$ & $-1.08$ \\
Medium ($48 \times 40$) & ENGINE & 0.233 & $-2.47$ & $-1.73$ & $-1.70$ \\
Medium ($48 \times 40$) & Curvilinear & 0.822 & $-2.48$ & $-1.73$ & $-1.70$ \\
Fine ($96 \times 80$) & ENGINE & 0.570 & $-3.14$ & $-2.38$ & $-2.11$ \\
Fine ($96 \times 80$) & Curvilinear & 2.540 & $-3.14$ & $-2.38$ & $-2.11$ \\
\bottomrule
\end{tabular}
\end{table}
```
