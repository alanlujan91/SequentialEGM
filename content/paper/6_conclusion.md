
(conclusion)=
# Conclusion

This paper introduces a novel method for solving dynamic stochastic optimization problems called the Sequential Endogenous Grid Method (EGM$^n$). Given a problem with multiple decisions (or control variables), the Sequential Endogenous Grid Method proposes separating the problem into a sequence of smaller subproblems that can be solved sequentially by using more than one EGM step. Then, depending on the resulting endogenous grid from each subproblem, this paper proposes different methods for interpolating functions on non-rectilinear grids, called the Warped Grid Interpolation (WGI) and the Gaussian Process Regression (GPR) method.

EGM$^n$ is similar to the Nested Endogenous Grid Method (NEGM) of {cite:t}`Druedahl2021` and the Generalized Endogenous Grid Method (G2EGM) of {cite:t}`Druedahl2017` in that it can solve problems with multiple decisions, but it differs from these methods in that by choosing the subproblems strategically, we can take advantage of multiple sequential EGM steps to solve complex multidimensional models in a fast and efficient manner. Additionally, the use of machine learning tools such as the GPR overcomes bottlenecks seen in unstructured interpolation using Delaunay triangulation and other similar methods.
