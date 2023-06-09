---
# title: The Sequential Endogenous Grid Method # a string (max 500 chars) page & project
# description: # a string (max 500 chars) page & project
short_title: Conclusion # a string (max 40 chars) page & project
# name:  # a string (max 500 chars) page & project
# tags:  # a list of strings page only
# thumbnail: # a link to a local or remote image page only
# subtitle: # a string (max 500 chars) page only
# date: # a valid date formatted string page can override project
# authors:  # a list of author objects page can override project
# doi:  # a valid DOI, either URL or id page can override project
# arxiv: # a valid arXiv reference, either URL or id page can override project
# open_access: # boolean (true/false) page can override project
# license: # a license object or a string page can override project
# github: # a valid GitHub URL or owner/reponame page can override project
# binder: # any valid URL page can override project
# subject: # a string (max 40 chars) page can override project
# venue: # a venue object page can override project
# biblio: # a biblio object with various fields page can override project
numbering:
    enumerator: "6.%s"
---

(conclusion)=

# Conclusion

% Summarize the method: Begin your conclusion by summarizing the new computational method you developed or proposed.
Provide a brief overview of the key features of the method and how it differs from existing methods.

This paper introduces a novel method for solving dynamic stochastic optimization problems called the Sequential
Endogenous Grid Method (EGM$^n$). Given a problem with multiple decisions (or control variables), the Sequential
Endogenous Grid Method proposes separating the problem into a sequence of smaller subproblems that can be solved
sequentially by using more than one EGM step. Then, depending on the resulting endogenous grid from each subproblem,
this paper proposes different methods for interpolating functions on non-rectilinear grids, called the Warped Grid
Interpolation (WGI) and the Gaussian Process Regression (GPR) method.

EGM$^n$ is similar to the Nested Endogenous Grid Method (NEGM)[^NEGM] and the Generalized Endogenous Grid Method (
G2EGM)[^G2EGM] in that it can solve problems with multiple decisions, but it differs from these methods in that by
choosing the subproblems strategically, we can take advantage of multiple sequential EGM steps to solve complex
multidimensional models in a fast and efficient manner. Additionally, the use of machine learning tools such as the GPR
overcomes bottlenecks seen in unstructured interpolation using Delauany triangulation and other similar methods.

[^NEGM]: {cite:t}`Druedahl2021`.

[^G2EGM]: {cite:t}`Druedahl2017`.

% Evaluate the method: Evaluate the strengths and limitations of the new computational method you developed or proposed.
Discuss how the method compares to existing methods in terms of accuracy, efficiency, and ease of use.

% Demonstrate the method: If possible, provide an example of how the new computational method can be used to solve a
problem or answer a research question. This will help the reader understand the practical implications of the method.

% Highlight potential applications: Discuss potential applications of the new computational method. This will help
demonstrate the broader impact of the method beyond the specific problem or research question addressed in your paper.

% Discuss future directions: Provide suggestions for future research based on the new computational method you developed
or proposed. This can include improvements to the method, potential extensions to other areas of research, or new
applications of the method.

% Conclude with final thoughts: End your conclusion with some final thoughts that tie together the main points of your
paper. This will help leave a lasting impression on the reader.
