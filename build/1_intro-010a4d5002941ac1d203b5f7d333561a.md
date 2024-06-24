

+++ {"part": "abstract"}

Heterogeneous agent models with multiple decisions are often solved using inefficient grid search methods that require many evaluations and are slow. This paper provides a novel method for solving such models using an extension of the Endogenous Grid Method (EGM) that uses Gaussian Process Regression (GPR) to interpolate functions on unstructured grids. First, I propose an intuitive and strategic procedure for decomposing a problem into subproblems which allows the use of efficient solution methods. Second, using an exogenous grid of post-decision states and solving for an endogenous grid of pre-decision states that obey a first-order condition greatly speeds up the solution process. Third, since the resulting endogenous grid can often be non-rectangular at best and unstructured at worst, GPR provides an efficient and accurate method for interpolating the value, marginal value, and decision functions. Applied sequentially to each decision within the problem, the method is able to solve heterogeneous agent models with multiple decisions in a fraction of the time and with less computational resources than are required by standard methods currently used. Software to reproduce these methods is available under the [`Econ-ARK/HARK`](https://econ-ark.org/) project for the `python` programming language.

+++

+++ {"part": "acknowledgements"}

I would like to thank Chris Carroll, Matthew White, and Simon Scheidegger for their helpful comments and suggestions. The remaining errors are my own. All figures and other numerical results were produced using the [`Econ-ARK/HARK`](https://econ-ark.org/) toolkit {cite:p}`Carroll2018`. Additional libraries used in the production of this paper include but are not limited to: [`scipy`](https://www.scipy.org/) {cite:p}`Virtanen2020`, [`numpy`](https://www.numpy.org/) {cite:p}`Harris2020`, [`numba`](https://numba.pydata.org/) {cite:p}`Lam2015`, [`cupy`](https://cupy.dev/) {cite:p}`Okuta2017`, [`scikit-learn`](https://scikit-learn.org/) {cite:p}`Pedregosa2011`, [`pytorch`](https://pytorch.org/) {cite:p}`Paszke2019`, and [`gpytorch`](https://gpytorch.ai/) {cite:p}`Gardner2018`

+++

(introduction)=
# Introduction

## Background

% Introduce the topic by providing a brief overview of the issue and why it is important to study it.

% Identify the research question: Clearly state the research question or problem being addressed in the current study.
% Provide context: Explain why the topic is important to study and what gap in the existing knowledge the current study aims to fill.
% Summarize the existing literature: Briefly describe what is currently known about the topic, including any relevant studies or theories that have been previously published.
% Highlight the limitations of previous research: Identify any limitations or gaps in the existing literature and explain how the current study will address these limitations.
% Provide a rationale for the study: Explain why the current study is needed and why it is a significant contribution to the existing knowledge on the topic.

% Use only the first paragraph to state the question and describe its importance. Don't weave
% around, be overly broad, or use prior literature to motivate it (the question is not important
% because so many papers looked at this issue before!).

Macroeconomic modeling aims to describe a complex world of agents interacting with each other and making decisions in a dynamic setting. The models are often very complex, require strong underlying assumptions, and use a lot of computational power to solve. One of the most common methods to solve these complex problems is using a grid search method to solve the model. The Endogenous Grid Method (EGM) developed by {cite:t}`Carroll2006` allows dynamic optimization problems to be solved in a more computationally efficient and faster manner than the previous method of convex optimization using grid search. Many problems that before took hours to compute became much easier to solve and allowed macroeconomists and computational economists to focus on estimation and simulation. However, the Endogenous Grid Method is limited to a few specific classes of problems. Recently, the classes of problems to which EGM can be applied have been expanded[^f1], but with every new method comes a new set of limitations. This paper introduces a new approach to EGM in a multivariate setting. The method is called Sequential EGM (or EGM$^n$) and introduces a novel way of breaking down complex problems into a sequence of simpler, smaller, and more tractable problems, along with an exploration of new multidimensional interpolation methods that can be used to solve these problems.

[^f1]: {cite:t}`Barillas2007, Maliar2013, Fella2014, White2015, Iskhakov2017`, among others.

## Literature Review

% Summarize the existing literature on the topic and highlight any gaps or limitations in the current research.

% Then use the second paragraph for a summary of the most relevant literature
% (not a full Section!). Hint: use present tense, to be consistent. "Smith (1986) presents a similar model, ..."

{cite:t}`Carroll2006` first introduced the Endogenous Grid Method as a way to speed up the solution of dynamic stochastic consumption-savings problems. The method consists of starting with an exogenous grid of post-decision states and using the inverse of the first-order condition to find the optimal consumption policy that rationalizes such post-decision states. Given the optimal policy and post-decision states, it is straightforward to calculate the initial pre-decision state that leads to the optimal policy. Although this method is certainly innovative, it only applied to a model with one control variable and one state variable. {cite:t}`Barillas2007` further extend this method by including more than one control variable in the form of a labor-leisure choice, as well as a second state variable for stochastic persistence.

{cite:t}`Hintermaier2010` introduce a model with collateral constraints and non-separable utility and solve using an EGM method that allows for occasionally binding constraints among endogenous variables. {cite:t}`Jorgensen2013` evaluates the performance of the Endogenous Grid Method against other methods for solving dynamic stochastic optimization problems and finds it to be fast and efficient. {cite:t}`Maliar2013` develop the Envelope Condition Method based on a similar idea as the Endogenous Grid Method, avoiding the need for costly numerical optimization and grid search. However, their model is limited to infinite horizon problems as it is a forward solution method.

Further development into a multivariate Endogenous Grid Method expanded the ability of researchers to solve models efficiently. {cite:t}`White2015` formally characterized the conditions for the Endogenous Grid Method and developed an interpolation method for structured non-rectilinear, or curvilinear, grids. {cite:t}`Iskhakov2015` additionally establishes conditions for solving multivariate models with EGM, requiring the invertibility of a triangular system of first-order conditions. {cite:t}`Ludwig2018` also develops a novel interpolating method using Delaunay triangulation of the resulting unstructured endogenous grid. However, the authors show that the gains from avoiding the grid search method are often offset by the costly construction of the triangulation.

For the papers discussed above, continuity and smoothness of the value and first-order conditions are strict requirements. {cite:t}`Fella2014` first introduced a method to solve non-convex problems using the Endogenous Grid Method. The idea is based on evaluating necessary but not sufficient candidates for the first-order condition in overlapping regions of the state space. {cite:t}`Arellano2016` use the Envelope Condition Method to solve a sovereign default risk model with similar efficiency gains to EGM. {cite:t}`Iskhakov2017` further advances the methodology by using extreme errors to solve discrete choice problems with Endogenous Grid Method. These methods however were only applied to a single control variable and a single state variable. {cite:t}`Druedahl2017` introduces the $G2EGM$ to handle non-convex problems with more than 1 control variable and more than 1 state variable. This method is also capable of handling occasionally binding constraints which previous multivariate EGM methods were not.

{cite:t}`Clausen2020` formalize the applicability of the Endogenous Grid Method and its extensions to discrete choice models and discuss the nesting of problems to efficiently find accurate solutions. {cite:t}`Druedahl2021` similarly suggest the nesting of problems to efficiently use the Endogenous Grid Method within problems with multiple control variables. However, while these nested methods reduce the complexity of solving these models, they often still require grid search methods as is the case with {cite:t}`Druedahl2021`.

% Finally, this paper contributes to the literature of solving dynamic optimization problems using machine learning tools. {cite:t}`Scheidegger2019` introduce the use of Gaussian Process Regression to compute global solutions for high-dimensional dynamic stochastic problems. {cite:t}`Maliar2021` use non-linear regression and neural networks to estimate systems of equations that characterize dynamic economic models.

## Research Question

% Clearly state the research question or problem being addressed in the current study.

% Next, while still on page one, the third paragraph must begin: "The purpose of this paper is ...",
% and summarize what you actually do. (Paragraphs 2 and 3 could be reversed.)

The purpose of this paper is to describe a new method for solving dynamic optimization problems efficiently and accurately while avoiding convex optimization and grid search methods with the use of the Endogenous Grid Method and first-order conditions. The method is called Sequential EGM (or EGM$^n$) and introduces a novel way of breaking down complex problems into a sequence of simpler, smaller, and more tractable problems, along with an exploration of new multidimensional interpolation methods that can be used to solve these problems. This paper also illustrates an example of how Sequential EGM can be used to solve a dynamic optimization problem in a multivariate setting.

## Methodology

% Briefly describe the research methodology used in the study, including any data sources, econometric techniques, or other methods used.

The sequential Endogenous Grid Method consists of 3 major parts: First, the problem to be solved should be broken up into a sequence of smaller problems that themselves don't add any additional state variables or introduce asynchronous dynamics with respect to the uncertainty. If the problem is broken up in such a way that uncertainty can happen in more than one period, then the solution to this sequence of problems might be different from the aggregate problem due to giving the agent additional information about the future by realizing some uncertainty. Second, I evaluate each of the smaller problems to see if they can be solved using the Endogenous Grid Method. This evaluation is of greater scope than the traditional Endogenous Grid Method, as it allows for the resulting exogenous grid to be non-regular. If the subproblem can not be solved with EGM, then convex optimization is used. Third, if the exogenous grid generated by the EGM is non-regular, then I use a multidimensional interpolation method that takes advantage of machine learning tools to generate an interpolating function. Solving each subproblem in this way, the sequential Endogenous Grid Method is capable of solving complex problems that are not solvable with the traditional Endogenous Grid Method and are difficult and time-consuming to solve with convex optimization and grid search methods.

## Contributions

% Discuss how the current study contributes to the existing literature and what new insights it provides.

% That sets you up for the fourth paragraph, which lists "The contributions of
% this work" – relative to that prior literature. Clarify what you do that's different

The Sequential Endogenous Grid Method is capable of solving multivariate dynamic optimization problems in an efficient and fast manner by avoiding grid search. This should allow researchers and practitioners to solve more complex problems that were previously not easily accessible to them, but more accurately capture the dynamics of the macroeconomy. By using advancements in machine learning techniques such as Gaussian Process Regression, the Sequential Endogenous Grid Method is capable of solving problems that were not previously able to be solved using the traditional Endogenous Grid Method. In particular, the Sequential Endogenous Grid Method is different from NEGM in that it allows for using more than one Endogenous Grid Method step to solve a problem, avoiding costly grid search methods to the extent that the problem allows.

Additionally, the Sequential Endogenous Grid Method often sheds light on the problem by breaking it down into a sequence of simpler problems that were not previously apparent. This is because intermediary steps in the solution process generate value and marginal value functions of different pre- and post-decision states that can be used to understand the problem better.

% The fifth paragraph then summarizes your results. Tell the answer, so they know what to expect,
% and how to think about each step along the way, what's driving your results.

## Outline

% Provide a brief overview of the results and conclusions that will be presented in the article.
% In the sixth and final paragraph, as an aid to the reader, plot the course for the rest of the paper.
% "The first Section below presents a theoretical model that can be used to generate specific
% hypotheses. Then [Section %s](#method) presents the econometric model, ..."

[Section %s](#method) presents a basic model that illustrates the sequential Endogenous Grid Method in one dimension. Then [Section %s](#multdim) introduces a more complex method with two state variables to demonstrate the use of machine learning tools to generate an interpolating function. In [Section %s](#multinterp) I present the unstructured interpolation methods using machine learning in more detail. [Section %s](#conditions) discusses the theoretical requirements to use the Sequential Endogenous Grid Method. Finally, [Section %s](#conclusion) concludes with some limitations and future work.
