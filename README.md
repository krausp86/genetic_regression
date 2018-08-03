GENETIC REGRESSION

In this Repo I try to optimize the capacity of a Regression to fit a given Problem.In order to do so, I will try to implement a genetic algorithm.

The Regression Object "Regressor" is defined by a set of "Genes" (input parameter).
This set of Genes defines which order of polynomial the Regression will try to fit.In an initial generation a population of 100 random individuals will try to fit a noisy function given by some arbitrary polynomial. The best performing 10 individuals will be allowed to "reproduce".
Gene combination will be realized as follows: If a gene is set (or unset) in both parents, set the gene accordingly, if they differ, choose by a 50-50 probability.
Afterwards, decide to flip each gene using a very small mutation probability.

Using 10 parents, recombining them with all remaining parents (including themselves) we get back to a population of 100 individuals.

The genesets can be identified by interpreting them as binary representations of numbers.
