## Overview of the first part of the DeepMoD project

- First version of DeepMoD up and running, applied to PDEs and ODEs. 
-  Simple single particle tracking analysis working for DeepMoD (works on diffusion, chemotaxis and PRW). This is essentially the project of Subham but it's not published nor availble somewhere.
- Improved densty estimation with temporal normalizing flows, unclear what will result from this. Interesting concept and reviewers did show interest but turns out to be slow to test and does not perform well througout the full time domain. 
- Started working on implementing group sparsity 
  - First step is to apply DeepMoD to a set of experiments with different initial conditions but govered by the same equation. This is a very common situation in many physical systems. 
  - Second step is to discover a set of equations which are governed by the same PDE but for which the constants are not necessarily the same. A perfect test case would be the real SPT diffusion data where we have particle size and corresponding trajectories. We have the data, equation should be universal throughout trajectories but diffusion coefficient should depend on the size of the particle. 
- Generallized Maxwell model: Application of DeepMoD on simple visco-elastic materials and electronical circuits. First experimental data looks promessing and on the artificial data all seems to be working fairly well. What remains to be done is: 
  - Generate more experimental trajectories for two capacitors and fit it to DeepMoD (after confinement)
  - Finish the results section and try to finalize the story.  
  - Potentially include the concept of group sparsity by applying a set of periodic function on the signal and train a single NN to learn the mapping (or different NNs and one single regression term)
- Bayesian DeepMoD v1 : Here we should first of all try to obtain a probablity distribution over the obtained coefficients for DeepMoD, could be a nice extension.
- Project DeepDrugs: Results from Camille showed that some simple bottom up modelling would be a more interesting approach at this stage to the inference of AB resistance compared to a model discovery approach.

## What would it take to make DeepMoD usefull for scientific community? 

1) Directly show that we can indeed handle experimental data. Try to find some simple test cases in the literature (e.g. https://www.ebi.ac.uk/biomodels/ ). This would probably require to further work on group sparsity first, since many experiments contain either trajectories with different initial conditions or model a process with different initial conditions 

2) Focus should be to keep the DeepMoD framework as intuitive and simple as possible considering we target a community that is as broad as possible. 

3) Analysing stochastic trajectories directly through the type of methods we use in the ANDI project are supposidly a better approach for SPT data, compared to using DeepMoD, since most practical stochastic processes do not have an obvious underlying PDE.  

4) Clean up all the peaces of work we discussed above and make it a consistent package that people can use (lot's of examples, blogpost about the topic etc. )