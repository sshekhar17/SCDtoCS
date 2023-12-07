## Reduction of sequential change detection to sequential estimation 

This repository contains an implementation of the general reduction from sequential change detection (SCD) to sequential estimation (via confidence sequences or CSs), proposed by [Shekhar and Ramdas (2023)](https://arxiv.org/pdf/2309.09111.pdf). This general reduction is valid for detecting changes in any parameter or functional $\theta$ associated with a data-generating source that can be estimated using Confidence Sequences. As a result, this reduction allows us to leverage the significant recent progress in constructing CSs in various settings, to immediately obtain powerful SCD schemes. 

This repository contains the following files: 

* `main.py`: this defines the following classes
    * `BaseChangeDetector`: abstract base class implementing the general scheme proposed in Definition 2.1 of [Shekhar and Ramdas (2023)](https://arxiv.org/pdf/2309.09111.pdf). 
    * `BoundedMeanHoeffdingSCD`: an instantiation of the general scheme using Hoeffding CS for detecting changes in means of bounded observations.  
    * `BoundedMeanBernsteinSCD`: an instantiation of the general scheme using Empirical-Bernstein CS for detecting changes in means of bounded observations.  

* `utils.py`: some helper functions 

* `example.ipynb`: an iPython notebook that includes experiments comparing the performance of the two SCD schemes for bounded observations.  



