### Machine Learning for Physical Sciences
*pip install mlphys*

Author: Sanjaya Lohani

*Please report bugs at slohani@mlphys.com

Papers:

1.   Lohani, S., Lukens, J.M., Jones, D.E., Searles, T.A., Glasser, R.T. and Kirby, B.T., 2021. Improving application performance with biased distributions of quantum states. *Physical Review Research*, 3(4), p.043145. 

2.  Lohani, S., Searles, T. A., Kirby, B. T., & Glasser, R. T. (2021). On the Experimental Feasibility of Quantum State Reconstruction via Machine Learning. *IEEE Transactions on Quantum Engineering*, 2, 1–10. 

Collaborator: Joseph M. Lukens, Daniel E. Jones, Ryan T. Glasser, Thomas A. Searles, and Brian T. Kirby

<!-- GETTING STARTED -->
## Getting Started

pip install mlphys

<!-- USAGE EXAMPLES -->
## Usage

```sh
import deepqis.Simulator.Distributions as dist
import deepqis.Simulator.Measurements as meas
import matplotlib.pyplot as plt
import deepqis.utils.Alpha_Measure as find_alpha
import deepqis.utils.Concurrence_Measure as find_con
import deepqis.utils.Purity_Measure as find_pm
import deepqis.network.inference as inference
import deepqis.utils.Fidelity_Measure as fm
...
```

_For examples (google colab), please refer to_ 
* [Generating Biased Distributions](https://github.com/slohani-ai/mlphys/blob/main/tutorials-google-colab-notebook/Biased_distributions_random_Q_states.ipynb). 
* [Inference Examples](https://github.com/slohani-ai/mlphys/blob/main/tutorials-google-colab-notebook/Inference_examples.ipynb).

<!--
_open in the google colab_
* [Generating Biased Distributions]
* [Inference_Examples]
-->