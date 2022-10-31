Author: Sanjaya Lohani

*Please report bugs at slohani@mlphys.com


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/">
    <img src="logo-image/logo_deepqis.png" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">Deep Quantum Information Science (DeepQis)</h3>

Papers:

1.   Lohani, S., Lukens, J.M., Jones, D.E., Searles, T.A., Glasser, R.T. and Kirby, B.T., 2021. Improving application performance with biased distributions of quantum states. *Physical Review Research*, 3(4), p.043145. 

2.  Lohani, S., Searles, T. A., Kirby, B. T., & Glasser, R. T. (2021). On the Experimental Feasibility of Quantum State Reconstruction via Machine Learning. *IEEE Transactions on Quantum Engineering*, 2, 1–10. 

Thanks to [Brian T. Kirby](https://briankirby.github.io/), [Ryan T. Glasser](http://www.tulane.edu/~rglasser97/), [Sean D. Huver](https://developer.nvidia.com/blog/author/shuver/) and [Thomas A. Searles](https://ece.uic.edu/profiles/searles-thomas/)

<!-- GETTING STARTED -->
## Getting Started

```pip install mlphys```

<!-- USAGE EXAMPLES -->
## Usage

```sh
import mlphys_nightly.deepqis.simulator.distributions as dist
import mlphys_nightly.deepqis.simulator.measurements as meas
import mlphys_nightly.deepqis.utils.Alpha_Measure as find_alpha
import mlphys_nightly.deepqis.utils.Concurrence_Measure as find_con
import mlphys_nightly.deepqis.utils.Purity_Measure as find_pm
import mlphys_nightly.deepqis.network.inference as inference
import mlphys_nightly.deepqis.utils.Fidelity_Measure as fm
...
```

_For examples (google colab), please refer to_ 
* [Generating Biased Distributions](https://github.com/slohani-ai/machine-learning-for-physical-sciences/blob/main/mlphys/deepqis/Biased_distributions_random_Q_states.ipynb). 
* [Inference Examples](https://github.com/slohani-ai/machine-learning-for-physical-sciences/blob/main/mlphys/deepqis/Inference_examples.ipynb).

<!--
_open in the google colab_
* [Generating Biased Distributions]
* [Inference_Examples]
-->