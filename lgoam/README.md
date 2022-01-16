<!--
*** Thanks for checking out. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request.
*** Thanks again!
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/">
    <img src="images/Image_1.png" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">OAM-Tensors (GPU/CPU)</h3>

  <p align="center">
    Let's leverage the power of tensor operations in simulating LG-OAM intensity modes, and phase patterns for a Spatial Light Modulator (SLM). Being tensors, the generated patterns can be directly fed into any machine learning frameworks.
    <br />
    <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/blob/main/Superposition_OAM_Tensors_GPU.ipynb">Superposition OAM Tensors</a>
    .
    <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/blob/main/NonSuperposition_OAM_Tensors-GPU.ipynb">Non-Superposition OAM Tensors</a>
    ·
    <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/issues">Report Bug</a>
    ·
    <a href="https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Optical communication relies on the generation, transmission, and detection of states of light to encode and send information. In order to increase the transfer rate one of the most promising methods is making use of orbital angular momentum (OAM) states of light. However, a primary technical difficulty is the accurate classification of OAM value  detected at the receiver. This project explores the various ways to simulate LG-OAM states (with GPU) that can be easily applied to train/test any machine learning setups at the receiver in prior. Additionally, the code generates phase patterns that can be uploaded to a SLM in order to have the correspoinding spatial distribution at the receiver. Overall this provides the robust way to simulate Laguerre-Gauss OAM modes to be used in the real-world communication setups.

Please cite the following(s) if you use any part of the code in your project:
  * Lohani, S., Knutson, E. M., & Glasser, R. T. (2020). Generative machine learning for robust free-space communication. Communications Physics, 3(1), 1-8. [nature-communications physics](https://www.nature.com/articles/s42005-020-00444-9).
  * Lohani, S., & Glasser, R. T. (2018). Turbulence correction with artificial neural networks. Optics letters, 43(11), 2611-2614. [OSA-Optics Letters](https://www.osapublishing.org/ol/abstract.cfm?uri=ol-43-11-2611) . [arXiv](https://arxiv.org/abs/1806.07456).
  * Lohani, S., Knutson, E. M., O’Donnell, M., Huver, S. D., & Glasser, R. T. (2018). On the use of deep neural networks in optical communications. Applied optics, 57(15), 4180-4190. [OSA-Applied Optics](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-15-4180) . [arXiv](https://arxiv.org/abs/1806.06663).

Thank you!

### Built With

* [Tensorflow 2.4](https://www.tensorflow.org/)
* [Tensorflow Probability](https://www.tensorflow.org/probability)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Please install libraries from the requirements.txt file
  ```sh
  pip install requirements.txt
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors.git
   ```
<!-- 
2. Install NPM packages
   ```sh
   npm install
   ```
-->


<!-- USAGE EXAMPLES -->
## Usage

```sh
from utils.Imaging import Save
from utils.Noise import Noise_Dist
from source.OAM_Intensity_Phase import LG_Lights_Tensorflow

lg = LG_Ligths_Tensorflow(xpixel, ypixel, dT, verbose)
```
where xpixel = width, ypixel = height, dT = SLM resolution (typically 8e-6 m), and versbose is False as default.

It only requires a single line of code to simulate everything.

- Superposition Modes. [Documentation: please see Superposition_OAM_Tensors_GPU](https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/blob/main/Superposition_OAM_Tensors_GPU.ipynb)
  1. A single batch of superposition of various modes
     <img src="https://render.githubusercontent.com/render/math?math=\psi=\alpha_1|LG_{0,1}^{\ell_1}\rangle%2B\alpha_2|LG_{0,1}^{\ell_2}\rangle %2B\alpha_2|LG_{0,1}^{\ell_3}\rangle%2B......%2B">
  
     ```sh
     Intensity, Phase = lg.Superposition(p_l_array, alpha_array, w, grating_period, save_image)
     ```
  2. Simultaneous simulation for multple batches of superposition modes
     <img src="https://render.githubusercontent.com/render/math?math=\psi_1=\alpha_1|LG_{0,1}^{\ell_1}\rangle%2B\alpha_2|LG_{0,1}^{\ell_2}\rangle %2B\alpha_2|LG_{0,1}^{\ell_3}\rangle%2B......%2B">;
     <img src="https://render.githubusercontent.com/render/math?math=\psi_2=\alpha_1|LG_{0,1}^{\ell_1}\rangle%2B\alpha_2|LG_{0,1}^{\ell_2}\rangle %2B\alpha_2|LG_{0,1}^{\ell_3}\rangle%2B......%2B">;
     <img src="https://render.githubusercontent.com/render/math?math=\psi_3=\alpha_1|LG_{0,1}^{\ell_1}\rangle%2B\alpha_2|LG_{0,1}^{\ell_2}\rangle %2B\alpha_2|LG_{0,1}^{\ell_3}\rangle%2B......%2B">
  .. .. ..
     ```sh
      Intensity, Phase = lg.Superposition_Batch(p_l_array, alpha_array, w, grating_period, save_image)
     ```
     where, 
     - p_l_array: Tensorflow tensor or Numpy array of size [None, 2] for a single batch and [Batch, No. of modes to be superposed per layer, 2] for superposition_batch. In [None, 2] first column represents p-value and second column repesents l-value,
     - alpha_array: an array or list representing mixture percentages of various modes,
     - w = beam width at z = 0,
     - grating_period: grating lines. Usful in implementing the simulated phase mask on the SLM,
     - save_image: False as default. If True, simulated OAM modes are automatically saved as images in the same dir.
     _for example the following script simultaneously generates multiple superposition OAM modes_.
     ```sh
     p_set = np.random.randint(0, 2, 36) 
     l_set = np.random.randint(-10, 10, 36)
     p_and_l_set = np.stack([p_set, l_set],axis=1)
     p_and_l_set = p_and_l_set.reshape(-1, 3, 2) #(batch, no. of oam modes to be superimposed, 2)
     
     alpha_array = np.ones([len(p_and_l_set), 3, 1])

     lg.verbose=False
     intensity_list, phase_list = lg.Superposition_Batch(p_l_array=p_and_l_set, alpha_array=alpha_array,\
                                                 w=0.00015, grating_period=0, save_image=False)

     print ('Total size of SUP-OAM modes: ',len(intensity_list))

     fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(8, 8))
     for i, axi in enumerate(ax.flat):
       axi.imshow(intensity_list[i])
       axi.set_title(f'SUP-OAM {i}')
     plt.tight_layout()
     plt.show()
     ```
     <p align="center">
     <img src="images/readme_image_2.png" alt="Intensity", width="400", height="400">
     <p> 
     and Corresponding Phase masks at grating period = 20 are shown below,
     
     ```sh
     lg.verbose=False
     intensity_list, phase_list = lg.Superposition_Batch(p_l_array=p_and_l_set, alpha_array=alpha_array,\
                                                 w=0.00015, grating_period=20, save_image=False)
     fig, ax = plt.subplots(nrows=3, ncols=4, figsize= (8, 8))
     for i, axi in enumerate(ax.flat):
     axi.imshow(phase_list[i])
     axi.set_title(f'SUP-Phase {i}')
     plt.tight_layout()
     plt.show()
     ```
     <p align="center">
     <img src="images/read_me_image_phase_mask.png" alt="Phase",width="400", height="400">
     <p>
   
- Non-superposition Modes. [Documentation: please see Non-Superposition_OAM_Tensors_GPU](https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/blob/main/NonSuperposition_OAM_Tensors_GPU.ipynb)
  - Multiple non-superposition OAM modes simultaneously,
  ```sh
  Intensity, Phase = lg.Non_Superposition(p_l_array, w, grating_period, save_image)
  ```

- Noisy OAM Modes. (_for example 200 noisy superposition modes per clean OAM image_).
```sh
intensity_with_gaussian = Noise_Dist().Guassian_Noise_Batch(intensity_list, mean=0., std=1., multiple=200, factor=5e5)
```
  This also supports Gaussain_Noise, Possion_Noise, Gamma_Noise, Poisson_Noise_Batch and Gamma_Noise_Batch as well.

- Saving Tensors as Images. (_for example saving 200 noisy superposition modes for each clean OAM images simultaneously_)
```sh
Save().Save_Tensor_Image(intensity_with_gaussian)
```

_For more examples, please refer to_ [Superposition Notebook](https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/blob/main/Superposition_OAM_Tensors_GPU.ipynb) . [Non-Superposition Notebook](https://github.com/slohani-ai/LG-OAM-simulations-with-Tensors/blob/main/NonSuperposition_OAM_Tensors_GPU.ipynb)



<!-- ROADMAP 
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).

-->

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit your Changes (`git commit -m 'Add some NewFeature'`)
4. Push to the Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

- Sanjaya Lohani - [@twitter_handle](https://twitter.com/slohani_ai) . slohani@mlphys.com / slohani@tulane.edu
<!-- - Ryan T. Glasser - [@twitter_handle](https://twitter.com/GlasserPhysics) . rglasser@tulane.edu -->

<!-- ACKNOWLEDGEMENTS 
## Acknowledgements

* [Lohani, S., Knutson, E. M., & Glasser, R. T. (2020). Generative machine learning for robust free-space communication. Communications Physics, 3(1), 1-8.](https://www.nature.com/articles/s42005-020-00444-9)
* [Lohani, S., & Glasser, R. T. (2018). Turbulence correction with artificial neural networks. Optics letters, 43(11), 2611-2614.](https://www.osapublishing.org/ol/abstract.cfm?uri=ol-43-11-2611)
* [Lohani, S., Knutson, E. M., O’Donnell, M., Huver, S. D., & Glasser, R. T. (2018). On the use of deep neural networks in optical communications. Applied optics, 57(15), 4180-4190.](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-57-15-4180)
-->






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
