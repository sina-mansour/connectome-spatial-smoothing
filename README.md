# Connectome Spatial Smoothing

Here, you may find the Python codes and sample scripts for Connectome Spatial Smoothing (CSS).

For more information you may check our article on **Connectome Spatial Smoothing (CSS): concepts, methods and evaluation** (currently under review). All resources are provided as complementary to the following article:

...article information to be added after publication...

The code used for this study is now released as a [python package](https://pypi.org/project/Connectome-Spatial-Smoothing/). If using the codes, please also cite the following:

...citeable code reference...


---

The codes in this repository mainly perform the following tasks:

- Map high-resolution structural connectomes from tractography

- Map atlas-resolution structural connectomes from tractography

- Compute the CSS smoothing kernel with selected parameters

- Perform CSS on high-resolution connectomes

- Perform CSS directly on atlas connectomes

---

## Installation

To use CSS functionality in your code, you can install the package with the following command:

`pip install Connectome-Spatial-Smoothing`

Then, you could simply use the package in your own code after importing:

`from Connectome_Spatial_Smoothing import CSS as css`

---

We have provided a short jupyter notebook showcasing all the functionalities described above. You may use the following link to open [this notebook](https://github.com/sina-mansour/connectome-based-smoothing/blob/main/notebooks/example.ipynb) in an interactive google colab session:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sina-mansour/connectome-based-smoothing/blob/main/notebooks/example.ipynb)