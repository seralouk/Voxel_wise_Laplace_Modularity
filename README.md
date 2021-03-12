# Fine-grained spectral mapping of voxel-wise connectivity: A framework and application on resting-state brain fMRI data

#### Code accompanying the Ph.D. disseration of Serafeim LOUKAS entitled: "Multivariate and predictive models for brain connectivity with application to neurodevelopment".
#### URL: TBA, Thèse n. 8854 (EPFL, 2021)

----

#### `main_modularity_acc.py`: the main script for modularity estimation using Accordance as the weighted, undirected adjacency matrix of the underlying network

#### `main_modularity_disc.py`: the main script for modularity estimation using Discordance as the weighted, undirected adjacency matrix of the underlying network

#### `main_laplacian_acc.py`: the main script for normalized laplacian estimation using Accordance as the weighted, undirected adjacency matrix of the underlying network

#### `main_laplacian_disc.py`: the main script for normalized laplacian estimation using Discordance as the weighted, undirected adjacency matrix of the underlying network

#### Dependencies: `numpy`, `scipy`, `os`, `nibabel` and `pandas`

----
#### Important: 

- To be able to run the example code you need to download the data from this link: https://drive.google.com/file/d/1B_KEiyiEzZReZ5L5Ayk0QGhLn7Bc4Ydv/view?usp=sharing and put the **.zip** file into the **Voxel_wise_Laplace_Modularity/Data/** directory that contains the GM mask. You should **not** unzip the data.

----

#### Note: 

- For the core framework only  `numpy` and `scipy` are needed.
- For loading some data and actually saving the eigenvectors as brain maps, `os`, `nibabel` and `pandas` are also needed.

----

#### Usage: In the main directory, lunch `python3 main_modularity_acc.py` depending on the desired case.
