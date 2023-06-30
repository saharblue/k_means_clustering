# K-means and K-means++ for Color Image Quantization 

This repository contains an implementation of the K-means and K-means++ algorithms for color image quantization.

The K-means and K-means++ algorithms are unsupervised learning methods used for clustering. The standard K-means algorithm iteratively assigns each data point to the nearest cluster center and then updates each cluster center by computing the mean of the data points assigned to it. However, the standard K-means algorithm can sometimes yield poor results due to its random initialization of cluster centers.

To address this issue, the K-means++ algorithm uses a sophisticated method for initializing the cluster centers. It starts by randomly selecting one data point as the first center, and each subsequent center is selected from the remaining data points with a probability proportional to its distance squared from the closest existing center. This initialization process helps spread out the initial cluster centers, which tends to lead to better and more consistent clustering results.

In this project, we apply the K-means and K-means++ algorithms to the task of color image quantization. Image quantization is a type of image compression that reduces the number of distinct colors used in an image, while still trying to maintain the overall visual appearance of the original image. This is achieved by clustering the colors present in the image, and then mapping each color to its nearest cluster center. The image is then reconstructed using only the colors of these cluster centers.

## Contents
The repository contains the following files:

1. `hw6.ipynb`: A Jupyter notebook that contains the code for the K-means and K-means++ algorithms, along with detailed explanations and visualizations.

2. `hw6.py`: A Python script that contains the same K-means and K-means++ algorithms as in the Jupyter notebook, but without the explanations and visualizations. This script can be run from the command line.

3. `data/small_duck.jpg`: A sample image that can be used to test the image compression algorithms.

## How to Run

1. Clone this repository.
2. Install the necessary Python packages. This code requires NumPy Matplotlib, and scikit-image. You can install these with pip by running `pip install numpy matplotlib scikit-image`.

To run the code in the Jupyter notebook, you'll also need to have Jupyter installed. You can install it with pip by running `pip install jupyter`.

3. Run the Python script or the Jupyter notebook. 

   To run the Python script, navigate to the directory containing `hw6.py` in the command line and run `python hw6.py`.

   To run the Jupyter notebook, navigate to the directory containing `hw6.ipynb` in the command line and run `jupyter notebook`. Then, open `hw6.ipynb` in the Jupyter notebook interface and run the cells.
