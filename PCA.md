### Principle Component Analysis
#### A simple implementation of PCA based on greyscale images and ED
##### Algorithm
- read the image, and convert it to a greyscale one (using PIL), construct the input matrix $X$ based on the greyscale image (using numpy), consider each row of the matrix represents a sample, and each column represents a different feature.
- calculate the mean of each feature(each column), then subtract it from the input matrix $X$
```
    def zeroMean(self):
        self.mean_val = np.mean(self.source, axis=0)
        self.phi_data = self.source - self.mean_val
```
- calculate the covariance matrix of the input matrix $X$
```
    def cov(self):
        self.covariance_matrix = np.cov(self.phi_data, rowvar=0)
```
- calculate the eigenvalues and eigenvectors of the covariance matrix
```
    def eigen(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)
```
- construct the score matrix $T$ based on the L largest eigenvalues and eigenvectors, and reconstruct the image
```
    def new_base(self):
        print("%d bases" % np.size(self.eigenvectors))
        new_n = int(input("The number of new bases: "))
        eig_seq = np.argsort(self.eigenvalues)
        eig_seq_indice = eig_seq[-1:-(new_n+1):-1]
        new_eig_vec = self.eigenvectors[:,eig_seq_indice]
        self.lower_data = np.dot(self.phi_data, new_eig_vec)
        self.reconstruction = np.dot(self.lower_data, new_eig_vec.T) + self.mean_val
```
##### Results
- PCs # 100
![](test7.PNG)
- PCs # 500
![](test8.PNG)
- PCs # 1000
![](test6.PNG)
#### Q1 : Why the variances $\sum_{ij}$ also defines the signal-to-noise ratio? And the properties of SNR w.r.t. $\sum_{ij}$
- failed to find relevant information on this topic

#### Q2 : Beside performing ED on $X^TX$, is there other way to obtain the principal components?
- **Singular value decomposition(SVD)** :
     - for the data matrix $X$ : 
     $$ X = U\Sigma W^T$$
        - $\Sigma$ is a n-by-p rectangular diagonal matrix of positive numbers $\sigma_k$ called the **singular values** of $X$
        - U is an n-by-n matrix, the columns of which are orthogonal unit vectors of length n called the **left singular vectors** of $X$
        - W is a p-by-p matrix whose columns are orthogonal unit vectors of length p and called the **right singular vectors** of X.
    - score matrix $T_L$:
    $$ T_L=XW_L=U_L\Sigma_L$$
    choosing the L largest singular values and singular vectors

#### Q3 : Can PCA handle the data drawn from multiple subspace?
- A **generalized principle component analysis(GPCA)** is required for multiple-subspce problems
    - the number of subspaces and their dimensions
    - a basis for each subspace
    - the segmentation of data points

#### Q4 : PCA is a unsupervised dimension reduction method, which may suffer from what problem or limitations?
- not robust to outliers -> **RPCA** method which separates the input data into a low-rank matrix and a sparse noise matrix
- could only be directly applied to one single-subspace problems -> **GPCA** method

#### Q5 : What distance is adopted by PCA to measure the relation among data points? Could such a measurement solve the linear inseparable issue? If Yes/NO, why?
- Euclidean distance (covariance matrix)
- PCA relies on finding orthogonal projections of the dataset that contains the highest variance possible. In other words, PCA aims at finding the hidden *linear correlations* of the dataset. Therefore, for linear inseparable issues, PCA is not enough. A **Kernel PCA** is required.

