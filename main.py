import numpy as np
from PIL import Image

class PCA:
    def __init__(self, name):
        self.name = name

    def read_image(self):
        im = Image.open(self.name)
        self.sign = input("For RGB PCA, press r; for greyscale image, press l\n")
        if self.sign == "r":
            self.width, self.height = im.size
            self.source = np.array(im.getdata())
            self.n, self.dimension = self.source.shape
        elif self.sign == "l":
            im_gray = im.convert("L")
            self.width, self.height = im_gray.size
            print(self.width)
            print(self.height)
            self.source = np.array(im_gray)
            #self.n, self.dimension = self.source.shape
        #im_data = [im_data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
        #print(self.n)
        #print(self.dimension)
        print(self.source.shape[0])
        print(self.source.shape[1])
        print(self.source)

    def zeroMean(self):
        self.mean_val = np.mean(self.source, axis=0)
        print(self.mean_val)
        print(self.mean_val.shape[0])
        self.phi_data = self.source - self.mean_val
        print(self.phi_data)

    def cov(self):
        self.covariance_matrix = np.cov(self.phi_data, rowvar=0)

    def eigen(self):
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance_matrix)
        print(self.eigenvalues)
        print(self.eigenvectors)
        print(self.eigenvectors.shape[0])
        print(self.eigenvectors.shape[1])

    def new_base(self):
        if self.sign == "r":
            new_eig_vec = self.eigenvectors[0:2]
            print(new_eig_vec.shape[0])
            print(new_eig_vec.shape[1])
            self.lower_data = np.dot(self.phi_data, new_eig_vec.T) + np.delete(self.mean_val, 2)
        elif self.sign == "l":
            print("%d bases" % np.size(self.eigenvectors))
            new_n = int(input("The number of new bases: "))
            eig_seq = np.argsort(self.eigenvalues)
            eig_seq_indice = eig_seq[-1:-(new_n+1):-1]
            new_eig_vec = self.eigenvectors[:,eig_seq_indice]
            self.lower_data = np.dot(self.phi_data, new_eig_vec)
            self.reconstruction = np.dot(self.lower_data, new_eig_vec.T) + self.mean_val
        #self.new_matrix = (lower_data * new_eig_vec.T) + self.mean_val

    def show(self):
        if self.sign == "r":
            temp = np.array([[0] * self.n])
            show_data = np.insert(self.lower_data, 2, values = temp, axis=1)
            print(show_data)
            show_data = show_data.astype('uint8')
            #show_data = show_data.reshape(self.width, self.height, 3)
            #show_data.reshape(show_data, show_data.shape + (1,))
            #print(show_data)
            target_shape = (self.width, self.height, 3)
            strides = show_data.itemsize * np.array([8277, 12000, 1])
            show_data = np.lib.stride_tricks.as_strided(show_data, shape=target_shape, strides=strides)
            new_im = Image.fromarray(show_data, mode='RGB')
        elif self.sign == "l":
            show_data = self.reconstruction.astype('uint8')
            new_im = Image.fromarray(show_data, mode="L")
        new_im.show()
        new_im.save("test8.PNG", "PNG")


    def analyze(self):
        self.read_image()
        self.zeroMean()
        self.cov()
        self.eigen()
        self.new_base()
        self.show()

if __name__ == '__main__':
    p = PCA("test.jpg")
    p.analyze()



