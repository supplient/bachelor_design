import numpy as np
from scipy.spatial.distance import mahalanobis


class DistCal:
    methods = [
        "euclidean",
        "cos",
        # "cos_sub_mean", # All input has been normalized, the mean is 0. So this has no meaning.
        "manhattan",
        # "std_euclidean", # The reason as above.
        "chebyshev",
        "mahalanobis"
    ]

    def __init__(self, all_data):
        self.all_data = np.array(all_data)
        self.mean = np.mean(self.all_data)
        self.std_dev = np.std(self.all_data, ddof=1)

        cov = np.cov(np.transpose(all_data), ddof=1)
        u, sigma, ut = np.linalg.svd(cov)
        inv_sigma = [1/x for x in sigma]
        inv_S = np.zeros(cov.shape)
        for i in range(inv_S.shape[0]):
            inv_S[i][i] = inv_sigma[i]
        self.inv_cov = np.dot(
            np.dot(u, inv_S),
            ut
        )

    def cal(self, a, b, method_name="cos"):
        # Select Method to Calculate Distance
        method = getattr(
            DistCal,
            "_method_" + method_name,
            None
        )
        if method == None:
            raise Exception("Invalid method name: " + method_name)
        return method(self, a, b)



    def _method_euclidean(self, a, b):
        return np.linalg.norm(np.subtract(a, b))

    def _method_cos(self, a, b):
        dot_sum = np.dot(a, b)
        norm_product = np.linalg.norm(a) * np.linalg.norm(b)
        cos_val = dot_sum / norm_product
        # Note, cos_val==1 means a and b are the same. 
        # However, here we want a distance, which should be 0 when there are two same points.
        return 1 - abs(cos_val) 

    def _method_cos_sub_mean(self, a, b):
        return self._method_cos(
            np.subtract(a, self.mean),
            np.subtract(b, self.mean)
        )

    def _method_manhattan(self, a, b):
        res = 0
        for i, j in zip(a, b):
            res += abs(i-j)
        return res

    def _method_std_euclidean(self, a, b):
        a = np.subtract(a, self.mean)
        a = np.divide(a, self.std_dev)
        b = np.subtract(b, self.mean)
        b = np.divide(b, self.std_dev)
        return self._method_euclidean(a, b)

    def _method_chebyshev(self, a, b):
        return max([abs(i-j) for i,j in zip(a,b)])

    def _method_mahalanobis(self, a, b):
        return mahalanobis(a, b, self.inv_cov)



    @classmethod
    def test(cls):
        a = [1, 0]
        b = [2, 2]
        
        dist_cal = DistCal([a, b])
        for method in DistCal.methods:
            dist = dist_cal.cal(a, b, method)
            print(method + ": " + str(dist))


if __name__ == "__main__":
    DistCal.test()