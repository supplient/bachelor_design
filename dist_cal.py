def _dot(a, b):
    res = 0
    for x, y in zip(a, b):
        res += x*y
    return res

def _mod(a):
    res = sum([x*x for x in a])
    return res**0.5

class DistCal:
    def __init__(self):
        self.method = DistCal.cos_dist

    @classmethod
    def cos_dist(cls, a, b):
        dot_sum = _dot(a, b)
        mod_product = _mod(a) * _mod(b)
        cos_val = dot_sum / mod_product
        # Note, cos_val==1 means a and b are the same. 
        # However, here we want a distance, which should be 0 when there are two same points.
        return 1 - abs(cos_val) 

    def cal(self, a, b):
        return self.method(a, b)

    @classmethod
    def test(cls):
        a = [1, 0]
        b = [2, 2]
        
        dist_cal = DistCal()
        dist = dist_cal.cal(a, b)
        print(dist)


if __name__ == "__main__":
    DistCal.test()