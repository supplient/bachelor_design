
class FUNCOMO:

    fp_p = 0.65
    fp_q = 0.01
    VAF_value = [0, 1, 2, 3, 4, 5]

    lang_factor = {
        "assemble" : 0.320,
        "C": 0.128,
        "C++": 0.064,
        "VB": 0.032,
        "JAVA": 0.030,
        "SQL": 0.012,
    }
    
    KLOC_p = 0.91
    KLOC_q = 0.01
    w_value = [0, 1, 2, 3, 4, 5]
    develop_mode_factor = {
        "organic": 2.4,
        "semi-detached": 3.0,
        "embedded": 3.6,
    }
    EM_factor = {
        "early-design": 7,
        "post-arch": 17,
    }

    @classmethod
    def FP2Scale(cls, fp_counts, p, q, VAF):
        VAF_sum = sum(VAF)
        factor = p + q * VAF_sum
        fp_sum = sum(fp_counts.values())
        return factor * fp_sum

    @classmethod
    def scale2KLOC(cls, scale, factor):
        return factor * scale

    @classmethod
    def KLOC2PM(cls, KLOC, a, EM, p, q, w):
        w_sum = sum(w)
        b = p + q * w_sum
        return a * EM * KLOC**b

    @classmethod
    def test(cls):
        fp_counts = {
            "EO": 10,
            "EQ": 90
        }
        VAF = [2, 2]
        scale = FUNCOMO.FP2Scale(
            fp_counts,
            FUNCOMO.fp_p,
            FUNCOMO.fp_q,
            VAF
        )

        lang = "C++"
        KLOC = FUNCOMO.scale2KLOC(
            scale,
            FUNCOMO.lang_factor[lang]
        )

        w = [2, 2, 2, 2, 2]
        develop_mode = "semi-detached"
        em = "early-design"
        PM = FUNCOMO.KLOC2PM(
            KLOC,
            FUNCOMO.develop_mode_factor[develop_mode],
            FUNCOMO.EM_factor[em],
            FUNCOMO.KLOC_p,
            FUNCOMO.KLOC_q,
            w
        )

        print({
            "scale": scale,
            "KLOC": KLOC,
            "PM": PM
        })


if __name__ == "__main__":
    FUNCOMO.test()