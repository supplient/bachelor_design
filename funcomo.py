
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
    
    develop_mode_factor = {
        "organic": (2.5, 1.05),
        "semi-detached": (3.0, 1.12),
        "embedded": (3.6, 1.2)
    }

    @classmethod
    def FP2AFP(cls, fp_counts, p, q, VAF):
        VAF_sum = sum(VAF)
        factor = p + q * VAF_sum
        fp_sum = sum(fp_counts.values())
        return factor * fp_sum

    @classmethod
    def scale2KLOC(cls, scale, factor):
        return factor * scale

    @classmethod
    def KLOC2PM(cls, KLOC, a, b, EAF):
        eaf = 1
        for ele in EAF.values():
            eaf *= ele
        return a * eaf * KLOC**b

    @classmethod
    def test(cls):
        fp_counts = {
            "EO": 10,
            "EQ": 90
        }
        VAF = [
            3, 3, 3,
            3, 3, 3,
            3, 3, 3,
            3, 3, 3,
            3, 3,
        ]
        AFP = FUNCOMO.FP2AFP(
            fp_counts,
            FUNCOMO.fp_p,
            FUNCOMO.fp_q,
            VAF
        )

        lang = "C++"
        KLOC = FUNCOMO.scale2KLOC(
            AFP,
            FUNCOMO.lang_factor[lang]
        )

        develop_mode = "semi-detached"
        EAF = {
            "required software reliability": 1.15,
            "run-time performance constraints": 1.11,
            "analyst capaility": 0.86,
            "applications experience": 0.82,
            "required development schedule": 1.04
        }
        PM = FUNCOMO.KLOC2PM(
            KLOC,
            FUNCOMO.develop_mode_factor[develop_mode][0],
            FUNCOMO.develop_mode_factor[develop_mode][1],
            EAF
        )

        print({
            "AFP": AFP,
            "KLOC": KLOC,
            "PM": PM
        })


if __name__ == "__main__":
    FUNCOMO.test()