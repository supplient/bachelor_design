from funcomo import FUNCOMO

# 一个信息管理系统
# 可靠性要求高，但对性能要求较低
# 需要较大的数据库，开发人员经验丰富
# 开发日程有规定，但并不紧凑
pro_A = {
   "fp_counts": {
       "ILF": 8,
       "EIF": 3,
       "EI": 20,
       "EO": 20,
       "EQ": 10,
   },
   "fp_p": FUNCOMO.fp_p,
   "fp_q": FUNCOMO.fp_q,
   "VAF": [
       5, 2, 5,
       0, 3, 3,
       3, 3, 3,
       0, 4, 3,
       2, 3,
   ],

   "lang": "JAVA",

    "develop_mode": "organic",
    "EAF": {
        "required software reliability": 1.15,
        "size of application database": 1.08,
        "analyst capaility": 0.86,
        "applications experience": 0.82,
        "required development schedule": 1.04
    }
}

# 一个实时反馈系统
# 可靠性要求与性能要求都很高
# 但不要求可移植性，也几乎不作更新
# 处理的数据也很简单
# 开发人员经验欠缺，但能力很强
pro_B = {
   "fp_counts": {
       "ILF": 3,
       "EIF": 2,
       "EI": 12,
       "EO": 15,
       "EQ": 16,
   },
   "fp_p": FUNCOMO.fp_p,
   "fp_q": FUNCOMO.fp_q,
   "VAF": [
       4, 2, 0,
       5, 3, 2,
       0, 0, 2,
       5, 1, 1,
       1, 1,
   ],

   "lang": "C++",

    "develop_mode": "semi-detached",
    "EAF": {
        "required software reliability": 1.40,
        "size of application database": 0.94,
        "run-time performance constraints": 1.66,
        "memory constraints": 1.56,
        "applications experience": 1.13,
        "software engineer capability": 0.86,
    }
}

# 一个学生作业
# 内容非常简单，也没有高的性能要求
# 但是开发人员对内容非常不熟悉
# 而且后续作业中也要继续使用
# 老师已经给出作业的框架设计
pro_C = {
   "fp_counts": {
       "ILF": 1,
       "EIF": 0,
       "EI": 3,
       "EO": 1,
       "EQ": 2,
   },
   "fp_p": FUNCOMO.fp_p,
   "fp_q": FUNCOMO.fp_q,
   "VAF": [
       0, 0, 0,
       0, 0, 0,
       0, 0, 1,
       2, 2, 3,
       3, 4,
   ],

   "lang": "VB",

    "develop_mode": "embedded",
    "EAF": {
        "required software reliability": 0.75,
        "applications experience": 1.29,
        "virtual machine experience": 1.21,
        "programming language experience": 1.14,
    }
}

def calPro(pro):
    res = {}
    res["AFP"] = FUNCOMO.FP2AFP(
        pro["fp_counts"],
        pro["fp_p"],
        pro["fp_q"],
        pro["VAF"],
    )
    res["KLOC"] = FUNCOMO.scale2KLOC(
        res["AFP"],
        FUNCOMO.lang_factor[pro["lang"]],
    )
    res["PM"] = FUNCOMO.KLOC2PM(
        res["KLOC"],
        FUNCOMO.develop_mode_factor[pro["develop_mode"]][0],
        FUNCOMO.develop_mode_factor[pro["develop_mode"]][1],
        pro["EAF"],
    )
    return res

if __name__ == "__main__":
    res_A = calPro(pro_A)
    res_B = calPro(pro_B)
    res_C = calPro(pro_C)
    print("A" + str(res_A))
    print("B" + str(res_B))
    print("C" + str(res_C))