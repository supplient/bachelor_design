from driver_amount import addh
import config
import json
from prettytable import PrettyTable

if __name__ == "__main__":
    rep = ""

    # Load train record
    train_rec = None
    with open(addh + config.EQUAL_TRAIN_REC_PATH, "r") as fd:
        train_rec = json.load(fd)

    # Get methods' name
    dist_methods = [x for x in train_rec.keys()]
    emb_methods = [x for x in train_rec[dist_methods[0]].keys()]

    # Cacluate metrics
    for dist_method in train_rec.keys():
        for emb_method in train_rec[dist_method].keys():
            rec = train_rec[dist_method][emb_method]

            # There are N positive examples and N*(N-1)/2 negative examples
            #   , which is too imbalance, to avoid biases, we multiply a factor
            N = rec["TP"] + rec["FN"]
            factor = (N-1)/2

            TP = rec["TP"] * factor
            FN = rec["FN"] * factor
            FP = rec["FP"]
            TN = rec["TN"]
            
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f1 = 2*p*r / (p+r)

            rec["precision"] = p
            rec["recall"] = r
            rec["F1"] = f1

    # Display tables
    ## First, display each method's sub table
    header = [""]
    header.extend(emb_methods)
    sub_row_names = train_rec[dist_methods[0]][emb_methods[0]].keys()
    for dist_method in dist_methods:
        header[0] = dist_method
        t = PrettyTable(
            field_names=header,
            header=True,
        )
        for row_name in sub_row_names:
            row = [row_name]
            for emb_method in emb_methods:
                row.append(
                    round(train_rec[dist_method][emb_method][row_name], 2)
                )
            t.add_row(row)
        rep += str(t) + "\n\n"

    ## Then, display a total table
    rep += "Total: " + "\n"
    header[0] = "F1"
    t = PrettyTable(
        field_names=header,
        header=True
    )
    for dist_method in dist_methods:
        row = [dist_method]
        for emb_method in emb_methods:
            row.append(
                round(train_rec[dist_method][emb_method]["F1"], 2)
            )
        t.add_row(row)
    rep += str(t)

    with open(addh + config.EQUAL_TRAIN_REP_PATH, "w") as fd:
        fd.write(rep)
    print(rep)
