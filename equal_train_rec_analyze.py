import config
import json
from prettytable import PrettyTable

def calMetrics(train_rec):
    for method1 in train_rec.keys():
        for method2 in train_rec[method1].keys():
            rec = train_rec[method1][method2]

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

def displayTable(train_rec, output_file=None, output_table=None, precision=2):
    rep = ""
    table_rows = []

    # Get methods' name
    row_labels = [x for x in train_rec.keys()]
    column_labels = [x for x in train_rec[row_labels[0]].keys()]

    # First, display each method's sub table
    header = [""]
    header.extend(column_labels)
    sub_row_names = train_rec[row_labels[0]][column_labels[0]].keys()
    for row_label in row_labels:
        header[0] = "[%s]" % row_label
        table_rows.append(header.copy())
        t = PrettyTable(
            field_names=header,
            header=True,
        )
        for row_name in sub_row_names:
            row = [row_name]
            for column_label in column_labels:
                row.append(
                    round(train_rec[row_label][column_label][row_name], precision)
                )
            t.add_row(row)
            table_rows.append(row)
        rep += str(t) + "\n\n"
        table_rows.append([None])

    # Then, display a total table
    rep += "Total: " + "\n"
    header[0] = "F1"
    table_rows.append(header.copy())
    t = PrettyTable(
        field_names=header,
        header=True
    )
    for row_label in row_labels:
        row = [row_label]
        for column_label in column_labels:
            row.append(
                round(train_rec[row_label][column_label]["F1"], precision)
            )
        t.add_row(row)
        table_rows.append(row)
    rep += str(t)

    if output_file:
        with open(output_file, "w") as fd:
            fd.write(rep)
    if output_table:
        import csv
        with open(output_table, "w", newline='') as fd:
            writer = csv.writer(fd)
            writer.writerows(table_rows)
    print(rep)

def makeTrainRep(train_rec_path, train_rep_path, train_table_path=None, precision=2):
    rep = ""

    # Load train record
    train_rec = None
    with open(train_rec_path, "r") as fd:
        train_rec = json.load(fd)

    # Cacluate metrics
    calMetrics(train_rec)

    # Display tables
    displayTable(
        train_rec, 
        output_file=train_rep_path, 
        output_table=train_table_path,
        precision=precision,
    )

if __name__ == "__main__":
    from driver_amount import addh
    makeTrainRep(
        addh + config.EQUAL_TRAIN_REC_PATH,
        addh + config.EQUAL_TRAIN_REP_PATH,
        addh + config.EQUAL_TRAIN_TABLE_PATH,
    )
    # makeTrainRep(
        # addh + config.EQUAL_SIF_TRAIN_REC_PATH,
        # addh + config.EQUAL_SIF_TRAIN_REP_PATH,
        # 4
    # )