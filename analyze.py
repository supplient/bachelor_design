
def cal_metrics(expect_tags, output_tags):
    # Cal cost precision
    ## Count each tag category's num
    total_expect = 0
    expect_tag_num = {}
    for tag in expect_tags:
        num = expect_tag_num.get(tag, 0)
        expect_tag_num[tag] = num + 1
        if tag != "":
            total_expect += 1
    output_tag_num = {}
    for tag in output_tags:
        num = output_tag_num.get(tag, 0)
        output_tag_num[tag] = num + 1

    ## Cal each tag's cost precision
    cost_p = {}
    for tag in expect_tag_num.keys():
        if tag == "":
            continue
        expect = expect_tag_num.get(tag, 0)
        output = output_tag_num.get(tag, 0)
            
        diff = abs(expect - output)
        if expect == 0:
            cost_p[tag] = 0
        else:
            cost_p[tag] = max(0, expect-diff)/total_expect

    ## Cal total cost precision
    total_cost_p = 0
    for tag, p in cost_p.items():
        total_cost_p += p

    # Cal label precision
    right_count = 0
    for expect, output in zip(expect_tags, output_tags):
        if expect == output:
            right_count += 1
    label_p = right_count/len(output_tags)

    return total_cost_p, label_p


def analyze(recpath, tablepath):
    import json
    import csv
    
    # Load train record
    train_rec = []
    with open(recpath, "r") as fd:
        train_rec = json.load(fd)

    # Calculate metrics
    cost_prec_list = []
    label_prec_list = []
    char_prec_list = []
    for rec in train_rec[1:]:
        cost_prec, label_prec = cal_metrics(
            rec["expect_tags"],
            rec["output_tags"]
        )
        char_prec = rec["logs"]["val_crf_viterbi_accuracy"]

        cost_prec_list.append(cost_prec)
        label_prec_list.append(label_prec)
        char_prec_list.append(char_prec)

    # Build rows
    rows = []
    header = ["epoch", "cost precision", "label precision", "character precision"]
    rows.append(header)
    for i in range(len(cost_prec_list)):
        rows.append([
            str(i+1),
            cost_prec_list[i],
            label_prec_list[i],
            char_prec_list[i]
        ])

    # Output CSV
    with open(tablepath, "w") as fd:
        writer = csv.writer(fd)
        writer.writerows(rows)
    

if __name__ == "__main__":
    from driver_amount import addh
    import config
    analyze(
        addh + config.TRAIN_REC_PATH,
        addh + config.TRAIN_TABLE_PATH
    )