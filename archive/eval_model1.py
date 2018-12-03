
with open(sys.argv[3], "wb+") as f:
    pickle.dump((dice, dicf, pfe), f)

count_null = 0
total_w = 0.0

# Inference
with open(sys.argv[3], "w+") as outfile:
    for sf, se in zip(corpf, corpe):
        probs = []
        aligns = []
        for t, wf in enumerate(se):
            max_pos = -1
            max_prob = 0.0
            for i, we in enumerate(sf):
                if (we, wf) not in pfe:
                    print("Error")

                prob = pfe[(we, wf)]

                if sf[i] == dicf[NULL]:
                    prob *= NULL_PRIOR
                # else:
                #     prob *= (1-NULL_PRIOR)

                if prob > max_prob:
                    max_prob = prob
                    max_pos = i

            if sf[max_pos] == dicf[NULL]:
                print("Null")
                continue

            # aligns.append(str(max_pos) + '-' + str(t))
            # probs.append(max_prob)
            outfile.write('%d-%d ' % (max_pos, t))
        # min_conf = probs.index(min(probs))
        # del aligns[min_conf]
        # outfile.write(" ".join(aligns))
        outfile.write('\n')
