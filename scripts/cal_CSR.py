import argparse

def main(args):
    hypcons=0
    totalcons=0
    with open(args.src,'r') as fsrc, open(args.tgt,'r') as ftrg, open(args.hyp,'r') as fhyp: 
        for _, src_line in enumerate(fsrc):
            trg_line=ftrg.readline().strip('\n')
            hyp_line=fhyp.readline().strip('\n')
            src_line=src_line.strip('\n')           
            if '<sep>' in src_line:
                cons=[x.strip() for x in src_line.split('<sep>')[1:]]
                cons_clean=' '.join(cons).split() 
                cons=[]
                for x in cons_clean:
                    if x not in cons and (x != '<unk>'):
                        cons.append(x)
                hypcons = hypcons + sum([int(x in hyp_line.split()) for x in cons])     
                totalcons = totalcons + sum([int(x in trg_line.split()) for x in cons])
    print(f"The copy success rate is: {hypcons/(totalcons+0.0001)}. Among given {totalcons} constraints, {hypcons} constraints are copied successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source sentences")
    parser.add_argument("--tgt", required=True, help="target sentences")
    parser.add_argument("--hyp", required=True, help="decoding sentences")
    args = parser.parse_args()
    main(args)