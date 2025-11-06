import os, sys
import pandas as pd


def read_fasta(fasta_file):
    sequences = []
    sid, seq = '', ''
    params = {}
    for line in open(fasta_file):
        line = line.rstrip()
        if line.startswith('>'):
            if seq != '':
                sequences.append([sid, seq])
                seq = ''
            sid = line.replace('>', '')
            try:
                sid, prm = sid.split()
                params[sid] = prm
            except:
                pass
        else:
            seq += line

    sequences.append([sid, seq])
    return sequences, params


def make_feature_table(msa_file):
    in_fasta = msa_file
    fasta, params = read_fasta(in_fasta)

    eval_fasta = []
    seq_labels = {}
    for sid, prm in params.items():
        seq_labels[sid] = prm

    for sid, seq in fasta:
        eval_fasta.append([sid, seq])

    alen = len(fasta[0][1])

    output = []
    headers = ['seq_id', 'obj_param',
               *[f"p{n+1}" for n in range(alen)]
               ]

    sys.stderr.write("Read MSA file...\n")
    for sid, seq in eval_fasta:
        if sid not in seq_labels.keys():
            sys.stderr.write(f"[Warning] No Param for {sid}. Skipped.\n")
            continue

        label = seq_labels[sid]
        output.append([
            sid, label, *seq
            ])

    feature_table = pd.DataFrame(output, columns=headers)
    sys.stderr.write(f"Feature table: {feature_table.shape}\n")

    return feature_table


if __name__ == '__main__':
    main()
