# MIGRATE

**M**achine learning-based **I**dentification of **G**lobally adaptive amino-acid **R**esidues
**A**ssociated with functional **T**raits and diverse **E**nvironmental factors

---

## Prerequisites

python 3.9 or later.



## Installation

```bash

tar zxf migrate_package.tgz

cd migrate_package

pip install .

```

## Usage

```bash

migrate msa_file --target seq_id [--mode classification]

```

### Example

```bash
migrate migrate/data/Myb_cls.fas --target Msp.Q0KIY5.3.MYG.KOGBR.pygmy.sperm.whale \
                                 --pdbid 6bmgA --pdb-start 0 \
                                 --mode classification \
                                 --seed 123 

migrate migrate/data/GFP_rgr.fas --target mKalama1 \
                                 --pdbid 4ornB \
                                 --pdb-start 0 \
                                 --mode regression \
                                 --seed 123 

```

### MSA data format

A multiple sequence alignment in FASTA format with target parameters in each header.

\>seqid1 param1  
SEQUENCE-ALIGNMENT...  
\>seqid2 param2  
SEQUENCE-ALIGNMENT...  
...

