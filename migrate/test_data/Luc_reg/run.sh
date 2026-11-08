#!/bin/sh -f


migrate --target Luciola_cruciata_AB220162.554_562nm.pdb2d1tA \
        --mode regression \
	--seed 123 \
	./Luc_reg.fas
