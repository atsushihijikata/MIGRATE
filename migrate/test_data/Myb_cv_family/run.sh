#!/bin/sh -f


migrate --target Msp.Q0KIY5.3.MYG.KOGBR.pygmy.sperm.whale \
        --cv-fold Myb_cls_family_cv.txt \
        --mode classification \
	--seed 123 \
	../Myg_cls.fas
