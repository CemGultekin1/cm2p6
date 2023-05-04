.ONESHELL:

.DEFAULT_GOAL := setup-greene

MEMORY = 15
GZFILE = overlay-$(MEMORY)GB-500K.ext3.gz
EXTFILE = overlay-$(MEMORY)GB-500K.ext3
GZBANK = /scratch/work/public/overlay-fs-ext3
GZPATH = $(GZBANK)/$(GZFILE)
CUDA_SINGULARITY = /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04.sif
SOURCE = /ext3/env.sh


define newline


endef

SOURCE_TEXT= "\#!/bin/bash $(newline) source /ext3/miniconda3/etc/profile.d/conda.sh $(newline) export PATH=/ext3/miniconda3/bin:\$$PATH $(newline) export PYTHONPATH=\$$(pwd)"

setup-miniconda:
	cp -rp $(GZPATH) .
	gunzip $(GZFILE)
	echo $(SOURCE_TEXT) > env.sh
	singularity exec --overlay $(EXTFILE) $(CUDA_SINGULARITY) /bin/bash -c "\
		wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh;\
		sh Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3;\
		export PATH=/ext3/miniconda3/bin:\$$PATH;
		conda update -n base conda -y;
		mv env.sh /ext3/env.sh;
		rm Miniconda3-latest-Linux-x86_64.sh;
		rm -rf $(GZFILE);
	"
	
setup-conda-env:
	singularity exec --overlay $(EXTFILE) $(CUDA_SINGULARITY) /bin/bash -c "\
		source /ext3/env.sh;
		pip install -r requirements.txt;
		pip install zarr xarray fsspec aiohttp requests;
		pip install gcm_filters;
		python -m pip install "xarray[io]";
	"
setup-greene: 
	make setup-miniconda
	make setup-conda-env
	echo "EXTFILE = $(EXTFILE)">root.txt
	echo "CUDA_SINGULARITY = $(CUDA_SINGULARITY)">>root.txt

interactive-singularity-writing-permitted:	
	echo run \"source /ext3/env.sh\"
	echo print \"exit\" to exit
	singularity exec --nv --overlay $(EXTFILE):rw $(CUDA_SINGULARITY) /bin/bash

interactive-singularity-read-only:	
	echo run \"source /ext3/env.sh\"
	echo print \"exit\" to exit
	singularity exec --nv --overlay $(EXTFILE):ro $(CUDA_SINGULARITY) /bin/bash

.SILENT: setup-conda-env setup-miniconda create-root-txt interactive-singularity-read-only interactive-singularity-writing-permitted setup-greene