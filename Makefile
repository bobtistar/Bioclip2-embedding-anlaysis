MODEL        ?= openclip-vitb32
SEED         ?= 42
N_SEEDS      ?= 5
CUB_ROOT     ?= data/CUB_200_2011
OUT          ?= results/$(MODEL)
TAX_CSV      := $(CUB_ROOT)/cub_taxonomy.csv
CACHE_EMB    := $(OUT)/img_emb.npz

.PHONY: mock toy cub quick install

install:
	pip install -r requirements.txt

mock:
	python src/run_experiment.py --mock --toy --seed $(SEED) --out results/mock

toy:
	python src/run_experiment.py --model $(MODEL) --toy --seed $(SEED) --out results/toy

$(TAX_CSV):
	python src/cub200_build_taxonomy.py --cub_root $(CUB_ROOT) --out_csv $(TAX_CSV)

cub: $(TAX_CSV)
	mkdir -p $(OUT)
	python src/run_experiment.py \
		--model $(MODEL) --seed $(SEED) --n_seeds $(N_SEEDS) \
		--csv $(TAX_CSV) --image_root $(CUB_ROOT)/images \
		--cache_emb $(CACHE_EMB) --out $(OUT)

quick:
	$(MAKE) cub MAX_PER_CLASS=5
