# BioCLIP2 Hierarchical Prompt Experiments

> **한 줄 요약** — BioCLIP2 같은 생물 이미지 모델에 종(species) 이름 대신 **분류학 계층 전체**(`Animalia Chordata Aves … Passer domesticus`)를 텍스트로 주면 성능이 좋아진다고 알려져 있다. 이 저장소는 그 효과가 **"텍스트가 정보를 더 줘서"** 생기는 것인지, 아니면 **"계층 구조가 임베딩 공간을 정리해서"** 생기는 것인지를 가르기 위한 실험 코드다.

---

## 1. 무엇을 알고 싶은가 (핵심 질문)

CLIP 계열 모델은 이미지와 텍스트를 같은 공간(embedding space)의 벡터로 바꾼다. 같은 종의 사진들은 이 공간에서 서로 가까이 모이고, 다른 종은 멀리 떨어진다. **BioCLIP2 논문**은 학습·평가 시 종 이름만 주는 대신 7단계 분류학(kingdom→phylum→class→order→family→genus→species)을 통째로 텍스트로 주면 인식이 더 잘 된다고 보고했다. 문제는 **왜** 좋아지는지가 불분명하다는 것이다. 경쟁하는 두 가설이 있다.

| 가설 | 주장 | 한 줄로 |
|------|------|---------|
| **A. 정보 채널 (Information channel)** | 계층 텍스트는 단지 토큰을 더 많이 공급해 텍스트 임베딩을 풍부하게 만든다. | *"텍스트가 많아서 좋아진다"* |
| **B. 의미 조직자 (Semantic organizer)** | 계층의 *구조 자체*가 임베딩을 재배치한다. 분류학적으로 가까운 종이 공간에서도 가까이 모이도록 정렬한다. 텍스트의 *내용*보다 *구조*가 본질이다. | *"계층이 공간을 정리해서 좋아진다"* |

> **비유 — 도서관에 책 꽂기**
> - 가설 A: "책 표지에 분류 라벨을 더 자세히 적으면 좋다" (정보가 많아짐)
> - 가설 B: "책장을 듀이 십진분류로 *정리*하면 좋다" (라벨 글자보다 *배치 구조*가 핵심)
>
> 이 연구는 둘 중 어느 쪽이 진짜 원인인지 가른다.

---

## 2. 어떻게 알아내는가 (실험 설계)

핵심 아이디어는 **프롬프트를 여러 방식으로 망가뜨려 보는 것**이다. 텍스트의 *내용*만 바꾸면 성능이 유지되고 *구조*를 깨면 무너진다면, 본질은 구조(가설 B)다. 5가지 프롬프트 조건을 비교한다.

| 조건 | 프롬프트 | 무엇을 조작했나 |
|------|----------|----------------|
| **C0** flat | `a photo of {species}` | 종 이름만 (기준선) |
| **C1** hierarchical | `a photo of {kingdom} {phylum} … {species}` | 정상 분류학 계층 (BioCLIP2 방식) |
| **C2** random-token | 상위 6개 rank를 의미 없는 토큰(`tax0 … tax5`)으로 치환 | **어휘는 무작위, 구조는 보존** |
| **C3** shuffled | 상위 rank 라벨을 다른 종에서 뽑아 끼워넣음 | **어휘는 진짜, 구조는 파괴** |
| **C4** word-bag | 7개 라벨의 순서만 매번 무작위로 섞음 | **순서만 제거, 어휘는 보존** |

이 조건들을 세 개의 실험(RQ = Research Question)에 걸쳐 측정한다.

| 실험 | 무엇을 하나 | 묻는 질문 |
|------|------------|-----------|
| **Exp1 (RQ1)** | C0 vs C1로 종 분리도(silhouette 등) 비교 | "계층 텍스트가 정말 종을 더 잘 분리하나?" |
| **Exp2 (RQ2)** | rank별(species/genus/family/order)로 분리도 비교 + 텍스트 없이 image embedding만 측정 | "효과가 어느 계층에서 나오나? 텍스트 없이도 분류학이 보이나?" |
| **Exp3 (RQ3)** | C0→C4 프롬프트 ablation | "텍스트의 *내용*이 중요한가, *구조*가 중요한가?" |

측정에 쓰는 지표(intra-class variance, inter-class margin, silhouette, RankMe, uniformity, kNN purity 등)와 통계 검정(paired permutation test, bootstrap CI)의 정확한 정의는 **[docs/metrics_guide.md](docs/metrics_guide.md)**에 정리돼 있다.

---

## 3. 무엇을 알아냈는가 (핵심 결과)

세 번의 실제 실행 — **BioCLIP2 × CUB-200**, **OpenCLIP × CUB-200**, **BioCLIP2 × Rare Species(multi-phylum)** — 에서 일관된 그림이 나왔다.

**겉으로 RQ1은 "실패"처럼 보인다.** 계층 프롬프트(C1)를 줬더니 종 분리도(silhouette)가 오히려 *떨어졌다* (CUB-200: 0.775 → 0.756). 하지만 이건 함정이다 — 진짜 신호는 Exp2·Exp3에 있다.

1. **Exp2(a) — rank별로 보면 그림이 뒤집힌다.** 계층 프롬프트는 species에선 손해지만 **genus·family·order 등 상위 rank에서는 일관되게 우세**하다. 즉 hierarchy는 "비슷한 종끼리 묶는" 일을 한다 — 그 대가로 종 내 fine-grained 분리를 조금 양보할 뿐. → **가설 B의 직접 증거.**

2. **Exp2(b) — 텍스트 없이도 분류학이 이미 거기 있다.** image embedding만으로 종 분리도를 재면 무작위 라벨 대비 **z-score ≈ 122 (CUB-200), 235 (Rare Species)** — 천문학적 수준의 유의성. → 텍스트가 분류학을 *가르친* 게 아니라, 임베딩에 *이미 잠재해 있던* 구조를 *꺼내 정렬*했을 뿐.

3. **Exp3 — 구조가 어휘보다 5배 중요.** silhouette 하락폭을 비교하면:
   - C2 (어휘 무작위, 구조 보존): **−0.078** ← 약간만 손해
   - C4 (순서 제거): −0.147 ← 중간
   - C3 (구조 파괴): **−0.354** ← **약 5배 더 큰 손해**

   토큰을 바꿔도 멀쩡한데 구조를 깨면 무너진다. → **텍스트의 *구조 정합성*이 본질.**

> ### 결론
> **BioCLIP2의 hierarchical prompt가 작동하는 이유는 텍스트가 "정보를 추가 공급"해서가 아니라, 분류학적 구조가 임베딩 공간에 *이미 잠재돼 있던 기하 구조*를 *꺼내 정렬*하기 때문이다. 텍스트는 정보 채널이 아니라 organizer prior다 (가설 B).**

전체 수치표, cross-model/cross-dataset 비교, 한계와 위협(seed-noise 통계 power, single-class confound 등)은 **[docs/results.md](docs/results.md)**에 상세히 기록돼 있다.

---

## 4. 빠른 시작

```bash
# 1) 설치
pip install -r requirements.txt

# 2) 모델·GPU 없이 파이프라인 검증 (mock 임베딩 + toy 데이터)
python src/run_experiment.py --mock --toy --seed 42 --out results/mock

# 3) 시각화까지 (UMAP / t-SNE PNG 생성)
python src/run_experiment.py --mock --toy --seed 42 \
  --visualize --vis_method tsne --out results/mock_vis
```

`Makefile`로도 동일하게 실행할 수 있다.

```bash
make install        # 의존성 설치
make mock           # mock + toy 빠른 검증
make cub MODEL=bioclip2   # CUB-200 전체 실행 (taxonomy 자동 빌드 → 인코딩 → Exp1~3)
```

### 실제 데이터로 실행하기

`--csv`에 아래 컬럼을 가진 파일을 주면 실제 이미지로 돈다. `file`은 `--image_root` 기준 상대 경로다.

```csv
file,kingdom,phylum,class,order,family,genus,species
img_0001.jpg,Animalia,Chordata,Aves,Passeriformes,Passeridae,Passer,Passer domesticus
```

```bash
python src/run_experiment.py \
  --model bioclip2 --seed 42 --n_seeds 5 \
  --csv data/CUB_200_2011/cub_taxonomy.csv \
  --image_root data/CUB_200_2011/images \
  --cache_emb results/cub200_bioclip2/img_emb.npz \
  --out results/cub200_bioclip2
```

지원 모델 키 (`--model`): `openclip-vitb32`, `openclip-vitl14`, `bioclip`, `bioclip2`.
한 번 인코딩한 image embedding은 `--cache_emb`의 `.npz`에 저장돼 재실행 시 재사용된다.

---

## 5. 출력 파일

기본 출력 디렉토리는 `results/<run-name>/`이며 `--out`으로 바꾼다.

```text
exp1_geometry.json          # C0 vs C1 평균/표준편차, permutation test, RQ1 성공 기준
exp2_rank_levels.json       # rank별 C0/C1 metric, latent taxonomy probe
exp3_counterfactuals.json   # C2/C3/C4 preservation ratio와 bootstrap CI
run_log.txt                 # 실행 인자, git hash, 라이브러리 버전
*.png                       # --visualize 사용 시 조건별/비교 grid 이미지
```

여러 run의 결과를 모델별 비교표로 요약하려면 `src/summarize_rare_species_results.py`를 쓴다.

---

## 6. 저장소 구조

```text
bioclip2-repo/
├── src/
│   ├── run_experiment.py          # 실험 전체를 조립하는 진입점 (Exp1~3 실행)
│   ├── prompt_variants.py         # 5가지 프롬프트 조건(C0~C4) + TaxonomyRecord 정의
│   ├── data_loader.py             # toy / 실제 CSV 데이터를 동일 형태로 로드
│   ├── extract_embeddings.py      # 모델 로드 + 이미지/텍스트 임베딩 추출 (mock 포함)
│   ├── metrics.py                 # geometry 지표 + 통계 검정 + 시각화
│   ├── cub200_build_taxonomy.py   # CUB-200에 GBIF로 7-rank taxonomy 자동 부착
│   ├── prepare_rare_species.py    # HuggingFace rare-species 데이터셋 export
│   ├── summarize_*.py             # 여러 run 결과를 markdown 비교표로 요약
│   └── models/baseline_flat.py    # C0 flat prompt helper
├── docs/
│   ├── protocol.md                # 데이터셋·모델·지표·통계의 정확한 실험 프로토콜
│   ├── metrics_guide.md           # 각 지표의 정의·논문 역할·파라미터
│   └── results.md                 # 전체 결과·해석·한계 (개념 설명 포함)
├── data/                          # CUB-200 등 원천 데이터
├── results/                       # run별 산출물 (JSON, npz, png)
├── Makefile                       # 실행 단축 명령 (mock / toy / cub)
└── requirements.txt               # 고정 버전 의존성
```

### 코드 동작 순서 (run_experiment.py)

1. `argparse`로 모델·데이터·seed·출력·캐시·시각화 옵션을 읽는다.
2. `.env`에서 `HF_TOKEN`을 로드하고 (`_load_env_file`, `_configure_hf_token`), `set_global_seeds()`로 numpy/torch/CUDA seed를 고정한다.
3. 데이터를 만든다 — `--csv`가 있으면 `load_real_dataset()`, 없으면 `get_toy_dataset()`(8종×6샘플).
4. image embedding을 만든다 — 캐시(`--cache_emb`) → mock(`--mock`) → 실제 모델(`load_model` + `encode_images`) 순.
5. Exp1 → Exp2 → Exp3을 순서대로 실행한다.
6. 결과를 JSON으로 저장하고, `--visualize`면 UMAP/t-SNE PNG를 추가 저장한다.
7. git hash와 라이브러리 버전을 `run_log.txt`에 남긴다.

---

## 7. 참고

- **데이터셋**: CUB-200-2011 (Wah et al., 2011, 학술용), Rare Species (본 연구 구축, multi-phylum). 자세한 라이선스·전처리는 [docs/protocol.md](docs/protocol.md).
- **모델**: BioCLIP2 ViT-L/14 (`hf:imageomics/bioclip-2`), OpenCLIP ViT-L/14 (`laion2b_s32b_b82k`). 둘 다 frozen 평가.
- **taxonomy 출처**: GBIF API (CC0).

> 재현 시 `requirements.txt`의 고정 버전과 `docs/protocol.md`의 seed·배치 설정을 그대로 사용할 것. 일부 통계(Cohen's d 등)는 seed-noise scale 때문에 inflate되므로 raw difference와 cross-run 재현성 위주로 해석한다 (자세한 내용은 [docs/results.md](docs/results.md)의 "한계 / 위협").
