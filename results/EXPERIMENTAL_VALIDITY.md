# Experimental Details & Validity

> 이 문서는 [ANALYSIS.md](ANALYSIS.md)(결과 분석)의 **부록**으로, (1) 각 실험이 *구체적으로 어떻게* 수행됐는지(데이터 구축·서브샘플링·rank skip 규칙·메모리 관리), (2) 그 결과가 *믿을 만한지*에 대한 **4종 타당성(validity) 검증**을 정리한다. 모든 수치는 `results/*/run_log.txt`와 소스 코드에서 직접 확인했다.

---

## Part I. 실험 세부 구현 (How exactly)

### 1.1 데이터셋별 구축 방식

세 데이터셋은 분류학 다양성과 taxonomy 부착 방식이 다르다.

| 데이터셋 | 종 / 이미지 (run_log 실측) | taxonomy 출처 | 분류학 다양성 |
|---|---|---|---|
| CUB-200 | 200종 / **11,788장** | GBIF API + fallback (`cub200_build_taxonomy.py`) | single-class (전부 Animalia/Chordata/Aves) |
| Rare Species | 400종 / **11,983장** | GBIF + 파이프라인 | multi-phylum (kingdom은 단일) |
| iNat21 subset | **821종 / 8,210장** | val.json 내장 taxonomy (API 불필요) | **multi-kingdom** (Animalia/Plantae/Fungi) |

- **CUB-200 / Rare Species**: GBIF API로 7-rank를 조회하고 캐시(`cub_gbif_cache.json`)에 영구 저장. 전체 이미지를 평가에 사용.
- **iNat21**: iNaturalist 2021 val.json은 이미 7-rank Linnaean taxonomy를 포함하므로 외부 API 없이 `inat21_build_metadata.py`가 `category_id → (kingdom…genus + "Genus epithet" species)`로 평탄화한다.

### 1.2 iNat21 실험방식

**문제**: iNat21 val.json 전체는 **100,000장 / 10,000종**이다. 본 파이프라인의 핵심 지표(silhouette, latent probe)는 **N×N cosine distance 행렬**을 요구한다 — N=100k면 $100{,}000^2 \times 4\text{B} \approx 40\,\text{GB}$로 RAM에 올릴 수 없다.

**해결**: `inat21_build_metadata.py`의 **stratified subsample**로 CUB-200과 비슷한 ~1만 스케일(행렬 ~400MB)로 줄였다. 실제 실행 설정(CSV `inat21_val_Aves-Insecta-Plantae-Fungi-Reptilia_s200_n15.csv`, run_log 기준):

```
도메인 5개: Aves, Insecta, Plantae, Fungi, Reptilia
  (Aves/Insecta/Reptilia = Linnaean class, Plantae/Fungi = kingdom → 매칭은 class→phylum→kingdom 순)
컷 순서:
  1) min_per_species = 5  : 5장 미만 종은 서브샘플 전 제거
  2) max_species_per_domain = 200 : 도메인당 종 무작위 ≤200
  3) max_per_species (n15) = 15   : 종당 이미지 무작위 ≤15
결과 (run_log 실측): 821 species, 8,210 images  (종당 평균 ~10장)
```

서브샘플은 `random.Random(seed=42)`로 고정돼 재현 가능하다. 결과 디렉토리는 `inat21_bioclip2/`로 리네임됐으나 원 실행 out 경로는 `inat21_Aves-Insecta-Plantae-Fungi-Reptilia_bioclip2`였다(run_log args 참조).

> **핵심**: iNat21 결과는 **전체가 아닌 5개 도메인 균형 서브샘플(821종)** 위에서 얻은 것이다. 절대 수치를 CUB-200/Rare Species와 직접 비교할 때 이 스케일·구성 차이를 감안해야 한다(§2.5 위협 참조).

### 1.3 왜 어떤 rank는 "skipped"인가 (degenerate rank 규칙)

Exp2는 rank마다 그 rank label로 silhouette을 다시 계산한다. 그러나 **한 rank의 라벨이 모든 샘플에서 같으면(=클래스가 1개) silhouette은 정의되지 않는다.** 코드의 정확한 조건([../src/run_experiment.py](../src/run_experiment.py), `experiment_2`):

```python
cls, cnt = np.unique(rank_int, return_counts=True)
if len(cls) < 2 or cnt.min() < 2:        # 클래스 2개 미만 OR 최소 클래스 샘플 2개 미만
    results[rank_name] = {"skipped": "degenerate at this rank"}
```

이 규칙이 데이터셋별로 어떤 rank를 떨어뜨리는지:

| Rank | CUB-200 | Rare Species | iNat21 | 이유 |
|---|---|---|---|---|
| kingdom | **skip** | **skip** | ✅ 측정 | CUB·Rare는 전 종이 Animalia(클래스 1개). iNat21만 Animalia/Plantae/Fungi 3개 → 측정 가능 |
| phylum | **skip** | ✅ | ✅ | CUB는 전부 Chordata. Rare는 multi-phylum |
| class | **skip** | ✅ | ✅ | CUB는 전부 Aves |
| order~species | ✅ | ✅ | ✅ | 모두 ≥2 클래스 |

> **즉 skip은 버그나 누락이 아니라 "그 데이터셋에 그 rank의 변별 대상이 존재하지 않는다"는 구조적 사실이다.** CUB-200이 전부 새(Aves)라 상위 3 rank가 단일값인 것이 RQ1 형식 fail의 원인이기도 하다([ANALYSIS.md](ANALYSIS.md) §2.1). iNat21을 추가한 핵심 동기가 바로 **kingdom rank를 처음으로 측정 가능하게** 만드는 것이었다.

### 1.4 메모리·연산 관리

- **cosine distance 행렬 1회 계산 후 재사용**: latent probe는 50회 permutation + 실제 라벨 silhouette을 계산하는데, sklearn은 매 호출마다 N×N 행렬을 재구성한다. 코드는 `cosine_distance_matrix(img_emb)`를 한 번 만들어 `precomputed_distance`로 넘기고, 사용 후 `del D_cos`로 즉시 해제한다(N=12k에서 ~556MB).
- **임베딩 캐시**: 이미지 인코딩 결과를 `img_emb_*.npz`에 저장해 5 seeds × 5 conditions 반복 시 재인코딩을 피한다.
- **AMP(float16)**: 모든 run에서 활성. 인코딩 시간 — iNat21 8,210장 31초, CUB-200 OpenCLIP 11,788장 2분14초, Rare Species 11,983장 11분8초(디스크 I/O 차이).

### 1.5 Zero-shot 구현 (iNat21 한정)

종별 텍스트 프로토타입(클래스명 임베딩)과 image embedding의 cosine 최근접으로 top-1을 매긴다([../src/run_experiment.py](../src/run_experiment.py) `zeroshot`). C0~C4 각 프롬프트 조건으로 프로토타입을 만들어 정확도를 측정(C5는 텍스트가 없어 제외). 이 지표만 geometry가 아닌 **실제 task 성능**을 본다.

---

## Part II. 유효성 검증 (Validity)

실험 결론의 신뢰도를 4종 타당성으로 점검한다.

### 2.1 Construct validity — 측정이 의도한 개념을 재나?

| 주장 | 측정 | 타당성 평가 |
|---|---|---|
| "종이 잘 뭉친다" | silhouette / intra_var / inter_margin (cosine) | ✅ 표준 cluster 지표, 다중 지표 교차 |
| "텍스트 없이 분류학이 잠재" | image-only silhouette을 **무작위 라벨 50 permutation**과 z-score 비교 | ✅ permutation control이 "우연히 높을" 가능성을 직접 차감 |
| "구조 vs 어휘" | C2(구조보존·어휘파괴) vs C3(어휘보존·구조파괴) **분리 설계** | ✅ counterfactual이 두 요인을 직교 분해 |

- **Sanity check 통과**: 무작위 라벨 silhouette은 모든 run에서 ≈0 또는 음수(−0.07~−0.18)로, "구조 없는 라벨엔 신호가 없다"는 기대와 일치. 실제 라벨만 크게 양(+) → 측정이 분류학 구조를 진짜로 포착.
- **다중 지표 수렴**: silhouette·inter_margin·knn_purity가 같은 방향을 가리켜 단일 지표 아티팩트 가능성 낮음.
- **⚠️ 약점**: zero-shot에서 C2(어휘파괴)는 급락(0.46)하나 geometry silhouette에서는 가장 잘 견딘다 — **geometry와 downstream이 서로 다른 축**을 측정함을 시사. "silhouette = 성능"으로 등치하면 안 됨.

### 2.2 Internal validity — 인과를 교란 없이 분리했나?

- ✅ **Paired 설계**: C0/C1을 *동일* image embedding에 텍스트만 바꿔 적용 → 이미지 분포 차이가 교란하지 않음.
- ✅ **counterfactual 통제군**: C2/C3/C4가 어휘·구조·순서를 각각 독립적으로 제거 → "무엇이 효과의 원인인가"를 통제.
- ⚠️ **seed noise가 비현실적으로 작음**: image embedding에 σ=1e-3 Gaussian만 더해 seed-std가 1e-7~1e-8 → **Cohen's d가 ×10⁵까지 inflate**(예: silhouette d ≈ −5.5×10⁵, `cub200_openclip` exp1). 이는 *효과가 크다*는 증거가 아니라 *분산이 인위적으로 0에 가깝다*는 아티팩트. → raw difference로만 해석.
- ⚠️ **preservation ratio ρ 무효화**: 전 run에서 분모 $(C_1-C_0)<0$이라 사전 정의 ρ 임계값이 부호 역전돼 `semantic_organizer_supported=False`로 찍힘. → drop(C3)/drop(C2) 비율로 대체 논증.

### 2.3 External validity — 일반화 가능한가?

- ✅ **데이터셋 사다리**: single-class(CUB) → multi-phylum(Rare) → multi-kingdom(iNat21)로 분류학 다양성을 단계적으로 확대. 핵심 패턴(상위 rank C1 우세, C3≫C2)이 **모든 단계에서 재현**.
- ✅ **모델 2종**: BioCLIP2(생물 도메인) + OpenCLIP(일반 도메인 LAION-2B). 정성 결론이 model-agnostic하게 재현 → 외적 타당도 보강.
- ⚠️ **여전히 척추동물 중심**: 세 데이터셋 모두 척추동물 비중이 큼. iNat21이 Insecta/Plantae/Fungi를 포함하나 **도메인당 ≤200종 균형 서브샘플**이라 도메인 전수가 아님. RQ4(5도메인 random-effects meta-analysis)는 미완.
- ⚠️ **Rare Species × OpenCLIP 미실행**: cross-model 비교가 CUB-200·iNat21에만 존재 → "BioCLIP2의 더 강한 신호가 *모델* 때문인지 *데이터셋* 때문인지" Rare Species에서는 분리 불가.

### 2.4 Statistical conclusion validity — 통계가 적절한가?

- ✅ **비모수 검정**: paired permutation test(B=1000) + bootstrap CI(B=1000)로 분포 가정 회피.
- ✅ **다중비교 보정**: Exp1 6개 metric에 Bonferroni α=0.01/6≈0.00167 적용.
- ⚠️ **검정력 손상**: §2.2의 작은 σ 때문에 permutation p가 모두 0.05~0.08 borderline → Bonferroni 후 **전 metric 비유의**. 통계적 유의성이 아니라 **cross-run 재현성·효과 방향**이 본 연구의 근거.
- ⚠️ **latent probe B=50**: permutation 50회로 z=50~235를 얻었으나, 더 큰 B로 robustness 확인 여지.

### 2.5 Reproducibility threats — run 간 환경 불일치

run_log 비교에서 드러난 차이(향후 재현·메타분석 시 통제 필요):

| 항목 | cub200_openclip | rare_species_bioclip2 | inat21_* |
|---|---|---|---|
| torch | (미기록) | (미기록) | **2.4.1+cu124** |
| sklearn | (미기록) | (미기록) | **1.8.0** |
| git hash | (미기록) | (미기록) | **9c1aa34** |
| batch_size | **128** | 64 | 64 |
| num_workers | 2 | 4 | 8 |
| Exp3 조건 | **C0…C5** | **C0…C5** | C0…C4 |

- ⚠️ **코드 버전 drift**: 구 run(cub200_openclip, rare_species)은 Exp3에 **C5(image-only) 포함**, 신 run(iNat21)은 C0…C4만. 즉 동일 파이프라인의 *다른 버전*으로 실행됨 → 조건 정의·코드 경로가 완전히 동일하다고 보장 못 함.
- ⚠️ **라이브러리 버전 차이**: requirements.txt는 torch 2.7.1을 고정하나 iNat21 run은 2.4.1+cu124, sklearn 1.8.0(requirements는 ≥1.4.0)으로 실행. silhouette 등 sklearn 구현 차이가 미세 수치에 영향 가능.
- ⚠️ **batch_size 차이(128 vs 64)**: AMP float16 누적 순서가 달라져 임베딩에 미세한 비결정성 유입 가능.

### 2.6 종합 — 무엇을 믿고, 무엇을 유보하나

**믿을 수 있는 결론** (다중 통제 + cross-run 재현):
1. **상위 rank에서 C1 우세, species에서만 역전** — 5개 run 일관, kingdom까지 iNat21로 검증.
2. **구조 파괴(C3) ≫ 어휘 파괴(C2)** — drop 비율 2.5~4.5×, geometry·zero-shot 양쪽 수렴.
3. **텍스트 없이도 강한 latent taxonomy** — permutation control 차감 후 z≥50.

**유보해야 할 것** (통제 한계):
1. **통계적 유의성 자체** — σ 설정으로 검정력이 손상돼 p·Cohen's d는 신뢰 불가. 방향성·재현성으로만 논증.
2. **절대 수치 cross-dataset 비교** — iNat21은 821종 서브샘플, 다른 둘은 전수 → 스케일 비대칭.
3. **모델 vs 데이터셋 효과 분리** — Rare Species × OpenCLIP 부재로 미완.
4. **도메인 일반화(RQ4)** — 척추동물 편향 + 도메인 서브샘플로 meta-analysis 미수행.

### 2.7 권장 후속 검증

1. **seed-noise 재설정**: σ=1e-3 → 모델 자연 stochasticity(dropout/AMP rounding/batch shuffle) 또는 σ=1e-2~5e-2로 재실행해 검정력 정상화 → Bonferroni 후에도 유의한 metric 탐색.
2. **단일 코드·환경 재실행**: 세 데이터셋을 동일 git hash·동일 torch/sklearn·동일 조건집합(C0…C4 통일)으로 재실행해 §2.5 drift 제거.
3. **Rare Species × OpenCLIP** 추가로 모델/데이터셋 효과 분리.
4. **iNat21 전수 또는 더 큰 서브샘플**(메모리 허용 범위에서 N 단계적 확대)로 서브샘플 의존성 점검.
5. **latent probe B 확대**(50→1000)로 z-score robustness 확인.
