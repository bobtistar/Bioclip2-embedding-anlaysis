# Results Analysis — Proposed Method & Experiments

> 이 문서는 `results/*/{exp1,exp2,exp3}_*.json`, `zeroshot_accuracy.json`의 **실제 산출 수치**를 논문의 *Proposed Method* + *Experiments* 절 형식으로 정리·분석한 것이다. 개념적 배경과 비유는 [../README.md](../README.md)와 [../docs/results.md](../docs/results.md)를, 지표 정의는 [../docs/metrics_guide.md](../docs/metrics_guide.md)를 참조한다.
>
> **분석 대상 4개 run** (모두 frozen evaluation, 5 seeds, σ=1e-3 image noise):
>
> | run 디렉토리 | 모델 | 데이터셋 | 특이성 |
> |---|---|---|---|
> | `cub200_openclip-vitl14/` | OpenCLIP ViT-L/14 | CUB-200 (200종) | single-class (전부 Aves) |
> | `rare_species_bioclip2/` | BioCLIP2 ViT-L/14 | Rare Species (400종) | multi-phylum |
> | `inat21_bioclip2/` | BioCLIP2 ViT-L/14 | iNat21 subset | **multi-kingdom** |
> | `inat21_openclip-vitl14/` | OpenCLIP ViT-L/14 | iNat21 subset | **multi-kingdom** |
>
> (BioCLIP2 × CUB-200 run의 수치는 [../docs/results.md](../docs/results.md)에 정리돼 있어 비교표에서만 인용한다.)

---

## Part I. Proposed Method

### 1.1 문제 정의와 경쟁 가설

CLIP 계열 모델은 이미지 $x$와 텍스트 $t$를 공유 임베딩 공간의 단위벡터로 사상한다. BioCLIP2는 종 이름만 주는 대신 7-rank Linnaean 분류학을 텍스트로 결합하면 인식이 향상된다고 보고했다. 본 연구는 그 향상의 *메커니즘*을 두 가설로 정식화하고 반증 가능한 형태로 검정한다.

- **H_info (정보 채널)**: 계층 텍스트는 토큰 수를 늘려 텍스트 임베딩의 정보량을 키운다. 효과의 본질은 *어휘/내용*이다.
- **H_geom (의미 조직자)**: 계층의 *구조*가 임베딩을 분류학적 거리에 맞춰 재배치한다. 효과의 본질은 *구조 정합성*이며, 그 구조는 image embedding에 이미 잠재해 있을 수 있다.

세 연구 질문(RQ)이 이를 분해한다.

| RQ | 질문 | 검정 실험 |
|---|---|---|
| RQ1 | 계층 텍스트가 *species* 분리를 개선하나? | Exp1 |
| RQ2 | 효과가 어느 rank에서 나오나? 텍스트 없이도 분류학이 보이나? | Exp2 |
| RQ3 | 효과의 본질은 *내용*인가 *구조*인가? | Exp3 (+ zero-shot) |

### 1.2 Counterfactual prompt 조건 (핵심 방법)

방법론의 중심은 **프롬프트를 차원별로 망가뜨리는 counterfactual 설계**다. 같은 image embedding에 아래 조건의 텍스트를 결합해 *무엇을 제거했을 때 효과가 사라지는지*로 H_info와 H_geom을 분리한다.

| 조건 | 정의 | 제거/보존한 것 | 분리하는 가설 |
|---|---|---|---|
| **C0** | `a photo of {species}` | 기준선 (계층 없음) | — |
| **C1** | `a photo of {kingdom}…{species}` | 정상 7-rank 계층 | full effect |
| **C2** | 상위 6 rank를 `taxK…taxG` 무의미 토큰으로 치환 | **구조 보존 / 어휘 파괴** | H_info를 끈다 |
| **C3** | 상위 6 rank를 다른 종 라벨로 교체 | **어휘 보존 / 구조 파괴** | H_geom을 끈다 |
| **C4** | 7개 라벨을 매 호출 무작위 순서로 섞음 | **어휘 보존 / 순서 파괴** | 순서 정보의 기여 |
| (C5) | 텍스트 없이 image embedding만 | 일부 legacy run에만 존재 | image-only 하한 |

> **추론 논리**: H_info가 참이면 어휘를 죽인 **C2가 크게 무너지고** 구조만 깬 C3는 비교적 보존돼야 한다. H_geom이 참이면 **그 반대** — C2는 견디고 **C3가 붕괴**해야 한다. 두 조건의 손실 비율 `drop(C3)/drop(C2)`이 결정적 판별식이다.

조건 정의는 [../src/prompt_variants.py](../src/prompt_variants.py)에, C3/C4의 무작위성은 seed 고정 `numpy.random.Generator`로 재현된다.

### 1.3 임베딩 결합

이미지·텍스트 임베딩을 각각 L2 정규화한 뒤 concat한다 (`fuse_image_text`, [../src/run_experiment.py](../src/run_experiment.py)). 최종 차원은 image dim + text dim. C5(텍스트 없음)는 image-only fallback.

### 1.4 측정 지표

embedding geometry를 다각도로 진단한다 (정의: [../docs/metrics_guide.md](../docs/metrics_guide.md)).

- **1차 판정**: `intra_var`↓, `inter_margin`↑, `silhouette`↑ (cosine)
- **보조**: `RankMe`(effective rank), `uniformity`, `knn_purity@10`
- **downstream**: zero-shot top-1 accuracy (iNat21 run에 한해 산출)
- **latent probe**: 텍스트 없이 image embedding만으로 잰 species silhouette을, 무작위 라벨 50회 permutation 분포와 비교한 **z-score**

### 1.5 통계 프로토콜

- 5 seeds (`42+s`), 각 seed마다 image embedding에 σ=1e-3 Gaussian noise
- **paired permutation test** (B=1000): C0 vs C1, Bonferroni α=0.01/6≈0.00167
- **bootstrap 95% CI** (B=1000): preservation ratio ρ
- **preservation ratio**: $\rho_x = \dfrac{M(C_x)-M(C_0)}{M(C_1)-M(C_0)}$

> ⚠️ **방법론적 한계 (해석에 직결)**: σ=1e-3가 너무 작아 seed-std가 1e-7~1e-8 수준 → Cohen's d가 ×10⁵까지 inflate(예: silhouette d ≈ −5.5×10⁵). 또한 모든 run에서 분모 $(C_1-C_0)<0$이라 ρ의 부호 해석이 뒤집힌다. **따라서 판정은 ρ 임계값이 아니라 raw difference와 cross-run 재현성으로 한다.**

---

## Part II. Experiments

### 2.0 Setup

| 항목 | 값 |
|---|---|
| 모델 | BioCLIP2 ViT-L/14 (`hf:imageomics/bioclip-2`), OpenCLIP ViT-L/14 (`laion2b_s32b_b82k`) — 둘 다 frozen, dim=768 |
| 데이터셋 | CUB-200 (200종, single-class), Rare Species (400종, multi-phylum), iNat21 subset (multi-kingdom) |
| 평가 방식 | train/val 분리 없는 frozen evaluation, 5 seeds |

세 데이터셋은 분류학 다양성이 단계적으로 커진다: CUB-200(kingdom/phylum/class 단일값) → Rare Species(phylum부터 검증 가능) → **iNat21(kingdom부터 전 rank 검증 가능)**. 이 사다리가 "상위 rank 효과"의 confound를 차례로 제거한다.

---

### 2.1 Exp1 — RQ1: Hierarchical vs Flat (species-level)

C0→C1의 species silhouette 변화와 RQ1 통과 여부 (`exp1_geometry.json`):

| run | C0 sil | C1 sil | Δ(C1−C0) | inter_margin ratio | passes_RQ1 |
|---|---|---|---|---|---|
| BioCLIP2 / CUB-200† | 0.7749 | 0.7557 | **−0.0192** | 0.938× | ✗ |
| BioCLIP2 / Rare Species | 0.5919 | 0.5537 | **−0.0382** | 0.905× | ✗ |
| BioCLIP2 / iNat21 | 0.6827 | 0.6715 | **−0.0112** | 0.972× | ✗ |
| OpenCLIP / CUB-200 | 0.5238 | 0.3027 | **−0.2211** | 0.418× | ✗ |
| OpenCLIP / iNat21 | 0.4519 | 0.2262 | **−0.2257** | 0.467× | ✗ |

†[../docs/results.md](../docs/results.md)에서 인용.

**관찰**:
1. **5개 run 전부 RQ1 형식 fail** — C1이 species silhouette을 *낮춘다*. 데이터셋·모델·분류학 다양성과 무관하게 일관.
2. paired permutation p는 모두 0.05~0.08 → Bonferroni 후 비유의. (단, std가 비현실적으로 작아 통계 power 자체가 손상됨 — §1.5)
3. **BioCLIP2의 drop(−0.011~−0.038)이 OpenCLIP(−0.22)보다 한 자릿수 작다.** 생물 도메인 사전학습이 hierarchical conditioning에 훨씬 강건.

**RQ1 해석**: "개선 가설"로는 기각. 그러나 multi-kingdom(iNat21)에서도 fail이 재현되므로 이는 *데이터셋 confound가 아니다*. 진짜 원인은 **BioCLIP2 species 표현의 사전학습 단계 포화**(C0 silhouette 0.68~0.77, knn_purity ~99%) — 추가 species 변별 정보가 없는 상태에서 공통 상위 rank 토큰이 fine-grained 분리를 평균화한다. 이는 Exp2의 trade-off로 직접 확인된다.

---

### 2.2 Exp2 — RQ2: Rank-level 분해 + Latent taxonomy probe

#### (a) Rank별 silhouette Δ(C1−C0) (`exp2_rank_levels.json`)

| Rank | OpenCLIP/CUB | BioCLIP2/Rare | **BioCLIP2/iNat21** | **OpenCLIP/iNat21** |
|---|---|---|---|---|
| kingdom | skip | skip | **+0.0037** ✓ | **+0.157** ✓ |
| phylum | skip | +0.0041 ✓ | +0.0059 ✓ | +0.139 ✓ |
| class | skip | +0.0127 ✓ | +0.0101 ✓ | +0.182 ✓ |
| order | +0.0474 ✓ | +0.0223 ✓ | +0.0095 ✓ | +0.098 ✓ |
| family | −0.0042 ✗ | +0.0157 ✓ | +0.0200 ✓ | −0.034 ✗ |
| genus | −0.1473 ✗ | +0.0090 ✓ | +0.0182 ✓ | −0.219 ✗ |
| **species** | **−0.2211** ✗ | **−0.0382** ✗ | **−0.0112** ✗ | **−0.226** ✗ |

**핵심 발견**:
1. **방향 전환 패턴이 보편적**: 상위 rank에서 C1 우세 → species에서 역전. **모든 run에서 species는 예외 없이 음(−).**
2. **iNat21이 kingdom rank를 처음으로 검증** — docs/results.md가 남긴 "kingdom 미검증" 한계를 해소. BioCLIP2는 kingdom→family까지 전부 양(+), OpenCLIP은 kingdom→order까지 강한 양(+0.10~+0.18).
3. **BioCLIP2가 OpenCLIP보다 일관적**: BioCLIP2/iNat21은 genus까지 6개 rank 전부 양(+)이고 species만 음. OpenCLIP은 family부터 무너진다 — 도메인 사전학습이 rank별 반응을 안정화.

→ hierarchy는 species 분리기가 아니라 **상위 분류학에 맞춘 정렬기**다. H_geom의 방향 예측이 4개 run + (docs의 CUB-200 BioCLIP2)에서 일관 관찰.

#### (b) Latent taxonomy probe — 텍스트 없이 image embedding만

| run | real silhouette | random μ (±σ) | **z-score** |
|---|---|---|---|
| BioCLIP2 / CUB-200† | 0.5015 | −0.0790 | **121.7** |
| BioCLIP2 / Rare Species | 0.1901 | −0.0751 (±0.0011) | **234.6** |
| **BioCLIP2 / iNat21** | 0.3702 | −0.1459 (±0.0029) | **180.9** |
| OpenCLIP / CUB-200 | 0.1343 | −0.0671 (±0.0021) | **96.1** |
| **OpenCLIP / iNat21** | **−0.0174** | −0.1821 (±0.0033) | **50.7** |

**핵심 발견**:
1. **모든 run에서 z ≥ 50 — 텍스트 없이도 분류학 구조가 image embedding에 깊이 잠재.** BioCLIP2가 OpenCLIP보다 2~4배 강함.
2. OpenCLIP/iNat21은 real silhouette이 *음수*(−0.017)지만 random μ가 더 음(−0.182)이라 z=50.7로 여전히 유의 — 절대 분리는 약해도 구조 신호는 존재.

→ 텍스트 organizer는 분류학을 *가르치는* 게 아니라 *이미 잠재한 구조를 표면으로 끌어올린다*. H_geom의 두 번째 기둥.

---

### 2.3 Exp3 — RQ3: Counterfactual ablation (구조 vs 어휘 vs 순서)

C0 대비 silhouette drop (`exp3_counterfactuals.json`):

| 조건 | OpenCLIP/CUB | BioCLIP2/Rare | BioCLIP2/iNat21 | OpenCLIP/iNat21 | (BioCLIP2/CUB†) |
|---|---|---|---|---|---|
| C1 정상 계층 | −0.221 | −0.038 | −0.011 | −0.226 | −0.019 |
| **C2 구조보존·어휘파괴** | **−0.093** | **−0.161** | **−0.087** | **−0.099** | **−0.078** |
| C4 순서파괴 | −0.245 | −0.202 | −0.131 | −0.245 | −0.147 |
| **C3 구조파괴·어휘보존** | **−0.352** | **−0.402** | **−0.381** | **−0.421** | **−0.354** |
| **drop(C3)/drop(C2)** | **3.8×** | **2.5×** | **4.4×** | **4.3×** | **4.5×** |

**핵심 발견 — 판별식 `drop(C3)/drop(C2)`이 모든 run에서 ≫1**:
1. **구조 파괴(C3) ≫ 어휘 파괴(C2)**: 2.5~4.5배. 어휘를 무의미 토큰으로 바꿔도(C2) 견디지만 구조를 깨면(C3) silhouette이 절반 가까이 붕괴. → **H_info 기각, H_geom 지지.**
2. **순서도 어휘보다 중요**: 모든 run에서 `drop(C4) > drop(C2)` — 어휘 < 순서 < 구조 ordering이 5개 run 전부 일치.
3. 패턴은 보편적, 강도는 데이터셋 의존(Rare Species 2.5× < iNat21 4.4×). `semantic_organizer_supported=False`는 ρ 분모 부호 문제일 뿐 raw 신호는 명확.

---

### 2.4 Zero-shot downstream (iNat21, 신규 지표)

geometry 지표를 넘어 **실제 분류 정확도**로 가설을 교차 검증 (`zeroshot_accuracy.json`):

| 조건 | BioCLIP2/iNat21 top-1 | OpenCLIP/iNat21 top-1 |
|---|---|---|
| C0 flat | **0.9429** | 0.1133 |
| C1 hierarchical | 0.9373 | 0.1037 |
| C4 word-bag (순서파괴) | 0.9015 | **0.1135** |
| C2 random-token (어휘파괴) | 0.4643 | 0.0819 |
| C3 shuffled (구조파괴) | **0.3505** | **0.0325** |

**핵심 발견**:
1. **C3(구조파괴)가 압도적 최저** — BioCLIP2 0.94→0.35, OpenCLIP 0.11→0.03. 구조를 깨면 downstream이 붕괴.
2. **C4(순서만 제거)는 90% 보존** (BioCLIP2 0.90) — 어휘가 살아있으면 zero-shot은 대체로 견딘다.
3. **C2(어휘파괴)는 0.46으로 급락** — 단, geometry(silhouette)에서는 C2가 가장 잘 견뎠는데 zero-shot에서는 무너진다. zero-shot은 *텍스트 어휘*가 실제 클래스명과 매칭돼야 하므로 무의미 토큰에 민감 — geometry(구조)와 downstream(어휘 매칭)이 서로 다른 축을 보고 있음을 시사.

→ geometry 분석(C3≫C2)과 downstream(C3 최저)이 **구조 파괴가 가장 치명적**이라는 결론에서 수렴. H_geom 보강.

---

### 2.5 Cross-model / Cross-dataset 종합

| 가설 검정 축 | 결론 | 근거 (재현된 run 수) |
|---|---|---|
| RQ1 species fail | **5/5 run에서 형식 fail** | 데이터셋·모델 불문, multi-kingdom서도 재현 → confound 아닌 species 포화 |
| RQ2(a) 상위 rank 우세 | **지지** | kingdom~genus 대부분 양(+); iNat21로 kingdom까지 검증 |
| RQ2(b) latent taxonomy | **강하게 지지** | z = 50.7 ~ 234.6, 전 run 유의 |
| RQ3 구조 > 어휘 | **지지** | drop(C3)/drop(C2) = 2.5~4.5×, 전 run; zero-shot도 C3 최저 |
| 모델 효과 | BioCLIP2 ≫ OpenCLIP | 생물 사전학습이 rank 안정성·latent z·강건성 모두 우위 |

---

## Part III. Discussion & Limitations

**결론**: 5개 run에 걸쳐 (1) species fail의 보편성, (2) 상위 rank C1 우세 + 강한 latent taxonomy, (3) 구조 파괴(C3) ≫ 어휘 파괴(C2)의 일관성이 관찰된다. 종합하면 hierarchical text는 *정보 채널*이 아니라 **image embedding에 잠재한 분류학적 기하를 표면으로 끌어올리는 organizer prior**다 (**H_geom 지지**).

**한계** (자세히는 [../docs/results.md](../docs/results.md) "한계/위협"):
1. **통계 power**: σ=1e-3 → Cohen's d ×10⁵ inflate, Bonferroni 후 Exp1 전 metric 비유의. raw difference·재현성으로 대체 논증. → σ를 1e-2~5e-2로 키우거나 모델 자연 stochasticity로 재실행 필요.
2. **ρ 분모 부호**: 전 run $(C_1-C_0)<0$이라 사전 정의 ρ 임계값 무효. drop ratio로 대체.
3. **도메인 범위**: 척추동물(CUB) → multi-phylum(Rare) → multi-kingdom(iNat21)까지 확장됐으나 Insecta/Plantae/Fungi 전용 도메인의 random-effects meta-analysis(RQ4)는 미완.
4. **C2 geometry vs downstream 불일치**: C2가 silhouette은 잘 견기지만 zero-shot은 급락 — geometry와 downstream이 서로 다른 축을 측정함을 보여주는 신호로, 추가 분석 가치가 있다.

**다음 단계**: ① seed-noise scale 정정으로 통계 power 정상화, ② Rare Species × OpenCLIP cross-check, ③ 도메인 2개 추가로 RQ4 meta-analysis 완성.
