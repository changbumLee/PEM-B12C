# B12-c + Boundary-Guided Residual Refinement v1 (Cityscapes 60k)

## 안전 주의
**현재 QMS/Support 학습이 실행 중이면 설치하지 마세요.** 이 설치 도구는 현재 PEM의 tracked 모델 소스를 B12-c `ddce1e5a8bf5d162ce4241c3c166114de1b4b6e3`로 교체합니다. 학습을 마친 뒤 진행하세요. 기존 checkpoint, output, Git HEAD/index, Python 환경은 변경하지 않습니다. 변경 대상의 원본을 `../PEM_source_backups/bgr_v1_...`에 백업합니다. 패키지를 압축해제하는 것만으로 모델 코드는 변경되지 않습니다.

## 핵심 변경
- 원본 B12-c의 SCDA 정렬은 그대로 유지. 기존 Transformer Decoder의 Top-1 Query 선택도 그대로 유지.
- stride-4 `x_4`에 **경계 예측 기반 residual**만 추가.
- `Boundary Expert`의 원본 SCDA 특징 기반 로짓을 sigmoid로 변환해 게이트로 사용. 해당 Expert의 GT 경계 감독은 기존 B12-c loss 그대로 적용.
- Shape Expert는 최종 보정된 특징을 입력받음. Prototype, Relational은 원래 B12-c의 SCDA 특징/정렬 보조 loss 계산을 유지.
- 새 residual 최종 1x1 conv를 전부 0으로 초기화하여 초기 forward는 기존 SCDA와 동일.
- 경계 게이트는 detach; GT label은 모델 추론 입력으로 사용하지 않음.
- 기존 B12-c 경계 감독 1000~5000 iter와 동일한 기간에 residual 경로를 0에서 점진적으로 켬.

### 수식
`F_BGR = F_SCDA + a(iter) * 0.25 * sigmoid(B8_boundary(F_SCDA)).detach() * R(F_fine, F_SCDA)`

`a(iter)`은 학습 중 1000~5000 iteration에서 선형으로 0→1, 검증·추론에서 1.0입니다.

## 파일 설치 (현재 PEM 루트 기준)
1. ZIP 압축 해제:

```bash
conda activate segman
cd ~/workspace/ai/projects/kroc_curve/PEM
unzip -o ~/Downloads/B12C_BGR_V1_CITYSCAPES_60k.zip -d .
```

2. 설치 전 검사 (실제 소스 변경 없음):

```bash
python B12C_BGR_V1_CITYSCAPES_60k/install_bgr_b12c.py --repo "$PWD" --check --self-test
```

3. 현재 모든 학습 프로세스 종료 확인 후 실제 설치:

```bash
python B12C_BGR_V1_CITYSCAPES_60k/install_bgr_b12c.py --repo "$PWD" --install --self-test
```

**`git reset --hard`는 절대 필요하지 않습니다.** 설치 도구가 고정된 Git 커밋의 여섯 개 tracked 모델 소스 파일을 복원하고, BGR 수정만 적용합니다. 이전 untracked QMS 파일이 남더라도 원본 Decoder에서는 import되지 않습니다.

## Cityscapes Batch 8 GPU Smoke Test
```bash
python B12C_BGR_V1_CITYSCAPES_60k/run_b12c_bgr_60k.py --smoke
```

Smoke Test는 실제 글로벌 Batch 8, 3 iter, Backward, 기존 B12-c 보조 손실 활성화를 검사합니다. **실제 GPU 시험 전에는 메모리 부족 해결을 보장하지 않습니다.** Smoke 결과는 별도 폴더에 저장됩니다. `[SMOKE PASS]` 확인 후 본학습을 실행하세요.

## 60,000-iteration 본학습
```bash
python B12C_BGR_V1_CITYSCAPES_60k/run_b12c_bgr_60k.py
```

기본 조건: ImageNet R50 초기 가중치, Cityscapes 19 classes, 60k, Batch 8, Workers 8, Seed 2026, Base LR 등 나머지는 기존 B12-c 출력 `config.yaml`을 그대로 사용합니다. 실행기는 **CSM Shape=True, Semantic Preserve=True, Relational=True, SSB=False**와 Cityscapes 60k를 검증합니다.

기준 YAML 위치 (기본값):
`output/kroc_curve/b12c_relation_l001_seed2026_20261005_162211_20261005_162212/config.yaml`

필요하면 `--config-file /절대/경로/config.yaml`로 변경합니다.

결과:
`output/kroc_curve/bgr_b12c_v1/train60k_bgr_seed2026_gain0p25_날짜_시간_.../`

`model_final.pth`, `metrics.json`, `config.yaml`이 저장되고 `records/`에 실험 실행 기록이 남습니다.

## 검증 범위
로컬 CPU 기능 테스트 16개 PASS. 고정 Git 커밋의 해당 소스 앵커는 GitHub에서 확인했지만, **사용자의 CUDA/Detectron2 환경에서 통합 학습 및 최종 mIoU는 실행되지 않았습니다.**

BGR는 연구용 첫 번째 실험 설계입니다. 60k 개선이 보장되지 않으며, 기존 B12-c 78.1869% 및 Pure SCDA 78.6667%와 검증 성능을 비교해야 합니다.
