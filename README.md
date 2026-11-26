# local-llm-internals

> 단일 GPU(RTX 5070 Ti 16GB VRAM) 환경에서 로컬 LLM의 내부 아키텍처, 추론 엔진, 파인튜닝 파이프라인을 직접 구현하고 분석하는 16주 연구·개발 프로젝트입니다.

## 프로젝트 개요

- **프로젝트명**: `local-llm-internals`
- **핵심 목표**: 프레임워크 추상화 뒤에 숨겨진 LLM의 실제 동작(`토크나이징 -> 텐서 연산 -> 샘플링 -> 양자화 -> 파인튜닝`)을 코드 레벨에서 직접 구현·디버깅하고, 재현 가능한 연구 노트로 GitHub에 아카이빙합니다.
- **상세 계획서**: [docs/ROADMAP.md](docs/ROADMAP.md)
- **실험 기록 로그**: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)

### 기본 개발 환경

| 구분 | 스펙 |
| :--- | :--- |
| **Hardware** | NVIDIA RTX 5070 Ti (16GB VRAM), 64GB RAM |
| **OS / Dev** | Windows 11 Pro (WSL2 Ubuntu + Docker), VS Code, Git/GitHub |
| **Stack** | Python (`PyTorch`, Hugging Face `transformers` / `peft` / `trl`), C++/CUDA (`llama.cpp`, `CMake`) |

---

## 16주 연구·개발 로드맵

| 단계 | 기간 | 핵심 연구 주제 | 주요 실습 및 구현 | GitHub 공개 산출물 |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | 1-4주 | LLM 핵심 아키텍처 구현 | BPE 토크나이저 원리 분석, PyTorch로 Decoder-only Transformer(RoPE, GQA 포함) 직접 코딩 및 소형 텍스트 학습 | [`01-mini-gpt/`](01-mini-gpt/README.md) 소스 코드, [텐서 차원 변화 도식화 문서](docs/01_transformer_math.md) |
| **Phase 2** | 5-8주 | 로컬 추론 엔진 해부 | `llama.cpp` C++/CUDA 소스 빌드, 디버거로 프롬프트 입력부터 토큰 출력까지 파이프라인 추적, GGUF 포맷 및 양자화(INT4/INT8) 구조 분석 | [`02-inference-internals/`](02-inference-internals/README.md) 빌드 스크립트, [GGUF 바이너리 파싱 노트](docs/02_gguf_format_spec.md), [KV Cache 메모리 프로파일링 리포트](docs/03_kv_cache_analysis.md) |
| **Phase 3** | 9-12주 | 파인튜닝 파이프라인 구축 | 16GB VRAM 최적화 QLoRA 설정, 커스텀 데이터셋 전처리, SFT(Supervised Fine-Tuning) 학습 및 모델 병합(GGUF 변환) | [`03-qlora-finetuning/`](03-qlora-finetuning/README.md) 학습 파이프라인 코드, VRAM/속도 벤치마크 표 |
| **Phase 4** | 13-16주 | 통합 툴킷 및 문서화 | 학습된 모델의 로컬 서빙 API 연동, 추론 로그/로짓(Logit) 시각화 도구 제작, 전체 연구 회고 및 레포지토리 정리 | [`04-logit-inspector/`](04-logit-inspector/README.md) 미니 툴, 완성된 프로젝트 `README.md` 및 기술 아티클 |

---

## 진행률 체크리스트

### Phase 1: LLM 핵심 아키텍처 구현 (1-4주)
- [ ] BPE 토크나이저 원리 분석 및 미니 구현
- [ ] PyTorch Decoder-only Transformer (RoPE, GQA 포함) 구현 (`01-mini-gpt/model.py`)
- [ ] 소형 텍스트 데이터셋 학습 루프 및 샘플링 구현 (`01-mini-gpt/train.py`)
- [ ] 텐서 차원 변화 및 수식 도식화 정리 ([`docs/01_transformer_math.md`](docs/01_transformer_math.md))

### Phase 2: 로컬 추론 엔진 해부 (5-8주)
- [ ] `llama.cpp` C++/CUDA 디버그 빌드 스크립트 작성 (`02-inference-internals/build_cuda_debug.sh`)
- [ ] 디버거 기반 추론 파이프라인 추적 (프롬프트 입력 -> 토큰 출력)
- [ ] GGUF 바이너리 파싱 스크립트 작성 (`02-inference-internals/gguf_inspector.py`) 및 포맷 문서화 ([`docs/02_gguf_format_spec.md`](docs/02_gguf_format_spec.md))
- [ ] INT4/INT8 양자화 구조 분석 및 KV Cache 메모리 프로파일링 ([`docs/03_kv_cache_analysis.md`](docs/03_kv_cache_analysis.md))

### Phase 3: 파인튜닝 파이프라인 구축 (9-12주)
- [ ] 16GB VRAM 최적화 QLoRA 설정 작성 (`03-qlora-finetuning/configs/`)
- [ ] 커스텀 데이터셋 전처리 및 SFT 학습 파이프라인 구현 (`03-qlora-finetuning/train_sft.py`)
- [ ] 학습 완료 어댑터 병합 및 GGUF 변환 자동화
- [ ] VRAM 사용량 및 학습 속도 벤치마크 정리

### Phase 4: 통합 툴킷 및 문서화 (13-16주)
- [ ] 학습된 모델의 로컬 서빙 API 연동
- [ ] 추론 로그 및 토큰별 로짓(Logit) 시각화 도구 구현 (`04-logit-inspector/`)
- [ ] 전체 16주 연구 회고 및 기술 아티클 작성
- [ ] 최종 레포지토리 문서화 및 릴리즈 정리

---

## 레포지토리 구조

```text
local-llm-internals/
├── README.md                   # 전체 로드맵, 진행률 체크리스트, 환경 스펙
├── CHANGELOG.md                # 버전별 변경 이력 및 릴리즈 노트
├── LICENSE                     # MIT 라이선스
├── .gitignore                  # 모델 가중치(.gguf, .safetensors), 환경변수 등 제외
├── docs/                       # 계획서, 개념 정리, 디버깅 분석 리포트
│   ├── ROADMAP.md              # 16주 상세 연구·개발 계획서
│   ├── EXPERIMENT_LOG.md       # 실패 실험, OOM 해결, 벤치마크 기록 템플릿
│   ├── 01_transformer_math.md  # Phase 1: 아키텍처 수식 및 텐서 차원 도식
│   ├── 02_gguf_format_spec.md  # Phase 2: GGUF 바이너리 포맷 및 양자화 분석
│   └── 03_kv_cache_analysis.md # Phase 2: KV Cache 메모리 프로파일링 리포트
├── 01-mini-gpt/                # Phase 1: PyTorch 기반 미니 LLM 구현
│   └── README.md
├── 02-inference-internals/     # Phase 2: llama.cpp 빌드/디버깅 스크립트 및 분석
│   └── README.md
├── 03-qlora-finetuning/        # Phase 3: QLoRA 학습 및 GGUF 변환 파이프라인
│   └── README.md
├── 04-logit-inspector/         # Phase 4: 로컬 서빙 API 및 로짓 시각화 미니 툴
│   └── README.md
└── docker/                     # 재현 가능한 CUDA/PyTorch 개발 환경
    └── README.md
```

---

## 공개 운영 및 기록 전략

1. **Issue 기반 마일스톤 관리**: 각 Phase의 세부 목표를 GitHub Issues와 Projects(칸반 보드)로 등록해 진행 상황을 투명하게 기록합니다.
2. **주 1회 이상 커밋 및 릴리즈 노트**: 완성된 코드뿐 아니라 실패한 실험과 해결 과정(OOM 에러 해결, 양자화 손실 비교 등)을 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)와 `docs/`에 꾸준히 기록합니다.
3. **재현성 확보**: 모든 실험 코드에는 실행 시 사용한 라이브러리 버전(`requirements.txt`), 시드(Seed) 값, GPU 메모리 사용량(Peak VRAM)을 명시합니다.
4. **보안 및 라이선스 준수**: 대용량 모델 가중치 파일(`.gguf`, `.safetensors` 등)과 API 키(`.env`)는 `.gitignore`로 엄격히 제외하며, 참고한 오픈소스 코드의 원본 라이선스를 명시합니다.

## 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.
