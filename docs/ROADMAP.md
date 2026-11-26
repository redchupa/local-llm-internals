# 16주 로컬 LLM 내부 구조 연구·개발 계획서

단일 GPU(`NVIDIA RTX 5070 Ti 16GB VRAM`) 환경에서 로컬 LLM의 내부 아키텍처, 추론 엔진, 파인튜닝 파이프라인을 처음부터 끝까지 구현·분석하고 재현 가능한 형태로 공개하기 위한 상세 로드맵입니다.

---

## 1. 프로젝트 개요

- **프로젝트명**: `local-llm-internals`
- **핵심 목표**: 프레임워크 추상화 뒤에 숨겨진 LLM의 실제 동작(`토크나이징 -> 텐서 연산 -> 샘플링 -> 양자화 -> 파인튜닝`)을 코드 레벨에서 직접 구현·디버깅하고, 재현 가능한 연구 노트로 GitHub에 아카이빙합니다.
- **타깃 환경**:
  - **Hardware**: NVIDIA RTX 5070 Ti (16GB VRAM), 64GB RAM
  - **OS / Dev**: Windows 11 Pro (WSL2 Ubuntu + Docker), VS Code, Git/GitHub
  - **Stack**: Python (`PyTorch`, Hugging Face `transformers`, `peft`, `trl`), C++/CUDA (`llama.cpp`, `CMake`)

---

## 2. 단계별 상세 계획 (16주)

### Phase 1: LLM 핵심 아키텍처 구현 (1-4주)

- **목표**: 외부 추상화 라이브러리 없이 순수 PyTorch로 현대적 Decoder-only Transformer를 구현하고 소형 텍스트로 직접 학습합니다.
- **주차별 세부 계획**:
  - **1주차**: 개발 환경 구축(WSL2/Docker/PyTorch CUDA 확인), BPE(Byte Pair Encoding) 토크나이저 원리 분석 및 미니 토크나이저 구현
  - **2주차**: Decoder-only Transformer 핵심 블록 구현 (RMSNorm, SwiGLU, RoPE 위치 인코딩, GQA 어텐션)
  - **3주차**: 학습 루프(`train.py`), 체크포인트 저장, 샘플링 전략(Temperature, Top-k, Top-p) 구현 및 소형 코퍼스 학습
  - **4주차**: Forward/Backward 과정에서 텐서 차원 변화 도식화, 학습 손실/생성 결과 분석 및 문서화
- **예정 산출물**:
  - `01-mini-gpt/model.py`, `01-mini-gpt/train.py`, `01-mini-gpt/README.md`
  - [`docs/01_transformer_math.md`](01_transformer_math.md)

### Phase 2: 로컬 추론 엔진 해부 (5-8주)

- **목표**: `llama.cpp`를 C++/CUDA 디버그 모드로 직접 빌드하고, 모델 로딩부터 토큰 생성까지의 내부 실행 흐름과 메모리 구조를 해부합니다.
- **주차별 세부 계획**:
  - **5주차**: `llama.cpp` C++/CUDA 소스 디버그 빌드 자동화 스크립트 작성 (`build_cuda_debug.sh`) 및 디버거(GDB/VS Code) 연동
  - **6주차**: 프롬프트 입력(Prefill)부터 토큰 출력(Decode)까지 주요 C++/CUDA 호출 스택 추적
  - **7주차**: GGUF 바이너리 포맷 명세 분석 및 Python 파서(`gguf_inspector.py`) 작성, INT4/INT8 양자화 블록 구조 비교
  - **8주차**: 컨텍스트 길이와 배치 크기에 따른 KV Cache 메모리 사용량 프로파일링(16GB VRAM 기준) 및 리포트 작성
- **예정 산출물**:
  - `02-inference-internals/build_cuda_debug.sh`, `02-inference-internals/gguf_inspector.py`
  - [`docs/02_gguf_format_spec.md`](02_gguf_format_spec.md)
  - [`docs/03_kv_cache_analysis.md`](03_kv_cache_analysis.md)

### Phase 3: 파인튜닝 파이프라인 구축 (9-12주)

- **목표**: 16GB VRAM 제약 내에서 안정적으로 동작하는 QLoRA 기반 SFT 파이프라인을 구축하고, 학습 모델을 로컬 추론용 GGUF로 변환합니다.
- **주차별 세부 계획**:
  - **9주차**: 16GB VRAM 최적화 QLoRA 하이퍼파라미터 및 메모리 절감 설정(4-bit NF4, Gradient Checkpointing, Paged AdamW) 설계
  - **10주차**: 커스텀 데이터셋 전처리, 포맷 검증 및 Chat Template 구성
  - **11주차**: `trl`/`peft` 기반 SFT 학습 파이프라인(`train_sft.py`) 실행, OOM 트러블슈팅 및 학습 로그 기록
  - **12주차**: LoRA 어댑터 병합, GGUF 변환/양자화 및 VRAM·처리량(tokens/s) 벤치마크 정리
- **예정 산출물**:
  - `03-qlora-finetuning/train_sft.py`, `03-qlora-finetuning/configs/`
  - VRAM/속도 벤치마크 표 및 실험 로그

### Phase 4: 통합 툴킷 및 문서화 (13-16주)

- **목표**: 학습·변환한 모델을 로컬 API로 서빙하고, 추론 과정의 로짓과 확률 분포를 눈으로 확인할 수 있는 경량 분석 도구를 제작합니다.
- **주차별 세부 계획**:
  - **13주차**: `llama-server` 또는 로컬 서빙 API 연동 및 추론 로그/로짓 수집 인터페이스 구현
  - **14주차**: 토큰별 Top-k 후보 확률, 엔트로피, 샘플링 결과를 시각화하는 `04-logit-inspector/` 미니 툴 제작
  - **15주차**: End-to-End 재현성 검증(클린 환경 Docker 테스트) 및 주요 기술 아티클 작성
  - **16주차**: 전체 16주 연구 회고 정리, `README.md` 최종 폴리싱 및 릴리즈 발행
- **예정 산출물**:
  - `04-logit-inspector/` 미니 툴
  - 완성된 프로젝트 `README.md` 및 기술 아티클

---

## 3. 운영 및 품질 원칙

1. **Issue 기반 마일스톤 관리**: Phase 1-4를 GitHub Milestone으로 등록하고, 주차별 작업을 Issue로 분해해 추적합니다.
2. **실패 과정의 자산화**: OOM 발생 조건, 수치 불안정성(NaN Loss), 양자화 품질 저하 사례와 해결 과정을 [`docs/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)에 기록합니다.
3. **재현성 메타데이터 필수 기입**: 모든 실험에 Python/CUDA/PyTorch 버전, `requirements.txt`, 난수 시드(`seed`), Peak VRAM 사용량을 남깁니다.
4. **보안 및 대용량 파일 관리**: `.gguf`, `.safetensors`, `.bin`, `.pt`, `.env` 등은 커밋하지 않으며, 오픈소스 코드 참고 시 출처와 라이선스를 명시합니다.
