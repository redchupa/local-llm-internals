# Changelog

이 프로젝트의 주요 변경 사항은 이 파일에 기록됩니다.

## [0.2.0] - 2026-11-26

### Added
- Phase 1 (1주차) 순수 Python 기반 Byte-level BPE 토크나이저 구현 (`01-mini-gpt/tokenizer.py`)
- 토크나이저 단위 테스트 스위트 추가 (`01-mini-gpt/tests/test_tokenizer.py`)
- Phase 1 의존성 파일(`01-mini-gpt/requirements.txt`) 및 RTX 5070 Ti CUDA 12.8용 `docker/Dockerfile` 추가
- `docs/01_transformer_math.md` 1절(BPE 토크나이저 원리 및 `vocab_size` 트레이드오프) 상세 정리

## [0.1.0] - 2026-11-26

### Added
- 16주 로컬 LLM 내부 구조 연구·개발 프로젝트 초기 스캐폴딩 구성
- 루트 `README.md` (프로젝트 개요, 환경 스펙, 16주 로드맵, 진행률 체크리스트, 운영 전략)
- 상세 계획서 및 연구 노트 템플릿 작성 (`docs/ROADMAP.md`, `docs/EXPERIMENT_LOG.md`, `docs/01_transformer_math.md`, `docs/02_gguf_format_spec.md`, `docs/03_kv_cache_analysis.md`)
- Phase 1-4 및 `docker/` 디렉토리 구조와 안내 `README.md` 생성
- 대용량 모델 가중치(`.gguf`, `.safetensors` 등) 및 비밀키(`.env`) 제외용 `.gitignore` 및 MIT `LICENSE` 추가
