# Phase 2: Inference Internals (로컬 추론 엔진 해부)

- **기간**: 5-8주
- **목표**: `llama.cpp` C++/CUDA 소스 디버그 빌드, 추론 파이프라인 추적, GGUF 포맷 및 양자화(INT4/INT8)·KV Cache 분석

## 예정 구성 파일

- `build_cuda_debug.sh`: `llama.cpp` CUDA 디버그 심볼 포함 빌드 스크립트
- `gguf_inspector.py`: GGUF 바이너리 헤더/메타데이터/텐서 정보 파싱 도구

## 관련 문서

- [전체 로드맵](../docs/ROADMAP.md)
- [GGUF 바이너리 포맷 명세](../docs/02_gguf_format_spec.md)
- [KV Cache 메모리 프로파일링 리포트](../docs/03_kv_cache_analysis.md)
- [실험 및 트러블슈팅 로그](../docs/EXPERIMENT_LOG.md)
