# 02. GGUF 바이너리 포맷 명세 및 양자화 구조 분석

> **Phase 2 (5-8주)** 산출물 — `02-inference-internals/gguf_inspector.py` 구현과 함께 정리하는 분석 노트입니다.

## 1. GGUF 파일 구조 개요
- **Header**: Magic Number (`GGUF`), Version, Tensor Count, Metadata KV Count
- **Metadata Key-Value Pairs**: 모델 아키텍처, 하이퍼파라미터, 토크나이저 어휘/병합 규칙
- **Tensor Info Table**: 텐서 이름, 차원(Dimensions), 양자화 타입(`ggml_type`), 데이터 오프셋
- **Alignment & Tensor Data**: 정렬 패딩 및 실제 가중치 바이너리 영역

## 2. 양자화(INT4 / INT8) 블록 구조 분석
- `Q8_0`: 블록 크기, 스케일 팩터(`fp16`), 양자화 오차 특성
- `Q4_0` / `Q4_K_M`: 슈퍼블록(Super-block) 구조, 비트 패킹, 역양자화(Dequantization) 수식
- FP16 대비 메모리 절감률 및 Perplexity 변화 측정 결과

## 3. 파싱 검증 메모
- `gguf_inspector.py` 실행 결과 예시 및 주요 메타데이터 해석
