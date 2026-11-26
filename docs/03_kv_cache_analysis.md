# 03. KV Cache 메모리 프로파일링 리포트

> **Phase 2 (5-8주)** 산출물 — 단일 GPU(RTX 5070 Ti 16GB VRAM) 환경에서의 추론 메모리 분석 리포트입니다.

## 1. KV Cache 메모리 이론식

$$\text{KV Cache Bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L_{\text{ctx}} \times B \times \text{bytes\_per\_elem}$$

- `2`: Key 텐서와 Value 텐서
- GQA 적용 여부에 따른 `n_kv_heads` 감소 효과
- FP16(2 bytes) vs Q8_0(1 byte) vs Q4_0(0.5 byte) KV Cache 비교

## 2. RTX 5070 Ti (16GB VRAM) 실측 벤치마크

| 모델 / 양자화 | 컨텍스트 길이 (`n_ctx`) | KV Cache 타입 | 실측 VRAM (GB) | Prefill (tok/s) | Decode (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| *(측정 후 기록)* | - | - | - | - | - |

## 3. 디버거 추적 요약
- `llama.cpp`에서 Prefill 단계와 Decode 단계의 메모리 할당 및 그래프 실행 차이
- OOM 임계점 분석 및 최적 컨텍스트/오프로딩 가이드
