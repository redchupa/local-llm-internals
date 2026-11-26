# 01. Transformer 핵심 수식 및 텐서 차원 변화

> **Phase 1 (1-4주)** 산출물 — `01-mini-gpt/` 구현과 함께 채워 나가는 기술 노트입니다.

## 1. BPE 토크나이저 원리
- 바이트 단위 인코딩 및 빈도 기반 병합(Merge) 규칙 정리
- 어휘 사전 크기(`vocab_size`)와 시퀀스 길이의 트레이드오프

## 2. Decoder-only Transformer 아키텍처
- 입력 임베딩: `(B, T) -> (B, T, D)`
- **RMSNorm**: LayerNorm 대비 계산량 및 안정성 비교
- **RoPE (Rotary Position Embedding)**: 복소수 회전 행렬 기반 상대 위치 인코딩
- **GQA (Grouped-Query Attention)**:
  - Query: `(B, n_heads, T, head_dim)`
  - Key / Value: `(B, n_kv_heads, T, head_dim)`
  - KV Head 공유에 따른 KV Cache 절감 비율 정리
- **SwiGLU FFN**: 게이트 활성함수 구조 및 파라미터 수 계산

## 3. 텐서 차원 변화 표

| 레이어 / 연산 | 입력 Shape | 출력 Shape | 비고 |
| :--- | :--- | :--- | :--- |
| Token Embedding | `(B, T)` | `(B, T, D)` | `vocab_size x D` |
| Q Projection | `(B, T, D)` | `(B, n_heads, T, d_head)` | RoPE 적용 |
| K, V Projection | `(B, T, D)` | `(B, n_kv_heads, T, d_head)` | GQA 공유 |
| Attention Output | `(B, n_heads, T, d_head)` | `(B, T, D)` | Causal Mask 적용 |
| LM Head | `(B, T, D)` | `(B, T, V)` | Logits 출력 |

## 4. 학습 및 샘플링 메모
- Cross-Entropy Loss 계산 및 Gradient Clipping
- Temperature, Top-k, Top-p 샘플링 동작 비교
