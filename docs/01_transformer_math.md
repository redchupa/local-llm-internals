# 01. Transformer 핵심 수식 및 텐서 차원 변화

> **Phase 1 (1-4주)** 산출물 — `01-mini-gpt/` 구현과 함께 채워 나가는 기술 노트입니다.

## 1. BPE 토크나이저 원리 (`01-mini-gpt/tokenizer.py`)

LLM은 문자열을 직접 이해하지 못하고 **정수(Token ID) 시퀀스**만 입력으로 받습니다. 현대 LLM(GPT-2, GPT-4, Llama 3 등)은 **Byte-level BPE (Byte Pair Encoding)** 방식을 사용합니다.

### 왜 Byte-level(UTF-8 바이트 기반)인가?
- 단어 단위(Word-level)로 사전을 만들면 처음 보는 신조어·오타·다국어가 들어올 때 `<UNK>`(Unknown) 토큰으로 뭉개집니다.
- 모든 텍스트를 UTF-8 바이트(`0`부터 `255`까지 총 256개 기본 토큰)로 분해해서 시작하면, **영어·한글·특수문자·이모지 어떤 문자열도 `<UNK>` 없이 100% 복원 가능**합니다.

### 학습(Train)과 인코딩(Encode) 알고리즘
1. **학습 (`train`)**:
   - 전체 코퍼스를 UTF-8 바이트 ID 리스트로 변환합니다.
   - 인접한 토큰 쌍 `(a, b)`의 등장 빈도를 센 뒤, 가장 자주 나온 쌍을 새 토큰 ID(`256`, `257`, ...)로 치환합니다.
   - 이 과정을 `vocab_size - 256`번 반복하며 병합 규칙(`merges`)을 순서대로 기록합니다.
2. **인코딩 (`encode`)**:
   - 입력 문자열을 UTF-8 바이트로 변환한 뒤, 학습 시 **가장 먼저 만들어진 병합 규칙(가장 낮은 `new_id`)**부터 우선 적용해 시퀀스를 압축합니다.
3. **디코딩 (`decode`)**:
   - 각 토큰 ID를 원래 `bytes`로 치환해 이어 붙인 뒤 UTF-8로 디코딩합니다(`errors="replace"`로 불완전한 바이트 시퀀스 방어).

### `vocab_size`의 트레이드오프
- **`vocab_size`가 클수록**: 같은 문장을 더 적은 토큰 수로 압축(Context Window 효율 증가, 추론 속도 향상)할 수 있지만, `Token Embedding (V x D)` 및 `LM Head (D x V)` 가중치 행렬과 마지막 Softmax 연산 비용이 커집니다.
- **`vocab_size`가 작을수록**: 임베딩 파라미터는 줄어들지만 시퀀스 길이 `T`가 길어져 Attention 연산량($O(T^2)$)과 KV Cache 메모리가 급증합니다.

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
