# Phase 1: Mini-GPT (LLM 핵심 아키텍처 구현)

- **기간**: 1-4주
- **목표**: BPE 토크나이저 원리 분석 및 순수 PyTorch 기반 Decoder-only Transformer(RoPE, GQA 포함) 직접 구현·학습

## 예정 구성 파일

- `model.py`: RMSNorm, RoPE, GQA, SwiGLU를 포함한 Decoder-only Transformer 모델 정의
- `train.py`: 소형 텍스트 데이터셋 학습 루프, 체크포인트 저장, 샘플링 추론 스크립트
- `requirements.txt`: 실행 시 사용한 정확한 라이브러리 버전 고정

## 관련 문서

- [전체 로드맵](../docs/ROADMAP.md)
- [Transformer 핵심 수식 및 텐서 차원 변화](../docs/01_transformer_math.md)
- [실험 및 트러블슈팅 로그](../docs/EXPERIMENT_LOG.md)
