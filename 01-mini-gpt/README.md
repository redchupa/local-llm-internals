# Phase 1: Mini-GPT (LLM 핵심 아키텍처 구현)

- **기간**: 1-4주
- **목표**: BPE 토크나이저 원리 분석 및 순수 PyTorch 기반 Decoder-only Transformer(RoPE, GQA 포함) 직접 구현·학습

## 구성 파일

- [`tokenizer.py`](tokenizer.py): 순수 Python으로 구현한 Byte-level BPE 토크나이저 (학습/인코딩/디코딩/JSON 저장)
- [`tests/test_tokenizer.py`](tests/test_tokenizer.py): 다국어·이모지 round-trip, 미학습 문자열, 저장/로드 일관성 단위 테스트
- [`requirements.txt`](requirements.txt): 실행 의존성 안내
- `model.py` *(2주차 예정)*: RMSNorm, RoPE, GQA, SwiGLU를 포함한 Decoder-only Transformer 모델 정의
- `train.py` *(3주차 예정)*: 소형 텍스트 데이터셋 학습 루프, 체크포인트 저장, 샘플링 추론 스크립트

## 실행 및 테스트

```bash
# BPE 토크나이저 데모 실행 (표준 라이브러리만 사용)
python 01-mini-gpt/tokenizer.py

# 단위 테스트 실행
python -m unittest discover -s 01-mini-gpt/tests -v
```

## 관련 문서

- [전체 로드맵](../docs/ROADMAP.md)
- [Transformer 핵심 수식 및 텐서 차원 변화](../docs/01_transformer_math.md)
- [실험 및 트러블슈팅 로그](../docs/EXPERIMENT_LOG.md)
