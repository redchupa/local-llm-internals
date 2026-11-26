# Phase 3: QLoRA Finetuning (16GB VRAM 파인튜닝 파이프라인)

- **기간**: 9-12주
- **목표**: RTX 5070 Ti (16GB VRAM) 환경 최적화 QLoRA 설정, 커스텀 데이터셋 전처리, SFT 학습 및 모델 병합(GGUF 변환)

## 예정 구성 파일

- `train_sft.py`: Hugging Face `transformers` / `peft` / `trl` 기반 QLoRA SFT 학습 파이프라인
- `configs/`: 모델별·데이터셋별 QLoRA 및 학습 설정 YAML/JSON 파일

## 관련 문서

- [전체 로드맵](../docs/ROADMAP.md)
- [실험 및 트러블슈팅 로그](../docs/EXPERIMENT_LOG.md)
