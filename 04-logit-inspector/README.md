# Phase 4: Logit Inspector (통합 툴킷 및 시각화)

- **기간**: 13-16주
- **목표**: 파인튜닝·변환한 모델의 로컬 서빙 API 연동, 추론 로그 및 토큰별 로짓(Logit)/확률 분포 시각화 도구 제작

## 예정 기능

- 로컬 서빙 API(`llama-server` 등) 연동 클라이언트
- 생성 토큰별 Top-k 후보 확률, 로짓, 엔트로피 시각화
- 샘플링 파라미터(Temperature, Top-k, Top-p) 변경에 따른 출력 비교

## 관련 문서

- [전체 로드맵](../docs/ROADMAP.md)
- [실험 및 트러블슈팅 로그](../docs/EXPERIMENT_LOG.md)
