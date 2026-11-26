# 실험 및 트러블슈팅 로그

> 완성된 코드만큼 **실패한 실험과 해결 과정(OOM 에러, NaN Loss, 양자화 손실 비교 등)**의 기록이 중요합니다. 모든 실험은 재현 가능한 메타데이터와 함께 기록합니다.

## 기록 템플릿

```markdown
### [YYYY-MM-DD] Phase X - 실험 제목

- **목적**: 무엇을 검증/해결하려는 실험인지 요약
- **환경**:
  - GPU / VRAM: RTX 5070 Ti (16GB)
  - OS / Container: WSL2 Ubuntu / Docker 이미지 태그
  - 주요 버전: Python X.Y, PyTorch X.Y, CUDA X.Y
  - Seed: `42`
- **설정 및 명령어**:
  ```bash
  # 실행 명령어
  ```
- **결과**:
  - Peak VRAM: `X.XX GB`
  - 속도 / 손실: `XXX tok/s`, `loss = X.XXX`
- **문제 및 해결 과정**:
  - 발생한 오류(OOM, 빌드 실패 등)와 원인 분석, 적용한 수정 사항
```

---

## 실험 기록

*(Phase 1 첫 실험부터 아래에 날짜 역순 또는 시간순으로 기록합니다.)*
