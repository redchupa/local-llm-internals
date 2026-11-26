# Docker 개발 환경

- **목표**: Windows 11 Pro (WSL2 Ubuntu + NVIDIA Container Toolkit) 환경에서 RTX 5070 Ti (16GB VRAM) GPU 가속을 재현 가능하게 구성합니다.

## 예정 구성 파일

- `Dockerfile`: CUDA + PyTorch + CMake/C++ 빌드 도구(`llama.cpp` 디버깅용)가 포함된 통합 개발 이미지

## 사전 확인 체크리스트

- [ ] WSL2 Ubuntu에서 `nvidia-smi` 정상 인식 확인
- [ ] Docker Desktop + NVIDIA Container Toolkit GPU 패스스루 확인
- [ ] PyTorch CUDA 가용성(`torch.cuda.is_available()`) 및 VRAM(16GB) 확인
