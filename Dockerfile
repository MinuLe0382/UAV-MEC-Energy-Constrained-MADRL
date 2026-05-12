FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dev Container: workspaceMount(bind mount)가 /workspace를 덮으므로 COPY . . 불필요
# Standalone 실행 시 (`docker run uav-mec`)에는 -v 옵션으로 마운트하거나
# 이 줄의 주석을 해제해 이미지에 코드 포함 가능
# COPY . .

# Dev Container에서는 VS Code가 keep-alive 명령으로 덮어쓰므로 무시됨
# Standalone 실행 시 기본 진입점으로 유지
CMD ["python", "train.py"]
