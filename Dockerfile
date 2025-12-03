FROM nvcr.io/nvidia/pytorch:22.12-py3

# 1. Configurações de Ambiente
ENV DEBIAN_FRONTEND=noninteractive

# 2. Instalação de Dependências do Sistema (Essencial para o Jogo)
RUN apt-get update && apt-get --no-install-recommends install -yq \
    git cmake build-essential \
    libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
    python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Atualização do Pip e Instalação de Pacotes Básicos
RUN python3 -m pip install --upgrade pip setuptools wheel psutil

# 4. Instalação das Bibliotecas de Inteligência Artificial (O QUE FALTAVA)
# Shimmy é necessário para conectar o GFootball ao Stable-Baselines3
RUN python3 -m pip install "stable-baselines3[extra]>=2.0.0" "shimmy>=2.0.0" tensorboard

# 5. Instalação do Google Football
# Copia os arquivos da pasta atual para dentro do container
COPY . /gfootball
RUN cd /gfootball && python3 -m pip install .

# 6. Correção do Erro de MPI/HPCX (O PULO DO GATO)
# Isso evita o erro de "undefined symbol" que travava o PyTorch
RUN mv /opt/hpcx /opt/hpcx_backup || true

# 7. Correção do OpenCV (Opcional, mas mantive do seu original)
RUN python3 -m pip install opencv-fixer
RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"

WORKDIR '/gfootball'
