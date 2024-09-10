FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel as builder

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1"

COPY pcdet pcdet
COPY setup.py setup.py

RUN python setup.py bdist_wheel


FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

COPY tools tools
COPY data data
COPY requirements.txt requirements.txt
COPY --from=builder /workspace/dist/ /workspace/dist/

RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libglib2.0-0 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "dist/$(ls dist)" > new_requirements.txt && \
    cat requirements.txt >> new_requirements.txt

RUN pip install -r new_requirements.txt

WORKDIR /workspace/tools
