FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN useradd -ms /bin/bash pyuser && mkdir /app && chown pyuser /app
COPY --chown=pyuser GFPGANv1.3.pth /app/
COPY --chown=pyuser models/ /app/models/
COPY --chown=pyuser requirements*.txt /app/
USER pyuser
WORKDIR /app
COPY --chown=pyuser webui-user.sh /app/
COPY --chown=pyuser docker_build.sh /app/
COPY --chown=pyuser launch.py /app/
RUN export PATH=$PATH:/home/pyuser/.local/bin; mkdir embeddings; mkdir repositories; mkdir tmp; mkdir outputs; bash docker_build.sh
COPY --chown=pyuser *.csv /app/
COPY --chown=pyuser *.sh /app/
COPY --chown=pyuser *.json /app/
COPY --chown=pyuser *.js /app/
COPY --chown=pyuser *.css /app/
COPY --chown=pyuser javascript/ /app/javascript/
COPY --chown=pyuser modules/ /app/modules/
COPY --chown=pyuser scripts/ /app/scripts/
COPY --chown=pyuser *.py /app/
COPY --chown=pyuser textual_inversion_templates/ /app/textual_inversion_templates/
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
EXPOSE 7860
EXPOSE 42587
ENTRYPOINT ["bash", "docker_launch.sh"]
