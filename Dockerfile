
# Use the official Python 3.12 image as a base
FROM python:3.12
# Install uv (https://github.com/astral-sh/uv)
RUN pip install --upgrade pip && pip install uv
# IPOPT
RUN apt-get update && apt-get install -y coinor-libipopt-dev pkg-config
# ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

ENV MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen


ENV MPLCONFIGDIR=/workspace/.mplcache
RUN mkdir -p /workspace/.mplcache

