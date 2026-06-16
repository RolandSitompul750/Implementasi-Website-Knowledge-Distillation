FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir numpy==2.0.2
RUN pip install --no-cache-dir tensorflow==2.20.0
RUN pip install --no-cache-dir keras==3.13.2
RUN pip install --no-cache-dir flask werkzeug gunicorn

COPY --chown=user . .

RUN mkdir -p static/uploads

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app:app"]