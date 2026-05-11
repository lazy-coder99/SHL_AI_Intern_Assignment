FROM python:3.10

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set cache directories to a writable location for Hugging Face Spaces
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
RUN mkdir -p /app/.cache/transformers /app/.cache/sentence_transformers

# Copy the rest of the application
COPY --chown=user . .

# Hugging Face Spaces expose port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
