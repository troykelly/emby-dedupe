# -- Build stage -- #
FROM python:3.9-slim as build-stage

# Set build-time metadata as defined at http://label-schema.org
ARG BUILD_DATE
ARG VCS_REF
ARG VCS_URL="https://github.com/troykelly/emby-dedupe"
ARG VERSION="edge"

LABEL org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.name="emby-dedupe" \
    org.label-schema.description="A Docker container to run the Emby deduplication script." \
    org.label-schema.url="https://github.com/troykelly/emby-dedupe#readme" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url=$VCS_URL \
    org.label-schema.vendor="Troy Kelly" \
    org.label-schema.version=$VERSION \
    org.label-schema.schema-version="1.0" \
    org.opencontainers.image.source=$VCS_URL

# Set the working directory
WORKDIR /build

# Create a virtual environment to isolate our package dependencies locally
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies:
# We use a requirements file to specify all the required Python modules.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files into the build stage
COPY scripts/dedupe.py ./dedupe.py


# -- Final stage -- #
FROM python:3.9-slim

WORKDIR /app

# Import the virtual environment from the build stage
COPY --from=build-stage /opt/venv /opt/venv

# Make sure scripts in the virtualenv are usable
ENV PATH="/opt/venv/bin:$PATH"

# Copy the built application from the build stage to the final stage
COPY --from=build-stage /build/dedupe.py ./dedupe.py

# Copy everything from rootfs to the root of the container
COPY rootfs/ /

# Set correct permissions for the entrypoint script
RUN chmod +x /usr/local/sbin/entrypoint

# We run our application as a non-root user for security reasons.
RUN useradd --create-home --shell /bin/bash embyuser
USER embyuser

# Set the entrypoint script as the Docker entrypoint
ENTRYPOINT ["/usr/local/sbin/entrypoint"]