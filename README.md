# Emby Deduplication Script Docker Container

This Docker container is designed to run the `emby-dedupe` script, which helps you deduplicate media in your Emby server libraries. It safely identifies and removes duplicate copies of media files from your Emby library, ensuring that each title is unique within your collection.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Examples](#examples)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have a working Docker environment.
- You have an Emby server setup with an accessible API.

## Installation

To use this Docker container, you can pull it from the GitHub Container Registry:

```shell
docker pull ghcr.io/troykelly/emby-dedupe:latest
```

Replace `latest` with a specific version tag if needed.

## Usage

To run the `emby-dedupe` script in a Docker container, you can use the `docker run` command with the necessary environment variables set:

Basic command structure:
```shell
docker run -e [ENV_VAR_NAME]=[VALUE] ghcr.io/troykelly/emby-dedupe
```

## Environment Variables

The following environment variables are used to configure the `emby-dedupe` script:

- `DEDUPE_EMBY_HOST`: The hostname of the Emby server (required).
- `DEDUPE_EMBY_API_KEY`: The Emby server API key (optional but recommended for full functionality).
- `DEDUPE_EMBY_PORT`: The port number for the Emby server (optional, defaults to 8096).
- `DEDUPE_EMBY_LIBRARY`: The name of the Emby library to deduplicate (required).
- `DEDUPE_DOIT`: Set to 'true' if you want to actually perform deletions (optional, defaults to false).
- `DEDUPE_EMBY_USERNAME`: The Emby username for authentication (required if no API key is provided).
- `DEDUPE_EMBY_PASSWORD`: The Emby password for authentication (required if no API key is provided).
- `DEDUPE_LOGGING`: The logging level (optional, defaults to ERROR; other options include WARNING, INFO, DEBUG).

## Examples

### Generate a List of Changes

To generate a report of what would be duplicated without making any changes:

```shell
docker run \
  -e DEDUPE_EMBY_HOST="http://embyserver:8096" \
  -e DEDUPE_EMBY_API_KEY="your_emby_api_key" \
  -e DEDUPE_EMBY_LIBRARY="My Movies" \
  ghcr.io/troykelly/emby-dedupe
```

### Perform the Deletion

To actually delete the duplicate items from your Emby library:

```shell
docker run \
  -e DEDUPE_EMBY_HOST="http://embyserver:8096" \
  -e DEDUPE_EMBY_API_KEY="your_emby_api_key" \
  -e DEDUPE_EMBY_LIBRARY="My Movies" \
  -e DEDUPE_DOIT="true" \
  ghcr.io/troykelly/emby-dedupe
```

## Acknowledgments

This project acknowledges:

- The Emby server, which provides a personal media solution (https://emby.media).
- The creators of Docker, for the containerization platform (https://docker.com).
- All contributors who report issues, suggest features, or contribute code to this project.

I would like to specifically thank the developers and maintainers of `httpx`, `tqdm`, and `backoff`, which make `emby-dedupe` robust and user-friendly.

## Contributing

Contributions are welcome! For bug reports or feature requests, please open an issue through the GitHub issue tracker. Feel free to fork the repository and submit pull requests.