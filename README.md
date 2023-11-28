# Emby Deduplication Script Docker Container

The Emby Deduplication Script Docker container assists in managing media libraries on Emby servers by identifying potential duplicate items. It compares media items within your Emby library and generates a report detailing duplicates that may warrant removal.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Examples](#examples)
- [Note on API Key](#note-on-api-key)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before using this container, you should have:

- Docker installed on your machine.
- Access to an Emby server with a username and password.

## Installation

To install the Docker container, pull the image from the GitHub Container Registry:

```shell
docker pull ghcr.io/troykelly/emby-dedupe:latest
```

Replace `latest` with the appropriate tag to use a specific version of the container.

## Usage

The `emby-dedupe` script is used within a Docker container by executing `docker run` with necessary environment variables to configure the connection to your Emby server along with parameters for the deduplication process.

## Environment Variables

The following variables are used to run the script:

- `DEDUPE_EMBY_HOST`: The hostname or IP of the Emby server.
- `DEDUPE_EMBY_PORT`: The port for the Emby server (defaults to 8096 if not specified).
- `DEDUPE_EMBY_LIBRARY`: The name of the library on the Emby server you want to deduplicate.
- `DEDUPE_EMBY_USERNAME`: Emby username for server access.
- `DEDUPE_EMBY_PASSWORD`: Emby password for server access.
- `DEDUPE_EMBY_API_KEY`: A placeholder for an API key; currently, any non-empty value will suffice.
- `DEDUPE_DOIT`: Set to 'true' to perform deduplication deletion actions (defaults to 'false').
- `DEDUPE_LOGGING`: The logging level (e.g., ERROR, WARNING, INFO, DEBUG), affecting verbosity.

## Examples

### Generating a List of Duplicates (Dry Run)

The following command simulates the deduplication process to provide a list of proposed changes without applying any:

```shell
docker run \
  -e DEDUPE_EMBY_HOST="http://your-emby-server" \
  -e DEDUPE_EMBY_LIBRARY="Your Library Name" \
  -e DEDUPE_EMBY_USERNAME="your_emby_username" \
  -e DEDUPE_EMBY_PASSWORD="your_emby_password" \
  -e DEDUPE_EMBY_API_KEY="notused" \
  ghcr.io/troykelly/emby-dedupe
```

### Performing Deduplication Actions

To perform the deletion of duplicates based on the script's output:

```shell
docker run \
  -e DEDUPE_EMBY_HOST="http://your-emby-server" \
  -e DEDUPE_EMBY_LIBRARY="Your Library Name" \
  -e DEDUPE_EMBY_USERNAME="your_emby_username" \
  -e DEDUPE_EMBY_PASSWORD="your_emby_password" \
  -e DEDUPE_EMBY_API_KEY="notused" \
  -e DEDUPE_DOIT="true" \
  ghcr.io/troykelly/emby-dedupe
```

## Note on API Key

The code currently contains a placeholder for an Emby API key. While the API key is required to be passed as an environment variable, it does not need to be valid, as it is not used in the current operation of this script. This will be addressed and potentially removed in future updates, transitioning to a more secure method of authentication.

## Acknowledgments

This project is possible thanks to the Emby media server and the Python libraries enabling easy HTTP communications and multi-threaded operations.

## Contributing

We welcome your contributions. If you encounter bugs or have suggestions for improvement, please feel free to open an issue on the [GitHub repository](https://github.com/troykelly/emby-dedupe). Pull requests are also greatly appreciated.
