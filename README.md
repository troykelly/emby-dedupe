# Emby Deduplication Script Docker Container

The Emby Deduplication Script Docker container assists in managing media libraries on Emby servers by identifying potential duplicate items. It compares media items within your Emby library and generates a report detailing duplicates that may warrant removal.

## Latest Versions

![GitHub release (latest by date)](https://img.shields.io/github/v/release/troykelly/emby-dedupe)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/troykelly/emby-dedupe?include_prereleases&label=pre-release)

## Multi-Architecture Build Status

![Build Status](https://github.com/troykelly/emby-dedupe/actions/workflows/release.yaml/badge.svg)

The following architectures are supported in the latest version:

| Architecture    | Supported          |
|-----------------|--------------------|
| `amd64`         | :white_check_mark: |
| `arm64`         | :white_check_mark: |
| `arm/v7`        | :white_check_mark: |
| `arm/v6`        | :white_check_mark: |
| `i386`          | :x:                |

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Examples](#examples)
- [API Key Requirement](#api-key-requirement)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before using this container, you should have:

- Docker installed on your machine.
- Access to an Emby server with an API key.

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
- `DEDUPE_EMBY_API_KEY`: The API key for the Emby server with appropriate permissions.
- `DEDUPE_DOIT`: Set to 'true' to perform deduplication deletion actions. Requires Emby username and password (defaults to 'false').
- `DEDUPE_LOGGING`: The logging level (e.g., ERROR, WARNING, INFO, DEBUG), affecting verbosity.
- `DEDUPE_EMBY_USERNAME`: Emby username for server access, required if `DEDUPE_DOIT` is 'true'.
- `DEDUPE_EMBY_PASSWORD`: Emby password for server access, required if `DEDUPE_DOIT` is 'true'.

## Examples

### Generating a List of Duplicates (Dry Run)

The following command simulates the deduplication process to provide a list of proposed changes without applying any:

```shell
docker run \
  -e DEDUPE_EMBY_HOST="http://your-emby-server" \
  -e DEDUPE_EMBY_LIBRARY="Your Library Name" \
  -e DEDUPE_EMBY_API_KEY="your_api_key" \
  ghcr.io/troykelly/emby-dedupe
```

### Performing Deduplication Actions

To perform the deletion of duplicates based on the script's output, provide the username and password in addition to other environment variables set earlier:

```shell
docker run \
  -e DEDUPE_EMBY_HOST="http://your-emby-server" \
  -e DEDUPE_EMBY_LIBRARY="Your Library Name" \
  -e DEDUPE_EMBY_API_KEY="your_api_key" \
  -e DEDUPE_EMBY_USERNAME="your_emby_username" \
  -e DEDUPE_EMBY_PASSWORD="your_emby_password" \
  -e DEDUPE_DOIT="true" \
  ghcr.io/troykelly/emby-dedupe
```

## API Key Requirement

A valid API key with enough permissions to access the necessary operations on the Emby server must be provided. This API key is used to authenticate the script with the Emby server for read and list actions. Deletion operations require username and password credentials for additional authentication.

## A Note With Regard To Folders

The script will not remove duplicate folders. Folders are ignored by the script. I'm not entirely sure how to handle them, and given the purpose of the script is to recover disk space, the benefit of a lot of additional logic to check if a folder is empty is yet to present itself. I'm happy to be swayed otherwise, please reach out if you think there's a need for this.

You can (and probably will) end up with empty folders, especially if you keep all your movies in folders.

I'd suggest something like:

```bash
MY_MEDIA=/data/Movies; find "$MY_MEDIA" -type d -empty
```

and if you are happy with that... (NOTE: This _will_ remove empty folders - you've been warned)

```bash
MY_MEDIA=/data/Movies; find "$MY_MEDIA" -type d -empty -delete
```

## Contributing

We welcome your contributions. If you encounter bugs or have suggestions for improvement, please feel free to open an issue on the [GitHub repository](https://github.com/troykelly/emby-dedupe). Pull requests are also greatly appreciated.