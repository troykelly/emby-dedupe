#!/usr/bin/env python

import os
import sys
import logging
import argparse
import httpx
from httpx import URL
from typing import Optional, Tuple
import json
import os
from typing import Any
import backoff


MAX_RETRIES = 5  # The maximum number of retries for HTTP requests

# Define constants for default values and environmental variable names
ENV_DEDUPE_LOGGING = "DEDUPE_LOGGING"
ENV_DEDUPE_EMBY_HOST = "DEDUPE_EMBY_HOST"
ENV_DEDUPE_EMBY_PORT = "DEDUPE_EMBY_PORT"
ENV_DEDUPE_EMBY_API_KEY = "DEDUPE_EMBY_API_KEY"
ENV_DEDUPE_EMBY_LIBRARY = "DEDUPE_EMBY_LIBRARY"
ENV_DEDUPE_DOIT = "DEDUPE_DOIT"

DEFAULT_PORT_HTTP = 80
DEFAULT_PORT_HTTPS = 443
DEFAULT_PORT_EMBY = 8096
LOGGING_LEVELS = {
    "": logging.ERROR,  # Default to ERROR if no verbosity
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


@backoff.on_exception(
    backoff.expo,  # Exponential backoff strategy
    httpx.HTTPError,  # Retry on HTTP error responses
    max_time=60,  # Maximum total time to backoff for
    max_tries=MAX_RETRIES,  # Maximum number of retries per call
)
def make_http_request(
    client: httpx.Client, method: str, url: str, **kwargs
) -> httpx.Response:
    """
    Make an HTTP request using the given httpx.Client, with exponential backoff in case of exceptions.

    Args:
        client (httpx.Client): The httpx client to use for sending the request.
        method (str): The HTTP method to use (e.g., 'GET', 'POST', 'DELETE').
        url (str): The URL to which the request is sent.
        **kwargs: Additional keyword arguments to pass to the client's request method.

    Returns:
        httpx.Response: The HTTP response from the server.
    """
    try:
        response = client.request(method, url, **kwargs)
        response.raise_for_status()  # Will raise an error for 4xx and 5xx responses
        return response
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error occurred: {e.response.content}")
        raise
    except httpx.RequestError as e:
        logging.error(f"Request error occurred: {str(e)}")
        raise


def create_http_client(api_key: str) -> httpx.Client:
    """
    Create an httpx.Client instance configured with retry logic and API key headers.

    Args:
        api_key (str): The API key for the Emby server.

    Returns:
        httpx.Client: A client instance configured for communication with Emby server.
    """
    headers = {"X-Emby-Token": api_key}
    retries = httpx.Retry(
        max_retries=MAX_RETRIES,
        backoff_factor=1,  # Exponential backoff factor
        status_forcelist=(500, 502, 503, 504),  # Retry on these status codes
        http_methods={"GET", "DELETE"},
    )
    client = httpx.Client(headers=headers, retries=retries)
    return client


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the parsed arguments.

    Returns:
        argparse.Namespace: An object holding attributes based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Emby Media Deduplication Script.")
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="Increase verbosity of logging for each occurrence.",
    )
    parser.add_argument("--host", type=str, help="The hostname of the Emby server.")
    parser.add_argument(
        "-p", "--port", type=int, help="The port number to use for the Emby server."
    )
    parser.add_argument("-a", "--api-key", type=str, help="The Emby server API key.")
    parser.add_argument(
        "-l", "--library", type=str, help="The Emby library to scan for duplicates."
    )
    parser.add_argument(
        "--doit",
        action="store_true",
        help="Must be provided for the script to remove media.",
    )
    return parser.parse_args()


def set_logging_level(verbosity_count: int, env_verbosity: Optional[str]) -> None:
    """Set logging level based on verbosity count and environment variable.

    Args:
        verbosity_count (int): Count of verbose flags (-v) in the command line.
        env_verbosity (Optional[str]): Verbosity level from the environment variable.
    """
    # Determine the logging level
    levels = ["", "WARNING", "INFO", "DEBUG"]
    level_name = env_verbosity or ""
    if verbosity_count:
        level_name = levels[min(verbosity_count, len(levels) - 1)]
    level = LOGGING_LEVELS.get(level_name, logging.ERROR)
    logging.basicConfig(level=level)
    logging.info(f"Logging level set to {logging.getLevelName(level)}")


def override_warning(arg_name: str, cmd_val: str, env_val: str) -> None:
    """Print a warning if a command-line argument overrides an environment variable.

    Args:
        arg_name (str): The name of the argument being overridden.
        cmd_val (str): The value from the command line.
        env_val (str): The value from the environment variable.
    """
    if cmd_val and env_val:
        logging.warning(
            f"Warning: The command-line argument {arg_name} ('{cmd_val}') "
            f"overrides the environment variable ('{env_val}')."
        )


def get_env_variable(name: str) -> Optional[str]:
    """Get the value of an environment variable.

    Args:
        name (str): The name of the environment variable to retrieve.

    Returns:
        Optional[str]: The value of the environment variable, if it exists.
    """
    return os.environ.get(name)


def validate_required_arguments(
    host: Optional[str], api_key: Optional[str], library: Optional[str]
):
    """Validate that required arguments are provided.

    Args:
        host (Optional[str]): The host of the Emby server.
        api_key (Optional[str]): The API key for the Emby server.
        library (Optional[str]): The library to scan for duplicates.
    """
    missing_args = []

    for arg, value in {"host": host, "api-key": api_key, "library": library}.items():
        if not value:
            missing_args.append(arg)

    if missing_args:
        missing_args_str = ", ".join(missing_args)
        print(f"Error: Missing required arguments: {missing_args_str}")
        print("Use -h for help.")
        sys.exit(1)


def handle_host_and_port(host: str, arg_port: Optional[int]) -> Tuple[str, int]:
    """
    Validate and handle the combination of host and port information.

    Args:
        host (str): The input host which may include protocol and port.
        arg_port (Optional[int]): The input port.

    Returns:
        Tuple[str, int]: The validated host and port.
    """
    url = URL(host)
    scheme = url.scheme or "http"
    final_host = (
        url.host or host
    )  # Default to using the original host if no scheme is provided.
    final_port = url.port

    # Determine default ports if not provided based on the scheme.
    if not final_port:
        if scheme == "https":
            final_port = DEFAULT_PORT_HTTPS
        elif scheme == "http":
            final_port = DEFAULT_PORT_HTTP
        else:
            final_port = DEFAULT_PORT_EMBY  # Fallback to default Emby port.

    # Use the port from arguments if it was explicitly provided and differs from the URL port.
    if arg_port is not None and final_port != arg_port:
        logging.warning(
            f"The port number from the URL '{final_port}' is overridden by the command-line argument port '{arg_port}'."
        )
        final_port = arg_port

    return f"{scheme}://{final_host}", final_port


# We should define an exception for handling API communication issues
class EmbyServerConnectionError(Exception):
    """Custom Exception for Emby Server Connection Errors."""


def check_emby_connection(client: httpx.Client, url: str) -> bool:
    """
    Check the connection to the Emby server by making a simple API request using the provided session.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        url (str): The URL to make the GET request to.

    Returns:
        bool: True if the server is reachable and the API key is valid, False otherwise.

    Raises:
        EmbyServerConnectionError: If there's an issue with connecting to the Emby server.
    """
    logging.debug(f"Checking connection to Emby server at {url}")
    try:
        response = make_http_request(client, "GET", url)
        # If the request was successful, we don't need to check response status
        # since 'make_http_request' already calls 'raise_for_status()'.
        logging.info("Successfully connected to the Emby server.")
        return True
    except httpx.HTTPStatusError as e:
        raise EmbyServerConnectionError(
            f"Failed to connect to Emby server: {e.response.content}"
        )
    except httpx.RequestError as e:
        raise EmbyServerConnectionError(
            f"An error occurred while communicating with Emby server: {str(e)}"
        )


def validate_library(base_url: str, api_key: str, library_name: str) -> bool:
    """
    Confirm that the specified library exists and it has at least one item.

    Args:
        base_url (str): Base URL of the Emby server.
        api_key (str): API key for the Emby server.
        library_name (str): The name of the library to be validated.

    Returns:
        bool: True if the library exists and contains items, False otherwise.

    Raises:
        EmbyServerConnectionError: If there's an issue with connecting to the Emby server.
    """
    headers = {"X-Emby-Token": api_key}
    response = httpx.get(f"{base_url}/Library/VirtualFolders", headers=headers)

    try:
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        virtual_folders = response.json()

        for folder in virtual_folders:
            if library_name == folder.get("Name"):
                # Here we could add additional checks for folder content if needed
                logging.info(f"Library '{library_name}' validated successfully.")
                return True

        logging.error(f"Library '{library_name}' is not found.")
        return False

    except httpx.HTTPStatusError as e:
        raise EmbyServerConnectionError(
            f"Failed to validate library: {e.response.content}"
        )
    except httpx.RequestError as e:
        raise EmbyServerConnectionError(
            f"An error occurred while checking the library: {str(e)}"
        )


def build_provider_id_tables(media_items: list, provider_tables: dict):
    """
    Builds tables that map provider IDs (Imdb, Tvdb, Tmdb) to lists of media item IDs,
    ignoring items with specific IMDb values.

    Args:
        media_items (list): A list of media items fetched from the Emby server.
        provider_tables (dict): A dictionary with keys 'imdb', 'tvdb', 'tmdb' to store the mappings.
    """
    IGNORED_IMDB_ID = "tt0000000"  # IMDb ID to ignore

    for item in media_items:
        provider_ids = item.get("ProviderIds", {})
        for provider, table_name in [
            ("Imdb", "imdb"),
            ("Tvdb", "tvdb"),
            ("Tmdb", "tmdb"),
        ]:
            id_value = provider_ids.get(provider)

            # Skip the IMDb ID if it is the one we're ignoring
            if provider == "Imdb" and id_value == IGNORED_IMDB_ID:
                continue

            if id_value:
                if id_value not in provider_tables[table_name]:
                    provider_tables[table_name][id_value] = []
                provider_tables[table_name][id_value].append(item["Id"])


def fetch_and_process_media_items(
    base_url: str, api_key: str, library_id: str
) -> Tuple[dict, dict, dict]:
    """
    Fetches media items in an efficient paginated manner and builds the provider ID tables.

    Args:
        base_url (str): Base URL of the Emby server.
        api_key (str): API key for the Emby server.
        library_id (str): The ID of the library/virtual folder from which to fetch media items.

    Returns:
        Tuple[dict, dict, dict]: Three dictionaries (imdb, tvdb, tmdb) containing provider ID mappings.
    """
    page_size = 100  # Efficient page size to minimize memory consumption
    start_index = 0  # Starting index for pagination

    # Initialize provider ID tables
    provider_tables = {"imdb": {}, "tvdb": {}, "tmdb": {}}
    headers = {"X-Emby-Token": api_key}

    # Fetch the total number of items.
    url = f"{base_url}/Items?StartIndex=0&Limit=0&Recursive=True&ParentId={library_id}&Fields=ProviderIds&Is3D=False"
    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    total_items = response.json().get("TotalRecordCount", 0)

    while start_index < total_items:
        url = f"{base_url}/Items?StartIndex={start_index}&Limit={page_size}&Recursive=True&ParentId={library_id}&Fields=ProviderIds&Is3D=False"
        response = httpx.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        media_items = data.get("Items", [])

        # Process the current chunk of items to build provider ID tables
        build_provider_id_tables(media_items, provider_tables)

        # Report progress
        start_index += page_size
        logging.info(f"Processed {start_index}/{total_items} items.")

    return provider_tables


def identify_duplicates(provider_tables: dict) -> dict:
    """
    Identifies duplicates by looking for provider IDs with multiple associated media item IDs.

    Args:
        provider_tables (dict): The table of provider IDs and associated media item IDs.

    Returns:
        dict: A dictionary of provider IDs and list of duplicate media item IDs.
    """
    duplicates = {}
    for provider, id_table in provider_tables.items():
        duplicates[provider] = {
            pid: items for pid, items in id_table.items() if len(items) > 1
        }
    return duplicates


def generate_report(unique_items: list, duplicates: list) -> None:
    """
    Generates a report of duplicates and unique items.

    Args:
        unique_items (list): A list of media items considered as unique.
        duplicates (list): A list of duplicate media items.
    """
    # Placeholder: Generate a report. You might want to write this to a file or print it.
    print(f"Unique Items ({len(unique_items)}): {json.dumps(unique_items, indent=4)}")
    print(f"Duplicates ({len(duplicates)}): {json.dumps(duplicates, indent=4)}")


def delete_items_if_doit(
    base_url: str, api_key: str, duplicates: list, doit: bool
) -> None:
    """
    Deletes duplicate media items if the 'doit' flag is true.

    Args:
        base_url (str): Base URL of the Emby server.
        api_key (str): API key for the Emby server.
        duplicates (list): A list of duplicate media items to delete.
        doit (bool): If True, actually perform the deletion.
    """
    if doit:
        # Placeholder: Perform the deletion of duplicate items using the Emby API.
        # Replace with actual API call to delete items.
        for item in duplicates:
            item_id = item[
                "Id"
            ]  # Placeholder: Replace with actual property that holds the Id.
            url = f"{base_url}/Items/{item_id}"
            headers = {"X-Emby-Token": api_key}
            try:
                response = httpx.delete(url, headers=headers)
                response.raise_for_status()
                logging.info(f"Deleted item with ID {item_id}")
            except httpx.HTTPStatusError as e:
                logging.error(
                    f"Failed to delete item with ID {item_id}: {e.response.content}"
                )
            except httpx.RequestError as e:
                logging.error(
                    f"An error occurred while deleting item with ID {item_id}: {str(e)}"
                )
    else:
        logging.info(
            "Deletion skipped. Items to be deleted are only listed in the report."
        )


def get_library_id(base_url: str, api_key: str, library_name: str) -> Optional[str]:
    """
    Retrieves the ID of the specified library by name.

    Args:
        base_url (str): Base URL of the Emby server.
        api_key (str): API key for the Emby server.
        library_name (str): The name of the library to retrieve the ID for.

    Returns:
        Optional[str]: The ID of the library if found, else None.
    """
    headers = {"X-Emby-Token": api_key}
    url = f"{base_url}/Library/VirtualFolders"
    try:
        response = httpx.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses.
        virtual_folders = response.json()

        for folder in virtual_folders:
            if folder.get("Name") == library_name:
                return folder.get("Id")

        logging.error(f"Library '{library_name}' not found.")
        return None  # Return None if library is not found

    except httpx.HTTPStatusError as e:
        logging.error(
            f"HTTP error occurred while retrieving library ID: {e.response.content}"
        )
    except httpx.RequestError as e:
        logging.error(f"HTTP request to Emby server failed: {str(e)}")

    return None  # Return None if any exception occurred


def dump_object_to_file(obj: Any, base_filename: str) -> None:
    """
    Attempts to serialize and save an object to a file.
    If the object is serializable to JSON, it's saved as pretty JSON with a '.json' extension.
    If it's a binary object, it's saved as is with a '.bin' extension.
    If it's a text object, it's saved as text with a '.txt' extension.

    Args:
        obj (Any): The object to be saved to a file.
        base_filename (str): The base name for the file to save to (without an extension).

    Raises:
        ValueError: If the object type cannot be determined or handled by the function.
    """
    # Try to save as JSON
    try:
        json_data = json.dumps(obj, indent=4)
        full_filename = f"{base_filename}.json"
        with open(full_filename, "w", encoding="utf-8") as file:
            file.write(json_data)
        print(f"Object saved as JSON to {full_filename}")
        return
    except TypeError:
        pass  # Object is not JSON serializable, moving to the next check

    # Check if it's binary data
    if isinstance(obj, bytes):
        full_filename = f"{base_filename}.bin"
        with open(full_filename, "wb") as file:
            file.write(obj)
        print(f"Object saved as binary to {full_filename}")
        return

    # Check if it's text data
    if isinstance(obj, str):
        full_filename = f"{base_filename}.txt"
        with open(full_filename, "w", encoding="utf-8") as file:
            file.write(obj)
        print(f"Object saved as text to {full_filename}")
        return

    # If none of the above, raise an error
    raise ValueError("Object type is not supported for dumping to a file.")


def determine_items_to_delete(base_url: str, api_key: str, duplicate_ids: list) -> dict:
    """
    Determines the best quality media item to keep and marks the rest for deletion.
    The criteria for the best quality is based on resolution, codec preference, interlacing,
    bitrate, bit depth, and file size.

    Args:
        base_url (str): Base URL of the Emby server.
        api_key (str): API key for the Emby server.
        duplicate_ids (list): A list of IDs of potentially duplicate media items.

    Returns:
        dict: A dictionary containing details of the item to keep and the ones to delete.
    """
    headers = {"X-Emby-Token": api_key}
    items = []

    # Fetch detailed item information
    for item_id in duplicate_ids:
        url = f"{base_url}/Items/{item_id}?Fields=MediaStreams,Path"
        response = httpx.get(url, headers=headers)
        response.raise_for_status()
        items.append(response.json())

    # Sort items by quality based on provided criteria
    items.sort(
        key=lambda i: (
            # Resolution (first by height then by width)
            max(
                stream.get("Height", 0)
                for stream in i["MediaStreams"]
                if stream["Type"] == "Video"
            ),
            max(
                stream.get("Width", 0)
                for stream in i["MediaStreams"]
                if stream["Type"] == "Video"
            ),
            # Codec preference (HEVC > h264)
            -1
            if any(
                stream.get("Codec") == "hevc"
                for stream in i["MediaStreams"]
                if stream["Type"] == "Video"
            )
            else 0,
            # Interlacing (prefer progressive)
            0
            if any(
                not stream.get("IsInterlaced", True)
                for stream in i["MediaStreams"]
                if stream["Type"] == "Video"
            )
            else 1,
            # Bit rate and bit depth (prefer higher)
            max(
                (stream.get("BitRate", 0), stream.get("BitDepth", 0))
                for stream in i["MediaStreams"]
                if stream["Type"] == "Video"
            ),
            # File size (prefer larger)
            i["Size"],
        ),
        reverse=True,
    )

    # The first item is the best quality; the rest are duplicates
    best_item = items[0]
    duplicate_items = items[1:]

    return {
        "keep": {
            "id": best_item["Id"],
            "name": best_item["Name"],
            "path": best_item["Path"],
            "quality": get_quality_description(best_item),
        },
        "delete": [
            {
                "id": item["Id"],
                "name": item["Name"],
                "quality": get_quality_description(item),
            }
            for item in duplicate_items
        ],
    }


def get_quality_description(item):
    """
    Get the quality description from the media item.

    Args:
        item (dict): A media item containing MediaStreams.

    Returns:
        dict: A description of the quality of the given media item.
    """
    video_stream = next((s for s in item["MediaStreams"] if s["Type"] == "Video"), None)
    audio_stream = next((s for s in item["MediaStreams"] if s["Type"] == "Audio"), None)

    return {
        "video": {
            "codec": video_stream.get("Codec") if video_stream else "unknown",
            "resolution": video_stream.get("DisplayTitle")
            if video_stream
            else "unknown",
            "bitrate": video_stream.get("BitRate") if video_stream else "unknown",
            "bitdepth": video_stream.get("BitDepth") if video_stream else "unknown",
            "interlaced": video_stream.get("IsInterlaced")
            if video_stream
            else "unknown",
        },
        "audio": {
            "codec": audio_stream.get("Codec") if audio_stream else "unknown",
            "channels": audio_stream.get("Channels") if audio_stream else "unknown",
            "bitrate": audio_stream.get("BitRate") if audio_stream else "unknown",
        },
        "size": item.get("Size"),
    }


def merge_duplicate_groups(duplicates_by_provider: dict) -> list:
    """
    Merges duplicate lists from all providers into groups of potential duplicates.
    Items are considered duplicates if they have a match in any provider.

    Args:
        duplicates_by_provider (dict): Dictionary of duplicates by provider, mapping provider IDs to item ID lists.

    Returns:
        list: A list of lists, where each inner list is a group of potential duplicate IDs.
    """
    duplicate_groups = []

    def merge_into_groups(dup_ids):
        nonlocal duplicate_groups
        merged_group = set()
        remaining_groups = []

        # Determine if the IDs should be merged with an existing group
        for existing_group in duplicate_groups:
            if any(dup_id in existing_group for dup_id in dup_ids):
                merged_group.update(existing_group)
            else:
                remaining_groups.append(existing_group)

        # Merge the IDs with any intersecting group or add as a new group
        if merged_group:
            merged_group.update(dup_ids)
            remaining_groups.append(merged_group)
        else:
            remaining_groups.append(set(dup_ids))

        duplicate_groups = remaining_groups

    # Iterate over each provider and merge their duplicates into the groups
    for provider_duplicates in duplicates_by_provider.values():
        for dup_ids in provider_duplicates.values():
            merge_into_groups(dup_ids)

    # Convert sets back to lists for JSON serialization
    return [list(group) for group in duplicate_groups]


def main():
    # The 'cmd_args' dictionary will store the command line arguments
    args = parse_args()

    # Retrieve environment variables
    env_verbosity = get_env_variable(ENV_DEDUPE_LOGGING)
    env_host = get_env_variable(ENV_DEDUPE_EMBY_HOST)
    env_port = get_env_variable(ENV_DEDUPE_EMBY_PORT)
    env_api_key = get_env_variable(ENV_DEDUPE_EMBY_API_KEY)
    env_library = get_env_variable(ENV_DEDUPE_EMBY_LIBRARY)
    env_doit = get_env_variable(ENV_DEDUPE_DOIT) in ("true", "True", "TRUE", "1")

    # Set logging verbosity
    set_logging_level(args.verbosity, env_verbosity)

    # Check for overrides with warnings
    override_warning(
        "--verbosity", args.verbosity and LOGGING_LEVELS[args.verbosity], env_verbosity
    )
    override_warning("--host", args.host, env_host)
    override_warning("--port", args.port and str(args.port), env_port)
    override_warning("--api-key", args.api_key, env_api_key)
    override_warning("--library", args.library, env_library)

    # Final values with command-line arguments taking precedence over environment variables
    logging.debug("Collecting final values for required settings")
    host = args.host or env_host
    port = args.port or env_port or None
    api_key = args.api_key or env_api_key
    library = args.library or env_library
    doit = args.doit or env_doit

    # Validate required arguments
    validate_required_arguments(host, api_key, library)

    # Validate and handle host and port information
    validated_host, validated_port = handle_host_and_port(host, port)

    logging.debug(
        f"Using the following configurations: "
        f"Host: {validated_host}, Port: {validated_port}, API Key: {api_key}, "
        f"Library: {library}, DoIt: {doit}"
    )

    try:
        client = create_http_client(api_key)

        # Determine the base URL, using HTTPS if the validated port is 443
        base_url = f"{validated_host}:{validated_port}"

        # Check connection to Emby server
        connection_url = f"{base_url}/System/Info"
        if not check_emby_connection(client, connection_url):
            logging.error(f"Unable to connect to the Emby server at {base_url}.")
            sys.exit(1)

        library_id = get_library_id(base_url, api_key, library)
        if library_id is None:
            logging.error(f"Unable to find library '{library}'.")
            sys.exit(1)

        # Fetch media items and process them to identify duplicates
        provider_tables = fetch_and_process_media_items(base_url, api_key, library_id)

        # Dump provider tables to files
        dump_object_to_file(provider_tables, "testing/provider_tables")

        # Identify duplicates
        duplicates = identify_duplicates(provider_tables)

        dump_object_to_file(duplicates, "testing/duplicates")

        # Aggregate duplicates
        duplicates = merge_duplicate_groups(duplicates)

        # Dump duplicates to files
        dump_object_to_file(duplicates, "testing/aggregate")

    except EmbyServerConnectionError as e:
        logging.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
