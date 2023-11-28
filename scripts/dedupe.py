#!/usr/bin/env python

import os
import sys
import logging
import argparse
import httpx
from httpx import URL
from tqdm import tqdm
from typing import Optional, Tuple
import json
import os
from typing import Any
import backoff

# At the top of your script, after the imports, establish a logger for the tool
logger = logging.getLogger("EmbyDedupe")
logger.setLevel(logging.ERROR)  # Default log level for the tool's logger

MAX_RETRIES = 20  # The maximum number of retries for HTTP requests
MAX_BACKOFF_TIME = 600  # Maximum total backoff time in seconds
PAGE_SIZE = 1000  # The page size for paginated requests
EMOJI_CHECK = "✅"
EMOJI_CROSS = "❌"

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


class DisjointSet:
    def __init__(self):
        # Initially, each item is its own parent
        self.parent = {}

    def find(self, item):
        # Find the root parent of the item recursively
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, set1, set2):
        # Perform union of two sets represented by their root items
        root1 = self.find(set1)
        root2 = self.find(set2)
        if root1 != root2:
            # Attach one tree's root to the other
            self.parent[root1] = root2


# Include the additional imports required for exception handling
from httpx import HTTPStatusError, ReadTimeout, RequestError


def should_give_up(e):
    # Determine whether the given exception should stop retries
    is_client_error = (
        isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500
    )
    return is_client_error


def handle_giveup(details):
    """
    A callback function that will be called when the retry loop has been
    terminated and is giving up.
    """
    logger.error(f"Giving up on request after retries: {details['tries']}")


@backoff.on_exception(
    backoff.expo,
    (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.RequestError),
    max_tries=MAX_RETRIES,
    max_time=MAX_BACKOFF_TIME,
    giveup=should_give_up,
    on_giveup=handle_giveup,
)
def make_http_request(
    client: httpx.Client, method: str, url: str, **kwargs
) -> httpx.Response:
    """
    Make an HTTP request using the given httpx.Client, equipped with exponential backoff
    and retry capabilities in case of certain exceptions.

    The exponential backoff policy will initiate a number of retries with increasing
    delay intervals if a `ReadTimeout` or other specified errors occur during the request.
    """
    try:
        response = client.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    except (HTTPStatusError, ReadTimeout, RequestError) as exc:
        logger.error(f"Request failed: {exc}. Retrying...")
        raise


def create_http_client(api_key: str) -> httpx.Client:
    """
    Create an httpx.Client instance with API key headers.

    Args:
        api_key (str): The API key for the Emby server.

    Returns:
        httpx.Client: A client instance configured for communication with Emby server.
    """
    headers = {"X-Emby-Token": api_key}
    client = httpx.Client(headers=headers)
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


def build_disjoint_set(media_items_by_provider):
    ds = DisjointSet()
    # Progress bar for initializing items in the disjoint set
    items_progress = tqdm(
        total=sum(
            len(items)
            for provider_dict in media_items_by_provider.values()
            for items in provider_dict.values()
        ),
        desc="Building sets",
        unit="item",
    )
    for provider in media_items_by_provider:
        for _, items in media_items_by_provider[provider].items():
            for item in items:
                if item not in ds.parent:
                    ds.parent[item] = item  # Initialize the item's parent to itself
                ds.union(
                    items[0], item
                )  # Union the first item with the rest in the list
                items_progress.update(
                    1
                )  # Update progress bar each time an item is processed
    items_progress.close()  # Close progress bar after all items have been processed
    return ds


def set_logging_level(verbosity_count: int, env_verbosity: Optional[str]) -> None:
    """Set logging level based on verbosity count and environment variable.

    Args:
        verbosity_count (int): Count of verbose flags (-v) in the command line.
        env_verbosity (Optional[str]): Verbosity level from the environment variable.
    """
    # Determine the logging level
    levels = ["ERROR", "WARNING", "INFO", "DEBUG"]
    level_name = env_verbosity or "ERROR"
    if verbosity_count:
        level_name = levels[min(verbosity_count, len(levels) - 1)]
    level = LOGGING_LEVELS.get(level_name, logging.ERROR)

    # Set the level for the tool's logger instead of the root logger
    logger.setLevel(level)

    # Configure a console handler for the tool's logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # To avoid duplicate logging if function is called multiple times, clear any previously added handlers
    logger.handlers.clear()
    logger.addHandler(console_handler)

    logger.info(f"Logging level set to {logging.getLevelName(level)}")


def override_warning(arg_name: str, cmd_val: str, env_val: str) -> None:
    """Print a warning if a command-line argument overrides an environment variable.

    Args:
        arg_name (str): The name of the argument being overridden.
        cmd_val (str): The value from the command line.
        env_val (str): The value from the environment variable.
    """
    if cmd_val and env_val:
        logger.warning(
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
        logger.warning(
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
    logger.debug(f"Checking connection to Emby server at {url}")
    try:
        response = make_http_request(client, "GET", url)
        # If the request was successful, we don't need to check response status
        # since 'make_http_request' already calls 'raise_for_status()'.
        logger.info("Successfully connected to the Emby server.")
        return True
    except httpx.HTTPStatusError as e:
        raise EmbyServerConnectionError(
            f"Failed to connect to Emby server: {e.response.content}"
        )
    except httpx.RequestError as e:
        raise EmbyServerConnectionError(
            f"An error occurred while communicating with Emby server: {str(e)}"
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
    client: httpx.Client, base_url: str, library_id: str
) -> Tuple[dict, dict, dict]:
    """
    Fetches media items in a paginated manner and builds the provider ID tables.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): Base URL of the Emby server.
        library_id (str): The ID of the library/virtual folder to fetch media items from.

    Returns:
        Tuple[dict, dict, dict]: Three dictionaries (imdb, tvdb, tmdb) containing provider ID mappings.
    """
    page_size = PAGE_SIZE  # Efficient page size to minimize memory consumption
    start_index = 0  # Starting index for pagination

    provider_tables = {"imdb": {}, "tvdb": {}, "tmdb": {}}

    # Fetch the total number of items
    url = f"{base_url}/Items?StartIndex=0&Limit=0&Recursive=True&ParentId={library_id}&Fields=ProviderIds&Is3D=False"
    try:
        response = make_http_request(client, "GET", url)
        total_items = response.json().get("TotalRecordCount", 0)
        logger.info(f"Total media items to fetch: {total_items}")
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        logger.error("Failed to fetch the total number of media items.")
        raise e

    # Initialize progress bar
    progress_bar = tqdm(total=total_items, desc="Fetching media items", unit="item")

    try:
        # Continue fetching until all pages are processed
        while start_index < total_items:
            url = f"{base_url}/Items?StartIndex={start_index}&Limit={page_size}&Recursive=True&ParentId={library_id}&Fields=ProviderIds&Is3D=False"
            try:
                response = make_http_request(client, "GET", url)
                media_items = response.json().get("Items", [])
                build_provider_id_tables(media_items, provider_tables)
                processed_items = len(media_items)  # Get the number of items processed
                start_index += processed_items
                progress_bar.update(processed_items)  # Update progress bar
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                logger.error(
                    f"Error fetching page of media items starting at index {start_index}"
                )
                raise e  # Raise the exception to indicate failure in the higher-level process
    finally:
        progress_bar.close()  # Ensure the progress bar is closed after processing is complete

    # Return provider ID tables
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
    client: httpx.Client, base_url: str, duplicate_items: list, doit: bool
) -> None:
    """
    Deletes duplicate media items if the 'doit' flag is true.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): Base URL of the Emby server.
        duplicate_items (list): A list of duplicate media items to delete.
        doit (bool): If True, actually perform the deletion.
    """
    if doit:
        for item in duplicate_items:
            item_id = item["id"]  # Assuming the item comes with the ID labeled as 'id'.
            url = f"{base_url}/Items/{item_id}"
            try:
                response = make_http_request(client, "DELETE", url)
                logger.info(f"Deleted item with ID {item_id}")
            except (httpx.HTTPStatusError, httpx.RequestError):
                logger.error(f"Failed to delete item with ID {item_id}")
    else:
        logger.info(
            "Deletion skipped. Items to be deleted are only listed in the report."
        )


def get_library_id(
    client: httpx.Client, base_url: str, library_name: str
) -> Optional[str]:
    """
    Retrieves the ID of the specified library by name using a provided HTTP session.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): Base URL of the Emby server.
        library_name (str): The name of the library to retrieve the ID for.

    Returns:
        Optional[str]: The ID of the library if found, else None.
    """
    url = f"{base_url}/Library/VirtualFolders"
    try:
        response = make_http_request(client, "GET", url)
        virtual_folders = response.json()

        for folder in virtual_folders:
            if folder.get("Name") == library_name:
                return folder.get("Id")

        logger.error(f"Library '{library_name}' not found.")
        return None  # Return None if library is not found

    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error occurred while retrieving library ID: {e.response.content}"
        )
    except httpx.RequestError as e:
        logger.error(f"HTTP request to Emby server failed: {str(e)}")

    return None  # Return None if any exception occurred


def dump_object_to_file(obj: Any, base_filename: str) -> None:
    """
    Attempts to serialize and save an object to a file. It saves objects as:
        - Pretty JSON if the object is a dictionary or list.
        - Plain text if it's a string.
        - Binary if the object is bytes.

    Args:
        obj (Any): The object to be saved to a file.
        base_filename (str): The base name for the file to save to (without an extension).

    Raises:
        ValueError: If the object type cannot be determined or handled by the function.
    """
    # Paths for different file types
    json_path = f"{base_filename}.json"
    text_path = f"{base_filename}.txt"
    bin_path = f"{base_filename}.bin"

    # Check if the object is serializable to JSON (dict or list)
    if isinstance(obj, (dict, list)):
        try:
            with open(json_path, "w", encoding="utf-8") as json_file:
                json.dump(obj, json_file, indent=4)
            logger.debug(f"Object saved as JSON to {json_path}")
            return
        except TypeError as e:
            logger.error(f"Failed to serialize object to JSON: {e}")
            # Fall through to other types if JSON serialization fails

    # Check if it's text data (string)
    if isinstance(obj, str):
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(obj)
        logger.debug(f"Text object saved to {text_path}")
        return

    # Check if it's binary data (bytes)
    if isinstance(obj, bytes):
        with open(bin_path, "wb") as bin_file:
            bin_file.write(obj)
        logger.debug(f"Binary object saved to {bin_path}")
        return

    # If none of the above, raise an error
    raise ValueError("Unsupported object type for dumping to a file.")


def read_json_file(file_path: str) -> Optional[Any]:
    """
    Attempts to read a JSON file and return its contents as a Python object.

    Args:
        file_path (str): Path to the JSON file to be read.

    Returns:
        Optional[Any]: Parsed JSON data if the file is successfully read and parsed, None otherwise.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except json.JSONDecodeError as exc:
        logger.error(f"Error parsing JSON file at {file_path}: {exc}")
    except Exception as exc:
        logger.error(
            f"An unexpected error occurred while reading file {file_path}: {exc}"
        )

    return None


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


def rationalize_duplicates(media_items_by_provider):
    ds = build_disjoint_set(media_items_by_provider)

    # Progress bar for grouping items
    grouping_progress = tqdm(
        total=len(ds.parent), desc="Grouping duplicates", unit="item"
    )

    groups = {}
    for item in ds.parent:
        root = ds.find(item)
        if root not in groups:
            groups[root] = {item}
        else:
            groups[root].add(item)
        grouping_progress.update(1)  # Update progress bar for each item grouped
    grouping_progress.close()  # Close progress bar after all items have been processed

    # Convert the sets to lists (because JSON can't represent sets)
    rationalized_list = [list(group) for group in groups.values() if len(group) > 1]
    return rationalized_list


def fetch_items_details(client: httpx.Client, base_url: str, item_ids: list) -> list:
    """
    Fetches the details for a list of media items by their IDs in one API request.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): Base URL of the Emby server.
        item_ids (list): List of item IDs to fetch details for.

    Returns:
        list: A list of media items with detailed information.
    """
    # Comma-separated item IDs for the query parameter
    ids_param = ",".join(item_ids)
    url = f"{base_url}/Items?Fields=MediaStreams,Path,ProviderIds&Ids={ids_param}"

    try:
        response = make_http_request(client, "GET", url)
        items = response.json().get("Items", [])
        return items
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        logger.error(f"Failed to fetch details for items: {e}")
        return []


def determine_items_to_delete(duplicate_ids: list, all_items_details: list) -> dict:
    """
    Determines the best quality media item to keep and marks the rest for deletion.
    The criteria for the best quality is based on resolution, audio channels, bitrate, and file size.

    Args:
        duplicate_ids (list): A list of IDs of potentially duplicate media items.
        all_items_details (list): Detailed media items information, including MediaStreams.

    Returns:
        dict: A dictionary containing details of the item to keep and the ones to delete.
    """
    # Process and rate each item based on quality factors
    rated_items = rate_media_items(all_items_details)

    # Sort items by their quality rating (higher is better)
    rated_items.sort(key=lambda x: x["rating"], reverse=True)

    # The first item in the sorted list is the best quality; the rest are duplicates
    item_to_keep = rated_items[0]
    items_to_delete = rated_items[1:]

    return {"keep": item_to_keep, "delete": items_to_delete}


def rate_media_items(items):
    """
    Assigns a quality rating to each media item based on its attributes.

    Args:
        items (list): List of media items.

    Returns:
        list: Rated media items, each with a 'rating' key indicating its quality score.
    """
    rated_items = []
    for item in items:
        video_stream = next(
            (s for s in item["MediaStreams"] if s["Type"] == "Video"), None
        )
        audio_stream = next(
            (s for s in item["MediaStreams"] if s["Type"] == "Audio"), None
        )

        # Define quality factors and their corresponding weights
        quality_factors = {
            "resolution": (
                video_stream.get("Height", 0) * video_stream.get("Width", 0)
                if video_stream
                else 0,
                1,
            ),
            "audio_channels": (
                audio_stream.get("Channels", 0) if audio_stream else 0,
                0.5,
            ),
            "bitrate": (item.get("Bitrate", 0), 0.2),
            "file_size": (item.get("Size", 0), 0.3),
        }

        # Calculate the weighted quality rating
        quality_rating = sum(
            value * weight for value, weight in quality_factors.values()
        )

        # Include the quality rating and relevant details in the result
        rated_items.append(
            {
                "id": item["Id"],
                "name": item["Name"],
                "path": item["Path"],
                "serverid": item["ServerId"],
                "rating": quality_rating,
                "quality_description": get_quality_description(
                    item
                ),  # function to extract quality description from item
            }
        )

    return rated_items


def process_duplicate_groups(
    client: httpx.Client, base_url: str, duplicate_groups: list
) -> list:
    """
    Processes each group of duplicate items to identify the item to keep and the ones to delete.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): Base URL of the Emby server.
        duplicate_groups (list): A list of lists, where each inner list contains IDs of potentially duplicate items.

    Returns:
        list: A list of dictionaries containing items to keep and delete for each group.
    """
    decisions = []
    # Progress bar for the overall process
    progress_bar = tqdm(
        total=len(duplicate_groups), desc="Processing duplicate groups", unit="group"
    )

    for group in duplicate_groups:
        # Fetch details for all items in the group
        items_details = fetch_items_details(client, base_url, group)
        if items_details:
            # Determine which items to delete within this group
            decision = determine_items_to_delete(group, items_details)
            decisions.append(decision)
        progress_bar.update(1)  # Update progress bar for each processed group

    progress_bar.close()  # Close progress bar after all groups have been processed
    return decisions


def delete_item(client: httpx.Client, base_url: str, item_id: str, doit: bool) -> dict:
    """
    Deletes a media item by its ID if the doit flag is True.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): The base URL of the Emby server.
        item_id (str): The ID of the media item to be deleted.
        doit (bool): If True, the media item will be deleted; otherwise, no action is taken.

    Returns:
        dict: The deletion status and any error message if the deletion failed.
    """
    deleted_status = {"id": item_id, "status": "skipped", "error": None}
    if doit:
        url = f"{base_url}/Items/{item_id}"
        try:
            response = client.delete(url)
            deleted_status["status"] = (
                "success" if response.status_code == 204 else "failed"
            )
        except Exception as e:
            deleted_status.update({"status": "failed", "error": str(e)})
    return deleted_status


def get_emoji_for_status(status):
    return EMOJI_CHECK if status == "success" else EMOJI_CROSS


def format_individual_item(item, base_url, decision):
    """
    Formats an individual item to be marked for deletion with an emoji and as a markdown link.

    Args:
        item (dict): The item information.
        base_url (str): The base URL of the Emby server.
        decision (dict): The decision information contains the item to keep.

    Returns:
        str: Formatted markdown string with emoji and link for the item.
    """
    name_match_emoji = (
        EMOJI_CHECK if item["name"] == decision["keep"]["name"] else EMOJI_CROSS
    )
    item_link = f"[{item['id']}]({base_url}/web/index.html#!/item?id={item['id']}&serverId={decision['keep']['serverid']})"
    deletion_status = next(
        (d for d in decision["deleted"] if d["id"] == item["id"]), {}
    )
    status_emoji = get_emoji_for_status(deletion_status.get("status", "skipped"))
    error_message = deletion_status.get("error", "")
    return (
        f"{name_match_emoji} {item_link} {item['name']}{status_emoji} {error_message}"
    )


def format_markdown_table(base_url: str, decisions: list) -> str:
    """
    Formats the decisions into a markdown table.

    Args:
        base_url (str): The base URL of the Emby server.
        decisions (list): List of decision objects containing items to keep and delete.

    Returns:
        str: A formatted markdown table as a string.
    """

    def get_deletion_status_emoji(status):
        return EMOJI_CHECK if status == "success" else EMOJI_CROSS

    # Headers
    headers = ["ID", "Title", "Codec", "Size", "Items to Delete"]
    max_widths = {header: len(header) for header in headers}

    # Determine the maximum width for each column
    for decision in decisions:
        keep = decision["keep"]
        max_widths["ID"] = max(max_widths["ID"], len(keep["id"]))
        max_widths["Title"] = max(max_widths["Title"], len(keep["name"]))
        max_widths["Codec"] = max(
            max_widths["Codec"], len(keep["quality_description"]["video"]["codec"])
        )
        max_widths["Size"] = max(
            max_widths["Size"], len(str(keep["quality_description"]["size"]))
        )

        deletion_entries = [
            format_individual_item(item, base_url, decision)
            for item in decision["delete"]
        ]
        max_widths["Items to Delete"] = max(
            max_widths["Items to Delete"], max(len(entry) for entry in deletion_entries)
        )

    header_line = (
        "| "
        + " | ".join(f"{header:<{max_widths[header]}}" for header in headers)
        + " |\n"
    )
    separator_line = (
        "|-" + "-|-".join(f"{'':-<{max_widths[header]}}" for header in headers) + "-|\n"
    )

    # Rows
    rows = []
    for decision in decisions:
        keep = decision["keep"]
        id_link = f"[{keep['id']}]({base_url}/web/index.html#!/item?id={keep['id']}&serverId={decision['keep']['serverid']})"
        keep_name = keep["name"]
        keep_codec = keep["quality_description"]["video"]["codec"]
        keep_size = str(keep["quality_description"]["size"])
        delete_entries = "<br>".join(
            format_individual_item(item, base_url, decision)
            for item in decision["delete"]
        )
        row = (
            f"| {id_link:<{max_widths['ID']}} "
            f"| {keep_name:<{max_widths['Title']}} "
            f"| {keep_codec:<{max_widths['Codec']}} "
            f"| {keep_size:<{max_widths['Size']}} "
            f"| {delete_entries:<{max_widths['Items to Delete']}} |\n"
        )
        rows.append(row)

    markdown_table = header_line + separator_line + "".join(rows)
    return markdown_table


def process_deletion_and_generate_report(
    client: httpx.Client, base_url: str, decisions: list, doit: bool
) -> str:
    """
    Processes deletions based on the decisions and generates a markdown report.

    Args:
        client (httpx.Client): The httpx client configured for the Emby server communication.
        base_url (str): The base URL of the Emby server.
        decisions (list): A list of decision objects containing items to keep and delete.
        doit (bool): If True, the deletion process will be attempted; otherwise, it is only simulated.

    Returns:
        str: The generated markdown report.
    """
    # Process deletions and update decisions with deletion results
    for decision in decisions:
        delete_results = []
        for item in decision["delete"]:
            deletion_result = delete_item(client, base_url, item["id"], doit)
            delete_results.append(deletion_result)
        # Update the decision object with deletion results
        decision["deleted"] = delete_results

    # Generate report in markdown format
    report = format_markdown_table(base_url, decisions)
    return report


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
    logger.debug("Collecting final values for required settings")
    host = args.host or env_host
    port = args.port or env_port or None
    api_key = args.api_key or env_api_key
    library = args.library or env_library
    doit = args.doit or env_doit

    # Validate required arguments
    validate_required_arguments(host, api_key, library)

    # Validate and handle host and port information
    validated_host, validated_port = handle_host_and_port(host, port)

    logger.debug(
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
            logger.error(f"Unable to connect to the Emby server at {base_url}.")
            sys.exit(1)

        library_id = get_library_id(client, base_url, library)
        if library_id is None:
            logger.error(f"Unable to find library '{library}'.")
            sys.exit(1)

        # Fetch media items and process them to identify duplicates
        provider_tables = fetch_and_process_media_items(client, base_url, library_id)

        # Dump provider tables to files
        dump_object_to_file(provider_tables, "testing/provider_tables")

        # Identify duplicates
        duplicates = identify_duplicates(provider_tables)

        dump_object_to_file(duplicates, "testing/duplicates")

        # Aggregate duplicates
        duplicates = rationalize_duplicates(duplicates)

        # Dump duplicates to files
        dump_object_to_file(duplicates, "testing/aggregate")

        decisions = process_duplicate_groups(client, base_url, duplicates)

        # Dump decisions to files
        dump_object_to_file(decisions, "testing/decisions")

        markdown_report = process_deletion_and_generate_report(
            client, base_url, decisions, doit
        )

        dump_object_to_file(markdown_report, "testing/report")

    except EmbyServerConnectionError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
