from urllib.parse import urlparse, parse_qs
 
def extract_api_version(url: str) -> str:
    """
    Extracts the value of 'api-version' from a given URL.
   
    Args:
        url (str): The URL containing the 'api-version' query parameter.
   
    Returns:
        str: The value of 'api-version', or None if not found.
    """
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get('api-version', [None])[0]