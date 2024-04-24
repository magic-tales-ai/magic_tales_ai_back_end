def convert_path_to_url(local_path):
    base_path = "/media/neoadmin/Treasure/NC-laptop/personal/00-MAGIC-TALES/magit_tales_pro/static/"
    http_base_url = "http://localhost:8000/media/"

    if local_path.startswith(base_path):
        # Replace the file system base path with the HTTP base URL
        return local_path.replace(base_path, http_base_url)
    else:
        raise ValueError("Path is not in the base directory")


# Example usage
local_path = "/media/neoadmin/Treasure/NC-laptop/personal/00-MAGIC-TALES/magit_tales_pro/static/stories/user_58/20240423-214742/story.pdf"
url = convert_path_to_url(local_path)
print(url)
