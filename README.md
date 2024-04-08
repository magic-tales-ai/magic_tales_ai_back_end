# Magic Tales - Personalized Story Generation

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#run-magic-tales)
- [Logging](#logging)
- [Development](#development)
- [License](#license)

## Overview

The Story Generation Application is a state-of-the-art system for generating compelling and engaging stories along with corresponding image prompts. It is built with robustness and modularity in mind, adhering to high production-level standards. The application is configurable, scalable, and can be extended for various storytelling and image generation requirements.

## Features

- Personality-based story generation
- Dynamic story feature gathering
- Synopsis and title generation
- Chapter-wise story creation
- Context-aware image prompt generation
- Image generation based on prompts
- Serialization of progress for resuming tasks
- Robust error handling and logging

## Technologies Used

- Python 3.x
- OmegaConf and Hydra
- Python-docx
- Various LLM and MLLM and Image Generation APIs

## Installation

- Clone this repository and navigate to the project directory. 
```bash
git clone https://github.com/munirjojoverge/magic_tales_ai_back_end.git
cd magic_tales_ai_back_end
```
- If you **DO NOT have** mamba or conda installed:
    - Got to:
    ```bash
    https://github.com/conda-forge/miniforge
    ```
    - Download the install script for your OS
    - Run the script

- Now, simply create the environment and install all the dependencies
```bash
mamba env create -n magic
mamba activate magic
pip install -r requirements.txt
```
## Configuration

All application configurations are managed via a YAML file. Modify the `config.yaml` file to suit your needs.

```yaml
output_artifacts:
  stories_folder_data_storage: "./stories"
  continue_where_we_left_of: true
image_prompt_gen:
  Other configurations...
```

- Edit .env file (the repo has a .env-example file, you can copy this and rename to .env)
- Set IP and PORT for the service

```
  SERVER_HOST="localhost"
  SERVER_PORT=8001
```

- Set data to connect to MySQL database: 

```
  DATABASE_HOST="127.0.0.1"
  DATABASE_PORT=3306
  DATABASE_USER="user"
  DATABASE_PASSWORD="password"
  DATABASE_NAME="database"
```

## Run Magic-Tales

- Go the the root folder
```bash
cd magic_tales_ai_back_end
```
- Run Magic-Tales
```python
python3 app.py
```
- Run the Front-End and the Std back-end
```python
python3 ...
```
- Interact with the front-end elements.

### Notes
- **Run the service on localhost and por 8001, so the front end can connect to it**


## Logging and Error Handling

The application is equipped with robust logging and error-handling mechanisms. Log files are generated for debugging and auditing purposes.

## Development and Contribution

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update the tests as appropriate.

## License

No License. This Software is Copyrighted: Magic-Tales LLC 2023(c) 