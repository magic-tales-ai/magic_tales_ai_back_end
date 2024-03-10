# Magi-Tales Core AI 

# Requirements  
- Running MySQL server  
- Python 3 (tested version: 3.12)
- Packages:

```
pip install uvicorn
pip install fastapi
pip install python-dotenv
pip install mysqlclient 
pip install pyjwt
pip install marshmallow
```

# Installation 
- Clone the repo
- Install required packages 

# Configuration 
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

# Start service  
- Run with: "python app.py"

# Notes
- **Run the service on localhost and por 8001, so the front end can connect to it**


  
