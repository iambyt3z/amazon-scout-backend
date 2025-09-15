Start server in local environment

```bash
uvicorn app.main:app --port 8000 --reload
```

Run the formatters
```bash
black .
isort .
```
