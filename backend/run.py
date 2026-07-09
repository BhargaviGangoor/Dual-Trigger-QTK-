import uvicorn

if __name__ == "__main__":
    # Start the FastAPI application on host 0.0.0.0 and port 8000
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
