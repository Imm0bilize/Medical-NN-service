import uvicorn
from fastapi import FastAPI
from loguru import logger


app = FastAPI()
logger.add('log/debug.log', format='{time} {level} {message}',
           level='DEBUG', rotation='00:00', compression='zip')


@app.get('/')
def start_page():
    return {'info': 'Service working'}


@logger.catch
def main():
    logger.debug('Starting service')
    uvicorn.run("app:app", host="localhost", port=8080, log_level="debug")


if __name__ == '__main__':
    main()
