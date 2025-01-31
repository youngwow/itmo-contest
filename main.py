import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response

from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger

from agents.itmo_agent import create_agent

# Initialize
app = FastAPI()
agent = create_agent()
logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")
    
        result = await agent.ainvoke({
            "query": body.query,
            "id": body.id
        })
        
        final_answer = result.get("final_answer", {})
        
        answer = final_answer.get("answer")
        if answer is not None:
            try:
                answer = int(answer)
            except (ValueError, TypeError):
                answer = None

        sources = final_answer.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=final_answer.get("reasoning", ""),
            sources=sources
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
