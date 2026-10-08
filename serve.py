"""
Web page and API for the trained models.

    python serve.py --pretrained ROBERTA_DIR --work-dir work [--models-dir DIR] [--port 8000]

Then open http://localhost:8000 for the page. The API:

    GET  /api/health                     what is loaded
    POST /api/extract                    {"text": "...", "sentiment": "negative"}
    POST /api/extract_batch              {"items": [{"text": "...", "sentiment": "..."}, ...]}

If the environment variable TSE_API_KEY is set, the /api/extract* calls need the
header X-API-Key with that value. Interactive API docs are at /docs.
"""
import argparse
import os
import sys
import threading
import time
from typing import List, Literal, Optional

import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from demo import Extractor, level2_seeds   # noqa: E402

Sentiment = Literal['positive', 'negative', 'neutral']


class Item(BaseModel):
    text: str = Field(..., min_length=1, max_length=280, description='the tweet')
    sentiment: Sentiment


class Batch(BaseModel):
    items: List[Item] = Field(..., min_length=1, max_length=64)


def create_app(extractor, api_key=None):
    app = FastAPI(title='Tweet sentiment phrase extraction',
                  description='Finds the phrase of a tweet that carries its sentiment.')
    app.add_middleware(CORSMiddleware, allow_origins=['*'],
                       allow_methods=['GET', 'POST'], allow_headers=['*'])
    with open(os.path.join(HERE, 'static', 'index.html'), encoding='utf-8') as fh:
        page = fh.read()
    lock = threading.Lock()   # one request at a time on the models

    def check_key(given):
        if api_key and given != api_key:
            raise HTTPException(status_code=401, detail='Missing or wrong X-API-Key header.')

    def run(item):
        text = item.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail='The text is empty.')
        t0 = time.perf_counter()
        try:
            with lock:
                level1, final = extractor.extract(text, item.sentiment)
        except ValueError:   # longer than the model's input size
            raise HTTPException(status_code=422,
                                detail='The text is too long for the model. Tweets up to about '
                                       '100 tokens (roughly 280 characters of normal text) work.')
        answer = final if final is not None else level1
        start = text.find(answer) if answer else -1
        return {'text': text,
                'sentiment': item.sentiment,
                'selected_text': answer,
                'start': start,
                'end': start + len(answer) if start >= 0 else -1,
                'level1_text': level1,
                'level2_used': final is not None,
                'ms': round((time.perf_counter() - t0) * 1000)}

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    def index():
        return page

    @app.get('/api/health')
    def health():
        return {**extractor.info, 'auth_required': bool(api_key)}

    @app.post('/api/extract')
    def extract(item: Item, x_api_key: Optional[str] = Header(None)):
        check_key(x_api_key)
        return run(item)

    @app.post('/api/extract_batch')
    def extract_batch(batch: Batch, x_api_key: Optional[str] = Header(None)):
        check_key(x_api_key)
        return {'results': [run(item) for item in batch.items]}

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pretrained', required=True,
                   help='RoBERTa folder (only config.json, vocab.json, merges.txt are read)')
    p.add_argument('--work-dir', default='work')
    p.add_argument('--models-dir', default=None, help='default WORK_DIR/models')
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='default: the seeds level two was trained on')
    p.add_argument('--folds', type=int, nargs='+', default=None,
                   help='load only these level-one folds (fewer = faster)')
    p.add_argument('--device', default='auto')
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    args = p.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = args.seeds if args.seeds is not None else level2_seeds(args.work_dir)
    extractor = Extractor(args.pretrained, args.work_dir, seeds, device,
                          models_dir=args.models_dir, folds=args.folds)
    app = create_app(extractor, api_key=os.environ.get('TSE_API_KEY') or None)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
