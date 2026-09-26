#!/usr/bin/env python3
"""Minimal OpenAI-compatible local BGE-M3 embedding service."""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoTokenizer, AutoModel

class App:
    def __init__(self, path, device):
        self.tok=AutoTokenizer.from_pretrained(path)
        self.model=AutoModel.from_pretrained(path, torch_dtype=torch.float16 if device.startswith('cuda') else None).to(device).eval()
        self.device=device
    def embed(self, texts):
        out=[]
        for i in range(0,len(texts),16):
            batch=self.tok(texts[i:i+16],padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
            with torch.inference_mode(): h=self.model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            v=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
            v=torch.nn.functional.normalize(v,dim=1).float().cpu().tolist(); out.extend(v)
        return out
class Handler(BaseHTTPRequestHandler):
    app=None
    def do_POST(self):
        n=int(self.headers.get('content-length','0')); body=json.loads(self.rfile.read(n))
        if self.path.endswith('/embeddings'):
            x=body.get('input',[]); x=[x] if isinstance(x,str) else x
            data=[{'index':i,'embedding':v} for i,v in enumerate(self.app.embed(x))]
            payload={'object':'list','data':data,'model':body.get('model','BAAI/bge-m3')}
        else: self.send_error(404); return
        raw=json.dumps(payload).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def log_message(self,*a): pass
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default=None); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8005); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    Handler.app=App(a.model,a.device); ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
if __name__=='__main__': main()
