#!/usr/bin/env python3
"""Local Qwen3 reranker with explicit, tested query/document formatting."""
import argparse
import json
import sys
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from best_memory.rerank.formatting import encode_pairs, INSTRUCTION, PROTOCOL

class App:
    def __init__(self,path,device,batch_size=16):
        self.tok=AutoTokenizer.from_pretrained(path,padding_side='left',local_files_only=True)
        self.m=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device).eval()
        self.d=device;self.batch_size=batch_size;self.lock=threading.Lock()
        self.yes=self.tok.convert_tokens_to_ids('yes');self.no=self.tok.convert_tokens_to_ids('no')
    def score(self,query,documents,instruction=INSTRUCTION):
        # Concurrent HTTP requests queue safely; one batch owns model/tokenizer state.
        with self.lock, torch.inference_mode():
            ids=encode_pairs(self.tok,query,documents,instruction)
            values=[None]*len(ids)
            ordered=sorted(enumerate(ids),key=lambda item:(len(item[1]),item[0]))
            for start in range(0,len(ordered),self.batch_size):
                chunk=ordered[start:start+self.batch_size]
                batch=self.tok.pad({'input_ids':[x[1] for x in chunk]},padding=True,return_tensors='pt').to(self.d)
                hidden=self.m.model(**batch).last_hidden_state[:,-1,:]
                logits=torch.nn.functional.linear(hidden,self.m.lm_head.weight[[self.no,self.yes]]).float()
                scores=logits.softmax(-1)[:,1].cpu().tolist()
                for (index,_),score in zip(chunk,scores):values[index]=score
            return values

class Handler(BaseHTTPRequestHandler):
    app=None
    def reply(self,status,payload):
        raw=json.dumps(payload).encode();self.send_response(status)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)))
        self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        self.reply(200,{'status':'ready','protocol':PROTOCOL})
    def do_POST(self):
        try:
            body=json.loads(self.rfile.read(int(self.headers.get('content-length','0'))))
            query=body.get('query') or body.get('text_1','')
            docs=body.get('documents') or body.get('text_2',[])
            if isinstance(docs,str):docs=[docs]
            scores=self.app.score(query,docs,body.get('instruction') or INSTRUCTION)
            self.reply(200,{'data':[{'index':i,'score':v} for i,v in enumerate(scores)],'protocol':PROTOCOL})
        except (ValueError,TypeError,KeyError) as exc:self.reply(400,{'error':str(exc)})
        except Exception as exc:self.reply(500,{'error':type(exc).__name__})
    def log_message(self,*args):pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default=None)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8002)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args();Handler.app=App(args.model,args.device,args.batch_size)
    ThreadingHTTPServer.request_queue_size=64
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(json.dumps({'status':'ready','port':args.port,'protocol':PROTOCOL}),flush=True)
    server.serve_forever()
if __name__=='__main__':main()
