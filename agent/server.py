"""Local streaming HTTP bridge; --dry-run explicitly uses a scripted model."""
import argparse
import json
import mimetypes
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,unquote
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent.agent import NoModelConfigured,RejectedClaim,check_listing,configured_model,model_is_configured
from agent.buyer_agent import BuyerSession,scripted_model,safe_path,valid_id,WORKSPACES
from agent.tools import ROOT


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass  # Never log requests or provider exceptions.

    def origin(self):
        origin=self.headers.get('Origin')
        if not origin:return None
        if origin=='null':return origin  # Demo opened as a local file.
        parsed=urlparse(origin)
        return origin if parsed.scheme in ('http','https') and parsed.hostname in ('localhost','127.0.0.1','[::1]','::1') else False

    def respond(self,status,value):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_headers(status,'application/json; charset=utf-8',len(data))
        self.wfile.write(data)

    def send_headers(self,status,content_type,size=None):
        self.send_response(status)
        origin=self.origin()
        if origin:self.send_header('Access-Control-Allow-Origin',origin)
        self.send_header('Vary','Origin')
        self.send_header('Content-Type',content_type)
        self.send_header('Cache-Control','no-store')
        if size is not None:self.send_header('Content-Length',str(size))
        self.end_headers()

    def model(self):
        if self.server.dry_run:return scripted_model(),'scripted-buyer'
        return configured_model()

    def stream(self,session,operation,reserved=False):
        events=queue.Queue()
        session.sinks.append(events.put)
        def worker():
            try:operation()
            except Exception:pass
            finally:
                if reserved:
                    with self.server.session_lock:session.approval_reserved=False
                events.put(None)
        self.send_headers(200,'application/x-ndjson; charset=utf-8')
        threading.Thread(target=worker,daemon=True).start()
        try:
            while True:
                item=events.get()
                if item is None:break
                self.wfile.write((json.dumps(item,ensure_ascii=False)+'\n').encode());self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):pass
        finally:session.sinks.remove(events.put)

    def do_OPTIONS(self):
        if self.origin() is False:return self.respond(403,{'error':'local demo origins only'})
        self.send_response(204)
        if self.origin():self.send_header('Access-Control-Allow-Origin',self.origin())
        self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers','Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.origin() is False:return self.respond(403,{'error':'local demo origins only'})
        path=unquote(urlparse(self.path).path)
        if path=='/status':
            dry_run=bool(self.server.dry_run)
            configured=dry_run or model_is_configured()
            return self.respond(200,dict(configured=configured,dry_run=dry_run,
                                         real_model_run=configured and not dry_run))
        try:
            self.model()
            if path.startswith('/files/'):
                _,_,thread,relative=path.split('/',3)
                file=safe_path(WORKSPACES/valid_id(thread),relative)
            elif path.startswith('/annotations/'):
                file=ROOT/'data/claims/annotations'/(valid_id(path[len('/annotations/'):])+'.json')
            else:return self.respond(404,{'error':'unknown endpoint'})
            if not file.is_file():return self.respond(404,{'error':'file not found'})
            if file.stat().st_size>8*1024**2:return self.respond(413,{'error':'file too large'})
            data=file.read_bytes()
            self.send_headers(200,mimetypes.guess_type(file.name)[0] or 'application/octet-stream',len(data))
            self.wfile.write(data)
        except NoModelConfigured as error:self.respond(503,{'error':str(error),'configured':False})
        except (ValueError,KeyError):self.respond(400,{'error':'invalid path'})
        except Exception:self.respond(502,{'error':'request failed; details withheld'})

    def do_POST(self):
        if self.origin() is False:return self.respond(403,{'error':'local demo origins only'})
        try:
            model,name=self.model()
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=65536:raise ValueError('invalid request size')
            body=json.loads(self.rfile.read(size))
            if not isinstance(body,dict):raise ValueError('Expected object')
            path=urlparse(self.path).path
            if path=='/ask':
                question=body.get('question')
                if not isinstance(question,str) or not question.strip():raise ValueError('Missing question')
                session=BuyerSession(question,model,name,dry_run=self.server.dry_run,profile_id=body.get('profile_id','default'))
                with self.server.session_lock:self.server.sessions[session.thread]=session
                return self.stream(session,session.run)
            if path=='/approve':
                with self.server.session_lock:
                    session=self.server.sessions.get(valid_id(body.get('thread')))
                    if session is None:raise ValueError('Unknown thread')
                    if session.execution_lock.locked() or getattr(session,'approval_reserved',False):return self.respond(409,{'error':'run still active'})
                    if not session.approval or body.get('approval_id')!=session.approval['id']:raise ValueError('No matching approval')
                    decision=body.get('decision')
                    if decision not in ('approve','edit','reject'):raise ValueError('Invalid decision')
                    edited=body.get('preferences') if session.approval['actions'][0]['name']=='save_preferences' else body.get('text')
                    if decision=='edit':
                        if session.approval['actions'][0]['name']=='publish_annotations':raise ValueError('Measured annotations accept approve or reject')
                        if isinstance(edited,dict):
                            from agent.buyer_tools import validate_preferences
                            validate_preferences(edited['needs'],edited['feedback'])
                        elif not isinstance(edited,str) or not edited.strip() or len(edited)>32000:raise ValueError('Edited content required')
                    session.approval_reserved=True
                return self.stream(session,lambda:session.approve(body['approval_id'],decision,edited),reserved=True)
            if path=='/check':
                sid=valid_id(body.get('sid'))
                if self.server.dry_run:
                    from agent.buyer_tools import inspect_claims
                    return self.respond(200,dict(real_model_run=False,**inspect_claims(sid)))
                return self.respond(200,check_listing(sid,model,name))
            if path=='/feedback':
                from agent.buyer_tools import validate_preferences
                needs=body.get('needs',{});feedback=body.get('feedback',[])
                validate_preferences(needs,feedback)
                request=dict(needs=needs,feedback=feedback)
                if self.server.dry_run:model=scripted_model('feedback')
                session=BuyerSession(json.dumps(request),model,name,dry_run=self.server.dry_run,
                                     profile_id=body.get('profile_id','default'),feedback_request=request)
                with self.server.session_lock:self.server.sessions[session.thread]=session
                return self.stream(session,session.run)
            return self.respond(404,{'error':'unknown endpoint'})
        except NoModelConfigured as error:self.respond(503,{'error':str(error),'configured':False})
        except (ValueError,RejectedClaim,KeyError):self.respond(400,{'error':'invalid request or ungrounded model output'})
        except Exception:self.respond(502,{'error':'model request failed; no provider details were logged'})


def create_server(port=8770,dry_run=False):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.sessions={};server.session_lock=threading.Lock();server.dry_run=dry_run
    return server


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8770)
    parser.add_argument('--dry-run',action='store_true',help='Scripted model; no external API calls')
    args=parser.parse_args();server=create_server(args.port,args.dry_run)
    print(json.dumps({'listening':'127.0.0.1','port':server.server_port,'dry_run':args.dry_run}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
