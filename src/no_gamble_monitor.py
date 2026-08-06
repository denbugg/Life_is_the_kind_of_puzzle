"""Tiny local web monitor for the no-gamble runner."""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import WORK_ROOT

STATUS_PATH = os.path.join(WORK_ROOT, "no_gamble_status.json")


def read_status():
    if not os.path.exists(STATUS_PATH):
        return {"state": "waiting", "stage": "none", "updated": time.time(),
                "message": f"waiting for {STATUS_PATH}"}
    try:
        with open(STATUS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"state": "error", "stage": "status_read", "updated": time.time(),
                "message": str(e)}


HTML = r"""<!doctype html><meta charset=utf-8>
<title>PAZZLE no-gamble monitor</title>
<style>
body{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.45 Segoe UI,Arial,sans-serif}
.wrap{max-width:980px;margin:0 auto;padding:22px}.top{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}
h1{font-size:20px;margin:0 0 4px}.mut{color:#8b949e}.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-top:14px}
.bar{height:10px;background:#30363d;border-radius:10px;overflow:hidden}.bar i{display:block;height:100%;background:#58a6ff}
.kv{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:12px;margin-top:14px}.kv b{display:block;font-size:22px}.kv span{color:#8b949e;font-size:12px}
pre{background:#010409;border:1px solid #30363d;border-radius:8px;padding:12px;overflow:auto;max-height:420px;white-space:pre-wrap}
.ok{color:#3fb950}.err{color:#f85149}.run{color:#d29922}
</style>
<div class=wrap>
  <div class=top><div><h1>PAZZLE no-gamble monitor</h1><div class=mut id=stamp>loading</div></div>
  <div class=mut>auto-refresh 2s</div></div>
  <div class=card>
    <h2 id=stage>stage</h2>
    <div class=bar><i id=bar style="width:0%"></i></div>
    <div class=kv>
      <div><b id=pct>0%</b><span>progress</span></div>
      <div><b id=done>0/0</b><span>items/steps</span></div>
      <div><b id=eta>-</b><span>ETA current stage</span></div>
      <div><b id=state>-</b><span>state</span></div>
    </div>
  </div>
  <div class=card><h2>Metrics</h2><pre id=metrics>{}</pre></div>
  <div class=card><h2>Command</h2><pre id=cmd></pre></div>
  <div class=card><h2>Last log lines</h2><pre id=tail></pre></div>
</div>
<script>
function fmt(sec){ if(sec==null) return '-'; sec=Math.max(0,sec); let m=Math.floor(sec/60),s=Math.floor(sec%60),h=Math.floor(m/60); m%=60; return h?`${h}h ${m}m`:`${m}m ${s}s`; }
async function tick(){
 let r=await fetch('/api',{cache:'no-store'}), d=await r.json();
 let p=d.progress||{}, pct=p.pct||0, age=(Date.now()/1000-(d.updated||0));
 document.getElementById('stamp').textContent=`updated ${new Date((d.updated||0)*1000).toLocaleTimeString()} · age ${age.toFixed(0)}s · log ${d.log_path||''}`;
 document.getElementById('stage').textContent=`${d.stage_index||0}/${d.stage_total||0} ${d.stage||'waiting'}`;
 document.getElementById('bar').style.width=`${pct}%`;
 document.getElementById('pct').textContent=`${pct.toFixed(1)}%`;
 document.getElementById('done').textContent=`${p.done||0}/${p.total||0}`;
 document.getElementById('eta').textContent=fmt(d.eta_sec);
 let st=d.state||'waiting'; document.getElementById('state').textContent=st;
 document.getElementById('state').className=st=='error'?'err':(st=='running'?'run':'ok');
 document.getElementById('metrics').textContent=JSON.stringify(d.metrics||{},null,2);
 document.getElementById('cmd').textContent=(d.command||[]).join(' ');
 document.getElementById('tail').textContent=(d.last_lines||[d.message||'']).join('\n');
}
tick(); setInterval(tick,2000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(read_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    print(f"monitor: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

