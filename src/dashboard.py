"""Live training/inference dashboard. Parses log files and serves an
auto-refreshing web page. No external deps. Open http://localhost:8000

Log sources are read from E:/pazzle_work/dash_sources.json:
    {"compat": "<path>", "restore": "<path>", "infer": "<path>"}
"""
import os, re, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from config import WORK_ROOT, CKPT_DIR, CACHE_DIR

SOURCES_JSON = os.path.join(WORK_ROOT, "dash_sources.json")
PORT = int(os.environ.get("DASH_PORT", "8000"))

RE = {
    "compat_step": re.compile(r"step (\d+)/(\d+) loss ([\d.]+) H@1 ([\d.]+) V@1 ([\d.]+) lr \S+ ([\d.]+)s/it"),
    "compat_val":  re.compile(r"\[VAL real\] H@1 ([\d.]+) V@1 ([\d.]+)"),
    "restore_step":re.compile(r"step (\d+)/(\d+) loss ([\d.]+) lr \S+ ([\d.]+)s/it"),
    "restore_val": re.compile(r"\[VAL\] SSIM base\(distorted\) ([\d.]+) -> restored ([\d.]+)"),
    "pair_step":   re.compile(r"step (\d+)/(\d+) loss ([\d.]+) acc@\d+ ([\d.]+) lr \S+ ([\d.]+)s/it"),
    "pair_val":    re.compile(r"\[VAL real\] acc@\d+ ([\d.]+)"),
    "infer_prog":  re.compile(r"(\d+)/(\d+)\s+([\d.]+)s/img\s+eta ([\d.]+)min"),
    "infer_done":  re.compile(r"wrote .*\((\d+) imgs, ([\d.]+) MB"),
}


def _downsample(rows, k=240):
    if len(rows) <= k:
        return rows
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)] + [rows[-1]]


def parse(kind, path):
    out = {"kind": kind, "steps": [], "val": [], "last": "", "status": "missing",
           "pct": 0, "eta_min": None, "extra": {}}
    if not path or not os.path.exists(path):
        return out
    mtime = os.path.getmtime(path)
    out["status"] = "running" if (time.time() - mtime) < 40 else "idle"
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return out
    last_step = 0
    for ln in lines:
        if kind == "compat":
            m = RE["compat_step"].search(ln)
            if m:
                s, tot, loss, h, v, sit = m.groups()
                last_step = int(s)
                out["steps"].append(dict(s=int(s), total=int(tot), loss=float(loss),
                                         h=float(h), v=float(v), sit=float(sit)))
                continue
            m = RE["compat_val"].search(ln)
            if m:
                out["val"].append(dict(s=last_step, a=float(m.group(1)), b=float(m.group(2))))
        elif kind == "restore":
            m = RE["restore_step"].search(ln)
            if m:
                s, tot, loss, sit = m.groups()
                last_step = int(s)
                out["steps"].append(dict(s=int(s), total=int(tot), loss=float(loss), sit=float(sit)))
                continue
            m = RE["restore_val"].search(ln)
            if m:
                out["val"].append(dict(s=last_step, a=float(m.group(1)), b=float(m.group(2))))
        elif kind == "pair":
            m = RE["pair_step"].search(ln)
            if m:
                s, tot, loss, acc, sit = m.groups()
                last_step = int(s)
                out["steps"].append(dict(s=int(s), total=int(tot), loss=float(loss),
                                         h=float(acc), sit=float(sit)))
                continue
            m = RE["pair_val"].search(ln)
            if m:
                out["val"].append(dict(s=last_step, a=float(m.group(1))))
        elif kind == "infer":
            m = RE["infer_prog"].search(ln)
            if m:
                i, n, sit, eta = m.groups()
                out["extra"] = dict(i=int(i), n=int(n), sit=float(sit))
                out["pct"] = round(100 * int(i) / int(n), 1)
                out["eta_min"] = float(eta)
            m = RE["infer_done"].search(ln)
            if m:
                out["extra"] = dict(done=True, imgs=int(m.group(1)), mb=float(m.group(2)))
                out["status"] = "done"; out["pct"] = 100
    if "DONE" in "".join(lines[-3:]) or "done." in "".join(lines[-3:]):
        if out["status"] != "done":
            out["status"] = "done"
    if out["steps"]:
        s = out["steps"][-1]
        out["pct"] = round(100 * s["s"] / max(1, s["total"]), 1)
        out["eta_min"] = round(s["sit"] * (s["total"] - s["s"]) / 60, 1)
    for ln in reversed(lines):
        if ln.strip():
            out["last"] = ln.strip()[:200]; break
    out["steps"] = _downsample(out["steps"])
    return out


def stages():
    def has(f): return os.path.exists(os.path.join(CKPT_DIR, f))
    return {
        "data extracted (E:)": os.path.exists(os.path.join(WORK_ROOT, "..", "pazzle_data")) or True,
        "perm cache": os.path.exists(os.path.join(CACHE_DIR, "perms.npz")),
        "compat model": has("compat_best.pt") or has("compat_last.pt"),
        "restore model": has("restore_best.pt") or has("restore_last.pt"),
    }


def api():
    src = {}
    if os.path.exists(SOURCES_JSON):
        try:
            src = json.load(open(SOURCES_JSON))
        except Exception:
            src = {}
    data = {k: parse(k, src.get(k)) for k in ("compat", "pair", "restore", "infer")}
    return {"t": time.time(), "jobs": data, "stages": stages()}


HTML = r"""<!doctype html><html><head><meta charset=utf-8>
<title>pazzle · training monitor</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e;
  --acc:#58a6ff;--grn:#3fb950;--yel:#d29922;--red:#f85149;--pur:#bc8cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:20px}
h1{font-size:20px;margin:0 0 2px}.sub{color:var(--mut);font-size:12px;margin-bottom:16px}
.grid{display:grid;gap:14px;grid-template-columns:1fr 1fr}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
.card h2{font-size:14px;margin:0 0 10px;display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%}
.running{background:var(--grn);box-shadow:0 0 8px var(--grn)}
.idle{background:var(--yel)}.done{background:var(--acc)}.missing{background:var(--bd)}
.kv{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:10px}
.kv div{min-width:70px}.kv b{display:block;font-size:19px}.kv span{color:var(--mut);font-size:11px}
.bar{height:6px;background:var(--bd);border-radius:4px;overflow:hidden;margin:8px 0}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--acc),var(--pur))}
canvas{width:100%;height:150px;display:block;margin-top:6px}
.last{color:var(--mut);font:11px/1.4 ui-monospace,Consolas,monospace;
  margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stages{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.chip{background:var(--card);border:1px solid var(--bd);border-radius:20px;
  padding:5px 12px;font-size:12px;display:flex;align-items:center;gap:6px}
.leg{font-size:11px;color:var(--mut);display:flex;gap:12px;margin-top:4px}
.leg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px}
</style></head><body><div class=wrap>
<h1>🧩 pazzle · training monitor</h1>
<div class=sub id=clock>connecting…</div>
<div class=stages id=stages></div>
<div class=grid id=cards></div>
</div>
<script>
const $=id=>document.getElementById(id);
function chart(cv,series,ymin,ymax){
 const dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
 cv.width=w*dpr;cv.height=h*dpr;const x=cv.getContext('2d');x.scale(dpr,dpr);
 x.clearRect(0,0,w,h);const pad=28,pl=34;
 let xs=[];series.forEach(s=>s.pts.forEach(p=>xs.push(p[0])));
 if(!xs.length){x.fillStyle='#8b949e';x.fillText('waiting for data…',pl,h/2);return;}
 const xmin=Math.min(...xs),xmax=Math.max(...xs)||1;
 x.strokeStyle='#30363d';x.lineWidth=1;x.strokeRect(pl,6,w-pl-8,h-pad);
 x.fillStyle='#8b949e';x.font='10px sans-serif';
 for(let g=0;g<=4;g++){const yy=6+(h-pad)*g/4,val=ymax-(ymax-ymin)*g/4;
  x.fillText(val.toFixed(2),2,yy+3);x.strokeStyle='#21262d';
  x.beginPath();x.moveTo(pl,yy);x.lineTo(w-8,yy);x.stroke();}
 const X=v=>pl+(w-pl-8)*(v-xmin)/(xmax-xmin||1),Y=v=>6+(h-pad)*(1-(v-ymin)/(ymax-ymin||1));
 series.forEach(s=>{x.strokeStyle=s.c;x.lineWidth=1.8;x.beginPath();
  s.pts.forEach((p,i)=>{const xx=X(p[0]),yy=Y(p[1]);i?x.lineTo(xx,yy):x.moveTo(xx,yy);});x.stroke();
  if(s.dot)s.pts.forEach(p=>{x.fillStyle=s.c;x.beginPath();x.arc(X(p[0]),Y(p[1]),2.5,0,7);x.fill();});});
 x.fillStyle='#8b949e';x.fillText(xmin,pl,h-8);x.fillText(xmax,w-40,h-8);
}
function card(name,j){
 const st=j.status,badge=`<span class="dot ${st}"></span>`;
 let kv='',charts='';
 if(j.kind=='compat'){
  const s=j.steps.at(-1)||{},v=j.val.at(-1)||{};
  const bh=Math.max(0,...j.steps.map(p=>p.h),...j.val.map(p=>p.a));
  kv=`<div><b>${(s.s||0)}</b><span>/ ${s.total||'?'} step</span></div>
      <div><b>${((v.a??s.h)||0).toFixed(3)}</b><span>H@1 val</span></div>
      <div><b>${((v.b??s.v)||0).toFixed(3)}</b><span>V@1 val</span></div>
      <div><b>${(s.loss||0).toFixed(2)}</b><span>loss</span></div>
      <div><b>${j.eta_min??'–'}</b><span>eta min</span></div>`;
  charts=`<canvas id=c_${name}></canvas>
   <div class=leg><span><i style=background:#58a6ff></i>H@1</span>
   <span><i style=background:#bc8cff></i>V@1</span>
   <span><i style=background:#3fb950></i>val H@1</span></div>`;
 }else if(j.kind=='restore'){
  const s=j.steps.at(-1)||{},v=j.val.at(-1)||{};
  kv=`<div><b>${(s.s||0)}</b><span>/ ${s.total||'?'} step</span></div>
      <div><b>${(v.b||0).toFixed(4)}</b><span>SSIM restored</span></div>
      <div><b>${(v.a||0).toFixed(4)}</b><span>SSIM distorted</span></div>
      <div><b>${(s.loss||0).toFixed(4)}</b><span>loss</span></div>
      <div><b>${j.eta_min??'–'}</b><span>eta min</span></div>`;
  charts=`<canvas id=c_${name}></canvas>
   <div class=leg><span><i style=background:#3fb950></i>restored SSIM</span>
   <span><i style=background:#8b949e></i>distorted SSIM</span></div>`;
 }else if(j.kind=='pair'){
  const s=j.steps.at(-1)||{},v=j.val.at(-1)||{};
  kv=`<div><b>${(s.s||0)}</b><span>/ ${s.total||'?'} step</span></div>
      <div><b>${((v.a??s.h)||0).toFixed(3)}</b><span>acc@cand</span></div>
      <div><b>${(s.loss||0).toFixed(2)}</b><span>loss</span></div>
      <div><b>${j.eta_min??'–'}</b><span>eta min</span></div>`;
  charts=`<canvas id=c_${name}></canvas>
   <div class=leg><span><i style=background:#58a6ff></i>train acc</span>
   <span><i style=background:#3fb950></i>val acc</span></div>`;
 }else{
  const e=j.extra||{};
  kv=`<div><b>${e.i||e.imgs||0}</b><span>/ ${e.n||700} imgs</span></div>
      <div><b>${e.sit?e.sit.toFixed(2):'–'}</b><span>s/img</span></div>
      <div><b>${j.eta_min??(e.done?'0':'–')}</b><span>eta min</span></div>
      <div><b>${e.mb?e.mb.toFixed(0)+'MB':'–'}</b><span>size</span></div>`;
 }
 return `<div class=card><h2>${badge}${name} <span style="color:var(--mut);
   font-weight:400;font-size:11px">${st}</span></h2>
   <div class=kv>${kv}</div>
   <div class=bar><i style="width:${j.pct||0}%"></i></div>
   ${charts}<div class=last>${(j.last||'').replace(/</g,'&lt;')}</div></div>`;
}
async function tick(){
 try{const r=await fetch('/api',{cache:'no-store'});const d=await r.json();
  $('clock').textContent='updated '+new Date().toLocaleTimeString()+' · auto-refresh 3s';
  $('stages').innerHTML=Object.entries(d.stages).map(([k,v])=>
    `<div class=chip><span class="dot ${v?'done':'missing'}"></span>${k}</div>`).join('');
  const order=['compat','pair','restore','infer'];
  $('cards').innerHTML=order.map(n=>card(n,d.jobs[n])).join('');
  const j=d.jobs;
  if(j.compat.steps.length)chart($('c_compat'),[
    {c:'#58a6ff',pts:j.compat.steps.map(p=>[p.s,p.h])},
    {c:'#bc8cff',pts:j.compat.steps.map(p=>[p.s,p.v])},
    {c:'#3fb950',dot:1,pts:j.compat.val.map(p=>[p.s,p.a])}],0,
    Math.max(0.3,...j.compat.steps.map(p=>p.h),...j.compat.val.map(p=>p.a))*1.1);
  if(j.pair.steps.length)chart($('c_pair'),[
    {c:'#58a6ff',pts:j.pair.steps.map(p=>[p.s,p.h])},
    {c:'#3fb950',dot:1,pts:j.pair.val.map(p=>[p.s,p.a])}],0,
    Math.max(0.3,...j.pair.steps.map(p=>p.h),...j.pair.val.map(p=>p.a))*1.1);
  if(j.restore.steps.length)chart($('c_restore'),[
    {c:'#3fb950',dot:1,pts:j.restore.val.map(p=>[p.s,p.b])},
    {c:'#8b949e',dot:1,pts:j.restore.val.map(p=>[p.s,p.a])}],0.4,
    Math.max(0.55,...j.restore.val.map(p=>p.b))*1.05);
 }catch(e){$('clock').textContent='waiting for server…';}
}
tick();setInterval(tick,3000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/api"):
            body = json.dumps(api()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    print(f"dashboard: http://localhost:{PORT}   (sources: {SOURCES_JSON})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
