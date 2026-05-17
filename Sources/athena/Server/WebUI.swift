import AthenaCore
import AthenaDeploy
import Foundation
import HTTPTypes
import Hummingbird
import NIOCore

// Monitoring dashboard (M11.2). Inbound-only — Athena still initiates
// nothing (passive oracle). Server-rendered: one embedded HTML page +
// vanilla JS polling `/ui/api/state`. Zero JS deps, zero build step,
// ships inside the single binary.

extension AthenaServer {
    /// Everything the dashboard needs in one poll. Reuses the
    /// existing Codable DTOs so the shapes never drift.
    struct UIState: Codable {
        let governor: GovernorSnapshot
        let metrics: AthenaMetrics.Snapshot
        let vectors: VectorStatsResponse
        let store: StoreStatsResponse
        let queue: [QueueJobSummary]
        let model: String
    }

    func handleUIState() async -> Response {
        let gov = await governor.snapshot()
        let met = await metrics.snapshot()
        let vs = await vectorStore.stats()
        let st = StoreStatsResponse(
            vectors: await store.vectorCount(),
            jobs: await store.jobCount(),
            bytes: storeBytes(),
            path: store.dbPath.path)
        let jobs = await queue.list(status: nil).suffix(25).map {
            QueueJobSummary(
                id: $0.id, kind: $0.kind, status: $0.status,
                created: $0.created, updated: $0.updated)
        }
        return Self.json(
            UIState(
                governor: gov, metrics: met,
                vectors: VectorStatsResponse(
                    count: vs.count, dim: vs.dim, bytes: vs.bytes,
                    cap_bytes: vs.capBytes),
                store: st, queue: Array(jobs), model: modelName))
    }

    static func html(_ s: String) -> Response {
        var buffer = ByteBuffer()
        buffer.writeBytes(Data(s.utf8))
        var headers = HTTPFields()
        headers[.contentType] = "text/html; charset=utf-8"
        return Response(
            status: .ok, headers: headers,
            body: ResponseBody(byteBuffer: buffer))
    }

    // MARK: - Config view + edit (M11.3)

    /// Current effective config values for the editable keys, plus the
    /// resolved file path. Unparseable/missing ⇒ empty values (the
    /// page still renders so the operator can populate it).
    func handleUIConfigGet() async -> Response {
        let url = ConfigEditor.resolvePath(nil)
        var values: [String: String] = [:]
        if let cfg = try? AthenaConfig.parse(file: url) {
            for k in ConfigEditor.knownKeys {
                values[k] = ConfigEditor.value(k, in: cfg) ?? ""
            }
        } else {
            for k in ConfigEditor.knownKeys { values[k] = "" }
        }
        struct R: Codable {
            let path: String
            let keys: [String]
            let values: [String: String]
        }
        return Self.json(
            R(
                path: url.path,
                keys: ConfigEditor.knownKeys.sorted(),
                values: values))
    }

    /// Write submitted scalars to the TOML via the THROWING editor
    /// (never exits — a bad field is a 4xx, not a dead daemon).
    /// Most keys need a daemon restart to take effect; the UI says so.
    func handleUIConfigPost(_ request: Request) async -> Response {
        let body: [String: String]
        do {
            let buf = try await request.body.collect(
                upTo: 64 * 1024)
            body =
                (try? JSONDecoder().decode(
                    [String: String].self,
                    from: Data(buffer: buf))) ?? [:]
        } catch {
            return Self.json(
                ["error": "invalid body"], status: .badRequest)
        }
        let url = ConfigEditor.resolvePath(nil)
        var saved: [String] = []
        var errors: [String: String] = [:]
        for (k, v) in body
        where ConfigEditor.knownKeys.contains(k)
            && !v.trimmingCharacters(in: .whitespaces).isEmpty {
            do {
                try ConfigEditor.setScalarThrowing(
                    key: k, value: v, in: url)
                saved.append(k)
            } catch {
                errors[k] = "\(error)"
            }
        }
        struct R: Codable {
            let saved: [String]
            let errors: [String: String]
            let path: String
            let note: String
        }
        return Self.json(
            R(
                saved: saved.sorted(), errors: errors,
                path: url.path,
                note:
                    "saved — restart the daemon to apply "
                    + "(athena stop && athena start)"),
            status: errors.isEmpty ? .ok : .badRequest)
    }

    static let configPage = #"""
        <!doctype html><html><head><meta charset="utf-8">
        <title>athena · config</title>
        <style>
        body{background:#0d1117;color:#c9d1d9;font:13px ui-monospace,
        Menlo,monospace;margin:0;padding:24px}
        h1{font-size:16px;margin:0 0 4px}
        a{color:#2f81f7}.sub{color:#8b949e;margin-bottom:20px}
        .card{background:#161b22;border:1px solid #30363d;
        border-radius:8px;padding:16px;max-width:640px}
        label{display:block;color:#8b949e;margin:10px 0 3px}
        input{width:100%;box-sizing:border-box;background:#0d1117;
        color:#e6edf3;border:1px solid #30363d;border-radius:6px;
        padding:7px;font:13px ui-monospace,monospace}
        button{margin-top:16px;background:#238636;color:#fff;
        border:0;border-radius:6px;padding:9px 16px;cursor:pointer;
        font:13px ui-monospace,monospace}
        #msg{margin-top:14px}.ok{color:#3fb950}.err{color:#f85149}
        .k{color:#8b949e}
        </style></head><body>
        <h1>athena · configuration</h1>
        <div class="sub"><a href="/ui">← dashboard</a> ·
          <span id="path" class="k"></span></div>
        <div class="card"><form id="f"></form>
          <button onclick="save()">Save</button>
          <div id="msg"></div></div>
        <script>
        const $=i=>document.getElementById(i);
        let KEYS=[];
        async function load(){
          const c=await (await fetch("/ui/api/config")).json();
          KEYS=c.keys; $("path").textContent=c.path;
          $("f").innerHTML=c.keys.map(k=>
            `<label>${k}</label><input id="in_${k}" value="${
              (c.values[k]||"").replace(/"/g,'&quot;')}">`).join("");
        }
        async function save(){
          const b={};
          KEYS.forEach(k=>{const v=$("in_"+k).value.trim();
            if(v)b[k]=v;});
          const r=await fetch("/ui/api/config",{method:"POST",
            headers:{"content-type":"application/json"},
            body:JSON.stringify(b)});
          const j=await r.json();
          if(r.ok)$("msg").innerHTML=
            `<span class=ok>saved ${j.saved.join(", ")}</span>`+
            `<br><span class=k>${j.note}</span>`;
          else $("msg").innerHTML=`<span class=err>`+
            Object.entries(j.errors||{"":j.error}).map(
              ([k,v])=>`${k}: ${v}`).join("<br>")+`</span>`;
        }
        load();
        </script></body></html>
        """#

    static let uiPage = #"""
        <!doctype html><html><head><meta charset="utf-8">
        <title>athena</title>
        <style>
        body{background:#0d1117;color:#c9d1d9;font:13px ui-monospace,
        Menlo,monospace;margin:0;padding:24px}
        h1{font-size:16px;margin:0 0 4px}
        .sub{color:#8b949e;margin-bottom:20px}
        .grid{display:grid;grid-template-columns:repeat(auto-fit,
        minmax(320px,1fr));gap:16px}
        .card{background:#161b22;border:1px solid #30363d;
        border-radius:8px;padding:14px}
        .card h2{font-size:12px;text-transform:uppercase;
        letter-spacing:.05em;color:#8b949e;margin:0 0 10px}
        table{width:100%;border-collapse:collapse}
        td,th{text-align:left;padding:3px 6px;border-bottom:
        1px solid #21262d}th{color:#8b949e;font-weight:600}
        .bar{height:10px;background:#21262d;border-radius:5px;
        overflow:hidden;margin:6px 0}
        .bar>i{display:block;height:100%;background:#2f81f7}
        .k{color:#8b949e}.v{color:#e6edf3}
        .ok{color:#3fb950}.warn{color:#d29922}.err{color:#f85149}
        </style></head><body>
        <h1>athena</h1>
        <div class="sub"><span id="sub">connecting…</span>
          &nbsp;·&nbsp;<a href="/ui/config"
          style="color:#2f81f7">config</a></div>
        <div class="grid">
          <div class="card"><h2>Memory governor</h2>
            <div class="bar"><i id="membar"></i></div>
            <div id="mem"></div>
            <table id="mods"></table></div>
          <div class="card"><h2>Requests</h2>
            <div id="met"></div><table id="bykind"></table></div>
          <div class="card"><h2>Store</h2><div id="store"></div></div>
          <div class="card"><h2>Queue (recent)</h2>
            <table id="queue"></table></div>
        </div>
        <script>
        const $=i=>document.getElementById(i);
        const mb=n=>(n/1048576).toFixed(0)+" MB";
        const row=(a,b)=>`<tr><td class=k>${a}</td>
          <td class=v>${b}</td></tr>`;
        async function tick(){
          let s;
          try{s=await (await fetch("/ui/api/state")).json();}
          catch(e){$("sub").textContent="daemon unreachable";return;}
          $("sub").textContent=s.model+"  ·  "+
            new Date().toLocaleTimeString();
          const g=s.governor, used=g.reservedBytes,
            tot=g.totalBudgetBytes;
          $("membar").style.width=
            (tot?100*used/tot:0).toFixed(1)+"%";
          $("mem").innerHTML="<table>"+
            row("reserved",mb(used))+row("free",mb(g.freeBytes))+
            row("budget",mb(tot))+
            row("prompt-cache cap",mb(g.promptCacheCapBytes))+
            "</table>";
          $("mods").innerHTML="<tr><th>module</th><th>state</th>"+
            "<th>resident</th></tr>"+g.modules.map(m=>
            `<tr><td>${m.id}</td><td class=${
              m.state=="loaded"?"ok":"k"}>${m.state}</td>
             <td>${mb(m.reservedBytes)}</td></tr>`).join("");
          const m=s.metrics;
          $("met").innerHTML="<table>"+
            row("requests",m.totalRequests)+
            row("errors",`<span class=${m.totalErrors?"err":"ok"}>`+
              m.totalErrors+"</span>")+
            row("avg",m.avgMs.toFixed(1)+" ms")+
            row("p50",m.p50Ms.toFixed(1)+" ms")+
            row("p95",m.p95Ms.toFixed(1)+" ms")+
            row("llm tokens",m.llmTokens)+"</table>";
          $("bykind").innerHTML="<tr><th>kind</th><th>count</th></tr>"+
            Object.entries(m.byKind).map(([k,v])=>
            `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
          $("store").innerHTML="<table>"+
            row("vectors",s.store.vectors+" (dim "+s.vectors.dim+")")+
            row("jobs",s.store.jobs)+
            row("db size",mb(s.store.bytes))+
            row("vec cap",mb(s.vectors.cap_bytes))+
            row("path",s.store.path)+"</table>";
          $("queue").innerHTML="<tr><th>id</th><th>kind</th>"+
            "<th>status</th></tr>"+(s.queue.length?s.queue.slice()
            .reverse().map(j=>`<tr><td>${j.id.slice(0,8)}</td>
            <td>${j.kind}</td><td class=${
              j.status=="error"?"err":j.status=="done"?"ok":"warn"
            }>${j.status}</td></tr>`).join("")
            :"<tr><td class=k>no jobs</td></tr>");
        }
        tick();setInterval(tick,2000);
        </script></body></html>
        """#
}
