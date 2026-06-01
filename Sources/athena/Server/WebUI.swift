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
        /// M59 follow-up — live resident model id per module class
        /// (module rawValue → the model actually loaded right now, e.g.
        /// the LLM slot's `Qwen3.5-27B-4bit-mtp` vs `…-8bit-mtp`). Absent
        /// entries = that slot is unloaded. Lets the dashboard show WHICH
        /// model is loaded, not just that a slot is occupied.
        let residentModels: [String: String]
    }

    func handleUIState() async -> Response {
        let gov = await governor.snapshot()
        let met = await metrics.snapshot()
        let resident = await residentModelMap()
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
                store: st, queue: Array(jobs), model: modelName,
                residentModels: resident))
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
        // CSRF (M18.1) + per-action RBAC re-check (defense-in-depth
        // ON TOP of AuthPolicy's /ui* daemonAdmin gate — never trust
        // the page; the logged-in user's OWN perms decide).
        guard csrfOK(request) else {
            return Self.json(
                ["error": "csrf token missing or invalid"],
                status: .forbidden)
        }
        let caller = await uiCaller(request)
        guard caller.perms.contains(.daemonAdmin) else {
            return Self.json(
                ["error": "insufficient permission (daemon.admin)"],
                status: .forbidden)
        }
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

    // MARK: - RBAC-aware shell + CSRF (M18.1)

    /// Custom CSRF header. A cross-site page cannot set a custom
    /// request header without a CORS preflight the daemon never
    /// grants, so requiring it (plus the HMAC value match) is
    /// defense-in-depth ON TOP of the SameSite=Strict cookie.
    static let csrfHeaderName = HTTPField.Name("X-CSRF-Token")!

    /// The logged-in /ui caller. AuthMiddleware has ALREADY enforced
    /// the AuthPolicy `/ui*` daemonAdmin gate (session cookie) before
    /// any page handler runs; this resolves WHICH user that was so
    /// pages render RBAC-aware AND each mutation re-checks that user's
    /// OWN permissions (never the page's word). Auth-off loopback =
    /// one trusted local operator ⇒ full perms (mirrors
    /// `callerPermissions`).
    func uiCaller(_ request: Request)
        async -> (user: String, perms: Set<Permission>)
    {
        guard auth.isEnabled else {
            return ("(local)", Set(Permission.allCases))
        }
        if let tok = Session.token(
            fromCookieHeader: request.headers[.cookie]),
            let user = session.validate(tok)
        {
            return (user, await auth.permissions(forUser: user))
        }
        return ("", [])
    }

    /// Verify the per-session CSRF token on a mutating /ui/api/*
    /// request. Auth-off loopback has no session/CSRF (single trusted
    /// operator) ⇒ allowed.
    func csrfOK(_ request: Request) -> Bool {
        guard auth.isEnabled else { return true }
        guard
            let tok = Session.token(
                fromCookieHeader: request.headers[.cookie]),
            let user = session.validate(tok)
        else { return false }
        return session.validateCSRF(
            request.headers[Self.csrfHeaderName], user: user)
    }

    /// One nav item: label, href, and the permission required to see
    /// (and reach) it. Rendered only if the user holds `perm`.
    private struct NavItem {
        let label: String
        let href: String
        let perm: Permission
    }

    /// The whole /ui surface in nav order. Pages land slice by slice
    /// (M18.2 models, M18.3 daemon, M18.4 users); listing them here
    /// keeps the bar RBAC-aware and avoids dead links (an item shows
    /// only when its page exists AND the user holds its perm).
    private static let navItems: [NavItem] = [
        NavItem(
            label: "dashboard", href: "/ui", perm: .metricsRead),
        NavItem(
            label: "models", href: "/ui/models", perm: .modelRead),
        NavItem(
            label: "allowlist", href: "/ui/allowlist",
            perm: .modelRead),
        NavItem(
            label: "daemon", href: "/ui/daemon",
            perm: .daemonAdmin),
        NavItem(
            label: "users", href: "/ui/users", perm: .usersRead),
        NavItem(
            label: "config", href: "/ui/config",
            perm: .daemonAdmin),
    ]

    /// Server-rendered page chrome: shared <head>/CSS + an
    /// RBAC-filtered nav + the signed-in user + logout. Zero JS in
    /// the shell itself (the page body may carry the inline poller).
    static func uiShell(
        title: String, user: String, csrf: String,
        perms: Set<Permission>, active: String, body: String
    ) -> String {
        let links =
            navItems
            .filter { perms.contains($0.perm) }
            .map { it -> String in
                let on = it.href == active ? " on" : ""
                return "<a href=\"\(it.href)\" class=\"nav\(on)\">"
                    + "\(it.label)</a>"
            }
            .joined()
        let who =
            user.isEmpty
            ? "" : "<span class=who>\(Self.esc(user))</span>"
        return "<!doctype html><html><head><meta charset=\"utf-8\">"
            + "<title>\(title)</title>"
            + "<meta name=\"csrf\" content=\"\(csrf)\">"
            + "<style>" + Self.uiCSS + "</style></head><body>"
            + "<nav class=topbar><span class=brand>athena</span>"
            + "<span class=navs>" + links + "</span>"
            + "<span class=right>" + who
            + "<a href=\"/ui/logout\" class=\"nav\">logout</a>"
            + "</span></nav><main>" + body + "</main></body></html>"
    }

    /// Minimal HTML escape for the one piece of caller-influenced
    /// text rendered into the shell (the username). The RBAC name
    /// guard already restricts users to `[A-Za-z0-9._-]`; this is
    /// belt-and-suspenders.
    static func esc(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }

    static let uiCSS = #"""
        body{background:#0d1117;color:#c9d1d9;font:13px ui-monospace,
        Menlo,monospace;margin:0}
        nav.topbar{display:flex;align-items:center;gap:14px;
        padding:12px 24px;border-bottom:1px solid #21262d;
        background:#161b22}
        .brand{font-weight:700;color:#e6edf3}
        .navs{display:flex;gap:14px;flex:1}
        a.nav{color:#8b949e;text-decoration:none}
        a.nav:hover{color:#e6edf3}a.nav.on{color:#2f81f7}
        .right{display:flex;gap:14px;align-items:center}
        .who{color:#8b949e}
        main{padding:24px}
        h1{font-size:16px;margin:0 0 4px}
        .sub{color:#8b949e;margin-bottom:20px}
        .grid{display:grid;grid-template-columns:repeat(auto-fit,
        minmax(320px,1fr));gap:16px}
        .card{background:#161b22;border:1px solid #30363d;
        border-radius:8px;padding:14px}
        .card.form{max-width:640px;padding:16px}
        .card h2{font-size:12px;text-transform:uppercase;
        letter-spacing:.05em;color:#8b949e;margin:0 0 10px}
        table{width:100%;border-collapse:collapse}
        td,th{text-align:left;padding:3px 6px;border-bottom:
        1px solid #21262d}th{color:#8b949e;font-weight:600}
        .bar{height:10px;background:#21262d;border-radius:5px;
        overflow:hidden;margin:6px 0}
        .bar>i{display:block;height:100%;background:#2f81f7}
        label{display:block;color:#8b949e;margin:10px 0 3px}
        input{width:100%;box-sizing:border-box;background:#0d1117;
        color:#e6edf3;border:1px solid #30363d;border-radius:6px;
        padding:7px;font:13px ui-monospace,monospace}
        button{margin-top:16px;background:#238636;color:#fff;
        border:0;border-radius:6px;padding:9px 16px;cursor:pointer;
        font:13px ui-monospace,monospace}
        button.danger{background:#da3633;margin:0;padding:4px 10px}
        #msg{margin-top:14px}
        .k{color:#8b949e}.v{color:#e6edf3}
        .ok{color:#3fb950}.warn{color:#d29922}.err{color:#f85149}
        """#

    // MARK: - Dashboard (M11.2 → reshelled M18.1)

    /// Live status/metrics dashboard, now inside the RBAC-aware
    /// shell. The inline poller (`/ui/api/state` every 2s) is
    /// unchanged — minimal hand-written vanilla JS, zero deps.
    func handleUIDashboard(_ request: Request) async -> Response {
        let c = await uiCaller(request)
        let csrf = c.user.isEmpty ? "" : session.csrf(user: c.user)
        return Self.html(
            Self.uiShell(
                title: "athena", user: c.user, csrf: csrf,
                perms: c.perms, active: "/ui",
                body: Self.dashBody))
    }

    static let dashBody = #"""
        <div class="sub"><span id="sub">connecting…</span></div>
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
          const g=s.governor, used=g.residentBytes,
            tot=g.totalBudgetBytes;
          $("membar").style.width=
            (tot?100*used/tot:0).toFixed(1)+"%";
          $("mem").innerHTML="<table>"+
            row("resident",mb(used))+row("free",mb(g.freeBytes))+
            row("budget",mb(tot))+
            row("prompt-cache cap",mb(g.promptCacheCapBytes))+
            (g.promptCachePoolEntries>0?row("prompt-cache pool",
              g.promptCachePoolEntries+" ent · "+mb(g.promptCachePoolBytes)):"")+
            "</table>";
          const rm=s.residentModels||{};
          $("mods").innerHTML="<tr><th>module</th><th>model</th>"+
            "<th>state</th><th>resident</th></tr>"+g.modules.map(m=>
            `<tr><td>${m.id}</td><td class=k>${rm[m.id]||"—"}</td>
             <td class=${m.state=="loaded"?"ok":"k"}>${m.state}${
              m.unloadedReason?" ("+m.unloadedReason+")":""}</td>
             <td>${mb(m.residentBytes)}</td></tr>`).join("");
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
        </script>
        """#

    // MARK: - Config view + edit (M11.3 → reshelled + CSRF M18.1)

    /// The config editor inside the shell. Defense-in-depth: even
    /// though AuthPolicy already gated /ui* on daemonAdmin, a user
    /// lacking it sees a notice (not the form) and the POST
    /// re-checks the CSRF token + daemonAdmin on the logged-in user.
    func handleUIConfigPage(_ request: Request) async -> Response {
        let c = await uiCaller(request)
        let csrf = c.user.isEmpty ? "" : session.csrf(user: c.user)
        let body =
            c.perms.contains(.daemonAdmin)
            ? Self.configBody
            : #"<div class="card">insufficient permission "#
                + #"(need daemon.admin)</div>"#
        return Self.html(
            Self.uiShell(
                title: "athena · config", user: c.user,
                csrf: csrf, perms: c.perms,
                active: "/ui/config", body: body))
    }

    static let configBody = #"""
        <div class="sub"><span id="path" class="k"></span></div>
        <div class="card form"><form id="f"></form>
          <button onclick="save()">Save</button>
          <div id="msg"></div></div>
        <script>
        const $=i=>document.getElementById(i);
        const CSRF=document.querySelector('meta[name=csrf]').content;
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
            headers:{"content-type":"application/json",
              "X-CSRF-Token":CSRF},
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
        </script>
        """#

    // MARK: - Model console page (M18.2)

    /// The model-store console inside the shell. List is gated on
    /// `.modelRead`; the mutate UI (pull/convert/copy/default/delete)
    /// renders only when the logged-in user holds `.modelWrite` —
    /// and every /ui/api/models* mutation re-checks it server-side
    /// (the page is never trusted; see `uiModelMutate`).
    func handleUIModelsPage(_ request: Request) async -> Response {
        let c = await uiCaller(request)
        let csrf = c.user.isEmpty ? "" : session.csrf(user: c.user)
        let body: String
        if c.perms.contains(.modelRead) {
            let w = c.perms.contains(.modelWrite) ? "1" : "0"
            body =
                "<meta name=\"write\" content=\"\(w)\">"
                + Self.modelsBody
        } else {
            body =
                #"<div class="card">insufficient permission "#
                + #"(need model.read)</div>"#
        }
        return Self.html(
            Self.uiShell(
                title: "athena · models", user: c.user,
                csrf: csrf, perms: c.perms,
                active: "/ui/models", body: body))
    }

    static let modelsBody = #"""
        <div class="sub">model store · <span id="def"
          class="k"></span></div>
        <div class="grid">
          <div class="card"><h2>Models</h2>
            <table id="ml"><tr><td class=k>loading…</td></tr>
            </table></div>
          <div class="card mut"><h2>Pull / convert</h2>
            <label>pull id (HF repo or path)</label>
            <input id="pid">
            <button onclick="op('pull','pid')">Pull</button>
            <label>convert id</label><input id="cid">
            <button onclick="op('convert','cid')">Convert</button>
            <div id="job" class="k"></div></div>
          <div class="card mut"><h2>Copy / default</h2>
            <label>copy src</label><input id="csrc">
            <label>copy dst</label><input id="cdst">
            <label><input type="checkbox" id="cdeep"> deep copy
              (else alias)</label>
            <button onclick="cp()">Copy</button>
            <label>set default model</label><input id="dname">
            <button onclick="setDefault()">Set default</button>
            <div id="cmsg" class="k"></div></div>
        </div>
        <script>
        const $=i=>document.getElementById(i);
        const CSRF=document.querySelector('meta[name=csrf]').content;
        const WM=document.querySelector('meta[name=write]');
        const WRITE=WM&&WM.content==="1";
        const jget=async u=>(await fetch(u)).json();
        const jpost=(u,b)=>fetch(u,{method:"POST",headers:{
          "content-type":"application/json","X-CSRF-Token":CSRF},
          body:JSON.stringify(b)});
        const em=j=>(j&&j.error&&j.error.message)||
          (j&&j.error)||"error";
        async function load(){
          const d=await jget("/ui/api/models/default");
          $("def").textContent="default: "+d.model+
            " ("+d.source+")";
          const m=await jget("/ui/api/models");
          const rows=(m.models||[]).map(x=>`<tr><td>${x.name}</td>
            <td>${(x.bytes/1048576).toFixed(0)} MB</td>
            <td class=k>${x.modified}</td>`+(WRITE?
            `<td><button class=danger onclick="rm('${x.name}')">
            delete</button></td>`:``)+`</tr>`).join("");
          $("ml").innerHTML="<tr><th>name</th><th>size</th>"+
            "<th>modified</th>"+(WRITE?"<th></th>":"")+"</tr>"+
            (rows||"<tr><td class=k>no models</td></tr>");
        }
        async function poll(id){
          const j=await jget("/ui/api/job?id="+
            encodeURIComponent(id));
          const cl=j.status=="error"?"err":
            j.status=="done"?"ok":"warn";
          $("job").innerHTML="job "+id.slice(0,8)+" · "+
            `<span class=${cl}>${j.status}</span>`+
            (j.error?(" · "+j.error):"");
          if(j.status!="done"&&j.status!="error"&&
             j.status!="canceled")setTimeout(()=>poll(id),2000);
          else load();
        }
        async function op(kind,inp){
          const id=$(inp).value.trim(); if(!id)return;
          const r=await jpost("/ui/api/models/"+kind,{id:id});
          const j=await r.json();
          if(r.ok&&j.job_id){$("job").textContent=
            "queued "+j.job_id; poll(j.job_id);}
          else $("job").innerHTML=
            "<span class=err>"+em(j)+"</span>";
        }
        async function cp(){
          const r=await jpost("/ui/api/models/copy",{
            src:$("csrc").value.trim(),dst:$("cdst").value.trim(),
            copy:$("cdeep").checked});
          const j=await r.json();
          $("cmsg").innerHTML=r.ok?
            `<span class=ok>copied → ${j.dst}</span>`:
            `<span class=err>${em(j)}</span>`;
          if(r.ok)load();
        }
        async function setDefault(){
          const r=await jpost("/ui/api/models/default",
            {name:$("dname").value.trim()});
          const j=await r.json();
          $("cmsg").innerHTML=r.ok?
            `<span class=ok>default → ${j.model}</span>`:
            `<span class=err>${em(j)}</span>`;
          if(r.ok)load();
        }
        async function rm(n){
          if(!confirm("Delete model '"+n+
            "'? This cannot be undone."))return;
          const r=await jpost("/ui/api/models/rm",{name:n});
          if(r.ok)load();
          else{const j=await r.json();
            alert("delete failed: "+em(j));}
        }
        load();
        </script>
        """#

    // MARK: - Allowlist page (M44.1)

    /// Persistent per-module model allowlist console. List is gated
    /// on `.modelRead`; the mutate UI (add/rm/default) renders only
    /// when the logged-in user holds `.modelWrite`, and every
    /// /ui/api/allowlist* mutation re-checks it server-side via
    /// `uiAllowlistMutate` (page is never trusted).
    func handleUIAllowlistPage(_ request: Request) async -> Response {
        let c = await uiCaller(request)
        let csrf = c.user.isEmpty ? "" : session.csrf(user: c.user)
        let body: String
        if c.perms.contains(.modelRead) {
            let w = c.perms.contains(.modelWrite) ? "1" : "0"
            body =
                "<meta name=\"write\" content=\"\(w)\">"
                + Self.allowlistBody
        } else {
            body =
                #"<div class="card">insufficient permission "#
                + #"(need model.read)</div>"#
        }
        return Self.html(
            Self.uiShell(
                title: "athena · allowlist", user: c.user,
                csrf: csrf, perms: c.perms,
                active: "/ui/allowlist", body: body))
    }

    static let allowlistBody = #"""
        <div class="sub">per-module model allowlist · the daemon
          refreshes its in-memory set after each change (no restart
          needed)</div>
        <div class="grid">
          <div class="card"><h2>Allowlist</h2>
            <table id="al"><tr><td class=k>loading…</td></tr>
            </table></div>
          <div class="card mut"><h2>Add</h2>
            <label>module</label>
            <select id="amod">
              <option value="llm">llm</option>
              <option value="textEmbedding">textEmbedding</option>
              <option value="transcription">transcription</option>
              <option value="diarization">diarization</option>
              <option value="speakerEmbedding">speakerEmbedding</option>
            </select>
            <label>id (HF id for aux modules, store name for llm)</label>
            <input id="aid">
            <label><input type="checkbox" id="adef"> mark as default
              for this module</label>
            <button onclick="addRow()">Add</button>
            <div id="amsg" class="k"></div></div>
        </div>
        <script>
        const $=i=>document.getElementById(i);
        const CSRF=document.querySelector('meta[name=csrf]').content;
        const WM=document.querySelector('meta[name=write]');
        const WRITE=WM&&WM.content==="1";
        const jget=async u=>(await fetch(u)).json();
        const jpost=(u,b)=>fetch(u,{method:"POST",headers:{
          "content-type":"application/json","X-CSRF-Token":CSRF},
          body:JSON.stringify(b)});
        const em=j=>(j&&j.error&&j.error.message)||
          (j&&j.error)||"error";
        const esc=s=>String(s).replace(/[&<>"]/g,c=>(
          {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"})[c]);
        async function load(){
          const m=await jget("/ui/api/allowlist");
          const rows=(m.allowlist||[]).map(x=>`<tr>
            <td>${esc(x.module)}</td>
            <td>${esc(x.id)}</td>
            <td class=k>${x.default?"*":""}</td>`+(WRITE?
            `<td>${x.default?"":`<button onclick="mkdef('`+
              `${esc(x.module)}','${esc(x.id)}')">default</button> `}`+
            `<button class=danger onclick="rm('${esc(x.module)}',`+
              `'${esc(x.id)}')">remove</button></td>`:``)+`</tr>`)
            .join("");
          $("al").innerHTML="<tr><th>module</th><th>id</th>"+
            "<th>default</th>"+(WRITE?"<th></th>":"")+"</tr>"+
            (rows||"<tr><td class=k>no entries</td></tr>");
        }
        async function addRow(){
          const mod=$("amod").value;
          const id=$("aid").value.trim();
          if(!id){$("amsg").innerHTML=
            "<span class=err>id required</span>"; return;}
          const r=await jpost("/ui/api/allowlist",{
            module:mod,id:id,default:$("adef").checked});
          const j=await r.json();
          if(r.ok){$("amsg").innerHTML=
            `<span class=ok>added ${esc(mod)}:${esc(id)}</span>`;
            $("aid").value=""; $("adef").checked=false; load();}
          else $("amsg").innerHTML=
            "<span class=err>"+em(j)+"</span>";
        }
        async function mkdef(mod,id){
          const r=await jpost("/ui/api/allowlist/default",
            {module:mod,id:id});
          if(r.ok)load();
          else{const j=await r.json();
            alert("set default failed: "+em(j));}
        }
        async function rm(mod,id){
          if(!confirm("Remove "+mod+":"+id+" from the allowlist?"))
            return;
          const r=await jpost("/ui/api/allowlist/rm",
            {module:mod,id:id});
          if(r.ok)load();
          else{const j=await r.json();
            alert("remove failed: "+em(j));}
        }
        load();
        </script>
        """#

    // MARK: - Daemon control page (M18.3)

    /// Daemon-control console inside the shell. Posture panel +
    /// warm/unload the model. Copy states the hard scope limit: the
    /// WebUI runs INSIDE the daemon so it can only control a RUNNING
    /// one — it cannot cold-start a stopped daemon. daemonAdmin only
    /// (re-checked server-side on every /ui/api/admin/* call).
    func handleUIDaemonPage(_ request: Request) async -> Response {
        let c = await uiCaller(request)
        let csrf = c.user.isEmpty ? "" : session.csrf(user: c.user)
        let body =
            c.perms.contains(.daemonAdmin)
            ? Self.daemonBody
            : #"<div class="card">insufficient permission "#
                + #"(need daemon.admin)</div>"#
        return Self.html(
            Self.uiShell(
                title: "athena · daemon", user: c.user,
                csrf: csrf, perms: c.perms,
                active: "/ui/daemon", body: body))
    }

    static let daemonBody = #"""
        <div class="sub">controls the RUNNING daemon — it cannot
          cold-start a stopped one (use launchd / the CLI for
          that)</div>
        <div class="grid">
          <div class="card"><h2>Posture</h2>
            <table id="st"><tr><td class=k>loading…</td></tr>
            </table></div>
          <div class="card"><h2>Model</h2>
            <p class="k">Warm pre-loads the model so the next
              request is hot. Unload frees its memory; the next
              request will lazily reload it.</p>
            <button onclick="warm()">Warm (load)</button>
            <button class=danger onclick="stop()"
              style="padding:9px 16px;margin-top:16px">
              Unload model</button>
            <div id="dmsg" class="k"></div></div>
        </div>
        <script>
        const $=i=>document.getElementById(i);
        const CSRF=document.querySelector('meta[name=csrf]').content;
        const jpost=u=>fetch(u,{method:"POST",
          headers:{"X-CSRF-Token":CSRF}});
        const em=j=>(j&&j.error&&j.error.message)||
          (j&&j.error)||"error";
        const R=(a,b)=>`<tr><td class=k>${a}</td>
          <td class=v>${b}</td></tr>`;
        async function load(){
          const s=await (await fetch(
            "/ui/api/admin/status")).json();
          $("st").innerHTML="<table>"+R("model",s.model)+
            R("listen",s.listen)+
            R("auth",s.auth_enabled?"enabled":"open (loopback)")+
            R("users",s.users)+R("tokens",s.tokens)+
            R("admins",s.admins)+"</table>";
        }
        async function warm(){
          $("dmsg").textContent="warming…";
          const r=await jpost("/ui/api/admin/load");
          const j=await r.json();
          $("dmsg").innerHTML=r.ok?
            `<span class=ok>model ${j.status}</span>`:
            `<span class=err>${em(j)}</span>`;
        }
        async function stop(){
          if(!confirm("Unload the model now? In-flight requests "+
            "may fail; the next request reloads it."))return;
          const r=await jpost("/ui/api/admin/stop");
          const j=await r.json();
          $("dmsg").innerHTML=r.ok?
            `<span class=ok>model ${j.status}</span>`:
            `<span class=err>${em(j)}</span>`;
          load();
        }
        load();
        </script>
        """#

    // MARK: - RBAC admin page (M18.4)

    /// User / role / token admin inside the shell. The user table +
    /// role catalog need `.usersRead`; the create/grant/revoke/
    /// delete UI shows only with `.usersAdmin`, the token UI only
    /// with `.tokensAdmin` — and EVERY /ui/api/{users,tokens,roles}
    /// call re-checks server-side AND runs the M16.4 canGrant /
    /// last-admin guards against the LOGGED-IN user (cookie-aware
    /// `callerPermissions`). A freshly minted token is shown ONCE.
    func handleUIUsersPage(_ request: Request) async -> Response {
        let c = await uiCaller(request)
        let csrf = c.user.isEmpty ? "" : session.csrf(user: c.user)
        let body: String
        if c.perms.contains(.usersRead) {
            let ua = c.perms.contains(.usersAdmin) ? "1" : "0"
            let ta = c.perms.contains(.tokensAdmin) ? "1" : "0"
            body =
                "<meta name=\"uadm\" content=\"\(ua)\">"
                + "<meta name=\"tadm\" content=\"\(ta)\">"
                + Self.usersBody
        } else {
            body =
                #"<div class="card">insufficient permission "#
                + #"(need users.read)</div>"#
        }
        return Self.html(
            Self.uiShell(
                title: "athena · users", user: c.user,
                csrf: csrf, perms: c.perms,
                active: "/ui/users", body: body))
    }

    static let usersBody = #"""
        <div class="sub">accounts, role grants & API tokens</div>
        <div class="grid">
          <div class="card"><h2>Users</h2>
            <table id="ul"><tr><td class=k>loading…</td></tr>
            </table></div>
          <div class="card mua"><h2>Create user</h2>
            <label>username</label><input id="nu">
            <label>password (≥ 8)</label>
            <input id="np" type="password">
            <label>role</label><input id="nr" value="member">
            <button onclick="mkUser()">Create</button>
            <div id="umsg" class="k"></div></div>
          <div class="card mua"><h2>Grant / revoke role</h2>
            <label>username</label><input id="gu">
            <label>role</label><input id="gr">
            <button onclick="role('grant')">Grant</button>
            <button onclick="role('revoke')"
              style="background:#6e40c9">Revoke</button>
            <div id="gmsg" class="k"></div></div>
          <div class="card"><h2>Role catalog</h2>
            <table id="rl"><tr><td class=k>loading…</td></tr>
            </table></div>
          <div class="card mta"><h2>Tokens</h2>
            <table id="tl"><tr><td class=k>loading…</td></tr>
            </table></div>
          <div class="card mta"><h2>Mint / revoke token</h2>
            <label>user</label><input id="tu">
            <label>scoped role (optional, narrows)</label>
            <input id="ts">
            <label>label (optional)</label><input id="tlb">
            <button onclick="mkTok()">Mint</button>
            <div id="tok" class="k"></div>
            <label>revoke by hash prefix (≥ 6 hex)</label>
            <input id="tp">
            <button class=danger onclick="rmTok()"
              style="padding:9px 16px">Revoke</button>
            <div id="tmsg" class="k"></div></div>
        </div>
        <script>
        const $=i=>document.getElementById(i);
        const CSRF=document.querySelector('meta[name=csrf]').content;
        const M=n=>{const e=document.querySelector(
          'meta[name='+n+']');return e&&e.content==="1";};
        const UA=M("uadm"),TA=M("tadm");
        const jget=async u=>(await fetch(u)).json();
        const jpost=(u,b)=>fetch(u,{method:"POST",headers:{
          "content-type":"application/json","X-CSRF-Token":CSRF},
          body:JSON.stringify(b)});
        const em=j=>(j&&j.error&&j.error.message)||
          (j&&j.error)||"error";
        function show(){
          [...document.querySelectorAll('.mua')].forEach(e=>
            e.style.display=UA?'':'none');
          [...document.querySelectorAll('.mta')].forEach(e=>
            e.style.display=TA?'':'none');
        }
        async function users(){
          const d=await jget("/ui/api/users");
          $("ul").innerHTML="<tr><th>user</th><th>roles</th>"+
            (UA?"<th></th>":"")+"</tr>"+
            ((d.users||[]).map(u=>`<tr><td>${u.username}</td>
              <td class=k>${(u.roles||[]).join(", ")}</td>`+
              (UA?`<td><button class=danger onclick=
                "delUser('${u.username}')">delete</button></td>`:``)+
              `</tr>`).join("")||
            "<tr><td class=k>no users</td></tr>");
        }
        async function roles(){
          const d=await jget("/ui/api/roles");
          $("rl").innerHTML="<tr><th>role</th><th>permissions</th>"+
            "</tr>"+(d.roles||[]).map(r=>`<tr><td>${r.role}</td>
            <td class=k>${(r.permissions||[]).join(" ")}</td></tr>`)
            .join("");
        }
        async function toks(){
          if(!TA)return;
          const d=await jget("/ui/api/tokens");
          $("tl").innerHTML="<tr><th>user</th><th>scope</th>"+
            "<th>prefix</th><th>label</th></tr>"+
            ((d.tokens||[]).map(t=>`<tr><td>${t.username}</td>
              <td class=k>${(t.scope||[]).join(",")||"—"}</td>
              <td class=k>${t.hash_prefix}</td>
              <td class=k>${t.label||""}</td></tr>`).join("")||
            "<tr><td class=k>no tokens</td></tr>");
        }
        async function mkUser(){
          const r=await jpost("/ui/api/users",{
            username:$("nu").value.trim(),
            password:$("np").value,role:$("nr").value.trim()});
          const j=await r.json();
          $("umsg").innerHTML=r.ok?
            `<span class=ok>created ${j.username}</span>`:
            `<span class=err>${em(j)}</span>`;
          if(r.ok)users();
        }
        async function delUser(n){
          if(!confirm("Delete user '"+n+"'?"))return;
          const r=await jpost("/ui/api/users/delete",{name:n});
          if(r.ok)users();
          else{const j=await r.json();alert("failed: "+em(j));}
        }
        async function role(act){
          const r=await jpost("/ui/api/users/role/"+act,{
            name:$("gu").value.trim(),role:$("gr").value.trim()});
          const j=await r.json();
          $("gmsg").innerHTML=r.ok?
            `<span class=ok>${act} ok</span>`:
            `<span class=err>${em(j)}</span>`;
          if(r.ok)users();
        }
        async function mkTok(){
          const b={user:$("tu").value.trim()};
          const s=$("ts").value.trim();
          if(s)b.role=[s];
          const l=$("tlb").value.trim(); if(l)b.label=l;
          const r=await jpost("/ui/api/tokens",b);
          const j=await r.json();
          if(r.ok)$("tok").innerHTML=
            `<span class=ok>token (shown once):</span><br>`+
            `<input readonly value="${j.token}" `+
            `onclick="this.select()" style="margin-top:6px">`;
          else $("tok").innerHTML=
            `<span class=err>${em(j)}</span>`;
          if(r.ok)toks();
        }
        async function rmTok(){
          const p=$("tp").value.trim();
          if(!confirm("Revoke token(s) with prefix '"+p+"'?"))
            return;
          const r=await jpost("/ui/api/tokens/delete",{prefix:p});
          const j=await r.json();
          $("tmsg").innerHTML=r.ok?
            `<span class=ok>revoked ${j.removed}</span>`:
            `<span class=err>${em(j)}</span>`;
          if(r.ok)toks();
        }
        show();users();roles();toks();
        </script>
        """#
}

// MARK: - WebUI session login (M12.2)

extension AthenaServer {
    static func htmlStatus(
        _ s: String, _ status: HTTPResponse.Status
    ) -> Response {
        var buf = ByteBuffer()
        buf.writeBytes(Data(s.utf8))
        var h = HTTPFields()
        h[.contentType] = "text/html; charset=utf-8"
        return Response(
            status: status, headers: h,
            body: ResponseBody(byteBuffer: buf))
    }

    static func loginPage(error: String?) -> String {
        let banner =
            error.map {
                "<p class=err>\($0)</p>"
            } ?? ""
        return #"""
            <!doctype html><html><head><meta charset="utf-8">
            <title>athena · sign in</title><style>
            body{background:#0d1117;color:#c9d1d9;font:13px
            ui-monospace,Menlo,monospace;display:flex;height:100vh;
            margin:0;align-items:center;justify-content:center}
            form{background:#161b22;border:1px solid #30363d;
            border-radius:8px;padding:28px;width:280px}
            h1{font-size:15px;margin:0 0 16px}
            input{width:100%;box-sizing:border-box;background:#0d1117;
            color:#e6edf3;border:1px solid #30363d;border-radius:6px;
            padding:8px;margin:6px 0;font:13px ui-monospace,monospace}
            button{width:100%;margin-top:14px;background:#238636;
            color:#fff;border:0;border-radius:6px;padding:9px;
            cursor:pointer;font:13px ui-monospace,monospace}
            .err{color:#f85149;margin:0 0 10px}
            </style></head><body>
            <form method="post" action="/ui/login">
            <h1>athena</h1>
            """# + banner + #"""
            <input name="username" placeholder="username" autofocus>
            <input name="password" type="password"
              placeholder="password">
            <button>Sign in</button></form></body></html>
            """#
    }

    private static func formField(
        _ name: String, in body: String
    ) -> String? {
        for pair in body.split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1)
            guard kv.first == name[...] else { continue }
            let raw =
                kv.count == 2 ? String(kv[1]) : ""
            return raw.replacingOccurrences(of: "+", with: " ")
                .removingPercentEncoding ?? raw
        }
        return nil
    }

    func handleUILoginPost(_ request: Request) async -> Response {
        let body: String
        if let buf = try? await request.body.collect(
            upTo: 64 * 1024)
        {
            body = String(buffer: buf)
        } else {
            body = ""
        }
        guard
            let user = Self.formField("username", in: body),
            let pass = Self.formField("password", in: body),
            !user.isEmpty
        else {
            return Self.htmlStatus(
                Self.loginPage(error: "missing credentials"),
                .badRequest)
        }
        guard let row = await store.getUser(username: user),
            Passwords.verify(
                password: pass, salt: row.salt, hash: row.hash,
                iters: row.iters)
        else {
            return Self.htmlStatus(
                Self.loginPage(error: "invalid credentials"),
                .unauthorized)
        }
        var h = HTTPFields()
        h[.location] = "/ui"
        h[.setCookie] = Session.setCookie(session.mint(user: user))
        return Response(status: .seeOther, headers: h)
    }

    static func logoutResponse() -> Response {
        var h = HTTPFields()
        h[.location] = "/ui/login"
        h[.setCookie] = Session.clearCookie
        return Response(status: .seeOther, headers: h)
    }
}
