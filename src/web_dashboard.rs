use axum::{response::Html, routing::get, Router};

pub fn create_dashboard() -> Router {
    Router::new()
        .route("/", get(index))
        .route("/round/{id}", get(round_detail))
}

async fn index() -> Html<String> { Html(DASHBOARD_HTML.to_string()) }
async fn round_detail() -> Html<String> { Html(ROUND_HTML.to_string()) }

const DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Embodied-FL — 具身智能联邦学习平台</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e1a;color:#e2e8f0}
.container{max-width:1280px;margin:0 auto;padding:20px}
header{display:flex;justify-content:space-between;align-items:center;padding:16px 0;border-bottom:1px solid #1e293b;margin-bottom:24px}
h1{font-size:22px;color:#38bdf8}
h1 span{color:#64748b;font-size:13px;font-weight:normal;margin-left:12px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}
.stat{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:16px}
.stat .label{font-size:12px;color:#64748b;margin-bottom:4px}
.stat .value{font-size:24px;font-weight:700}
.stat .value.green{color:#22c55e}.stat .value.blue{color:#38bdf8}.stat .value.amber{color:#f59e0b}.stat .value.purple{color:#a78bfa}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:20px;margin-bottom:16px}
.card h2{font-size:16px;color:#94a3b8;margin-bottom:16px;font-weight:500}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:11px;color:#64748b;text-transform:uppercase;padding:8px 12px;border-bottom:1px solid #1e293b}
td{padding:10px 12px;font-size:13px;border-bottom:1px solid #0f172a}
tr:hover{background:#0f172a}
.badge{display:inline-block;padding:2px 8px;border-radius:9999px;font-size:11px;font-weight:600}
.badge.grasping{background:#164e63;color:#22d3ee}
.badge.navigation{background:#365314;color:#a3e635}
.badge.inspection{background:#4c1d95;color:#c084fc}
.badge.assembly{background:#7c2d12;color:#fb923c}
.badge.manipulation{background:#831843;color:#f472b6}
.loading{text-align:center;padding:40px;color:#475569}
.pulse{animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
</style>
</head>
<body>
<div class="container">
<header>
<h1>🤖 Embodied-FL <span>Federated Learning for Embodied Intelligence</span></h1>
<div style="font-size:12px;color:#475569" id="status">Connecting...</div>
</header>
<div class="stats" id="stats"></div>
<div class="grid">
<div>
<div class="card"><h2>📋 任务列表</h2><div id="tasks"><div class="loading pulse">Loading...</div></div></div>
<div class="card"><h2>🔄 训练轮次</h2><div id="rounds"><div class="loading pulse">Loading...</div></div></div>
</div>
<div>
<div class="card"><h2>🏆 贡献排行榜</h2><div id="leaderboard"><div class="loading pulse">Loading...</div></div></div>
<div class="card"><h2>🔐 审计链</h2><div id="audit"><div class="loading pulse">Loading...</div></div></div>
</div>
</div>
</div>
<script>
const API='/api/v1';
async function load(){
  try{
    const [stats,tasks,lb,audit]=await Promise.all([
      fetch(API+'/stats').then(r=>r.json()),
      fetch(API+'/tasks?limit=20').then(r=>r.json()),
      fetch(API+'/leaderboard').then(r=>r.json()),
      fetch(API+'/audit/logs').then(r=>r.json()),
    ]);
    document.getElementById('status').textContent='● Connected';
    document.getElementById('status').style.color='#22c55e';
    // Stats
    document.getElementById('stats').innerHTML=`
      <div class="stat"><div class="label">当前轮次</div><div class="value blue">${stats.current_round}</div></div>
      <div class="stat"><div class="label">在线客户端</div><div class="value green">${stats.online_clients}/${stats.total_clients}</div></div>
      <div class="stat"><div class="label">活跃任务</div><div class="value amber">${stats.active_tasks}</div></div>
      <div class="stat"><div class="label">向量索引</div><div class="value purple">${stats.total_vectors}</div></div>
      <div class="stat"><div class="label">审计链</div><div class="value ${stats.audit_chain_valid?'green':'red'}">${stats.audit_chain_length} ✓</div></div>`;
    // Tasks
    if(tasks.tasks&&tasks.tasks.length>0){
      document.getElementById('tasks').innerHTML='<table><thead><tr><th>ID</th><th>客户端</th><th>类型</th><th>状态</th><th>轮次</th><th>贡献</th></tr></thead><tbody>'+
        tasks.tasks.map(t=>`<tr><td style="font-family:monospace;font-size:11px">${t.task_id.slice(0,8)}</td><td>${t.client_id}</td><td><span class="badge ${t.task_type}">${t.task_type}</span></td><td>${t.status}</td><td>${t.rounds_participated}</td><td>${t.total_contribution.toFixed(2)}</td></tr>`).join('')+'</tbody></table>';
    }else{document.getElementById('tasks').innerHTML='<div class="loading">No tasks registered</div>'}
    // Leaderboard
    if(lb.leaderboard&&lb.leaderboard.length>0){
      document.getElementById('leaderboard').innerHTML='<table><thead><tr><th>#</th><th>客户端</th><th>贡献</th><th>轮次</th></tr></thead><tbody>'+
        lb.leaderboard.map((e,i)=>`<tr><td>${i+1}</td><td>${e.client_name||e.client_id}</td><td style="color:#f59e0b;font-weight:600">${e.total_contribution.toFixed(2)}</td><td>${e.rounds_participated}</td></tr>`).join('')+'</tbody></table>';
    }else{document.getElementById('leaderboard').innerHTML='<div class="loading">No data yet</div>'}
    // Audit
    if(audit.logs&&audit.logs.length>0){
      document.getElementById('audit').innerHTML='<table><thead><tr><th>时间</th><th>操作</th></tr></thead><tbody>'+
        audit.logs.slice(0,8).map(l=>`<tr><td style="font-size:11px;color:#64748b">${l.timestamp.slice(11,19)}</td><td style="font-size:12px">${l.operation}</td></tr>`).join('')+'</tbody></table>';
    }else{document.getElementById('audit').innerHTML='<div class="loading">No audit entries</div>'}
  }catch(e){
    document.getElementById('status').textContent='● Disconnected';
    document.getElementById('status').style.color='#ef4444';
    console.error(e);
  }
}
load();setInterval(load,10000);
</script>
</body>
</html>"##;

const ROUND_HTML: &str = r##"<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Round Detail — Embodied-FL</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e1a;color:#e2e8f0}
.container{max-width:960px;margin:0 auto;padding:20px}
a{color:#38bdf8;text-decoration:none}
.card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:20px;margin-bottom:16px}
h1{font-size:20px;margin-bottom:20px}
h2{font-size:15px;color:#94a3b8;margin-bottom:12px;font-weight:500}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:11px;color:#64748b;text-transform:uppercase;padding:8px 12px;border-bottom:1px solid #1e293b}
td{padding:10px 12px;font-size:13px;border-bottom:1px solid #0f172a}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.metric{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:16px;text-align:center}
.metric .label{font-size:11px;color:#64748b;margin-bottom:4px}
.metric .value{font-size:22px;font-weight:700}
</style>
</head>
<body>
<div class="container">
<a href="/">← Back to Dashboard</a>
<h1 id="title">Round Detail</h1>
<div class="metrics" id="metrics"></div>
<div class="card"><h2>📤 Client Updates</h2><div id="updates"><div class="loading">Loading...</div></div></div>
</div>
<script>
const id=window.location.pathname.split('/').pop();
fetch('/api/v1/rounds/'+id).then(r=>r.json()).then(data=>{
  const r=data.round;
  document.getElementById('title').textContent='Round #'+r.round_id+' — '+r.status;
  document.getElementById('metrics').innerHTML=`
    <div class="metric"><div class="label">Loss</div><div class="value" style="color:#38bdf8">${r.loss.toFixed(4)}</div></div>
    <div class="metric"><div class="label">Accuracy</div><div class="value" style="color:#22c55e">${(r.accuracy*100).toFixed(1)}%</div></div>
    <div class="metric"><div class="label">Participants</div><div class="value" style="color:#f59e0b">${r.participants}/${r.total_clients}</div></div>
    <div class="metric"><div class="label">Status</div><div class="value">${r.status}</div></div>`;
  if(data.updates&&data.updates.length>0){
    document.getElementById('updates').innerHTML='<table><thead><tr><th>Client</th><th>Task</th><th>Samples</th><th>Loss</th><th>Accuracy</th></tr></thead><tbody>'+
      data.updates.map(u=>`<tr><td>${u.client_id}</td><td>${u.task_type}</td><td>${u.num_samples}</td><td>${u.local_loss.toFixed(4)}</td><td>${(u.local_accuracy*100).toFixed(1)}%</td></tr>`).join('')+'</tbody></table>';
  }
}).catch(e=>console.error(e));
</script>
</body>
</html>"##;
