//! Minimal HTTP server with SSE streaming for web chat UI.

use crate::forward::InferenceState;
use crate::forward_llama;
use crate::model::BitNetModel;
use crate::tokenizer::Tokenizer;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Instant;

pub fn run(
    model: &BitNetModel,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    max_seq_len: usize,
    port: u16,
) {
    run_inner(model, tokenizer, max_tokens, temperature, repetition_penalty, max_seq_len, port, false);
}

pub fn run_q4k(
    model: &BitNetModel,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    max_seq_len: usize,
    port: u16,
) {
    run_inner(model, tokenizer, max_tokens, temperature, repetition_penalty, max_seq_len, port, true);
}

fn run_inner(
    model: &BitNetModel,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    temperature: f32,
    repetition_penalty: f32,
    max_seq_len: usize,
    port: u16,
    use_q4k: bool,
) {
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("cougar> failed to bind {addr}: {e}");
        std::process::exit(1);
    });
    eprintln!("cougar> serving on http://localhost:{port}");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));

        let mut buf = [0u8; 8192];
        let mut n = 0;
        loop {
            let r = match stream.read(&mut buf[n..]) {
                Ok(0) => break,
                Ok(r) => r,
                Err(_) => break,
            };
            n += r;
            if n >= 4 && buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
            if n >= buf.len() { break; }
        }
        let req = match std::str::from_utf8(&buf[..n]) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let first_line = req.lines().next().unwrap_or("");
        let mut parts = first_line.split_whitespace();
        let method = parts.next().unwrap_or("");
        let path = parts.next().unwrap_or("");

        match (method, path) {
            ("GET", "/") => {
                let body = CHAT_HTML;
                let _ = write!(stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
            ("GET", "/api/model") => {
                let body = format!(
                    "{{\"layers\":{},\"hidden_dim\":{},\"heads\":{},\"vocab_size\":{}}}",
                    model.n_layers, model.hidden_dim, model.n_heads, model.vocab_size
                );
                let _ = write!(stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
            ("POST", "/api/generate") => {
                // Read POST body based on Content-Length
                let content_len = parse_content_length(req);
                let header_end = req.find("\r\n\r\n").unwrap_or(n) + 4;
                let already = n - header_end;
                let mut body_buf = vec![0u8; content_len];
                if already > 0 && already <= content_len {
                    body_buf[..already].copy_from_slice(&buf[header_end..n]);
                }
                if already < content_len {
                    let _ = stream.read_exact(&mut body_buf[already..]);
                }
                let body_str = std::str::from_utf8(&body_buf).unwrap_or("");

                let prompt_text = extract_json_string(body_str, "prompt").unwrap_or_default();
                let req_max = extract_json_number(body_str, "max_tokens")
                    .map(|v| v as usize).unwrap_or(max_tokens);
                let req_temp = extract_json_float(body_str, "temperature").unwrap_or(temperature);
                let req_rep = extract_json_float(body_str, "repetition_penalty")
                    .unwrap_or(repetition_penalty);

                let _ = write!(stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                     Cache-Control: no-cache\r\nConnection: close\r\n\r\n");
                let _ = stream.flush();

                if prompt_text.is_empty() {
                    let _ = write!(stream, "data: [DONE]\n\n");
                } else {
                    let mut tokens = vec![tokenizer.bos_id];
                    tokens.extend(tokenizer.encode(&prompt_text));
                    let gen_start = Instant::now();
                    let mut gen_count = 0u64;

                    let on_tok = |tok_id: u32| {
                        gen_count += 1;
                        let text = tokenizer.decode(&[tok_id]);
                        let elapsed = gen_start.elapsed().as_secs_f64();
                        let tps = if elapsed > 0.0 { gen_count as f64 / elapsed } else { 0.0 };
                        let escaped = escape_json(&text);
                        let _ = write!(stream,
                            "data: {{\"token\":\"{escaped}\",\"tps\":{tps:.1}}}\n\n");
                        let _ = stream.flush();
                    };
                    if use_q4k {
                        let _ = forward_llama::generate(
                            model, &tokens, req_max, req_temp, req_rep,
                            tokenizer.eos_id, max_seq_len, on_tok,
                        );
                    } else {
                        let _ = InferenceState::generate(
                            model, &tokens, req_max, req_temp, req_rep,
                            tokenizer.eos_id, max_seq_len, on_tok,
                        );
                    }
                    let _ = write!(stream, "data: [DONE]\n\n");
                    let _ = stream.flush();
                }
            }
            _ => {
                let _ = write!(stream,
                    "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\
                     Connection: close\r\n\r\nNot Found");
            }
        }
    }
}

fn parse_content_length(req: &str) -> usize {
    for line in req.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("content-length:") {
            return line[15..].trim().parse().unwrap_or(0);
        }
    }
    0
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    if !after_colon.starts_with('"') { return None; }
    let content = &after_colon[1..];
    let mut result = String::new();
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(esc) = chars.next() {
                    match esc {
                        '"' => result.push('"'),
                        '\\' => result.push('\\'),
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        _ => { result.push('\\'); result.push(esc); }
                    }
                }
            }
            '"' => break,
            _ => result.push(c),
        }
    }
    Some(result)
}

fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_json_float(json: &str, key: &str) -> Option<f32> {
    extract_json_number(json, key).map(|v| v as f32)
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

const CHAT_HTML: &str = r##"<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cougar Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#eee;font-family:system-ui,-apple-system,sans-serif;height:100vh;display:flex;flex-direction:column}
#header{padding:12px 20px;border-bottom:1px solid #0f3460;display:flex;align-items:center;gap:16px}
#logo{font-family:monospace;font-size:13px;white-space:pre;color:#e94560;line-height:1.15}
#model-info{font-size:13px;color:#888}
#chat{flex:1;overflow-y:auto;padding:16px 20px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:75%;padding:10px 14px;border-radius:12px;font-size:14px;line-height:1.5;white-space:pre-wrap;word-wrap:break-word}
.msg.user{align-self:flex-end;background:#16213e;border-bottom-right-radius:4px}
.msg.assistant{align-self:flex-start;background:#0f3460;border-bottom-left-radius:4px}
#tps-bar{padding:6px 20px;display:none;align-items:center;gap:10px;font-size:13px;color:#888}
#tps-track{flex:1;height:6px;background:#16213e;border-radius:3px;overflow:hidden}
#tps-fill{height:100%;width:0%;background:#e94560;border-radius:3px;transition:width .3s ease}
#tps-label{min-width:90px;text-align:right}
#settings-toggle{padding:6px 20px}
#settings-toggle button{background:none;border:none;color:#e94560;cursor:pointer;font-size:13px}
#settings{display:none;padding:8px 20px 12px;gap:12px;flex-wrap:wrap}
#settings.open{display:flex}
.setting{display:flex;flex-direction:column;gap:4px;font-size:13px}
.setting label{color:#888}
.setting input[type=range]{width:160px;accent-color:#e94560}
.setting input[type=number]{width:80px;background:#16213e;color:#eee;border:1px solid #0f3460;border-radius:4px;padding:4px 8px}
#input-area{padding:12px 20px;border-top:1px solid #0f3460;display:flex;gap:10px}
#prompt{flex:1;background:#16213e;color:#eee;border:1px solid #0f3460;border-radius:8px;padding:10px 14px;font-size:14px;resize:none;min-height:44px;max-height:120px;font-family:inherit}
#prompt:focus{outline:none;border-color:#e94560}
#send{background:#e94560;color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;cursor:pointer;font-weight:600}
#send:disabled{opacity:.4;cursor:default}
</style></head><body>
<div id="header">
<div id="logo">  /\_/\
 ( o.o )  COUGAR
  > ^ <   BitNet b1.58</div>
<div id="model-info">loading model info...</div>
</div>
<div id="chat"></div>
<div id="tps-bar"><div id="tps-track"><div id="tps-fill"></div></div><div id="tps-label"></div></div>
<div id="settings-toggle"><button onclick="document.getElementById('settings').classList.toggle('open')">&#9881; Settings</button></div>
<div id="settings">
<div class="setting"><label>Temperature: <span id="temp-val">0.0</span></label><input type="range" id="temp" min="0" max="2" step="0.1" value="0" oninput="document.getElementById('temp-val').textContent=this.value"></div>
<div class="setting"><label>Rep. Penalty: <span id="rep-val">1.1</span></label><input type="range" id="rep" min="1" max="2" step="0.05" value="1.1" oninput="document.getElementById('rep-val').textContent=this.value"></div>
<div class="setting"><label>Max Tokens</label><input type="number" id="maxtok" value="128" min="1" max="4096"></div>
</div>
<div id="input-area">
<textarea id="prompt" rows="1" placeholder="Type a message..."></textarea>
<button id="send" onclick="sendMsg()">Send</button>
</div>
<script>
const chat=document.getElementById('chat'),tpsBar=document.getElementById('tps-bar'),
  tpsFill=document.getElementById('tps-fill'),tpsLabel=document.getElementById('tps-label'),
  promptEl=document.getElementById('prompt'),sendBtn=document.getElementById('send');
let generating=false;

fetch('/api/model').then(r=>r.json()).then(d=>{
  document.getElementById('model-info').textContent=
    d.layers+' layers | '+d.hidden_dim+'d | '+d.heads+' heads | '+d.vocab_size+' vocab';
}).catch(()=>{document.getElementById('model-info').textContent='model info unavailable';});

promptEl.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}
});

function addBubble(role,text){
  const d=document.createElement('div');d.className='msg '+role;d.textContent=text;
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;
}

async function sendMsg(){
  if(generating)return;
  const text=promptEl.value.trim();if(!text)return;
  promptEl.value='';sendBtn.disabled=true;generating=true;
  addBubble('user',text);
  const bubble=addBubble('assistant','');
  tpsBar.style.display='flex';tpsFill.style.width='0%';tpsLabel.textContent='';

  try{
    const res=await fetch('/api/generate',{method:'POST',body:JSON.stringify({
      prompt:text,
      temperature:parseFloat(document.getElementById('temp').value),
      repetition_penalty:parseFloat(document.getElementById('rep').value),
      max_tokens:parseInt(document.getElementById('maxtok').value)||128
    })});
    const reader=res.body.getReader(),dec=new TextDecoder();
    let buf='';
    while(true){
      const{done,value}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n');buf=lines.pop();
      for(const line of lines){
        if(!line.startsWith('data: '))continue;
        const payload=line.slice(6);
        if(payload==='[DONE]')continue;
        try{
          const ev=JSON.parse(payload);
          bubble.textContent+=ev.token;
          chat.scrollTop=chat.scrollHeight;
          if(ev.tps>0){
            const pct=Math.min(100,ev.tps/20*100);
            tpsFill.style.width=pct+'%';
            tpsLabel.textContent=ev.tps.toFixed(1)+' tok/s';
          }
        }catch(e){}
      }
    }
  }catch(e){bubble.textContent+=' [error: '+e.message+']';}
  generating=false;sendBtn.disabled=false;
  setTimeout(()=>{tpsBar.style.display='none';},2000);
}
</script></body></html>"##;
