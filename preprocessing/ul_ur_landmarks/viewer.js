'use strict';
const $ = id => document.getElementById(id);
const VIEWS = [['CAM_AV','AV'],['CAM_UL','UL'],['CAM_UR','UR'],['CAM_LL','LL'],['CAM_LR','LR']];
const VIEW_MODES = {all:VIEWS.map(([v])=>v),aerial:['CAM_AV'],upper:['CAM_UL','CAM_UR'],lower:['CAM_LL','CAM_LR']};
const MP_BODY_EDGES = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]];
const BODY_EDGES = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const HAND_EDGES = [1,5,9,13,17].flatMap(i => [[0,i],[i,i+1],[i+1,i+2],[i+2,i+3]]);
const BODY_NAMES = ['nose','left eye','right eye','left ear','right ear','left shoulder','right shoulder','left elbow','right elbow','left wrist','right wrist','left hip','right hip','left knee','right knee','left ankle','right ankle'];
const COLORS = {body:'#f2c66d',face:'#96dc82',participant:'#58d9ef',other_actor:'#f190cf',unknown:'#ffab70',objects:'#c4afff'};
let pairs=[], info=null, current=null, frame=0, playing=false, clipRevision=0, frameRevision=0, controller=null;
let images={}, hitPoints=Object.fromEntries(VIEWS.map(([v])=>[v,[]])), playTimer=null, playGeneration=0;
const on = name => $(name).checked;
const error = e => { if(e.name!=='AbortError'){stop();$('status').textContent=e.message;$('status').classList.add('error');} };
async function api(path,signal){const response=await fetch(path,{signal});if(!response.ok)throw new Error(`Could not load view (${response.status}). Check that the selected outputs and source video are available.`);return response.json();}
function option(value,text){const e=document.createElement('option');e.value=value;e.textContent=text;return e;}
function stop(){playing=false;++playGeneration;clearTimeout(playTimer);$('play').textContent='Play';}
function filterPairs(){
  stop();const previous=$('pair').value, text=$('search').value.toLowerCase();
  const chosen=pairs.filter(p=>(!$('participant').value||p.participant===$('participant').value)&&(!$('task').value||p.task===$('task').value)&&p.pair_id.toLowerCase().includes(text));
  $('matches').textContent=`${chosen.length.toLocaleString()} matching clips${chosen.length>250?' · first 250 shown':''}`;
  $('pair').replaceChildren(...chosen.slice(0,250).map(p=>option(p.id,p.pair_id)));
  if(chosen.slice(0,250).some(p=>String(p.id)===previous))$('pair').value=previous;
  if(chosen.length)loadPair().catch(error);
  else{++clipRevision;controller?.abort();info=null;current=null;$('play').disabled=true;$('pairName').textContent='No matching clips';drawAll();}
}
async function loadPair(initialFrame=0){
  stop();const revision=++clipRevision;controller?.abort();$('play').disabled=true;info=null;current=null;
  const id=$('pair').value;if(id==='')return;
  $('status').classList.remove('error');$('status').textContent='Checking source identity and video timing…';
  const next=await api(`/api/clip?id=${id}`);if(revision!==clipRevision)return;
  info=next;current=null;images={};$('pairName').textContent=info.pair_id;$('aerialStatus').textContent=info.aerial_status;
  if(!VIEW_MODES[$('viewMode').value].some(v=>info.views[v])){$('viewMode').value='all';viewMode();}
  const last=info.views[info.primary_view].frames-1;$('scrub').max=last;$('frameNumber').max=last;
  $('play').disabled=false;await showFrame(Math.min(initialFrame,last));
}
async function showFrame(index){
  if(!info)return;const id=$('pair').value, clipToken=clipRevision, token=++frameRevision;
  controller?.abort();controller=new AbortController();
  index=Math.max(0,Math.min(info.views[info.primary_view].frames-1,Math.round(index)));
  const data=await api(`/api/frame?id=${id}&frame=${index}&image=${on('background')?1:0}`,controller.signal);
  const loaded={};
  await Promise.all(VIEWS.map(async ([view])=>{if(data[view]?.image){const img=new Image();img.src=data[view].image;await img.decode();loaded[view]=img;}}));
  if(token!==frameRevision||clipToken!==clipRevision)return;
  frame=index;current=data;images=loaded;$('scrub').value=index;$('frameNumber').value=index;
  const u=new URL(location.href);u.searchParams.set('pair',id);u.searchParams.set('frame',index);history.replaceState(null,'',u);
  $('status').classList.remove('error');$('status').textContent=`Frame ${index+1} of ${info.views[info.primary_view].frames} · ${info.pair_id}`;
  drawAll();
}
function drawPoints(ctx, group, start, end, edges, color, label, view, radius=3){
  ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;
  for(const [a,b] of edges){const x=group.xy[start+a],y=group.xy[start+b];if(x&&y&&group.valid[start+a]&&group.valid[start+b]){ctx.beginPath();ctx.moveTo(...x);ctx.lineTo(...y);ctx.stroke();}}
  for(let i=start;i<end;i++){const p=group.xy[i];if(!p||!group.valid[i])continue;ctx.beginPath();ctx.arc(p[0],p[1],radius,0,Math.PI*2);ctx.fill();hitPoints[view].push({xy:p,text:`${label} · point ${i-start}${label==='Body'?' ('+BODY_NAMES[i]+')':''} · (${p[0].toFixed(1)}, ${p[1].toFixed(1)}) px · score ${group.confidence[i]?.toFixed(3)??'unavailable'}`});}
}
function box(ctx,xy,color,label,dashed=false){if(!xy)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dashed?[9,5]:[]);ctx.strokeRect(xy[0],xy[1],xy[2]-xy[0],xy[3]-xy[1]);ctx.setLineDash([]);if(on('labels')){ctx.font='18px system-ui';const width=ctx.measureText(label).width;const y=Math.max(22,xy[1]);ctx.fillStyle='#080c12dc';ctx.fillRect(xy[0],y-22,width+10,24);ctx.fillStyle=color;ctx.fillText(label,xy[0]+5,y-4);}}
function draw(view,suffix){
  const canvas=$('canvas'+suffix),ctx=canvas.getContext('2d'),data=current?.[view];hitPoints[view]=[];
  if(data){canvas.width=data.width;canvas.height=data.height;}
  ctx.fillStyle='#080c12';ctx.fillRect(0,0,canvas.width,canvas.height);
  if(!data){$('time'+suffix).textContent='No matched frame';$('stats'+suffix).textContent=info?.view_errors?.[view]||(view==='CAM_AV'?(info?.aerial_status||'No aerial frame loaded.'):'No frame at this relative timestamp.');return;}
  if(on('background')&&images[view])ctx.drawImage(images[view],0,0,canvas.width,canvas.height);
  if(on('objects'))for(const o of data.objects){if(['lefthand','righthand','surrogate_hand'].includes(o.class_name))continue;box(ctx,o.bbox,COLORS.objects,`${o.class_name}${o.confidence==null?'':' '+o.confidence.toFixed(2)}`);}
  if(data.pose_topology==='mediapipe_pose_33'){
    if(on('body'))drawPoints(ctx,data.pose,0,33,MP_BODY_EDGES,COLORS.body,'Body · MediaPipe 33',view,4);
  }else if(data.pose){
  if(on('body'))drawPoints(ctx,data.pose,0,17,BODY_EDGES,COLORS.body,'Body',view,4);
  if(on('feet'))drawPoints(ctx,data.pose,17,23,[],COLORS.body,'Feet',view);
  if(on('coarse'))drawPoints(ctx,data.pose,23,91,[],COLORS.face,'Coarse face · COCO 68',view,2);
  if(on('native'))for(const start of [91,112])drawPoints(ctx,data.pose,start,start+21,HAND_EDGES,'#ddedff',`Native ${start===91?'left':'right'} hand · RTMW`,view,2);
  }
  if(data.face&&on('face'))drawPoints(ctx,data.face,0,478,[],COLORS.face,'Dense face · MediaPipe 478',view,1.7);
  for(const h of data.hands){const color=COLORS[h.actor],native=h.model_id==='wholebody_native_fallback';const actor=h.actor==='other_actor'?'surrogate':h.actor;const label=`${actor}${h.track_id==null?'':' · T'+h.track_id} · ${h.handedness}${native?' · native':''}${h.model_id==='legacy_av'?' · AV':''}`;if(on('boxes'))box(ctx,h.bbox,color,label,native);if(on('hands'))drawPoints(ctx,h,0,21,HAND_EDGES,color,`${label} · ${h.topology} · ${h.model_id}`,view,4);}
  if(on('gaze')&&data.gaze){const [x,y]=data.gaze;ctx.strokeStyle='#ffdc62';ctx.lineWidth=3;ctx.beginPath();ctx.arc(x,y,12,0,Math.PI*2);ctx.moveTo(x-19,y);ctx.lineTo(x+19,y);ctx.moveTo(x,y-19);ctx.lineTo(x,y+19);ctx.stroke();hitPoints[view].push({xy:data.gaze,text:`AV gaze · (${x.toFixed(1)}, ${y.toFixed(1)}) px · legacy normalized gaze; confidence unavailable`});}
  $('time'+suffix).textContent=`frame ${data.frame_index} · ${(data.timestamp_ms/1000).toFixed(3)} s`;
  const objects=data.objects.filter(o=>!['lefthand','righthand','surrogate_hand'].includes(o.class_name));
  $('stats'+suffix).textContent=data.kind==='legacy_lower'?`${data.pose.valid.filter(Boolean).length}/33 body · ${data.face.valid.filter(Boolean).length}/478 face · ${data.hands.length} hands (actor unknown) · legacy MediaPipe`:data.kind==='legacy_aerial'?`${data.hands.filter(h=>h.actor==='participant').length} participant hands · ${data.hands.filter(h=>h.actor==='other_actor').length} surrogate boxes · ${objects.length} stored detection boxes · gaze ${data.gaze?'present':'absent/unknown'}`:`${data.pose.valid.slice(0,17).filter(Boolean).length}/17 body · ${data.face.valid.filter(Boolean).length}/478 face · ${data.hands.filter(h=>h.actor==='participant').length} participant / ${data.hands.filter(h=>h.actor==='other_actor').length} surrogate hands · ${objects.length} LEGO/assembly boxes`;
}
function drawAll(){for(const [view,suffix] of VIEWS)draw(view,suffix);}
function viewMode(){const mode=$('viewMode').value;$('views').dataset.mode=mode;for(const [view,suffix] of VIEWS)$('view'+suffix).hidden=!VIEW_MODES[mode].includes(view);}
async function tick(generation=playGeneration){
  if(!playing||!info||generation!==playGeneration)return;
  if(frame>=info.views[info.primary_view].frames-1){stop();return;}
  const start=performance.now(),delta=info.views[info.primary_view].timestamps_ms[frame+1]-info.views[info.primary_view].timestamps_ms[frame];
  try{await showFrame(frame+1);}catch(e){error(e);return;}
  if(playing&&generation===playGeneration)playTimer=setTimeout(()=>tick(generation),Math.max(0,delta/Number($('speed').value)-(performance.now()-start)));
}
function togglePlay(){if(!info)return;if(playing){stop();return;}playing=true;$('play').textContent='Pause';if(frame===info.views[info.primary_view].frames-1){showFrame(0).then(tick).catch(error);}else tick();}
function step(amount){stop();showFrame(frame+amount).catch(error);}
function saveImage(){
  if(!current)return;const selected=VIEWS.filter(([,suffix])=>!$('view'+suffix).hidden);
  const out=document.createElement('canvas');out.width=selected.reduce((n,[,suffix])=>n+$('canvas'+suffix).width,0);
  out.height=Math.max(...selected.map(([,suffix])=>$('canvas'+suffix).height))+52;
  const ctx=out.getContext('2d');ctx.fillStyle='#10151d';ctx.fillRect(0,0,out.width,out.height);ctx.font='20px system-ui';ctx.fillStyle='#e8edf5';
  let x=0;for(const [view,suffix] of selected){const c=$('canvas'+suffix);ctx.fillText(`QUB-PHEO · ${view} · ${$('time'+suffix).textContent}`,x+16,32);ctx.drawImage(c,x,52);x+=c.width;}
  const a=document.createElement('a');a.download=`qub-pheo_views${$('pair').value}_frame${frame}.png`;a.href=out.toDataURL('image/png');a.click();
}
async function coverage(){try{const r=await api('/api/quality');$('coverage').textContent=`${r.audited_clips.toLocaleString()} / ${r.expected_clips.toLocaleString()} clips audited. `+Object.entries(r.views).map(([v,d])=>`${v}: body predictions on ${(100*d.body_prediction_frame_fraction).toFixed(1)}% of frames; dense face on ${(100*d.dense_face_prediction_frame_fraction).toFixed(1)}%.`).join(' ');}catch{$('coverage').textContent='The collection-wide coverage audit is not available yet.';}}
for(const id of ['participant','task'])$(id).addEventListener('change',filterPairs);
$('search').addEventListener('input',filterPairs);$('pair').addEventListener('change',()=>loadPair().catch(error));
$('previous').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('play').onclick=togglePlay;$('save').onclick=saveImage;
for(const id of ['scrub','frameNumber'])$(id).addEventListener('input',()=>{stop();showFrame(Number($(id).value)).catch(error);});
for(const input of document.querySelectorAll('.layers input'))input.addEventListener('change',()=>{if(input.id==='background'){stop();showFrame(frame).catch(error);}else drawAll();});
$('viewMode').addEventListener('change',viewMode);
for(const [view,suffix] of VIEWS)$('canvas'+suffix).addEventListener('click',event=>{const c=event.currentTarget,r=c.getBoundingClientRect(),x=(event.clientX-r.left)*c.width/r.width,y=(event.clientY-r.top)*c.height/r.height;const chosen=hitPoints[view].map(p=>({...p,d:Math.hypot(p.xy[0]-x,p.xy[1]-y)})).sort((a,b)=>a.d-b.d)[0];$('inspect').textContent=chosen&&chosen.d<20?`${view} · ${chosen.text}`:'No displayed landmark near this position.';});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','BUTTON'].includes(event.target.tagName))return;if(event.code==='Space'){event.preventDefault();togglePlay();}else if(event.code==='ArrowRight'){event.preventDefault();step(1);}else if(event.code==='ArrowLeft'){event.preventDefault();step(-1);}});
async function start(){const index=await api('/api/index');pairs=index.pairs;$('modelLabel').textContent=index.model_label||(index.hand_refinement?'Refined collection · 2D source pixels':'Collection · 2D source pixels');for(const [id,key] of [['participant','participant'],['task','task']])for(const value of [...new Set(pairs.map(p=>p[key]))].sort())$(id).append(option(value,value));const params=new URLSearchParams(location.search),chosen=pairs.find(p=>String(p.id)===params.get('pair'))||pairs[0];if(chosen){if(chosen.views?.length===1 && chosen.views[0]==='CAM_AV'){$('viewMode').value='aerial';viewMode();}$('participant').value=chosen.participant;$('task').value=chosen.task;const selected=pairs.filter(p=>p.participant===chosen.participant&&p.task===chosen.task);$('pair').replaceChildren(...selected.map(p=>option(p.id,p.pair_id)));$('pair').value=chosen.id;$('matches').textContent=`${selected.length} matching clips`;await loadPair(Number(params.get('frame'))||0);}await coverage();}
start().catch(error);
