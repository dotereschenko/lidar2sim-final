export function initSimulationPanel({THREE,api,openJob,view}){
 const $=id=>document.getElementById(id);
 let catalog=null,overlay=null,animation=0,poll=0,active=null;
 const fail=error=>{$('sim-error').hidden=!error;$('sim-error').textContent=error?.message||error||'';};
 function clear(){
  cancelAnimationFrame(animation);animation=0;
  if(overlay){view().scene?.remove(overlay);overlay.traverse(x=>{x.geometry?.dispose();x.material?.dispose();});overlay=null;}
  if(view().renderer)view().renderer.clippingPlanes=[];
  $('sim-stop').disabled=true;
 }
 function endpoints(){
  const points=catalog?.routes[Number($('sim-route').value)]?.points||[];
  for(const id of ['sim-start','sim-end']){
   $(id).replaceChildren();points.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}: (${p[0].toFixed(2)}, ${p[1].toFixed(2)})` ;$(id).append(o);});
  }
  $('sim-end').value=String(Math.max(0,points.length-1));clear();
 }
 function selection(){
  const route=Number($('sim-route').value),start=Number($('sim-start').value),end=Number($('sim-end').value);
  if(!catalog?.scene)throw new Error('Сцена A ещё не подготовлена');
  if(!Number.isInteger(start)||!Number.isInteger(end)||start>=end)throw new Error('Цель должна быть после старта');
  return {robot:$('sim-robot').value,route,start,end};
 }
 async function refresh(){
  try{
   fail('');catalog=await api('/api/simulation-scene');
   $('sim-run').textContent=catalog.backend==='newton'?'Запустить физику · Newton GPU':'Запустить Isaac Sim';
   $('sim-route').replaceChildren();catalog.routes.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.name;$('sim-route').append(o);});endpoints();
   $('sim-open').disabled=$('sim-preview').disabled=!catalog.scene;
   $('sim-run').disabled=!catalog.scene||!catalog.runner_configured||!!active;
   $('sim-backend').textContent=!catalog.scene?'Сцена A ещё не подготовлена.':catalog.runner_configured?`Сцена готова. Исполнитель: ${catalog.backend==='newton'?'Newton · CUDA, без RTX-видео':'Isaac Sim'}. Результат маршрута определяется физикой.`:catalog.reason;
   $('sim-archive').hidden=!catalog.archive;
   const history=await api('/api/simulations');$('sim-history').replaceChildren();
   for(const job of history){const b=document.createElement('button');b.textContent=`${job.id.slice(0,8)} · ${job.status} · маршрут ${job.route+1}`;b.disabled=!!active;b.onclick=()=>watch(job.id).catch(fail);$('sim-history').append(b);}
   const running=history.find(j=>['queued','running'].includes(j.status));if(running&&!active)watch(running.id).catch(fail);
  }catch(e){fail(e);$('sim-run').disabled=true;}
 }
 async function preview(){
  const selected=selection();clear();await openJob(catalog.scene.id);
  const {scene,camera,controls,renderer}=view();if(!renderer)throw new Error('Для предпросмотра нужен WebGL');
  const points=catalog.routes[selected.route].points.slice(selected.start,selected.end+1).map(p=>new THREE.Vector3(p[0],p[1],.24));
  overlay=new THREE.Group();scene.add(overlay);
  overlay.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xb4ed86})));
  const robot=new THREE.Group();overlay.add(robot);
  robot.add(new THREE.Mesh(new THREE.BoxGeometry(...catalog.body.size),new THREE.MeshStandardMaterial({color:0xff983e})));
  for(const y of [-.23,.23])for(const x of [-.13,.13]){
   const wheel=new THREE.Mesh(new THREE.CylinderGeometry(.09,.09,.055,16),new THREE.MeshStandardMaterial({color:0x20242b}));wheel.position.set(x,y,-.12);robot.add(wheel);
  }
  for(const [point,color] of [[points[0],0x7ce2ae],[points.at(-1),0xff7f75]]){const marker=new THREE.Mesh(new THREE.SphereGeometry(.13,12,8),new THREE.MeshBasicMaterial({color}));marker.position.copy(point);overlay.add(marker);}
  robot.position.copy(points[0]);controls.target.copy(points[0]);camera.position.copy(points[0]).add(new THREE.Vector3(4,-6,6));controls.update();
  renderer.clippingPlanes=[new THREE.Plane(new THREE.Vector3(0,0,-1),1.6)];
  const lengths=points.slice(1).map((p,i)=>p.distanceTo(points[i])),total=lengths.reduce((a,b)=>a+b,0),began=performance.now();
  $('sim-stop').disabled=false;
  function tick(now){
   if(!overlay)return;let distance=(now-began)/1000*catalog.speed*4,remaining=distance,index=0;
   while(index<lengths.length-1&&remaining>lengths[index])remaining-=lengths[index++];
   const previous=robot.position.clone();const t=Math.min(1,remaining/Math.max(lengths[index],1e-9));robot.position.lerpVectors(points[index],points[index+1],t);camera.position.add(robot.position.clone().sub(previous));controls.target.copy(robot.position);
   const delta=points[index+1].clone().sub(points[index]);robot.rotation.z=Math.atan2(delta.y,delta.x);
   $('sim-status').textContent=`Предпросмотр ×4 · ${Math.min(distance,total).toFixed(1)} / ${total.toFixed(1)} м. Без физики и проверки столкновений. Срез стен: 1,6 м.`;
   if(distance<total)animation=requestAnimationFrame(tick);else $('sim-stop').disabled=true;
  }
  animation=requestAnimationFrame(tick);$('viewer').scrollIntoView({block:'center',behavior:'auto'});
 }
 async function watch(id){
  const generation=++poll;active=id;$('sim-replay').hidden=true;$('sim-run').disabled=true;$('sim-video').pause();$('sim-video').hidden=$('sim-frame').hidden=true;$('sim-video-label').textContent=`Прогон ${id.slice(0,8)}`;for(const b of $('sim-history').querySelectorAll('button'))b.disabled=true;
  while(generation===poll){
   const job=await api('/api/simulations/'+id);$('sim-status').textContent=({queued:'Ожидание физического запуска',running:'Выполняется физическая симуляция',completed:'Физический прогон записан. См. результат маршрута в отчёте.',failed:'Ошибка физического запуска',cancelled:'Запуск отменён'})[job.status];
   fail(job.error);$('sim-log').textContent=job.log_tail||'';$('sim-cancel').hidden=!['queued','running'].includes(job.status);
   const root=`/api/simulations/${id}/files/`;
   const frame=job.files.includes('preview.jpg')?'preview.jpg':job.files.includes('overview.png')?'overview.png':null;if(frame){$('sim-frame').src=root+frame+'?t='+Date.now();$('sim-frame').hidden=false;}
   $('sim-files').replaceChildren();for(const name of job.files){const a=document.createElement('a');a.textContent='↓ '+name;a.href=root+name;a.download=name;$('sim-files').append(a);}
   if(!['queued','running'].includes(job.status)){
    if(job.status==='completed'){if(job.files.includes('simulation.mp4')){$('sim-video').src=root+'simulation.mp4';$('sim-video').hidden=false;}if(job.backend==='newton'){$('sim-replay').hidden=false;$('sim-replay').onclick=()=>replay(job).catch(fail);}$('sim-video-label').textContent=`Запись выбранного прогона · ${id.slice(0,8)} · ${({reached:'Цель достигнута',blocked:'Движение заблокировано',fell:'Проваливание',timeout:'Цель не достигнута за отведённое время'})[job.result?.outcome]||'См. отчёт'}`;}
    active=null;for(const b of $('sim-history').querySelectorAll('button'))b.disabled=false;$('sim-run').disabled=!catalog?.runner_configured;return;
   }
   await new Promise(r=>setTimeout(r,2000));
  }
 }
 async function replay(job){
  clear();fail('');const record=await api(`/api/simulations/${job.id}/files/recording.json`);
  await openJob(job.scene_job);
  const {scene,camera,controls,renderer}=view();if(!renderer)throw new Error('Для воспроизведения нужен WebGL');
  const poses=record.trajectory,points=poses.map(p=>new THREE.Vector3(...p.position));
  overlay=new THREE.Group();scene.add(overlay);
  overlay.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xff983e})));
  const robot=new THREE.Mesh(new THREE.BoxGeometry(...record.body_size_m),new THREE.MeshStandardMaterial({color:0xff983e}));overlay.add(robot);
  robot.position.copy(points[0]);camera.position.copy(points[0]).add(new THREE.Vector3(4,-6,6));controls.target.copy(points[0]);controls.update();
  renderer.clippingPlanes=[new THREE.Plane(new THREE.Vector3(0,0,-1),1.6)];
  const began=performance.now();let index=0;$('sim-stop').disabled=false;
  function tick(now){
   if(!overlay)return;const t=(now-began)/1000;
   while(index<poses.length-1&&poses[index+1].t<=t)index++;
   const pose=poses[index],previous=robot.position.clone();robot.position.copy(points[index]);robot.quaternion.set(...pose.quaternion_xyzw);
   camera.position.add(robot.position.clone().sub(previous));controls.target.copy(robot.position);
   $('sim-status').textContent=`Запись физики Newton · ${pose.t.toFixed(2)} / ${record.duration_s.toFixed(2)} с · контактов: ${pose.contacts} · результат: ${record.outcome}. Измеренные состояния, не прямой эфир.`;
   if(index<poses.length-1)animation=requestAnimationFrame(tick);else $('sim-stop').disabled=true;
  }
  animation=requestAnimationFrame(tick);$('viewer').scrollIntoView({block:'center',behavior:'auto'});
 }
 $('sim-open').onclick=()=>{clear();openJob(catalog.scene.id).catch(fail);};
 $('sim-preview').onclick=()=>preview().catch(fail);$('sim-stop').onclick=()=>{clear();$('sim-status').textContent='Воспроизведение остановлено.';};
 $('sim-route').onchange=endpoints;for(const id of ['sim-start','sim-end'])$(id).onchange=clear;
 $('sim-refresh').onclick=refresh;
 $('sim-run').onclick=async()=>{try{const params=selection();clear();$('sim-run').disabled=true;fail('');$('sim-video').hidden=$('sim-frame').hidden=true;$('sim-video-label').textContent='Ожидаем запись нового прогона…';const job=await api('/api/simulations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});await watch(job.id);}catch(e){active=null;fail(e);$('sim-run').disabled=!catalog?.runner_configured;}};
 $('sim-cancel').onclick=async()=>{try{if(active)await api(`/api/simulations/${active}/cancel`,{method:'POST'});}catch(e){fail(e);}};
 refresh();return clear;
}
