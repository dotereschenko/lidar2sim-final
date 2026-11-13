export function initSimulationPanel({THREE,api,openJob,view,selectedJob}){
 const $=id=>document.getElementById(id);
 let catalog=null,overlay=null,animation=0,poll=0,active=null,lastScene=null,picking=null;
 const fail=error=>{$('sim-error').hidden=!error;$('sim-error').textContent=error?.message||error||'';};
 function clear(){
  cancelAnimationFrame(animation);animation=0;
  if(overlay){$('sim-status').textContent='Воспроизведение остановлено.';view().scene?.remove(overlay);overlay.traverse(x=>{x.geometry?.dispose();x.material?.dispose();});overlay=null;}
  if(view().renderer)view().renderer.clippingPlanes=[];
  $('sim-stop').disabled=true;
 }
 function customPoints(){
  const points=$('sim-points').value.trim().split(/\n/).filter(s=>s.trim()).map(s=>s.trim().split(/[ ,;]+/).map(Number));
  if(points.length<2||points.length>32||points.some(p=>p.length!==3||p.some(v=>!Number.isFinite(v))))throw new Error('Задайте 2–32 точки пола XYZ, по одной на строку.');
  return points;
 }
 function clip(){
  const renderer=view().renderer;if(!renderer)return;
  renderer.clippingPlanes=$('sim-cut').value===''?[]:[new THREE.Plane(new THREE.Vector3(0,0,-1),Number($('sim-cut').value))];
 }
 function sceneChanged(){
  const job=selectedJob(),ready=job?.status==='completed',isA=ready&&job.id===catalog?.scene?.id;
  if(job?.id!==lastScene){lastScene=job?.id;picking=null;$('sim-points').value='';$('sim-mode').value='custom';$('sim-cut').value='';clear();}
  $('sim-mode').options[1].disabled=!isA;
  $('sim-custom').hidden=$('sim-mode').value!=='custom';$('sim-preset').hidden=$('sim-mode').value==='custom';
  $('sim-preview').disabled=!ready;
  $('sim-run').disabled=!ready||!catalog?.runner_configured||!!active;
  for(const id of ['sim-pick-start','sim-pick-end','sim-pick-add'])$(id).disabled=!ready;
  $('sim-scene').textContent=ready?`Выбрана реконструкция ${job.id}. Новый прогон использует её collision mesh и заданный ниже маршрут.`:'Откройте завершённую реконструкцию. Маршрут принадлежит только выбранной сцене.';
 }
 function endpoints(){
  const points=catalog?.routes[Number($('sim-route').value)]?.points||[];
  for(const id of ['sim-start','sim-end']){
   $(id).replaceChildren();points.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1}: (${p[0].toFixed(2)}, ${p[1].toFixed(2)})` ;$(id).append(o);});
  }
  $('sim-end').value=String(Math.max(0,points.length-1));clear();
 }
 function selection(){
  const job=selectedJob();if(job?.status!=='completed')throw new Error('Откройте завершённую реконструкцию.');
  if($('sim-mode').value==='custom')return {scene_job:job.id,robot:'velocity_body',waypoints:customPoints()};
  const route=Number($('sim-route').value),start=Number($('sim-start').value),end=Number($('sim-end').value);
  if(job.id!==catalog?.scene?.id)throw new Error('Готовые маршруты относятся только к примеру A.');
  if(start>=end)throw new Error('Цель должна быть после старта');
  return {scene_job:job.id,robot:'velocity_body',route,start,end};
 }
 async function refresh(){
  try{
   fail('');catalog=await api('/api/simulation-scene');
   $('sim-run').textContent=catalog.backend==='newton'?'Запустить физику · Newton GPU':'Запустить Isaac Sim';
   $('sim-route').replaceChildren();catalog.routes.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.name;$('sim-route').append(o);});endpoints();
   $('sim-open').disabled=!catalog.scene;
   sceneChanged();
   $('sim-backend').textContent=catalog.runner_configured?`Сцена готова. Исполнитель: ${catalog.backend==='newton'?'Newton · CUDA, без RTX-видео':'Isaac Sim'}. Результат маршрута определяется физикой.`:catalog.reason;
   $('sim-archive').hidden=!catalog.archive;
   const history=await api('/api/simulations');$('sim-history').replaceChildren();
   for(const job of history){const b=document.createElement('button');b.textContent=`${job.id.slice(0,8)} · ${job.status} · сцена ${job.scene_job.slice(0,8)} · маршрут ${job.route+1}`;b.disabled=!!active;b.onclick=()=>watch(job.id).catch(fail);$('sim-history').append(b);}
   const running=history.find(j=>['queued','running'].includes(j.status));if(running&&!active)watch(running.id).catch(fail);
  }catch(e){fail(e);$('sim-run').disabled=true;}
 }
 async function preview(){
  const selected=selection();clear();clip();
  const {scene,renderer}=view();if(!renderer)throw new Error('Для предпросмотра нужен WebGL');
  const raw=selected.waypoints||catalog.routes[selected.route].points.slice(selected.start,selected.end+1).map(p=>[...p,.03]);
  const points=raw.map(p=>new THREE.Vector3(p[0],p[1],p[2]+.03));
  overlay=new THREE.Group();scene.add(overlay);
  overlay.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xb4ed86})));
  for(const [point,color] of [[points[0],0x7ce2ae],[points.at(-1),0xff7f75]]){const marker=new THREE.Mesh(new THREE.SphereGeometry(.08,12,8),new THREE.MeshBasicMaterial({color}));marker.position.copy(point);overlay.add(marker);}
  $('sim-stop').disabled=false;$('sim-status').textContent=`Заданный маршрут · сцена ${selected.scene_job.slice(0,8)}. Это точки пользователя, физика ещё не рассчитана.`;
  $('viewer').scrollIntoView({block:'center',behavior:'auto'});
 }
 async function watch(id){
  clear();const generation=++poll;active=id;$('sim-replay').hidden=true;$('sim-run').disabled=true;$('sim-video').pause();$('sim-video').hidden=$('sim-frame').hidden=true;$('sim-video-label').textContent=`Прогон ${id.slice(0,8)}`;for(const b of $('sim-history').querySelectorAll('button'))b.disabled=true;
  while(generation===poll){
   const job=await api('/api/simulations/'+id);if(generation!==poll)return;$('sim-status').textContent=({queued:'Ожидание физического запуска',running:'Выполняется физическая симуляция',completed:job.result?.outcome==='reached'?'Цель достигнута в физическом прогоне.':'Прогон завершён, цель НЕ достигнута: '+job.result?.outcome,failed:'Ошибка физического запуска',cancelled:'Запуск отменён'})[job.status];
   $('sim-video-label').textContent=`Прогон ${id.slice(0,8)} · сцена ${job.scene_job} · ${job.backend}`;fail(job.error);$('sim-log').textContent=job.log_tail||'';$('sim-cancel').hidden=!['queued','running'].includes(job.status);
   const root=`/api/simulations/${id}/files/`;
   const frame=job.files.includes('preview.jpg')?'preview.jpg':job.files.includes('overview.png')?'overview.png':null;if(frame){$('sim-frame').src=root+frame+'?t='+Date.now();$('sim-frame').hidden=false;}
   $('sim-files').replaceChildren();for(const name of job.files){const a=document.createElement('a');a.textContent='↓ '+name;a.href=root+name;a.download=name;$('sim-files').append(a);}
   if(!['queued','running'].includes(job.status)){
    if(job.status==='completed'){if(job.files.includes('simulation.mp4')){$('sim-video').src=root+'simulation.mp4';$('sim-video').hidden=false;}if(job.backend==='newton'){$('sim-replay').hidden=false;$('sim-replay').onclick=()=>replay(job).catch(fail);}$('sim-video-label').textContent=`Запись ${job.backend} · прогон ${id.slice(0,8)} · сцена ${job.scene_job} · маршрут ${job.route+1}, точки ${job.start+1}–${job.end+1} · ${({reached:'Цель достигнута',blocked:'Движение заблокировано',fell:'Проваливание',timeout:'Цель не достигнута за отведённое время'})[job.result?.outcome]||'См. отчёт'}`;}
    active=null;for(const b of $('sim-history').querySelectorAll('button'))b.disabled=false;sceneChanged();return;
   }
   await new Promise(r=>setTimeout(r,2000));
  }
 }
 async function replay(job){
  clear();fail('');const record=await api(`/api/simulations/${job.id}/files/recording.json`);
  await openJob(job.scene_job);
  if(selectedJob()?.id!==job.scene_job)throw new Error('Не удалось открыть сцену записи. Воспроизведение отменено.');
  const {scene,camera,controls,renderer}=view();if(!renderer)throw new Error('Для воспроизведения нужен WebGL');
  const poses=record.trajectory,points=poses.map(p=>new THREE.Vector3(...p.position));
  overlay=new THREE.Group();scene.add(overlay);
  overlay.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xff983e})));
  const robot=new THREE.Group(), size=record.body_size_m;overlay.add(robot);
  const material=()=>new THREE.MeshStandardMaterial({color:0xff983e});
  if(record.body_shape==='box_with_spherical_skids'){
   const r=Math.min(...size)*.2,box=new THREE.Mesh(new THREE.BoxGeometry(size[0],size[1],size[2]-r),material());box.position.z=r/2;robot.add(box);
   for(const x of [-size[0]/2+r,size[0]/2-r])for(const y of [-size[1]/2+r,size[1]/2-r]){const skid=new THREE.Mesh(new THREE.SphereGeometry(r,16,12),material());skid.position.set(x,y,-size[2]/2+r);robot.add(skid);}
  }else robot.add(new THREE.Mesh(new THREE.BoxGeometry(...size),material()));
  robot.position.copy(points[0]);camera.position.copy(points[0]).add(new THREE.Vector3(4,-6,6));controls.target.copy(points[0]);controls.update();
  $('sim-cut').value=String(poses[0].position[2]+1.3);clip();
  const began=performance.now();let index=0;$('sim-stop').disabled=false;
  function tick(now){
   if(!overlay)return;const t=(now-began)/1000;
   while(index<poses.length-1&&poses[index+1].t<=t)index++;
   const pose=poses[index],previous=robot.position.clone();robot.position.copy(points[index]);robot.quaternion.set(...pose.quaternion_xyzw);
   camera.position.add(robot.position.clone().sub(previous));controls.target.copy(robot.position);
   $('sim-status').textContent=`Запись физики Newton · сцена ${job.scene_job.slice(0,8)} · ${pose.t.toFixed(2)} / ${record.duration_s.toFixed(2)} с · контактов: ${pose.contacts} · результат: ${record.outcome}. Измеренные состояния, не прямой эфир.`;
   if(index<poses.length-1)animation=requestAnimationFrame(tick);else $('sim-stop').disabled=true;
  }
  animation=requestAnimationFrame(tick);$('viewer').scrollIntoView({block:'center',behavior:'auto'});
 }
 $('sim-open').onclick=()=>{clear();openJob(catalog.scene.id).catch(fail);};
 $('sim-preview').onclick=()=>preview().catch(fail);$('sim-stop').onclick=()=>{clear();$('sim-status').textContent='Воспроизведение остановлено.';};
 $('sim-mode').onchange=()=>{clear();sceneChanged();};
 $('sim-cut').oninput=clip;
 $('sim-route').onchange=endpoints;for(const id of ['sim-start','sim-end'])$(id).onchange=clear;
 $('sim-refresh').onclick=refresh;
 $('sim-run').onclick=async()=>{try{const params=selection();clear();$('sim-run').disabled=true;fail('');$('sim-replay').hidden=true;$('sim-files').replaceChildren();$('sim-log').textContent='';$('sim-video').hidden=$('sim-frame').hidden=true;$('sim-video-label').textContent='Проверяем маршрут перед запуском…';const job=await api('/api/simulations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params)});await watch(job.id);}catch(e){active=null;fail(e);$('sim-status').textContent='Физический прогон не запущен или не завершён. См. причину.';$('sim-video-label').textContent='Новая запись не получена.';sceneChanged();}};
 $('sim-cancel').onclick=async()=>{try{if(active)await api(`/api/simulations/${active}/cancel`,{method:'POST'});}catch(e){fail(e);}};
 for(const mode of ['start','end','add'])$('sim-pick-'+mode).onclick=()=>{picking=mode;fail('');$('sim-status').textContent='Нажмите на пол в 3D. Поворот мышью не выбирает точку.';$('viewer').scrollIntoView({block:'center'});};
 let down=null;
 $('viewer').addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
 $('viewer').addEventListener('pointerup',e=>{
  if(!picking||!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;
  try{
   const {renderer,camera,object}=view();if(!object?.isMesh)throw new Error('Переключитесь на слой Collision или Visual для выбора пола.');
   const rect=renderer.domElement.getBoundingClientRect(),ray=new THREE.Raycaster();
   ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
   const hit=ray.intersectObject(object,false).find(h=>$('sim-cut').value===''||h.point.z<=Number($('sim-cut').value));
   if(!hit||Math.abs(hit.face.normal.z)<.85)throw new Error('Выберите горизонтальную поверхность пола, не стену. При необходимости используйте срез сверху.');
   const row=hit.point.toArray().map(v=>v.toFixed(4)).join(' '),rows=$('sim-points').value.trim().split(/\n/).filter(s=>s.trim());
   if(picking==='start'){if(rows.length)rows[0]=row;else rows.push(row);}
   else if(picking==='end'&&rows.length>1)rows[rows.length-1]=row;else rows.push(row);
   $('sim-points').value=rows.join('\n');picking=null;fail('');$('sim-status').textContent='Точка выбрана. Проверьте порядок точек и запустите проверку с физикой.';
   if(rows.length>=2)preview().catch(fail);
  }catch(e){fail(e);}
 });
 refresh();return {clear,sceneChanged};
}
