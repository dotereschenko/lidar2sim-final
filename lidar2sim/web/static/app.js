import {initSimulationPanel} from '/simulation.js';
import * as THREE from 'three';
import {PLYLoader} from '/vendor/PLYLoader.js';
import {OrbitControls} from '/vendor/OrbitControls.js';
const $ = id => document.getElementById(id);
const labels = {uploaded:'Файл загружен',queued:'В очереди — ожидает свободного обработчика',running:'Реконструкция и экспорт…',completed:'Обработка завершена',failed:'Ошибка обработки',cancelled:'Обработка отменена'};
const stages={loading:'Чтение облака…',reconstructing:'Реконструкция…',normals:'Подготовка и ориентация нормалей…',nksr:'Нейросетевая реконструкция NKSR…',collision:'Подготовка сеток и коллизий…',exporting:'Экспорт и проверка файлов…'};
let current=null, localBuffer=null, object=null, camera, controls, renderer, scene, viewVersion=0, pollVersion=0;
let clearSimulation=()=>{};
const loader=new PLYLoader();
function error(message){$('error').textContent=message; $('error').hidden=!message;}
try {
 renderer=new THREE.WebGLRenderer({antialias:true,alpha:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); $('viewer').prepend(renderer.domElement);
 scene=new THREE.Scene(); camera=new THREE.PerspectiveCamera(45,1,.01,10000); camera.up.set(0,0,1);
 controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
 scene.add(new THREE.HemisphereLight(0xd9edff,0x334455,2)); const light=new THREE.DirectionalLight(0xffffff,2.3);light.position.set(10,-15,25);scene.add(light);
 new ResizeObserver(()=>{const w=$('viewer').clientWidth,h=$('viewer').clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe($('viewer'));
 renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
} catch(e){error('3D-просмотр недоступен: '+e.message+'. Обработка и скачивание остаются доступны.');}
function clearView(){clearSimulation();if(object){scene.remove(object);object.geometry.dispose();object.material.dispose();object=null;}$('empty').hidden=false;}
function fit(){if(!object||!camera)return; const box=new THREE.Box3().setFromObject(object),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()).length()||1;controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(size*.65,-size*.85,size*.65));camera.near=Math.max(size/10000,.0001);camera.far=size*100;camera.updateProjectionMatrix();controls.update();}
function show(buffer, points, local=false){
 if(!renderer)return;
 let geometry=loader.parse(buffer), pos=geometry.getAttribute('position');
 if(!pos||!pos.count)throw new Error('В PLY нет точек');
 if(points){const stride=Math.max(1,Math.ceil(pos.count/100000)), vertices=[];for(let i=0;i<pos.count;i+=stride)vertices.push(pos.getX(i),pos.getY(i),pos.getZ(i));geometry.dispose();geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));}
 if(local){geometry.scale(Number($('scale').value),Number($('scale').value),Number($('scale').value));if($('up').value==='Y')geometry.rotateX(Math.PI/2);}
 geometry.computeBoundingSphere();const radius=geometry.boundingSphere.radius;
 if(!Number.isFinite(radius)) {geometry.dispose();throw new Error('PLY содержит некорректные координаты');}
 if(object){scene.remove(object);object.geometry.dispose();object.material.dispose();}
 if(points)object=new THREE.Points(geometry,new THREE.PointsMaterial({color:0xa7dcc9,size:Math.max(radius/550,.001)}));
 else{geometry.computeVertexNormals();object=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:geometry.hasAttribute('color')?0xffffff:0xadc9d3,vertexColors:geometry.hasAttribute('color'),roughness:.85,side:THREE.DoubleSide,wireframe:$('wire').checked}));}
 scene.add(object);$('empty').hidden=true;fit();
}
async function api(url,options){const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(data.error||'Ошибка сервера');return data;}
function executionSummary(result){
 if(result?.solution!=='nksr')return '';
 const timing=result.timings_s||{}, seconds=value=>Number.isFinite(value)?`${value.toFixed(2)} с`:'нет замера';
 return `GPU: ${result.neural_backend?.device||'CUDA'}. NKSR: ${seconds(timing.nksr)} · Подготовка (CPU): ${seconds(timing.preprocess)} · Сетки (CPU): ${seconds(timing.decimate)}.`;
}
function layers(layer){document.querySelectorAll('[data-layer]').forEach(b=>{b.classList.toggle('active',b.dataset.layer===layer);b.setAttribute('aria-pressed',String(b.dataset.layer===layer));});$('view-label').textContent=layer==='input.ply'?'ОБЛАКО · МЕТРЫ · Z ВВЕРХ':layer==='visual.ply'?'ВИЗУАЛЬНАЯ СЕТКА':'СЕТКА КОЛЛИЗИЙ';}
async function loadLayer(layer){const version=++viewVersion;try{if(layer==='input.ply'&&localBuffer&&!current){show(localBuffer,true,true);layers(layer);return;}if(!current)return;const response=await fetch(`/api/jobs/${current.id}/files/${layer}`);if(!response.ok)throw new Error('Файл ещё не готов');const buffer=await response.arrayBuffer();if(version!==viewVersion)return;show(buffer,layer==='input.ply');layers(layer);}catch(e){error(e.message);}}
function renderJob(job){if(current?.id!==job.id){++viewVersion;clearView();}current=job;$('cancel').hidden=!['queued','running'].includes(job.status);$('status').textContent=(job.status==='running'&&stages[job.stage])||labels[job.status]||job.status;error(job.error||'');document.querySelectorAll('[data-layer]').forEach(b=>b.disabled=!job.files.includes(b.dataset.layer));$('downloads').replaceChildren();for(const file of job.files.filter(x=>x!=='input.ply')){const a=document.createElement('a');a.textContent='↓ '+file;a.href=`/api/jobs/${job.id}/files/${file}`;a.download=file;$('downloads').append(a);}$('metrics').replaceChildren();$('result-title').textContent=job.status==='completed'?'Сцена готова':job.status==='failed'?'Не удалось построить сцену':job.status==='cancelled'?'Обработка отменена':'Обработка сцены';if(job.result){for(const [label,value] of [['Время',`${job.result.total_s.toFixed(1)} с`],['Точек',job.result.input.points.toLocaleString('ru')],['Visual, треугольников',job.result.outputs.visual_triangles.toLocaleString('ru')],['Collision, треугольников',job.result.outputs.collision_triangles.toLocaleString('ru')]]){const div=document.createElement('div'),strong=document.createElement('strong');strong.textContent=value;div.append(strong,document.createTextNode(label));$('metrics').append(div);}}$('execution').textContent=executionSummary(job.result);$('execution').hidden=!$('execution').textContent;$('quality').textContent='Качество не проверено: GT/B и физические тесты не запускались. Успешная реконструкция не означает прохождение приёмки.'+(job.result?.normal_orientation?.warning?' '+job.result.normal_orientation.warning:'');}
async function watch(id){const version=++pollVersion;while(version===pollVersion){const job=await api('/api/jobs/'+id);if(version!==pollVersion)return;renderJob(job);if(['completed','failed','cancelled','uploaded'].includes(job.status)){if(job.status==='completed')await loadLayer('visual.ply');await history();return;}await new Promise(resolve=>setTimeout(resolve,1500));}}
async function history(){const jobs=await api('/api/jobs');$('history').replaceChildren();for(const job of jobs){const button=document.createElement('button');button.textContent=`${labels[job.status]} · ${new Date(job.created_at).toLocaleString('ru')} · ${job.settings.method}`;button.onclick=()=>{localBuffer=null;watch(job.id).catch(e=>error(e.message));};$('history').append(button);}}
async function preview(){const file=$('cloud').files[0];if(!file)return;if(!file.name.toLowerCase().endsWith('.ply')||file.size>512*1048576){error('Выберите PLY размером до 512 МБ');return;}++pollVersion;current=null;const version=++viewVersion;clearView();error('');$('filename').textContent=file.name;$('status').textContent='Загрузка предпросмотра…';$('downloads').replaceChildren();$('metrics').replaceChildren();$('execution').textContent='';$('execution').hidden=true;$('result-title').textContent='Сцена ещё не построена';document.querySelectorAll('[data-layer]').forEach(b=>b.disabled=b.dataset.layer!=='input.ply');try{const buffer=await file.arrayBuffer();if(version!==viewVersion)return;localBuffer=buffer;show(localBuffer,true,true);layers('input.ply');$('status').textContent=`${(file.size/1048576).toFixed(1)} МБ · готово к обработке`;}catch(e){localBuffer=null;error('Не удалось открыть PLY: '+e.message);}}
$('cloud').onchange=preview;for(const id of ['scale','up'])$(id).onchange=()=>{if(localBuffer&&!current)show(localBuffer,true,true);};
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('drag');};$('drop').ondragleave=()=>$('drop').classList.remove('drag');$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('drag');if(e.dataTransfer.files.length){$('cloud').files=e.dataTransfer.files;preview();}};
$('method').onchange=()=>{$('submit').disabled=$('method').selectedOptions[0].disabled;$('method-note').textContent={'baseline':'Контрольная версия: фильтрация, Poisson и упрощение.','improved':'Удаление ложных поверхностей, восстановление пола и отдельная сетка коллизий.','nksr':'Предобученная модель kitchen-sink. Построит сетки и USD; точность и физику нужно проверить отдельно.','high-detail':'Более подробные проёмы и сетка коллизий. Требует больше времени и памяти; результат зависит от облака.'}[$('method').value];};
$('form').onsubmit=async e=>{e.preventDefault();const file=$('cloud').files[0];if(!file)return;$('submit').disabled=true;error('');++pollVersion;try{$('status').textContent='Передача файла…';const params=new URLSearchParams({method:$('method').value,scale:$('scale').value,up:$('up').value});const job=await api('/api/jobs?'+params,{method:'POST',body:file});const trajectory=$('trajectory').files[0];if(trajectory)await api(`/api/jobs/${job.id}/trajectory`,{method:'POST',body:trajectory});await api(`/api/jobs/${job.id}/start`,{method:'POST'});localBuffer=null;await watch(job.id);}catch(e){error(e.message);$('status').textContent='Не удалось завершить обработку';}finally{$('submit').disabled=false;}};
for(const button of document.querySelectorAll('[data-layer]'))button.onclick=()=>loadLayer(button.dataset.layer);
$('reset').onclick=fit;$('wire').onchange=()=>{if(object?.isMesh)object.material.wireframe=$('wire').checked;};
history().catch(e=>error(e.message));

api('/api/backends').then(backends=>{const info=backends.nksr;document.querySelector('#method option[value=nksr]').disabled=!info.available;$('method').disabled=false;$('method').onchange();$('nksr-status').textContent=info.available?'NKSR доступен · исследовательский режим':info.reason||'NKSR недоступен';}).catch(()=>{$('nksr-status').textContent='Не удалось проверить доступность NKSR';});

$('cancel').onclick=async()=>{if(!current)return;try{renderJob(await api(`/api/jobs/${current.id}/cancel`,{method:'POST'}));}catch(e){error(e.message);}};

clearSimulation=initSimulationPanel({THREE,api,openJob:async id=>{localBuffer=null;await watch(id);},view:()=>({scene,camera,controls,renderer})});
