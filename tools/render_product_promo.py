"""Render an editorial promo from real exported geometry and Newton states.

uv run --with pillow --with numpy python tools/render_product_promo.py out/product-demo
Requires geometry.npz (v/f/p/cv/cf), reconstruction.json, simulation.json, scene.usd.
This is an offline visualization, not a browser recording or a new physics run.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);ap.add_argument('--preview',type=float);args=ap.parse_args();root=args.directory
geo=np.load(root/'geometry.npz');job=json.loads((root/'reconstruction.json').read_text());sim=json.loads((root/'simulation.json').read_text());r=sim['result']
assert sim['scene_job']==job['id'] and sim['status']=='completed' and sim['goal_reached']
assert hashlib.sha256((root/'scene.usd').read_bytes()).hexdigest()==r['scene_sha256']
assert r['outcome']=='reached' and r['body_shape']=='box_with_spherical_skids'
traj=r['trajectory'];assert len(traj)>1 and all(np.isfinite(x['position']).all() for x in traj)
W,H,FPS,DURATION=1920,1080,20,45
fontfile='/System/Library/Fonts/Supplemental/Arial.ttf'
fonts={n:ImageFont.truetype(fontfile,n) for n in [19,22,24,28,32,40,54,66,90]}
bg=Image.new('RGB',(W,H));a=np.zeros((H,W,3),dtype=np.uint8)
y,x=np.mgrid[:H,:W];glow=np.exp(-((x-1370)**2+(y-430)**2)/700000)
for i,c in enumerate([9,14,20]):a[:,:,i]=c+glow*[8,17,20][i]
bg=Image.fromarray(a);ink='#eaf0f5';muted='#93a5b5';green='#b6ee83';cyan='#66d4dc'
v,f,p=geo['v'],geo['f'],geo['p'];cv,cf=geo['cv'],geo['cf'];center=(v.min(0)+v.max(0))/2
# Captions describe this verified example, not arbitrary inputs.
assert job['result']['input']['points']==750000 and len(f)==89618 and len(cf)==59999 and len(p)==93750
assert abs(r['duration_s']-8.1)<.01 and abs(job['result']['total_s']-25.4)<.1
route=np.array([z['position'] for z in traj]);mid=(route[0]+route[-1])/2

def project(pts,t,close=False):
 az=(-.65+t*.025) if not close else -.72;el=.70 if not close else .85
 u=np.array([np.cos(az),np.sin(az),0]);w=np.array([-np.sin(az)*np.sin(el),np.cos(az)*np.sin(el),np.cos(el)]);depth=np.cross(u,w)
 q=pts-(mid if close else center);scale=235 if close else 43
 return np.stack([1300+q@u*scale,570-q@w*scale],axis=1),q@depth

def mesh(draw,verts,faces,t,close=False,wire=False,body=False):
 pts=verts[faces];cent=pts.mean(1)
 if not body:
  keep=cent[:,2]<1.65
  faces=faces[keep];pts=pts[keep]
 xy,z=project(verts,t,close);depth=z[faces].mean(1)
 n=np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9)
 light=.35+.65*np.abs(n@np.array([.2,-.35,.91]));base=np.array([182,238,131]) if body else np.array([72,137,154])
 colors=(base[None,:]*light[:,None]).astype(int)
 polygons=np.rint(xy[faces]).astype(int)
 for k in np.argsort(depth):
  poly=[tuple(q) for q in polygons[k]]
  draw.polygon(poly,fill=tuple(colors[k]),outline=(43,83,99) if wire else None)

def robot(draw,state,t):
 pos=np.array(state['position']);x,y,z,w=state['quaternion_xyzw'];R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
 corners=np.array([[a,b,c] for a in [-.2,.2] for b in [-.2,.2] for c in [-.09,.15]])
 faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
 for sx in [-.14,.14]:
  for sy in [-.14,.14]:
   pt,_=project((np.array([[sx,sy,-.09]])@R.T+pos),t,True);xx,yy=pt[0];draw.ellipse((xx-18,yy-18,xx+18,yy+18),fill='#84996d',outline=green,width=2)
 mesh(draw,corners@R.T+pos,faces,t,True,body=True)

def txt(d,xy,s,size=28,fill=ink):d.text(xy,s,font=fonts[size],fill=fill)
def frame(t):
 im=bg.copy();d=ImageDraw.Draw(im);phase=0 if t<5 else 1 if t<13 else 2 if t<21 else 3 if t<27 else 4 if t<38 else 5
 if phase in [0,1]:
  xy,z=project(p,t);valid=(xy[:,0]>650)&(xy[:,0]<1890)&(xy[:,1]>150)&(xy[:,1]<900);ids=np.flatnonzero(valid);ids=ids[np.argsort(z[ids])]
  for idx in ids[::2]:
   xx,yy=xy[idx];d.point((int(xx),int(yy)),fill=(110,190,int(170+min(60,max(0,p[idx,2]*20)))))
 elif phase==4:
  mesh(d,cv,cf,t,True)
  points,_=project(np.array([route[0],r['goal_position_m']]),t,True);d.line([tuple(q) for q in points],fill=cyan,width=4)
  for q in points:d.ellipse((q[0]-9,q[1]-9,q[0]+9,q[1]+9),outline=cyan,width=3)
  state=traj[min(len(traj)-1,max(0,int((t-27)*FPS)))];robot(d,state,t)
  txt(d,(1020,900),f"NEWTON / XPBD     t = {state['t']:.2f} с     контакты: {state['contacts']}",24,cyan)
 else:mesh(d,cv if phase==3 else v,cf if phase==3 else f,t,wire=phase==3)
 # fixed editorial panel, separate from the product UI
 d.rectangle((0,0,650,H),fill='#0c1119');d.line((650,150,650,910),fill='#23333e',width=2)
 d.rectangle((650,0,W,140),fill='#0c151d');d.rectangle((650,935,W,H),fill='#0c151d');d.line([(66,68),(79,53),(92,68),(79,83),(66,68)],fill=green,width=3);txt(d,(112,43),'Lidar2Sim',40);txt(d,(1395,57),'СКАН → СЦЕНА → ПРОВЕРКА',22,green)
 labels=['ИЗ ТОЧЕК — В СИМУЛЯЦИЮ','01 / ОБЛАКО ТОЧЕК','02 / НЕЙРОРЕКОНСТРУКЦИЯ','03 / КОЛЛИЗИИ','04 / ФИЗИЧЕСКИЙ ПРОГОН','ОТ СКАНА К ЭКСПЕРИМЕНТУ'];txt(d,(64,190),labels[phase],22,green)
 titles=[['Пространство.','В цифровом','виде.'],['Загрузите','скан','помещения.'],['Восстановите','поверхности.'],['Подготовьте','сцену','к физике.'],['Проверьте','движение.'],['Исследуйте.','Проверяйте.','Повторяйте.']]
 for i,line in enumerate(titles[phase]):txt(d,(60,245+i*79),line,54 if phase==2 else 66)
 descriptions=[['Облако точек → геометрия →','физический эксперимент.'],['PLY · единицы · ось вверх','750 000 исходных точек'],['Предобученный NKSR','на NVIDIA A100'],['Отдельная collision mesh','в координатах исходной сцены'],['Сохранённая траектория Newton','на выбранной реконструкции'],['Visual + Collision + USD','Отчёт и состояния симуляции']]
 for i,line in enumerate(descriptions[phase]):txt(d,(64,550+i*43),line,28,muted)
 metrics=[('PLY → USD','Конвейер реконструкции'),('750 000','точек в исходном скане'),('25,4 с','весь конвейер · NKSR 1,86 с'),('59 999','треугольников collision mesh'),('Цель достигнута' if t>=35.1 else 'Newton · CUDA','8,1 с симуляции · маршрут 1 м'),('Lidar2Sim','Исследовательский прототип')]
 d.rounded_rectangle((64,700,593,850),radius=18,fill='#16222d',outline='#2b414c');txt(d,(90,726),metrics[phase][0],40,green);txt(d,(90,789),metrics[phase][1],22,muted)
 notes=['Визуализация сохранённых данных','Предпросмотр: 93 750 точек','Срез по высоте — только отображение','Срез по высоте — только отображение','Корпус на сферических опорах, без приводов колёс','Качество каждого скана проверяется отдельно']
 txt(d,(700,950),notes[phase],24,muted)
 d.line((64,1004,1856,1004),fill='#283540',width=3);d.line((64,1004,64+1792*t/DURATION,1004),fill=green,width=3)
 txt(d,(64,1027),'ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ · НЕ ЗАПИСЬ UI',19,muted);txt(d,(1230,1027),'NKSR + NEWTON   /   A100   /   2026',19,muted)
 return im

if args.preview is not None:
 frame(args.preview).save(root/f'preview-{args.preview:g}.png')
else:
 out=root/'lidar2sim-promo.mp4'
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
 try:
  for i in range(FPS*DURATION):
   proc.stdin.write(frame(i/FPS).tobytes())
   if i%(FPS*5)==0:print(f'{i/FPS:.0f}/{DURATION} s',flush=True)
 finally:proc.stdin.close()
 assert proc.wait()==0
 (root/'provenance.json').write_text(json.dumps({'type':'offline editorial visualization; not UI capture','reconstruction':job['id'],'simulation':sim['id'],'source_scene_sha256':r['scene_sha256'],'video_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'duration_s':DURATION,'fps':FPS,'dimensions':[W,H],'limitations':'Display cutaway only. Simplified force-driven body on spherical skids. No new simulation executed.'},indent=2))
 print(out)
