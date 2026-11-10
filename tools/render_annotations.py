"""Top-down GT/A/B/common-region overlay; labels are dimension numbers 01–10."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

# Tiny numeric bitmap labels keep diagnostic rendering dependency-free.
FONT = ['111101101101111','010110010010111','111001111100111','111001111001111',
        '101101111001001','111100111001111','111100111101111','111001001001001',
        '111101111101111','111101111001111']


def render(scans, annotations, output):
    scans, annotations, output = map(Path,(scans,annotations,output))
    report = json.loads((annotations/'annotations.json').read_text())
    data = np.load(annotations/'eval_region.npz',allow_pickle=False)
    points = data['points_m']
    a,b = [np.asarray(o3d.io.read_point_cloud(str(scans/f'{n}.ply')).points) for n in ('A','B')]
    # All panels share an equal-aspect XY projection, meters, north/up.
    panels = [points,a,b,points[data['common_mask']]]
    image = np.full((1800,1000,3),255,dtype=np.uint8)
    def pixels(xyz,panel):
        x = ((np.asarray(xyz)[:,0]+25)*40+30).astype(int)+(panel%2)*500
        y = (880-(np.asarray(xyz)[:,1]-7)*40).astype(int)+(panel//2)*900
        return x,y
    def draw(xyz,panel,color):
        x,y = pixels(xyz,panel)
        valid=(x>=panel%2*500)&(x<(panel%2+1)*500)&(y>=panel//2*900)&(y<(panel//2+1)*900)
        image[y[valid],x[valid]]=color
    def line(start,end,panel,color):
        draw(np.linspace(start,end,1000),panel,color)
    for panel,cloud in enumerate(panels):
        for x in range(-24,-13):line([x,7,0],[x,28,0],panel,[230]*3)
        for y in range(8,28):line([-25,y,0],[-13,y,0],panel,[230]*3)
        slab=cloud[(cloud[:,2]>.7)&(cloud[:,2]<1.8)]
        draw(slab,panel,[70]*3 if panel==0 else ([40,95,180] if panel<3 else [35,140,70]))
        for item in report['doors']:
            corners=item['corners_m']
            line(corners[0],corners[1],panel,[240,140,0])
        for item in report['dimensions']:
            p,q=np.array(item['surface_points_m'])
            line(p,q,panel,[210,35,35])
            # Labels above the line midpoint; D09/D10 offset to avoid overlap.
            x,y=pixels([(p+q)/2],panel);x,y=int(x[0])+3,int(y[0])-13
            if item['id']=='D10':y+=24
            for digit in item['id'][1:]:
                bitmap=np.array(list(FONT[int(digit)])).reshape(5,3)=='1'
                for row,col in zip(*np.where(bitmap)):
                    image[y+row*2:y+row*2+2,x+col*2:x+col*2+2]=[190,0,0]
                x+=8
    image[899:902]=180;image[:,499:502]=180
    output.parent.mkdir(parents=True,exist_ok=True)
    if not o3d.io.write_image(str(output),o3d.geometry.Image(image)):
        raise OSError('Cannot write annotation overlay')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scans');p.add_argument('annotations');p.add_argument('output')
    a=p.parse_args();render(a.scans,a.annotations,a.output)
