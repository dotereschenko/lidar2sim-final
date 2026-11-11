"""Measure annotated spans from reconstructed triangle hits, never GT endpoints."""
import numpy as np
import open3d as o3d
from .annotations import cloud_array, unit_vector

METHOD = {'id':'local-ray-bundle-v1','probe_offset_m':.005,'probes_per_side':5,
          'minimum_valid_probes':4,'minimum_normal_alignment':.8,'maximum_surface_spread_m':.02}


def validate_dimensions(items, required_count):
    if not isinstance(items,list) or len(items)!=required_count or len({x['id'] for x in items})!=required_count:
        raise ValueError('Expected required number of uniquely identified dimensions')
    for item in items:
        cloud_array([item['origin_m']]);unit_vector(item['axis'])
        if not np.isfinite(item['reference_m']) or item['reference_m']<=0:
            raise ValueError('Dimension reference must be positive and finite')
        boxes=item['search_boxes']
        if len(boxes)!=2:raise ValueError('Expected two dimension search boxes')
        for box in boxes:
            low,high=cloud_array([box['min_m'],box['max_m']])
            if (low>=high).any():raise ValueError('Invalid dimension search box bounds')


def measure_dimension(scene, item, tolerance):
    validate_dimensions([item],1)
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid dimension tolerance')
    origin=np.asarray(item['origin_m'],dtype=float);axis=unit_vector(item['axis'])
    tangent=np.cross(axis,np.eye(3)[np.argmin(np.abs(axis))]);tangent/=np.linalg.norm(tangent)
    other=np.cross(axis,tangent)
    offsets=np.array([np.zeros(3),tangent,-tangent,other,-other])*METHOD['probe_offset_m']
    result={'id':item['id'],'label':item.get('label',item['id']),'reference_m':float(item['reference_m']),
            'measured_m':None,'absolute_error_m':None,'tolerance_m':float(tolerance),
            'status':'error','passed':False,'surface_points_m':[],'sides':[]}
    for sign,box in zip([-1,1],item['search_boxes']):
        starts=origin+offsets;direction=sign*axis
        hits=scene.cast_rays(o3d.core.Tensor(np.column_stack([starts,np.tile(direction,(5,1))]).astype(np.float32)),nthreads=1)
        distances=hits['t_hit'].numpy().astype(float)
        finite=np.isfinite(distances)&(distances>.01)
        # Missing rays remain invalid without inf*0 or NaN entering the report.
        points=starts+np.where(finite,distances,0)[:,None]*direction
        valid=finite & (points>=np.array(box['min_m'])-1e-6).all(1) & (points<=np.array(box['max_m'])+1e-6).all(1)
        valid &= np.abs(hits['primitive_normals'].numpy() @ axis)>=METHOD['minimum_normal_alignment']
        count=int(valid.sum())
        side={'valid_probes':count,'spread_m':None}
        result['sides'].append(side)
        if not valid[0] or count<METHOD['minimum_valid_probes']:
            result['reason']='missing_or_unaligned_surface_in_search_box';return result
        spread=float(np.ptp(points[valid]@axis));side['spread_m']=spread
        if spread>METHOD['maximum_surface_spread_m']:
            result['reason']='ambiguous_surface_probe_spread';return result
        result['surface_points_m'].append(points[0].tolist())
    measured=float((np.array(result['surface_points_m'][1])-result['surface_points_m'][0])@axis)
    if measured<=0:
        result['reason']='nonpositive_measured_span';return result
    error=abs(measured-item['reference_m'])
    result.update(measured_m=measured,absolute_error_m=error,passed=bool(error<=tolerance),
                  status='passed' if error<=tolerance else 'failed')
    return result


def measure_dimensions(scene, items, rules):
    validate_dimensions(items,rules['required_count'])
    values=[measure_dimension(scene,item,rules['absolute_error_max_m']) for item in items]
    return {'method':METHOD,'items':values,'passed_count':sum(v['passed'] for v in values),
            'required_count':len(values),'status':'passed' if all(v['passed'] for v in values) else 'failed'}
