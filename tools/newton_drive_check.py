"""Real CUDA controller regression: progress with floor, stop at a wall."""
import argparse
import json
from pathlib import Path
from newton_record import add_mobile_body, drive, newton, wp, np


def run(wall):
    b=newton.ModelBuilder()
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=.3))
    if wall:
        b.add_shape_box(-1,hx=.1,hy=2,hz=1,xform=wp.transform(wp.vec3(.8,0,1),wp.quat_identity()))
    add_mobile_body(b,[0,0,.23],np.array([.4,.4,.3]),20.)
    m=b.finalize(device='cuda:0');assert m.device.is_cuda
    assert abs(float(m.body_mass.numpy()[0])-20)<.001
    solver=newton.solvers.SolverXPBD(m,iterations=10)
    current,following=m.state(),m.state();control,contacts=m.control(),m.contacts()
    for i in range(840):
        current.clear_forces()
        if i>=240:wp.launch(drive,1,inputs=[current.body_qd,current.body_f,wp.vec3(.2,0,0),20.],device='cuda:0')
        m.collide(current,contacts);solver.step(current,following,control,contacts,1/120);current,following=following,current
    q=current.body_q.numpy()[0]
    assert np.isfinite(q).all() and .12<q[2]<.19
    return q[:3].tolist()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    free=run(False);wall=run(True)
    assert free[0]>.65,free
    assert wall[0]<.53,wall
    report={'status':'passed','device':'cuda:0','free_final_xyz':free,'wall_final_xyz':wall,
            'body_shape':'box_with_spherical_skids','body_mass_kg':20,'friction':.3,
            'steps':840,'dt_s':1/120,'driven_s':5}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(report)
