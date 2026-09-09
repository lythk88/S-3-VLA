import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ap=argparse.ArgumentParser(); ap.add_argument('--debug-json',required=True); ap.add_argument('--primitive-json',required=True); ap.add_argument('--output',required=True); ap.add_argument('--orange-juice-box',action='store_true'); ap.add_argument('--orange-juice-static-center',nargs=3,type=float); a=ap.parse_args()
d=json.loads(Path(a.debug_json).read_text()); prim=json.loads(Path(a.primitive_json).read_text())
pi05=[]; qp=[]; exe=[]; gripper=[]
for c in d.get('chunks',[]):
  physical=c.get('physical_actions',[])
  chunk_pi05=c.get('pi05_nominal_trajectory_positions',[])
  for e in c.get('executed',[]):
    if e.get('predicted_eef_position') is not None and e.get('actual_eef_position') is not None:
      horizon_index=int(e.get('horizon_index',0))
      nominal_position=e.get('pi05_nominal_eef_position')
      if nominal_position is None and horizon_index < len(chunk_pi05):
        nominal_position=chunk_pi05[horizon_index]
      if nominal_position is None:
        raise RuntimeError(
          'Debug JSON lacks true PI0.5 nominal trajectory positions; rerun with updated main_aegis.py'
        )
      pi05.append(nominal_position); qp.append(e['predicted_eef_position']); exe.append(e['actual_eef_position'])
      gripper.append(float(physical[horizon_index][6]) if horizon_index < len(physical) else -1.0)
pi05=np.asarray(pi05); qp=np.asarray(qp); exe=np.asarray(exe); center=np.asarray(prim['center']); half=np.asarray(prim['size'])
fig=plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection='3d');
allp=np.vstack([pi05,qp,exe,center[None,:]]); lo=allp.min(0)-.12; hi=allp.max(0)+.12
ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2]); ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
corn=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half+center
faces=[[corn[i] for i in f] for f in [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]]
ax.add_collection3d(Poly3DCollection(faces,alpha=.22,facecolor='red',edgecolor='darkred',label='estimated obstacle OBB (+1 cm)'))
ln0,=ax.plot([],[],[],color='darkorange',linewidth=2,label='PI0.5 nominal (pre-guidance)'); ln1,=ax.plot([],[],[],color='purple',linewidth=2,label='QP-corrected prediction'); ln2,=ax.plot([],[],[],color='blue',linewidth=2,label='actual EEF response to QP command'); dot0,=ax.plot([],[],[], 'o',color='darkorange'); dot1,=ax.plot([],[],[], 'o',color='purple');dot2,=ax.plot([],[],[], 'o',color='blue'); ax.legend();
orange_poly=None; orange_centers=None; grasp_step=None; release_step=None
if a.orange_juice_box:
  if a.orange_juice_static_center is not None:
    orange_centers=np.repeat(np.asarray(a.orange_juice_static_center,dtype=float)[None,:],len(exe),axis=0)
    grasp_step=len(exe)+1
  else:
    grasp_step=next((i for i,v in enumerate(gripper) if v > 0.0),None)
    release_step=next((i for i,v in enumerate(gripper) if grasp_step is not None and i > grasp_step and v < 0.0),None)
  if orange_centers is not None or grasp_step is not None:
    if orange_centers is None:
      orange_centers=np.repeat(exe[grasp_step][None,:],len(exe),axis=0)
      for i in range(grasp_step,len(exe)):
        source=min(i,release_step) if release_step is not None else i
        orange_centers[i]=exe[grasp_step] + (exe[source]-exe[grasp_step])
    # Exact union AABB of the four MuJoCo collision boxes after the task's
    # fixed x=90-degree initialization rotation: 53.07 x 52.50 x 131.19 mm.
    orange_half=np.array([0.02625,0.026535,0.065595])
    orange_poly=Poly3DCollection([],alpha=.42,facecolor='orange',edgecolor='darkorange',label='orange juice MuJoCo AABB')
    ax.add_collection3d(orange_poly)
    ax.legend()
writer=FFMpegWriter(fps=20,metadata={'title':'PI0.5 nominal vs QP prediction vs actual execution'})
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
with writer.saving(fig,a.output,dpi=120):
  for i in range(len(pi05)):
    ln0.set_data(pi05[:i+1,0],pi05[:i+1,1]);ln0.set_3d_properties(pi05[:i+1,2]);ln1.set_data(qp[:i+1,0],qp[:i+1,1]);ln1.set_3d_properties(qp[:i+1,2]);ln2.set_data(exe[:i+1,0],exe[:i+1,1]);ln2.set_3d_properties(exe[:i+1,2]);dot0.set_data([pi05[i,0]],[pi05[i,1]]);dot0.set_3d_properties([pi05[i,2]]);dot1.set_data([qp[i,0]],[qp[i,1]]);dot1.set_3d_properties([qp[i,2]]);dot2.set_data([exe[i,0]],[exe[i,1]]);dot2.set_3d_properties([exe[i,2]])
    phase=''
    if orange_poly is not None:
      oc=orange_centers[i]; corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*orange_half+oc
      orange_poly.set_verts([[corners[j] for j in f] for f in [(0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5)]])
      phase='fixed initial orange-juice pose' if a.orange_juice_static_center is not None else ('pre-grasp' if i < grasp_step else ('released' if release_step is not None and i >= release_step else 'carried'))
    nominal_qp_mm=1000.0*np.linalg.norm(pi05[i]-qp[i]); qp_execution_mm=1000.0*np.linalg.norm(qp[i]-exe[i])
    ax.set_title(f'PI0.5 vs QP vs executed — step {i}\nPI0.5→QP: {nominal_qp_mm:.1f} mm | QP→executed: {qp_execution_mm:.1f} mm {phase}');writer.grab_frame()
plt.close(fig)
