from pathlib import Path
import csv,json,subprocess,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from matplotlib import pyplot as plt
rows=list(csv.DictReader((R/'legacy_reference/matched_summary.csv').open()))
ranked=[]
for r in rows:
 out=dict(path=r['path'],delay=r['delay'],mass=r['mass'])
 for k in ['new_lans','legacy_ppo','fast_lqr','mpc']:
  out[k+'_rank']=1+sum(float(r[j+'_rmse'])<float(r[k+'_rmse'])-1e-12 for j in ['new_lans','legacy_ppo','fast_lqr','mpc'])
  out[k+'_penalized_rmse']=r[k+'_rmse'];out[k+'_completed']=r[k+'_completed'];out[k+'_episodes']=r[k+'_episodes']
 ranked.append(out)
with (R/'final_rankings_against_original_baselines.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(ranked[0]));w.writeheader();w.writerows(ranked)
fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
labels=['New LANS','Original LANS','Original PPO','MPC']
for ax,path in zip(axes.ravel(),['dlc','step','sine','slalom']):
 row=next(r for r in rows if r['path']==path and r['delay']=='fixed200' and float(r['mass'])==1650)
 kinds=['new_lans','legacy_lans','legacy_ppo','mpc']
 vals=[float(row[k+'_rmse']) for k in kinds]
 bars=ax.bar(labels,vals,color=['#1766b6','#999999','#31945e','#df8a2b'])
 ax.set_title(path+' — constant 200 ms')
 ax.set_ylabel('Mean failure-penalized RMSE (m)')
 ax.tick_params(axis='x',labelsize=8)
 for bar,k in zip(bars,kinds):
  ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.03,f"{row[k+'_completed']}/{row[k+'_episodes']} completed",ha='center',fontsize=7)
 ax.set_ylim(0,2.85);ax.grid(axis='y',alpha=.2)
out=ROOT/'figures';out.mkdir(exist_ok=True)
fig.savefig(out/'matched_200ms.png',dpi=180,bbox_inches='tight',pad_inches=.15);fig.savefig(out/'matched_200ms.pdf',bbox_inches='tight',pad_inches=.15);plt.close(fig)
p=ROOT/'Robust_Fallback_Report.md'
text=p.read_text()
if '\n## Matched high-delay plot' not in text:text+='\n## Matched high-delay plot\n\n![Fresh cases; completion counts include three policy repetitions for learned controllers, five physical seeds and two reference scalings. At200ms neural authority vanishes, so the new LQR fallback explains LANS improvement.](figures/matched_200ms.png)\n'
p.write_text(text)
for root,stem in [(ROOT,'Robust_Fallback_Report'),(ROOT.parent/'lans-improvement','Improvement_Report')]:
 subprocess.run(['pandoc',str(root/f'{stem}.md'),'-o',str(root/f'{stem}.docx'),'--resource-path='+str(root)],check=True)
 from docx import Document
 from docx.shared import Pt,Inches
 d=Document(root/f'{stem}.docx')
 for section in d.sections:section.left_margin=section.right_margin=Inches(.6)
 d.styles['Normal'].font.name='Times New Roman';d.styles['Normal'].font.size=Pt(10)
 for table in d.tables:
  for row in table.rows:
   for cell in row.cells:
    for para in cell.paragraphs:
     for run in para.runs:run.font.size=Pt(8)
 d.save(root/f'{stem}.docx')
 with (root/'results/pdf_build.log').open('w') as log:
  subprocess.run(['pandoc',str(root/f'{stem}.md'),'-o',str(root/f'{stem}.pdf'),'--pdf-engine=xelatex','--resource-path='+str(root),'-V','geometry:margin=0.65in','-V','geometry:landscape','-V','fontsize=10pt','-V','mainfont=DejaVu Serif'],stdout=log,stderr=subprocess.STDOUT,check=True)
print('Created both reports, final rank table and matched plot.')
