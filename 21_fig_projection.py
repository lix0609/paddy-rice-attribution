import json,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.linewidth':0.7,
 'axes.spines.top':False,'axes.spines.right':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PUB=PROJ+'/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-20,4),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
g=gpd.read_file(ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp').to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
pc=pd.read_csv(MID+'/projection_county.csv'); pc['CountyCode']=pc['CountyCode'].astype(str)
m=g.merge(pc,on='CountyCode',how='left')
S=json.load(open(MID+'/projection_summary.json'))
fig,ax=plt.subplots(2,2,figsize=(DC*0.8,DC*0.68),constrained_layout=True)
vmax=max(abs(pc[['dY_245','dY_585']].min().min()),abs(pc[['dY_245','dY_585']].max().max()))
for a,col,ttl,L in [(ax[0,0],'dY_245','SSP2-4.5 (+2.1 °C)','a'),(ax[0,1],'dY_585','SSP5-8.5 (+2.9 °C)','b')]:
    m.plot(column=col,ax=a,cmap='RdBu',vmin=-vmax,vmax=vmax,edgecolor='0.5',lw=0.25,legend=True,
           legend_kwds={'label':'Climate-driven Δyield (%)','shrink':0.7})
    a.set_title('Yield change: '+ttl); a.axis('off'); plab(a,L)
# c: province summary bars (Δyield & suitable%)
axc=ax[1,0]; x=np.arange(2); 
dy=[S['dY_245_mean'],S['dY_585_mean']]
b1=axc.bar(x-0.2,dy,0.4,color=OI['verm'],label='Δyield (%)',edgecolor='white')
axc.set_ylabel('Climate-driven Δyield (%)',color=OI['verm']); axc.tick_params(axis='y',colors=OI['verm'])
axc.axhline(0,color='k',lw=0.5); axc.set_xticks(x); axc.set_xticklabels(['SSP2-4.5','SSP5-8.5'])
for xi,v in zip(x-0.2,dy): axc.text(xi,v-0.15,f'{v:.1f}%',ha='center',va='top',fontsize=6.5,color=OI['verm'])
suit=[98,100]; ax2=axc.twinx(); ax2.plot(x+0.0,[80,80],'o',color='0.6',ms=4); ax2.plot(x+0.2,suit,'s-',color=OI['green'],ms=5)
ax2.set_ylabel('Thermally suitable counties (%)',color=OI['green']); ax2.tick_params(axis='y',colors=OI['green']); ax2.set_ylim(70,105); ax2.spines['top'].set_visible(False)
for xi,v in zip(x+0.2,suit): ax2.text(xi,v+0.5,f'{v}%',ha='center',fontsize=6,color=OI['green'])
axc.text(0.02,0.06,'baseline suitable = 80%',transform=axc.transAxes,fontsize=5.8,color='0.4')
axc.set_title('Province: yield vs thermal suitability'); plab(axc,'c')
# d: Δyield vs future warmest-month temp (heat penalty gradient)
axd=ax[1,1]; axd.axhline(0,color='0.6',lw=0.6); axd.axvline(21.9,color=OI['orange'],lw=1,ls=':')
axd.scatter(pc.t_warm_245,pc.dY_245,s=14,color=OI['blue'],edgecolor='white',lw=0.3,label='SSP2-4.5')
axd.scatter(pc.t_warm_585,pc.dY_585,s=14,color=OI['verm'],edgecolor='white',lw=0.3,label='SSP5-8.5')
axd.set_xlabel('Future warmest-month T (°C)'); axd.set_ylabel('Climate-driven Δyield (%)')
axd.set_title('Heat drives the yield loss'); axd.legend(fontsize=6,loc='upper right')
axd.text(21.95,axd.get_ylim()[1]*0.9,'optimum\n21.9 °C',fontsize=5.8,color=OI['orange']); plab(axd,'d')
fig.savefig(PUB+'/Fig11_projection.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig11_projection.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/Fig11_projection.svg',bbox_inches='tight'); plt.close()
print('Fig11_projection done')
