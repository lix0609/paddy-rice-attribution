# -*- coding: utf-8 -*-
"""Fig 7 升级: 定量化预测 (县级图 + 分带 + 产量吨数 + 净产量对冲 + 热惩罚散点)。"""
import json,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
 'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6,
 'axes.linewidth':0.8,'legend.frameon':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00','grey':'#666666'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PUB=PROJ+'/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-20,5),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
S=json.load(open(MID+'/projection_upgrade.json'))
pc=pd.read_csv(MID+'/projection_county_v2.csv'); pc['CountyCode']=pc['CountyCode'].astype(str)
g=gpd.read_file(ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp').to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
m=g.merge(pc,on='CountyCode',how='left')

fig,ax=plt.subplots(2,3,figsize=(DC,DC*0.70),constrained_layout=True)
# (a,b) county maps
vmax=float(np.nanmax(np.abs(pc[['dY245','dY585']].values)))
for a,col,ttl,L in [(ax[0,0],'dY245','SSP2-4.5 (+2.1 °C)','a'),(ax[0,1],'dY585','SSP5-8.5 (+2.9 °C)','b')]:
    m.plot(column=col,ax=a,cmap='RdBu',vmin=-vmax,vmax=vmax,edgecolor='0.5',lw=0.25,legend=True,
           legend_kwds={'label':'Climate-driven Δyield (%)','shrink':0.62})
    a.set_title('Yield change: '+ttl); a.axis('off'); plab(a,L)

# (c) band bars ΔY North/Central/South
axc=ax[0,2]; bnames=['North','Central','South']; xb=np.arange(3); w=0.38
d245=[S['bands'][b]['dY245_mean'] for b in bnames]; d585=[S['bands'][b]['dY585_mean'] for b in bnames]
axc.bar(xb-w/2,d245,w,color=OI['blue'],edgecolor='white',label='SSP2-4.5')
axc.bar(xb+w/2,d585,w,color=OI['verm'],edgecolor='white',label='SSP5-8.5')
axc.axhline(0,color='k',lw=0.6); axc.set_xticks(xb); axc.set_xticklabels(bnames)
axc.set_ylabel('Climate-driven Δyield (%)'); axc.set_ylim(-4.3,0.5); axc.legend(loc='lower right')
for xi,v in zip(xb-w/2,d245): axc.text(xi,v-0.08,f'{v:.1f}',ha='center',va='top',fontsize=5.6,color=OI['blue'])
for xi,v in zip(xb+w/2,d585): axc.text(xi,v-0.08,f'{v:.1f}',ha='center',va='top',fontsize=5.6,color=OI['verm'])
axc.set_title('Loss largest in the north'); plab(axc,'c')

# (d) production tonnage loss + suitability
axd=ax[1,0]; x=np.arange(2); loss_kt=[S['production']['yield_loss_t']['ssp245']/1e3,S['production']['yield_loss_t']['ssp585']/1e3]
axd.bar(x-0.2,loss_kt,0.4,color=OI['verm'],edgecolor='white'); axd.axhline(0,color='k',lw=0.6)
axd.set_ylabel('Climate-driven production\nchange (kt yr$^{-1}$)',color=OI['verm']); axd.tick_params(axis='y',colors=OI['verm'])
axd.set_xticks(x); axd.set_xticklabels(['SSP2-4.5','SSP5-8.5']); axd.set_xlim(-0.6,1.6); axd.set_ylim(-260,60)
for xi,v in zip(x-0.2,loss_kt): axd.text(xi,v-8,f'{v:.0f} kt',ha='center',va='top',fontsize=6,color=OI['verm'])
ax2=axd.twinx(); suit=[S['suitability_pct']['ssp245'],S['suitability_pct']['ssp585']]
ax2.plot(x+0.2,[S['suitability_pct']['base']]*2,'o',color='0.6',ms=4,zorder=3)
ax2.plot(x+0.2,suit,'s',color=OI['green'],ms=6,zorder=3)
ax2.set_ylabel('Thermally suitable counties (%)',color=OI['green']); ax2.tick_params(axis='y',colors=OI['green']); ax2.set_ylim(70,112)
for xi,v in zip(x+0.2,suit): ax2.text(xi,v+1.5,f'{v:.0f}%',ha='center',va='bottom',fontsize=6,color=OI['green'])
ax2.text(0.03,0.06,'baseline 80% (○)',transform=ax2.transAxes,fontsize=5.4,color='0.4')
axd.set_title('Production loss vs suitability gain'); plab(axd,'d')

# (e) net outlook: expansion potential vs yield penalty (Mt)
axe=ax[1,1]; xs=np.arange(2)
gain=[S['net_outlook_t']['ssp245']['expansion_gain']/1e6,S['net_outlook_t']['ssp585']['expansion_gain']/1e6]
pen=[S['net_outlook_t']['ssp245']['yield_effect']/1e6,S['net_outlook_t']['ssp585']['yield_effect']/1e6]
net=[S['net_outlook_t']['ssp245']['net']/1e6,S['net_outlook_t']['ssp585']['net']/1e6]
axe.bar(xs-0.2,gain,0.4,color=OI['green'],edgecolor='white',label='Expansion potential (+)')
axe.bar(xs+0.2,pen,0.4,color=OI['verm'],edgecolor='white',label='Yield penalty (−)')
axe.plot(xs,net,'D',color='k',ms=5,zorder=4,label='Net')
axe.axhline(0,color='k',lw=0.6); axe.set_xticks(xs); axe.set_xticklabels(['SSP2-4.5','SSP5-8.5'])
axe.set_ylabel('Climate-attributable\nproduction (Mt yr$^{-1}$)'); axe.set_ylim(-0.7,4.6); axe.legend(loc='upper left',fontsize=5.4)
for xi,v in zip(xs-0.2,gain): axe.text(xi,v+0.12,f'+{v:.1f}',ha='center',va='bottom',fontsize=5.8,color=OI['green'])
for xi,v in zip(xs,net): axe.text(xi+0.28,v,f'net {v:+.1f}',ha='left',va='center',fontsize=5.8)
axe.set_title('Expansion outweighs heat penalty'); plab(axe,'e')

# (f) heat scatter with optimum + most-exposed ring
axf=ax[1,2]; axf.axhline(0,color='0.6',lw=0.6); axf.axvline(21.9,color=OI['orange'],lw=1,ls=':')
axf.scatter(pc.t_warm_245,pc.dY245,s=13,color=OI['blue'],edgecolor='white',lw=0.3,label='SSP2-4.5')
axf.scatter(pc.t_warm_585,pc.dY585,s=13,color=OI['verm'],edgecolor='white',lw=0.3,label='SSP5-8.5')
exp=pc.nsmallest(5,'dY585')
axf.scatter(exp.t_warm_585,exp.dY585,s=42,facecolors='none',edgecolors='k',lw=0.9,zorder=5)
axf.set_xlabel('Future warmest-month T (°C)'); axf.set_ylabel('Climate-driven Δyield (%)')
axf.set_title('Heat drives the loss'); axf.legend(loc='lower left')
yl=axf.get_ylim()
axf.annotate('optimum 21.9 °C',xy=(21.9,yl[0]*0.5),xytext=(22.2,yl[0]*0.3),fontsize=6,color=OI['orange'],
             arrowprops=dict(arrowstyle='-',color=OI['orange'],lw=0.6))
axf.annotate('most-exposed\nwestern plains',xy=(exp.t_warm_585.iloc[0],exp.dY585.iloc[0]),
             xytext=(25.2,0.4),fontsize=5.8,ha='center',va='center',arrowprops=dict(arrowstyle='->',lw=0.6,color='0.3'))
plab(axf,'f')

for a in [axc,axd,axe,axf]:
    for s in a.spines.values(): s.set_visible(True)
fig.savefig(PUB+'/Fig11_projection.pdf',bbox_inches='tight')
fig.savefig(PUB+'/Fig11_projection.png',dpi=600,bbox_inches='tight')
fig.savefig(PUB+'/Fig11_projection.svg',bbox_inches='tight'); plt.close()
print('Fig7 v2 saved')