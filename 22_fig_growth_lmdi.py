import json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.3,
 'axes.linewidth':0.7,'legend.frameon':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00','sky':'#56B4E9','grey':'#7A7A7A'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-30,5),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
def sen(x,y):
    from itertools import combinations;x=np.asarray(x,float);y=np.asarray(y,float);m=~np.isnan(y);x,y=x[m],y[m]
    s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(range(len(x)),2)]);return s,np.median(y)-s*np.median(x)
mk=json.load(open(MID+'/mk_trends.json')); h=json.load(open(MID+'/headline_numbers.json'))
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); pa['CountyCode']=pa['CountyCode'].astype(str); rs=pa.groupby('year')['paddy_ha'].sum().reset_index()
yb=pd.read_csv(MID+'/yield_panel_province.csv')
fig,ax=plt.subplots(2,2,figsize=(DC*0.78,DC*0.60),constrained_layout=True)
def tp(a,x,y,c,ylab,ttl,star=None):
    a.plot(x,y,'o-',color=c,mfc=c,mec='white',mew=0.4,ms=3,zorder=3)
    s,ic=sen(x,y); xr=np.array([min(x),max(x)]); a.plot(xr,ic+s*xr,'--',color='0.4',lw=0.9)
    a.set_ylabel(ylab); a.set_title(ttl); a.set_xlabel('Year'); a.xaxis.set_major_locator(MaxNLocator(5))
    if star: a.text(0.05,0.9,star,transform=a.transAxes,fontsize=6.2,va='top',color='0.3')
tp(ax[0,0],rs.year,rs.paddy_ha/1e4,OI['blue'],'Paddy area (10$^4$ ha)','Paddy area','MK ***')
ax[0,0].plot(yb.year,yb.area_ha/1e4,'s',color=OI['sky'],mec='white',mew=0.3,ms=3,zorder=3)
ax[0,0].plot([],[],'o-',color=OI['blue'],label='Satellite'); ax[0,0].plot([],[],'s',color=OI['sky'],label='Yearbook'); ax[0,0].legend(loc='lower right')
plab(ax[0,0],'a')
yy=yb.dropna(subset=['prod_kg']); tp(ax[0,1],yy.year,yy.prod_kg/1e9,OI['verm'],'Production (10$^9$ kg)','Rice production'); plab(ax[0,1],'b')
yz=yb.dropna(subset=['yield_kg_ha']); tp(ax[1,0],yz.year,yz.yield_kg_ha/1000,OI['green'],'Unit yield (t ha$^{-1}$)','Unit yield'); plab(ax[1,0],'c')
# d: LMDI waterfall
yb2=yb.dropna(subset=['area_ha','prod_kg']).sort_values('year'); lo3=yb2.head(3); hi3=yb2.tail(3)
A0=lo3.area_ha.mean(); P0kg=lo3.prod_kg.mean(); A1=hi3.area_ha.mean(); P1kg=hi3.prod_kg.mean(); Y0=P0kg/A0; Y1=P1kg/A1
Lw=(P1kg-P0kg)/(np.log(P1kg)-np.log(P0kg)); dA=Lw*np.log(A1/A0); dY=Lw*np.log(Y1/Y0); tot=P1kg-P0kg
P0=P0kg/1e9; P1=P1kg/1e9; Ae=dA/1e9; Ye=dY/1e9; area_share=100*dA/tot; yield_share=100*dY/tot
axd=ax[1,1]; labels=['1985–90','Area','Yield','2018–20']; bottoms=[0,P0,P0+Ae,0]; heights=[P0,Ae,Ye,P1]; colors=[OI['grey'],OI['blue'],OI['green'],OI['grey']]
for i,(b,hh,c) in enumerate(zip(bottoms,heights,colors)): axd.bar(i,hh,bottom=b,color=c,width=0.62,edgecolor='white')
for i,cum in enumerate([P0,P0+Ae]): axd.plot([i+0.31,i+1-0.31],[cum,cum],color='0.5',lw=0.7,ls='--')
for i,(b,hh) in enumerate(zip(bottoms,heights)):
    lab=f'{hh:.1f}'+(f'\n{area_share:.0f}%' if i==1 else (f'\n{yield_share:.0f}%' if i==2 else ''))
    axd.text(i,b+hh+0.1,lab,ha='center',fontsize=6.2)
axd.set_xticks(range(4)); axd.set_xticklabels(labels); axd.set_ylabel('Production (10$^9$ kg)'); axd.set_title('LMDI decomposition'); axd.set_ylim(0,P1*1.18); plab(axd,'d')
fig.savefig(PUB+'/Fig2_growth_lmdi.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig2_growth_lmdi.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/Fig2_growth_lmdi.svg',bbox_inches='tight'); plt.close()
print('Fig2_growth_lmdi done')
