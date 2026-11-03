"""Posture ledger 1958-2023 under the coding rule of Appendix A.
Limited conflict (L): the year contains at least one of
  L-a  North Korean armed action that killed ROK or U.S./UNC military or police personnel;
  L-b  an engagement between the two sides' regular naval or ground combat units with fatalities on either side;
  L-c  a North Korean state-directed attack (bombing, raid, assassination) that killed five or more people or targeted the ROK head of state.
Tension (T): not L, and at least one nuclear test, or at least three ballistic missiles (incl. satellite launch vehicles)
  launched (CNS/NTI database), or five or more provocation events in the CSIS Beyond Parallel record.
Peacetime (P): otherwise.
Sources: CSIS = CSIS Beyond Parallel provocations database; CRS = Nanto, CRS RL30004 (2003);
NTI = CNS/NTI North Korea missile test database; FRUS = FRUS 1961-63 vol. XXII, doc. 312;
KWE = Korean War Educator, U.S. hostile deaths in the DMZ since 1968 (cross-check for omissions).
"""
import json, pathlib, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / 'source'
L = {
 1960:('L-b','30 Jul: ROK destroyer sinks a DPRK patrol vessel in the first post-armistice naval clash; at least 14 DPRK crew killed','CSIS'),
 1963:('L-a','29-30 Jul: DMZ ambush (two U.S. soldiers killed) and follow-up clash (one U.S. soldier and one ROK policeman killed)','CSIS; FRUS'),
 1966:('L-a','2 Nov: two DMZ ambushes kill six U.S. soldiers, one KATUSA, and two ROK soldiers','CSIS; Bolger'),
 1967:('L-a','DMZ ambushes and barracks bombing kill U.S. and ROK soldiers and police (e.g., 22 May, 10 Aug, 28 Aug)','CSIS'),
 1968:('L-c','21 Jan Blue House raid targeting the president; Jan Pueblo seizure; DMZ ambushes; Oct-Nov Uljin-Samcheok landings','CSIS; CRS'),
 1969:('L-a','15 Mar DMZ fire kills a U.S. soldier; 18 Mar raid kills a ROK policeman; 15 Apr EC-121 shot down (31 U.S. killed)','CSIS; CRS'),
 1970:('L-a','29 Jun sea battle near Yeongheung Island; two ROK policemen and one militiaman killed','CSIS'),
 1971:('L-a','16 Aug DMZ firefights; two ROK soldiers killed','CSIS'),
 1973:('L-a','7 Mar DPRK gunfire kills two ROK soldiers on a DMZ work detail','CSIS'),
 1974:('L-c','15 Aug assassination attempt on President Park (First Lady killed); 28 Jun coast-guard vessel sunk with crew; 20 Nov tunnel booby trap kills U.S. and ROK officers','CSIS; CRS'),
 1975:('L-a','Sep Gochang interception; two ROK soldiers killed (CRS; CSIS reports four wounded)','CRS; CSIS'),
 1976:('L-a','18 Aug Panmunjom axe murders kill two U.S. officers; Jun DMZ encounters kill ROK soldiers','CSIS; CRS'),
 1977:('L-a','14 Jul U.S. Army CH-47 shot down after straying north of the MDL; three U.S. soldiers killed','KWE; AP (Deseret News, 30 Dec 1994)'),
 1978:('L-a','28 Apr firefight with an infiltration boat off Yeosu; one ROK seaman killed','CSIS'),
 1979:('L-a','21 Jul gun battle with a spy vessel off Namhae; two ROK soldiers killed','CSIS'),
 1980:('L-a','Dec: three DPRK agents intercepted off the South Gyeongsang coast near Namhae; two ROK soldiers killed and two wounded','CRS'),
 1983:('L-c','9 Oct Rangoon bombing targeting President Chun; 17 ROK officials killed','CSIS; CRS'),
 1984:('L-a','23 Nov Panmunjom firefight after a Soviet defection; one ROK soldier killed','CSIS'),
 1986:('L-c','14 Sep Gimpo airport bombing; five killed (attributed to North Korea by ROK police, as reported by CRS)','CSIS; CRS'),
 1987:('L-c','29 Nov KAL 858 bombing; 115 killed','CSIS; CRS'),
 1994:('L-a','17 Dec U.S. Army OH-58 shot down after straying north; one U.S. pilot killed','CSIS; KWE'),
 1995:('L-a','24 Oct Buyeo counter-infiltration operation (one DPRK agent killed, one captured); two ROK policemen killed by the agents\' gunfire','CRS; Yonhap (24 Feb 2017)'),
 1996:('L-a','Sep-Nov Gangneung submarine infiltration; ROK military personnel and civilians killed in the manhunt','CSIS; CRS'),
 1999:('L-b','15 Jun first Yeonpyeong naval battle between regular naval units; DPRK fatalities','CSIS'),
 2002:('L-a','29 Jun second Yeonpyeong naval battle; six ROK sailors killed','CSIS; CRS'),
 2009:('L-b','10 Nov Daecheong naval engagement; one DPRK sailor killed','CSIS'),
 2010:('L-a','26 Mar Cheonan sinking (46 ROK sailors killed); 23 Nov Yeonpyeong shelling','CSIS'),
}
# checked and NOT limited conflict under the rule (fatalities only among infiltrators, or civilians below the L-c threshold)
NOT_L = {
 1966:'17 May infiltrator killed (also L-a event in Nov)',
 1981:'infiltrators killed (CRS)', 1982:'one infiltrator killed (CRS)',
 1985:'spy ship sunk; no ROK deaths', 1992:'22 May three DPRK infiltrators killed in ROK uniforms; ROK soldiers wounded',
 1997:'Feb defector Lee Han-young shot (one killed, below L-c threshold); DMZ exchanges without deaths',
 1998:'Jun midget-submarine crew found dead (no engagement); Dec semi-submersible sunk (DPRK crew only)',
 2014:'exchanges of fire without deaths', 2015:'landmine (two ROK soldiers maimed, none killed); artillery exchange without deaths',
}
csis=pd.read_excel(SRC / 'csis_beyond_parallel_provocations.xlsx')
csis['year']=pd.to_datetime(csis['Date'],errors='coerce').dt.year
n_events=csis.groupby('year').size()
nuc=csis[csis.Type.str.contains('Nuclear',na=False)].groupby('year').size()
m=pd.read_excel(SRC / 'cns_nti_missile_tests.xlsx',sheet_name='Missile Tests',header=1)
m['year']=pd.to_datetime(m['Date'],errors='coerce').dt.year
n_bm=m.groupby('year').size()
rows=[]; code={}
for y in range(1958,2024):
    ne=int(n_events.get(y,0)); nn=int(nuc.get(y,0)); nb=int(n_bm.get(y,0))
    if y in L: st='L'; crit,ev,src=L[y]
    elif nn>0 or nb>=3 or ne>=5:
        st='T'; crit='T-'+('nuclear' if nn>0 else ('missiles' if nb>=3 else 'events'))
        ev=f'nuclear tests {nn}; ballistic missiles/SLVs {nb}; CSIS events {ne}'+('; '+NOT_L[y] if y in NOT_L else ''); src='CSIS; NTI'
    else:
        st='P'; crit='P'; ev=f'CSIS events {ne}; ballistic missiles/SLVs {nb}'+('; '+NOT_L[y] if y in NOT_L else ''); src='CSIS; NTI'+('; CRS' if y in (1981,1982) else '')
    code[y]=st; rows.append(dict(year=y,posture=st,criterion=crit,decisive_events=ev,csis_events=ne,nuclear_tests=nn,ballistic_launches=nb,sources=src))
df=pd.DataFrame(rows); df.to_csv(HERE / 'posture_ledger.csv',index=False)
alt={int(k):'PTL'[v-1] for k,v in json.load(open(HERE / 'posture_coding_alt.json')).items()}
ch=[(y,alt[y],code[y]) for y in code if alt[y]!=code[y]]
print('counts', {s:sum(1 for v in code.values() if v==s) for s in 'PTL'}); print('differs from the alternative coding:',ch)
json.dump({str(y):{'P':1,'T':2,'L':3}[s] for y,s in code.items()},open(HERE / 'posture_coding.json','w'))
