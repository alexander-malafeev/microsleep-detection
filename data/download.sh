
file_list=(0ncr.mat  5bSg.mat  Af8a.mat  C1Wu.mat  DjrT.mat\
	EMcQ.mat  G7PJ.mat  ibbz.mat  JCpz.mat  MS6u.mat  Nzhl.mat\
	QDsa.mat  SOZ3.mat  uXdB.mat  y5We.mat  ZYFG.mat 0pai.mat\
	6iwd.mat  AsLD.mat  cblr.mat  Dr51.mat  EyTS.mat  go56.mat\
	IhpU.mat  kj2l.mat  Msy4.mat  oOMR.mat  R2f3.mat  svlu.mat\
	X7s0.mat  YHLr.mat 3J4W.mat  6JVj.mat  AXbm.mat  csxQ.mat\
	DSfb.mat  f8H5.mat  hcml.mat  ipxV.mat  Kn9O.mat  muls.mat\
	Otq3.mat  RfL0.mat  tG6i.mat  Xg1l.mat  YOh8.mat 3P0D.mat\
	9098.mat  bkx9.mat  d3ET.mat  DYYI.mat  fNe4.mat  hRMy.mat\
	iSqw.mat  LR2s.mat  mZje.mat  pPpj.mat  RM1S.mat  UsSz.mat\
	Xii6.mat  zaca.mat 40kO.mat  9JQY.mat  BSvO.mat  ddTG.mat\
	EHED.mat  fT68.mat  hT38.mat  Ivfn.mat  mBks.mat  N1nM.mat\
	puoa.mat  sNMf.mat  UwK6.mat  Y4FK.mat  Zpwh.mat)

mkdir files

for file in "${file_list[@]}"; 
do 
	wget "https://zenodo.org/record/3251716/files/$file" --directory-prefix=./files/; 
done
