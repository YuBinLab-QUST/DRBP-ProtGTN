clear all
clc
lamdashu=5;
%Reading protein sequence of PSSM
WEISHU=186;
name=textread('','%s')    %Read protein name
for i=1:186
    nnn=name(i);
    nnn1=strcat(nnn,'.pssm');  %Read PSSM file  
    %nnn1 = sprintf('%d.pssm', i);
    nnn1=char(nnn1);
    disp(nnn1)
    fid{i}=importdata(nnn1);
end
%All protein sequences normalized
c=cell(WEISHU,1);
for t=1:WEISHU
    clear shu d
shu=fid{t}.data;
%disp(shu)
%Know the quantity of each protein, the extracted matrix, pay attention to the order of the protein
[M,N]=size(shu);
shuju=shu(1:M-5,1:20);
d=[];
%Normalized
for i=1:M-5
   for j=1:20
       d(i,j)=1/(1+exp(-shuju(i,j)));
   end
end
c{t}=d(:,:);
end
%Generate PSSM-AAC
for i=1:WEISHU
[MM,NN]=size(c{i});
 for  j=1:20
   x(i,j)=sum(c{i}(:,j))/MM;
 end
end
%After PsePSSM 20*lamda
xx=[];
sheta=[];
shetaxin=[];
for lamda=1:lamdashu;
for t=1:WEISHU
  [MM,NN]=size(c{t});
  clear xx
   for  j=1:20
      for i=1:MM-lamda
       xx(i,j)=(c{t}(i,j)-c{t}(i+lamda,j))^2;
      end
      sheta(t,j)=sum(xx(1:MM-lamda,j))/(MM-lamda);
   end
end
shetaxin=[shetaxin,sheta];
end
psepssm=[x,shetaxin];
%xlswrite('psepssm.xlsx',psepssm)
csvwrite('PsePSSM186.csv',psepssm)
      
