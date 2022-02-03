
function main
clc
clear 
close all
tic
diary Optimal_Diary

%% --- Definition of parametes and variables
T =18; %Number of time steps considered 
c_k=[5000,2500,4000,4500,3000,5500,3000]; %intervention cost per time unit
cl_j = 5*[5000,2500,4000,4500,3000,5500,3000,5000,2500,4000,4500,3000]; % shutdown cost per time unit
G_min=[3,2,4,3,3,4,2]; %minimum intervention time gap
G_max=[5,6,6,4,3,6,4]; %maximum intervention time gap
%G_max=[3,6,4,4,2,6,2]; %maximum intervention time gap

Cent=[5]; %index of central activities (or find((G_min-G_max)==0)
Anch=[1]; %anchor of central activities: the step at which an activity must start

nvars=length(c_k)*T; %n variables
nsimulations=10; %number of tries of the GA 300 simulations=19 minutes parallel computing
noperators=3;
rng(970)

collectData=zeros(9,4); %to collect data for Table paper

G_ij=[
    1 1 0 0 0 0 1 0 0 0 0 0
    1 1 0 0 0 0 0 0 0 0 1 0
    0 0 1 0 0 0 0 0 1 0 0 0
    0 0 0 1 0 0 0 0 0 1 0 0
    0 0 0 0 1 1 0 0 0 1 0 1
    0 0 0 0 1 1 0 1 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 1 0
    0 0 0 0 0 0 0 0 0 1 0 1
    0 0 0 0 0 0 0 0 1 0 1 1
    0 0 0 0 0 0 0 0 0 1 1 1]; %matrix of interaction between subsystems i and j.

R_ik=[
    0 1 0 0 0 0 0
    0 0 1 0 0 0 0
    0 0 0 0 0 1 0
    0 0 0 0 0 1 0
    0 0 0 0 0 1 0
    0 0 0 0 0 0 0
    1 0 0 1 0 0 0
    0 0 0 1 0 0 0
    0 0 0 1 1 0 0
    0 0 0 1 0 0 0
    0 0 0 0 1 0 0 
    0 0 0 0 0 0 1]; %relation matrix indicating upon which subsystem i each activity type k intervenes

[nsubsystems,~]=size(G_ij);
%relation between operators and subsystems
delta_opi=zeros(noperators,nsubsystems);delta_opi(1,1:6)=ones(1,6);delta_opi(2,7:10)=ones(1,4);delta_opi(3,11:12)=ones(1,2);

%% --- Generating initial sets of mi (mi:solution)
np=10; %number of initial points (at least 3 and less/equal than T+2)
NonOptCosts=zeros(np,1); %to compare how the optimazation improves the initial solution


% least compact solution Gmax (standard plan)

for i=1 
     [m]=optim2(c_k,T,G_max,G_max,Cent,Anch);
     mi(:,i)=m;
end


m_ind=(reshape(mi(:,1),T,length(c_k)))'; %store the most compact qarrangement as m_ind

% totalC=TotalCost(m_ind,c_k,cl_j,G_ij,R_ik)
operator_intervention=IntervPerOper(c_k,m_ind,R_ik,delta_opi);
operator_shutdown=ShutDPerOper(cl_j,G_ij,R_ik,m_ind,delta_opi);
%collection of data
collectData(1:3,1)=operator_intervention';collectData(4,1)=sum(operator_intervention);
collectData(5:7,1)=operator_shutdown';collectData(8,1)=sum(operator_shutdown);
collectData(9,1)=collectData(4,1)+collectData(8,1);

%Most compact solution Gmin
for i=2 
     [m]=optim2(c_k,T,G_min,G_min,Cent, Anch);
     mi(:,i)=m;
end

%other random initial solutions
for i=3:np 
     [m]=optim2(c_k,T,G_max,G_min,Cent, Anch);
     mi(:,i)=m;
end

for i=1:np
    m_kt=(reshape(mi(:,i),T,length(c_k)))';
    NonOptCosts(i)=TotalCost(m_kt,c_k,cl_j,G_ij,R_ik);%associated cost
end



%% --- Optimization problem
% allocating memory for f and ms sets for final results
OptCosts=zeros(nsimulations,1); %total cost
ms=cell(1, nsimulations); %activity arrangement
parfor i=1:nsimulations
    [m_kt,fval,exitflag]=optim_GA(mi,T,c_k,cl_j,G_max,G_min,G_ij,R_ik,Cent, Anch);
    
    % Control of feasibility
    if exitflag~=-2
        OptCosts(i)=fval;
    else
        m_kt=m_kt*NaN;
        OptCosts(i)=NaN;
    end
    ms{i}=(reshape(m_kt,T,length(c_k)))';
end
%Optimal result
[MinCost,pos]=min(OptCosts);
m_kt_opt=reshape(ms{pos},length(c_k),T);
operator_intervention=IntervPerOper(c_k,m_kt_opt,R_ik,delta_opi);
operator_shutdown=ShutDPerOper(cl_j,G_ij,R_ik,m_kt_opt,delta_opi);

%collection of data
collectData(1:3,2)=operator_intervention';collectData(4,2)=sum(operator_intervention);
collectData(5:7,2)=operator_shutdown';collectData(8,2)=sum(operator_shutdown);
collectData(9,2)=collectData(4,2)+collectData(8,2);
%
collectData(:,3)=collectData(:,2)./collectData(:,1);
collectData(:,4)=collectData(:,2)-collectData(:,1); 

%Nonoptimum results
operator_intervention_list=zeros(nsimulations,3);

for i=1:nsimulations
    m_kt_opt=reshape(ms{i},length(c_k),T);
    operator_intervention_list(i,:)=IntervPerOper(c_k,m_kt_opt,R_ik,delta_opi);
    CollectData_2(i,1)=i;
    CollectData_2(i,7)=immse(operator_intervention_list(i,:), (collectData(1:3,1))');
end

%collection of data


%CollectData_2(1:nsimulations,2)=sum(operator_shutdown_list');
CollectData_2(1:nsimulations,3)=OptCosts;
CollectData_2(1:nsimulations,4:6)=operator_intervention_list;


CollectData_2 = sortrows(CollectData_2,[3,7]); %sort rows according to the total cost then according to the error
[~,idx] = unique(CollectData_2(:,3));   %remove  rows whose total cost has been used (remove duplication)
CollectData_2 = CollectData_2(idx,:); 

% Results
figure,hold on
plot(1:np,NonOptCosts,'-k','LineWidth',2),plot([1,np],OptCosts*[1 1],'-r','LineWidth',1),plot([1,np],MinCost*[1 1],'-r','LineWidth',3)
xlabel('Potential solution'),ylabel('Total Cost'), grid on, box on
figure

%compute the difference in intervention cost between the individual and
%the optimized arrangement

for k=1:length(c_k)
    ind_cost(k)=sum(m_ind(k,:))*c_k(k);
    opt_cost(k)=sum(m_kt_opt(k,:))*c_k(k);
end

%% Plots and results
%compute cost at every time step and for all iterations

f1=zeros(nsimulations,T);
f2=zeros(nsimulations,T);
f=zeros(nsimulations,T);

f1t=zeros(nsimulations,1);
f2t=zeros(nsimulations,1);
ft=zeros(nsimulations,1);

for i=1:nsimulations %nsimulations iterations
    for t=1:T % iteratre over time steps
      f1(i,t)=sum(c_k*ms{i}(:,t)); % cost of intervention activities at time step t
      f2(i,t)=sum(cl_j*Kron(G_ij'*R_ik*ms{i}(:,t)));% cost of interruption at time step t
      f(i,t)=f1(i,t)+f2(i,t); %total cost at time step t
    end
    f1t(i)=sum(f1(i,:)); % cost of intervention activities for all time steps
    f2t(i)=sum(f2(i,:)); % cost of interruption for all time steps
    ft(i)=sum(f(i,:)); % Total cost for all time steps
end


% color_str=["r","b","#77AC30","k","#4DBEEE","m","#7E2F8E"];
color_str={[0.4940 0.1840 0.5560],[0 0 1]...
    ,[0.4660 0.6740 0.1880],[0 0 0]...
    ,[0.3010 0.7450 0.9330],[1 0 1],[0.6350 0.0780 0.1840]};
legend_str=["Activity 1","Activity 2","Activity 3","Activity 4","Activity 5","Activity 6","Activity 7"];

%--Creating the vertical bars

p=zeros(length(c_k),T); %allocating memory for the plots
for k=1:length(c_k)  %iterating over number of activities
    for t=1:T
        p(k,t)=plot([t,t],2e5*[k-1,k]*ms{pos}(k,t),'color',color_str{k},'DisplayName',legend_str(k),'LineWidth',7);
        hold on
    end
    
end
xlim([1 T])
xticks(1:1:T)
xlabel('Time step','FontSize',20)
ylabel('Cumulative cost (monetary unit)','FontSize',20)
ax = gca;
ax.FontSize = 20; 
grid on
%legend(p(1:k),'FontSize',16)


p_opt_f=plot(cumsum(f(pos,:)), 'Color' , 'r','DisplayName','Total cost ({\it f_1+f_2})');
p_opt_f.LineWidth = 3;
legend (p_opt_f)
hold on


p_opt_f1=plot(cumsum(f1(pos,:)),'--','Color' , 'r','DisplayName','Cost of interventions ({\it f_1})');
p_opt_f1.LineWidth = 3;

hold on


p_opt_f2=plot(cumsum(f2(pos,:)),':','Color' , 'r','DisplayName','Total cost of service interruption ({\it f_2})');
p_opt_f2.LineWidth = 3;


hold off

txt = {'Int 1'}; text(T+0.5,100000,txt,'FontSize',15)
txt = {'Int 2'}; text(T+0.5,300000,txt,'FontSize',15)
txt = {'Int 3'}; text(T+0.5,500000,txt,'FontSize',15)
txt = {'Int 4'}; text(T+0.5,700000,txt,'FontSize',15)
txt = {'Int 5'}; text(T+0.5,900000,txt,'FontSize',15)
txt = {'Int 6'}; text(T+0.5,1100000,txt,'FontSize',15)
txt = {'Int 7'}; text(T+0.5,1300000,txt,'FontSize',15)


figure

g=zeros(1,nsimulations); %allocating memory for the plots of the cost of the different iterations

for k=1:nsimulations
      g(k)=plot(cumsum(f(k,:)), 'lineWidth',1, 'Color' , [0.7 0.7 0.7],'DisplayName','non-optimal plan');
      hold on    
end
[M,I]=min(sum(f(:,1:11),2)); %find the location of the minimum number at month 11
g(I)=plot(cumsum(f(I,:)),'--', 'lineWidth',3,'Color' , [0.7 0.7 0.7],'DisplayName','optimal plan T=11 time units');


g(pos)=plot(cumsum(f(pos,:)), 'lineWidth',3,'Color' , 'k','DisplayName','optimal plan T=18 time units');
legend(g(1:pos),'FontSize',20)
legend([g(1) g(pos) g(I)],'FontSize',20)

xlim([1 T])
xticks(1:1:T)
xlabel('Time step','FontSize',20)
ylabel('Cumulative cost (monetary unit)','FontSize',20)
bx = gca;
bx.FontSize = 20; 
grid on
hold off

function res=Kron(M)
res=M>1e-7; %1e-7 instead of 0 because of a numerical problem
end  

%% Closure
toc
save('OptimalSolution.mat')
diary off
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%---Optimization problem definition
function [m_kt,fval,exitflag]=optim_GA(m0,T,c_k,cl_j,G_max,G_min,G_ij,R_ik,Cent, Anch)
   
%--- Definition of constrains
[A,b]=DefinAb(c_k,T,G_max,G_min,Cent, Anch); %linear equalities
Aeq=[];
beq=[];

nonlcon=[]; %non linear constrains;
    
%--- Boundary constraints and vble definition
nvars=length(c_k)*T;
IntCon=1:nvars; %definition of integers
lb = zeros(1,nvars); 
ub = ones(1,nvars);

%--- Objective function
fun = @objfun;

%--- Options
[nvars,sizepopul]=size(m0);
options = optimoptions('ga','InitialPopulationMatrix',m0',...    %'Display','iter','FunctionTolerance',1e-12
    'PopulationSize',1000,'MigrationDirection','both',... % MigrationDirection: More explorative, EliteCount: how many solutions evolve to the next step (to avoid being trapped)
    'ConstraintTolerance',0,'EliteCount',ceil(0.20*sizepopul));    %'PlotFcn',@gaplotbestfun %'MaxStallGenerations',50,'FunctionTolerance',1e-10,'MaxGenerations',300,

%--- Optimatization funcion
[m_kt,fval,exitflag]=ga(fun,nvars,A,b,Aeq,beq,lb,ub,nonlcon,IntCon,options);

%--- Nested functions
%objective function
function [f]=objfun(m_kt)    
    m_kt=(reshape(m_kt,T,length(c_k)))';
    f1=sum(c_k*m_kt); %total cost of intervention activities
    f2=sum(cl_j*Kron(G_ij'*R_ik*m_kt));%total cost of interruption
    f=f1+f2;
    
    %---
    function res=Kron(M)
    res=M>1e-7; %1e-7 instead of 0 because of a numerical problem
    end  
end

end

function [m,fval]=optim2(c_k,T,G_max,G_min,Cent, Anch)%data


%--- definition of constrains
[A,b]=DefinAb(c_k,T,G_max,G_min,Cent,Anch);
    
%--- Boundary constraints.
nvars=length(c_k)*T;
lb = zeros(1,nvars); 
ub = ones(1,nvars);

%--- Objective function
fun=randi([0 1],1,nvars);


%--- Optimatization funcion
[m,fval] = linprog(fun,A,b,[],[],lb,ub);

end

function [A,b]=DefinAb(c_k,T,G_max,G_min,Cent, Anch)
n=4; %number of inequalities

%Building the Big matrix A:
 A = cell((n*length(c_k)), length(c_k));
%Dividing A into smaller matrices. A matrix for every contraint
 for j=1:length(c_k)
    for k = 1 : length(c_k)*n
        A{k,j} = zeros(T,T);
    end
 end
 

%Building the Constraints matrices
for g=1:length(c_k)
    %Constraint 1 and 2: 1=<sum (m_kt(i:i+G_min,o ,o))=<2 for i=1:T-G_min,o
    for i=1:T-G_min(1,g)+1
        A{(g-1)*n+1,g}(i,i:i+G_min(1,g)-1)=ones;   %Eq. (8)
        A{(g-1)*n+2,g}(i,i:i+G_min(1,g)-1)=-ones;  %Eq. (8)
    end
    %Constraint 3
    for i=1:T-G_max(1,g)+1
        A{(g-1)*n+3,g}(i,i:i+G_max(1,g)-1)=-ones; %Eq. (9)
    end

end
    
    %Constraint 4: extra constraint for anchoring central activities
    for g=1:length(Cent)
    A{(Cent(g)-1)*n+4,Cent(g)}(1,Anch(g))=-1;
    end
    
A=cell2mat(A);


%Buildin the Big matrix b:
 b = cell(n*length(c_k),1) ;
%Dividing b into smaller vectors. A vector for every variable

for k = 1 : n*length(c_k)
     b{k,1} = zeros(T,1) ;
end

for g=1:length(c_k)
    %Constraint 1 and 2:  
    for i=1:T-G_min(1,g)+1
        b{(g-1)*n+1,1}(i)=1;  %Eq. (8)
        b{(g-1)*n+2,1}(i)=0;  %Eq. (8)
    end
    %Constraint 3
    for i=1:T-G_max(1,g)+1
        b{(g-1)*n+3,1}(i)=-1;
    end
    

end   
%Constraint 4: extra constraint for anchoring central activities
for g=1:length(Cent)
    b{(Cent(g)-1)*n+4,1}(1)=-1; 
end

b=cell2mat(b);

%--- delete the null rows of A and B
valid=find(sum(A,2)~=0);
A=A(valid,:);
b=b(valid);

end

%--- Other functions
function totalC=TotalCost(m_kt,c_k,cl_j,G_ij,R_ik)
% [r,c]=size(m_kt);
% m_kt=(reshape(m_kt,c,r))';
f1=sum(c_k*m_kt); %total cost of intervention activities
f2=sum(cl_j*Kron(G_ij'*R_ik*m_kt));%total cost of interruption
totalC=f1+f2;
  
%---
function res=Kron(M)
res=M>1e-7; %1e-7 instead of 0 because of a numerical problem
end
end

function operator_shutdown=ShutDPerOper(cl_j,G_ij,R_ik,m_kt,delta_opi)

sh_i=cl_j.*sum(Kron(G_ij'*R_ik*m_kt)');  %total shutdown of each subsystem
operator_shutdown=sh_i*delta_opi';  % total cost of shutdown per operator

%---
function res=Kron(M)
res=M>1e-7; %1e-7 instead of 0 because of a numerical problem
end

end

function operator_intervention=IntervPerOper(c_k,m_kt,R_ik,delta_opi)

Int_i=c_k.*sum(m_kt');  %total intervention cost of each activity
aux=Kron(delta_opi*R_ik);delta_opk=aux./sum(aux); %relation operators and activities
operator_intervention=Int_i*delta_opk';  % total cost of shutdown per operator


%---
function res=Kron(M)
res=M>1e-7; %1e-7 instead of 0 because of a numerical problem
end
end
