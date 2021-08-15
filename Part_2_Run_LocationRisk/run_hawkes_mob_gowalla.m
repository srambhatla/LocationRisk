%function [error_In, risk_In_all, risk_In_pred, mus, mob_val, fK0, infections, r, mdl] = run_hawkes_mob(Alpha, Beta, all_mobility, DaysPred)
wr = 0; % plot stuff?
true_days = 900; % number of days in the simulation
Alpha = 10 ; % 8, 4 2,2 worked with drycorrect 5? 9, 2, 15/18
Beta = 1 ;
dry_correct = 5;
Delta =  10; % the time delay parameter
EMitr =  500; % number of EM iterations
SimTimes =  100;

days_for_exp = 60; % Days for this particular experiment

% number of mobility indices
mob_l = 5;

DaysPred =  5;
all_mobility =4; % 1 (one type), 2 (three types), 4 (5 types, our main)

clus = '5';

time = '2021_06_08_12_31.csv'; % 5 (for exps 2)

lst = dir(['./']);
[~,~,~,~,m_file, i_file] = lst.name;
save_K0_file = ['./covid_data/gowalla/', 'risk_exp2', '.csv'] ;
inf_file = ['./covid_data/gowalla/cluster_inf_events_',clus, '_',time ] ;
checkin_file = ['./covid_data/gowalla/cluster_checkin_events_',clus, '_',time ] ;

% Load data
OutputPath_mdl =  './gowalla/output/mdl.mat';
OutputPath_pred = './gowalla/output/pred.csv';


% Preprocess data

% Read-in mobility
InputPath_mobility = checkin_file;
Mobi = readtable(InputPath_mobility,'ReadVariableNames',true);
Mobi = Mobi(:,1:days_for_exp+2);

% infections
inf_data = readtable(inf_file, 'ReadVariableNames', true);
inf_data = table2array(inf_data);
inf_data = inf_data(1:days_for_exp,:);
day_timestamps = inf_data(:,1);
infections = smoothdata(inf_data(:,2:end), 'movmedian', 6)';
covid = infections(:,1:end-DaysPred);


days = size(infections,2); % number of days

close all
plot(infections')

%% Read in parameter
EMitr = EMitr;
if strcmp(Alpha,'') && strcmp(Beta,'')
	disp('No shape and scale parameter for Weibull distribution provided. Use MLE to infer alpha and beta ... ')
    alphaScale_in = 0;
    betaShape_in  = 0;
else
	alphaScale_in = Alpha;
	betaShape_in  = Beta;
end 

if strcmp(Delta,'') 
        disp('No shift parameter for mobility provided.  It will set to zero ... ')
	mobiShift_in = 0;
else
	mobiShift_in = Delta;
end

%%
% Pad to shift 
mob_head = Mobi(:,1:2);
%mob_val = table2array(Mobi(:,7:end));

mob_val = table2array(Mobi(:,3:end));
mob_val = smoothdata(mob_val', 'movmedian', 6)';
%mob_val = (mob_val - repmat(mean(mob_val'), [31 1])');

ii = 1;
if all_mobility == 2
    
mob_val_new = zeros(size(infections,1)*3, size(infections,2));
VarNamesOld = [ {['checkins']}; {['to']}; {['from']}; {['Qprob']}];

    for id = 1:str2double(clus)
        idx = 5*(id-1) + 1;
        mob_val_new(ii,:) = mob_val(idx+1,:);
        mob_val_new(ii+1, :) = (mob_val(idx+2,:))';
        mob_val_new(ii+2, :) = (mob_val(idx+3,:))';
    
        ii = ii + 3;
    end

elseif all_mobility == 1 % ratio
    mob_val_new = zeros(size(infections,1), size(infections,2));
    VarNamesOld = [ {['checkins']}; {['Qprob']}];
    
    for id = 1:str2double(clus)
        idx = 5*(id-1) + 1;
        mob_val_new(ii,:) = mob_val(idx+1,:);
       % mob_val_new(ii+1, :) = ((mob_val(idx+2,:)) + (mob_val(idx+3,:)))./mob_val(idx+1,:);

        ii = ii + 1;
    end
elseif all_mobility == 3 % ratio
    mob_val_new = zeros(size(infections,1)*16, size(infections,2));
    Mobi_Type_list = table2cell(Mobi(2:17,2));
    VarNamesOld = [ Mobi_Type_list; {['Qprob']}];
    %VarNamesOld = [ {['checkins']}; {['ratio']}; {['Qprob']}];
    
    for id = 1:str2double(clus)
        idx = mob_l*(id-1) + 1;
        mob_val_new(ii,:) = mob_val(idx+1,:);
        mob_val_new(ii+1:ii+15, :) = ((mob_val(idx+2:idx+16,:)));%+(mob_val(idx+17:idx+31,:)))./repmat(mob_val(idx+1,:), [size(infections,1),1]);

        ii = ii + 16;
    end
elseif all_mobility==4
    mob_val_new = zeros(size(infections,1)*5, size(infections,2));
    VarNamesOld = [ {['checkins']}; {['to']}; {['from']}; {['inf_mob']}; {['self_inf_mob']};{['Qprob']}];
    %VarNamesOld = [ {['checkins']}; {['to']}; {['from']}; {['Qprob']}];
    %VarNamesOld = [{['inf_mob']}; {['self_inf_mob']};{['Qprob']}];
    %VarNamesOld = [{['inf_mob']}; {['self_inf_mob']};{['Qprob']}];
    %VarNamesOld = [ {['inf_mob']}; {['Qprob']}];
    for id = 1:str2double(clus)
        idx = 5*(id-1) + 1;
        mob_val_new(ii,:) = mob_val(idx,:);
        mob_val_new(ii + 1, :) = (mob_val(idx+1,:))';
        mob_val_new(ii + 2, :) = (mob_val(idx+2,:))';
        mob_val_new(ii + 3, :) = (mob_val(idx+3,:))';
        mob_val_new(ii + 4, :) = (mob_val(idx+4,:))';
        ii = ii + 5;
    end
else
    mob_val_new = zeros(size(infections,1), size(infections,2));
    VarNamesOld = [ {['checkins']}; {['Qprob']}];

    for id = 1:str2double(clus)
        idx = 4*(id-1) + 1;
        mob_val_new(ii,:) = mob_val(idx+1,:);
    
        ii = ii + 1;
    end
end

mob_val = mob_val_new;

for pad = 1:mobiShift_in
    mob_val = [ mean(mob_val(:,1:7),2) mob_val ];
end
mob_val(4:5:end,1:Delta) = 0;
mob_val(5:5:end,1:Delta) = 0;

% Get Key and Date

Mobi_Type_list = table2cell(Mobi(1:mob_l,2));
Mobi_Date_list = Mobi.Properties.VariableNames(3:end);
Mobi_Key_list = table2cell(Mobi(1:mob_l:end,1));
%%
% Get number of counties and number of days
[n_cty, n_day]=size(covid);
n_mobitype = size(mob_val,1)/n_cty;

disp(['There ' num2str(n_cty) ' clusters, ' num2str(n_mobitype) ' types of Mobility indices, and ' num2str(n_day) ' days in the covid reports.' ])

% Train & Test Split
n_tr = size(covid,2);
mob_tr = mob_val(:, 1:n_tr);
mob_te = mob_val(:, n_tr+1:n_tr+DaysPred);


% Normalization
mob_tr_reshape = reshape(mob_tr, n_mobitype, size(mob_tr,1)/n_mobitype * size(mob_tr,2) ).';
mob_te_reshape = reshape(mob_te, n_mobitype, size(mob_te,1)/n_mobitype * size(mob_te,2) ).';

covid_tr = covid;
%
Covar_tr = [mob_tr_reshape];
Covar_te = [mob_te_reshape];
%
Covar_tr_mean = mean(Covar_tr,1);
Covar_tr_std = std(Covar_tr,1);
%
Covar_tr = (Covar_tr-Covar_tr_mean) ./ Covar_tr_std;
Covar_te = (Covar_te-Covar_tr_mean) ./ Covar_tr_std;


% Get Variable names
%Mobi_Type_list = {'checkins', 'ratio'};
%VarNamesOld = [ Mobi_Type_list; {['Qprob']}];

VarNames=[];
% Rename
for i = 1:size(VarNamesOld,1)
    newStr = replace( VarNamesOld{i} , ' & ' , '_' );
    newStr = replace( newStr , ' ' , '_' );
    newStr = regexprep(newStr, '^_', '');
    VarNames=[VarNames; {newStr}];
end


%% Define Parameters
n_day_tr = n_day;
T = n_day_tr;
% Boundary correction, the number of days before the total number of days (n_day)
%dry_correct = 5;

% EM step iterations
emiter = EMitr; %Nt=covid(:,2:end);
break_diff = 10^-4;
% Boundary correction: T-dry_correct
% Mobility has only 6 weeks so we take the less by min(T-dry_correct, size(mobi_in,2) )
day_for_tr = min(T-dry_correct, size(mob_tr,2) );

%% Initialize Inferred Parameters

if (alphaScale_in==0) && (betaShape_in==0)
    % Weibull Distribution (scale, shape) parameters as (alphas, betas)
    alpha  = 2;
    beta = 2;
else
    alpha  = alphaScale_in;
    beta = betaShape_in;
end
% K0 reproduction number, a fuction of time and mobility.
% Estimat for each county at each day.
K0 = ones(n_cty, n_day_tr);

% p is the n_day by n_day matrix.
% p( i, j), i > j stands the probability of ONE SIGLE event at day i triggered by ALL events at day j
% i.e., Prob for (j_1, j_2, ...) triggered by each i
p=[];
for i = 1:n_cty
    p{i} = zeros(n_day_tr,n_day_tr);
end

% q is the n_day by n_day matrix.
% q( i, j), i > j stands the probability of ONE SIGLE event at day i triggered by ONE SIGLE event at day j
% i.e., Prob for each j triggered by each i
q=[];
for i = 1:n_cty
    q{i} = zeros(n_day_tr,n_day_tr);
end


% Mu is the back ground rate
mus=0.5*ones(n_cty,1);

% lam is the event intensity
lam = zeros(n_cty,T);

%% EM interation
alpha_delta = []; alpha_prev = [];
beta_delta = [];  beta_prev = [];
mus_delta = [];   mus_prev = [];
K0_delta = [];    K0_prev = [];
theta_delta = []; theta_prev = [];
for itr = 1:emiter
    tic
    %% E-step
    % county levelitr
    for c = 1:n_cty
        if( sum(covid_tr(c,:)) ~= 0)
            [p{c}, q{c}, lam(c,:)] = updatep( covid_tr(c,:) , p{c}, q{c}, K0(c,:), alpha, beta, mus(c) );
        end
    end
    
    %% M-step
    % Calculate Q, which stands for the average number (observed) of children generated by a SINGLE event j
    % Note taht the last "dry_correct" days of Q will be accurate
    % Since we haven't observed their children yet
    Q = [];
    for c = 1:n_cty
        Qprob = q{c} - diag(diag(q{c}));
        
        % Note that q is prob for one event to one event
        % The average number (observed) of children generated by j would be q(i,j)*t(i)
        n_i = covid_tr(c,:).';
        Q = [Q; sum( Qprob.* n_i, 1)];
        
    end
    
    
    %% Estimate K0 and Coefficients in Possion regression
    
    % parameters for possion regression
    opts = statset('glmfit');
    opts.MaxIter = 300;
    
    % boundaty correct
    glm_tr = Covar_tr(1: n_cty*day_for_tr ,:);
    glm_y = Q(:, 1:day_for_tr);
    glm_y = reshape(glm_y, prod(size(glm_y)), 1);
    
    % weight for observation, which is the number of evets at day j
    freqs = covid_tr(:, 1:day_for_tr);
    freqs = reshape(freqs, prod(size(freqs)), 1);
    
    mdl = fitglm( glm_tr, glm_y,'linear', 'Distribution', 'poisson', 'options', opts, 'VarNames', VarNames, 'Weights', freqs);
    
    %% Estimate K0
    %mdl = fitglm( Covar_tr(1: n_cty*day_for_tr ,6:7), glm_y,'linear', 'Distribution', 'poisson', 'options', opts, 'Weights', freqs);
    [ypred,yci] = predict(mdl,Covar_tr);
    K0 = reshape(ypred, n_cty, n_day_tr);

    %Bound K0
    K0 = smoothdata(K0, 2);
    %
    %% Estimate mu, the background rate
    
    for c = 1:n_cty
        mus(c) = sum(( diag(p{c}).' .* covid_tr(c,:) )) / (n_day_tr) ;
        %mus(c) = sum(( diag(p{c}) )) / (n_day_tr) ;
    end
    
    %% Take all the average
    %%mus = repmat( mean(mus), size(mus, 1), size(mus, 2) );
    
    %% Estimate alpha and beta in Weibull Distribution
    
    % Get all pairs
    combos = sortrows( nchoosek( (1:n_day_tr),2 ),2);
    
    % Get those within boundary
    combos = combos(find( prod(combos<=day_for_tr,2)),: );
    combos = combos(:,end:-1:1);
    
    % allocate memory first for spead-up
    obs_sample = zeros( size(combos,1)*n_cty ,1 );
    freq_sample = zeros( size(combos,1)*n_cty ,1 );
    
    for c = 1:n_cty
        
        nc = covid_tr(c,:);
        
        % obs vation is i-j
        obs =[combos(:,1) - combos(:,2)];
        
        % freq
        Prob = p{c};
        freq = Prob(  sub2ind(size(Prob), combos(:,1), combos(:,2) ) ) .* nc(combos(:,1)).';
        if(sum(isnan(freq))>0)
            disp(sum(isnan(freq)))
        end
        % Record obs, freq
        obs_sample( (c-1)*size(combos,1)+1:c*size(combos,1) ) = obs;
        freq_sample( (c-1)*size(combos,1)+1:c*size(combos,1) ) = freq;
        
    end
    
    if (alphaScale_in==0) && (betaShape_in==0)
        % Fit weibull
        [coef,~] = wblfit(obs_sample,[],[],freq_sample);
        alpha=coef(1); % scale
        beta=coef(2);  % shape
        if beta > 100
            beta = 100;
        end
        if alpha > 100
            alpha = 100;
        end
    else
        alpha  = alphaScale_in;
        beta = betaShape_in;
    end
    
    
    
    %% check for the convergence
    if(itr==1)
        % save the first value
        alpha_prev = alpha; beta_prev = beta; mus_prev = mus; K0_prev = K0; theta_prev = mdl.Coefficients.Estimate;
    else
        % Calculate the RMSR
        alpha_delta = [alpha_delta sqrt( (alpha-alpha_prev).^2 )];
        beta_delta  = [beta_delta  sqrt( (beta-beta_prev).^2 )];
        mus_delta   = [mus_delta sqrt( sum((mus_prev - mus).^2)/numel(mus) )];
        K0_delta    = [K0_delta sqrt( sum(sum((K0_prev - K0).^2))/numel(K0) )];
        theta_delta   = [theta_delta sqrt( sum((theta_prev - mdl.Coefficients.Estimate).^2)/numel(mdl.Coefficients.Estimate) )];
        % save the current
        alpha_prev = alpha; beta_prev = beta; mus_prev = mus; K0_prev = K0; theta_prev = mdl.Coefficients.Estimate;
    end
    %disp(max(K0(:)))
    [wblstat_M, wblstat_V]=wblstat(alpha, beta);
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%

    %%%%%%%%%%%%%%%%%%
    % Early Stop
    %%%%%%%%%%%%%%%%%%

    if (itr > 5)
        rule = all( alpha_delta(end-4:end) < break_diff) & all( beta_delta(end-4:end) < break_diff);
        rule = rule &  all( mus_delta(end-4:end) < break_diff) & all( K0_delta(end-4:end) < break_diff);
        rule = rule &  all( theta_delta(end-4:end) < break_diff);

        if( rule )
            disp(['Convergence Criterion Meet. Break out EM iteration ...'])
            break;
        end
    end
    t = toc;
    %disp(['Iterattion ' num2str(itr) ', Elapse time: ' num2str(t)])
end
if(itr == emiter)
    disp(['Reach maximun EM iteration.'])
end
%mdl
%save(OutputPath_mdl,'mus','alpha','beta','K0','mdl','VarNames','alpha_delta','beta_delta','mus_delta','K0_delta','theta_delta')

% Start Simulation
%load(OutputPath_mdl, 'mus','alpha','beta','K0','mdl','VarNames','alpha_delta','beta_delta','mus_delta','K0_delta','theta_delta')

%% Get K0
Covar_all = [Covar_tr; Covar_te];
n_day = n_day_tr+DaysPred;
T_sim = n_day;
%% Predict
[ypred,yci] = predict(mdl,Covar_all);
fK0 = reshape(ypred, n_cty, n_day);

% Simulation results
sim = zeros(n_cty, T_sim, SimTimes);

% Loop for simulation
for c = 1:n_cty
    tic
    tr_in = covid_tr(c,:);
    for itr = 1:SimTimes
        rng(itr);
        [times_sim] = Hawkes_Sim_Corona( mus(c), alpha, beta, T_sim, fK0(c,:), T_sim-DaysPred, itr,  tr_in);
        %
        [Nt]=discrete_hawkes(times_sim,T_sim);
        %
        sim(c,:,itr) = Nt;
    end
    t = toc;
   % disp(['Simulation county ' num2str(c) ', Elapse time: ' num2str(t)])
end

% Format the output 
sim_mean = mean(sim,3);
sim_mean = sim_mean(:, end-DaysPred+1:end);

sim_std = std(sim,1,3);
sim_std = sim_std(:, end-DaysPred+1:end);
mdl
%close all 
%plotResiduals(mdl,'probability')
%% plot stuff

if wr == 1
close all
hh = figure('Renderer', 'painters', 'Position', [10 10 900 500])
ax = gca;
filename = './figure/preds_hawkes_cluster.gif';
for i = 1 : size(sim_mean,1)

%mae = mean(abs(sim_mean(i,:)- infections(i,end-6:end)));
mae = mean(abs(sim_mean(i,:)- infections(i,end-DaysPred+1:end))./infections(i,end-DaysPred+1:end));
X = [1:(DaysPred+n_tr), fliplr(1:(DaysPred+n_tr))];
Y = [[infections(i,1:end-DaysPred), sim_mean(i,:)] + [zeros(1, n_tr), sim_std(i,:)], ...
    fliplr([infections(i,1:end-DaysPred), sim_mean(i,:)]- [zeros(1, n_tr), sim_std(i,:)])];
h = fill(X,Y,'r', 'LineWidth', 0.1);
hold all
plot([infections(i,1:end-DaysPred), infections(i,end-DaysPred+1:end)], 'LineWidth', 2, 'Marker', 'o')
plot([infections(i,1:end-DaysPred), sim_mean(i,:)], 'LineWidth', 2, 'Marker', 'o')
%plot(0.005*mob_val(i,:), 'LineWidth', 2, 'Marker', '+')
set(h, 'FaceAlpha', 0.1)
set(gca, 'FontSize', 14)
ax.XLim = [0 true_days];
ax.YLim = [0,max(floor(infections(:)))];

ylabel('Infections')
plot(n_tr*ones(max(floor(infections(:))),1), 0:max(floor(infections(:)))-1, 'k')
grid minor
xlabel('Days')
%axis tight
title({['7-day Predictions for Grid = ', num2str(i)], ['Relative MAE = ', num2str(mae)]})
hold off
pause(0.5)
% subplot(212)
% plot(K0(i,:))
% set(gca, 'FontSize', 14)
% ylabel('R0')
% xlabel('Days')
% axis tight
legend('Standard Deviation', 'Groundtruth', 'Predicted')
drawnow
if wr == 1
    % Capture the plot as an image 
      frame = getframe(hh); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if i == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',1.5); 
      end 
end  
hold off
end
end
%error = sqrt(norm(sim_mean - infections(:,end-6:end), 'fro'))% /norm(infections(:,end-6:end), 'fro')
%error = sqrt(mean(sum((abs(sim_mean - infections(:,end-DaysPred+1:end))./infections(:,end-DaysPred+1:end)).^2,2)))
%%
risk = @(x) (x - min(x(:)))./(max(x(:)) - min(x(:)));
div_max = @(x) x./max(x)
%fK0_sm = smoothdata(fK0,2);
fK0_sm = fK0;
%pre_r = (fK0_sm./max(fK0_sm(:))) + (repmat(mus./(max(mus(:))), [1, days]));
pre_r = (fK0_sm);
r = risk(pre_r);

%% Get Lambda for evaluating risk
p=[];
for i = 1:n_cty
    p{i} = zeros(DaysPred,DaysPred);
end

% q is the n_day by n_day matrix.
% q( i, j), i > j stands the probability of ONE SIGLE event at day i triggered by ONE SIGLE event at day j
% i.e., Prob for each j triggered by each i
q=[];
for i = 1:n_cty
    q{i} = zeros(DaysPred,DaysPred);
end

ex_sim_mean = [covid_tr(:,end:end) sim_mean];
lam_pred = zeros(str2num(clus),DaysPred + 1);

for c = 1:n_cty
        if( sum(sim_mean(c,:)) ~= 0)
            [~, ~, lam_pred(c,:)] = updatep(ex_sim_mean(c,:) , p{c}, q{c}, fK0(c,:), Alpha, Beta, mus(c));      
        end
end

lam_pred = lam_pred(:,end-DaysPred+1:end);
infec = risk(smoothdata(infections, 2));
flam = [lam lam_pred];

flam = risk(flam);


%% Calculate and Print Risk

risk = @(x) (x - min(x(:)))./(max(x(:)) - min(x(:)));
r = flam;
rep_p = [5];

error_In = zeros(numel(rep_p),1);
risk_In_all = zeros(numel(rep_p), 1);
risk_In_pred = zeros(numel(rep_p), 1);
eps = 1e-3;

[x, y] = sort(sum(infections'));
In = y(end-rep_p(1)+1:end);
abs_err = abs(sim_mean(In,:) - infections(In,end-DaysPred+1:end));
abs_inf = abs(infections(In,end-DaysPred+1:end) + eps);
abs_inf_all = abs(infections(In,:) + eps);

sim_std_In = sim_std(In,:);
error_In(1) = mean(abs_err(:)./abs_inf(:));
error_In(2) = mean(sim_std_In(:)./abs_inf(:));

 
abs_err_risk_pred = abs(infec(In,end-DaysPred+1:end) - flam(In,end-DaysPred+1:end));
abs_err_risk_all = abs(infec(In,:) - flam(In,:));
risk_In_pred(1) = mean(abs_err_risk_pred(:));
risk_In_all(1) = mean(abs_err_risk_all(:));

% Print all metrics in the order: 1) the RMAE error in predicted infections(I), 
% 2) the corresponding standard deviation, 3) MAE of risk for the test set, and
% 4) MAE of risk for the entire period
 all_metrics = [error_In'; risk_In_pred; risk_In_all]

if (all_mobility == 4)
    csvwrite(save_K0_file,flam)
end
%% Plot stuff

% close all
% figure, 
%  subplot(211)
%  imagesc(infec)
%  subplot(212)
%  imagesc(flam)
