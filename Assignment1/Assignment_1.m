%% Q1 — PCA by SVD for first 360 days
data = load("Seattle_traffic_flow.mat").data;

% Take first 360 days
X = data(:,1:360);

% Remove row mean (center each feature)
X = X - mean(X,2);

% SVD
[U,S,V] = svd(X,'econ');

% First 4 PCs
U4 = U(:,1:4);

% Plot PC1–PC4
figure;
plot(U4(:,1),'LineWidth',1.5); hold on;
plot(U4(:,2),'LineWidth',1.5);
plot(U4(:,3),'LineWidth',1.5);
plot(U4(:,4),'LineWidth',1.5);
grid on;
xlabel('Time index');
ylabel('PC loadings');
title('First 4 Principal Components (U1–U4)');
legend('PC1','PC2','PC3','PC4');

%% Q2 — Singular values & Cumulative Explained Variance
clear;clc
data = load("Seattle_traffic_flow.mat").data;
X = data(:,1:360);

% Remove mean
X = X - mean(X,2);

% SVD
[U,S,V] = svd(X,'econ');
singular_values = diag(S);

% Explained variance
explained_variance = singular_values.^2 / sum(singular_values.^2);
cumulative_variance = cumsum(explained_variance);

% Plot
figure;
subplot(1,2,1);
plot(singular_values,'o-','LineWidth',1.5);
xlabel('Principal Component Index');
ylabel('Singular Value');
title('Singular Values');
grid on;

subplot(1,2,2);
plot(cumulative_variance,'s-','LineWidth',1.5);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance');
title('Cumulative Explained Variance');
ylim([0 1]); grid on;

%% Q3 — Low rank approximation of Day-10 and Day-20
clear;clc
data = load("Seattle_traffic_flow.mat").data;

X = data(:,1:360);
mu = mean(X,2);
X = X - mu;
[U,S,V] = svd(X,'econ');

r_values = [10,20,30,50];
day10_orig = data(:,10);
day20_orig = data(:,20);

figure;
subplot(1,2,1);
plot(day10_orig,'k-','LineWidth',1.5); hold on;
for r = r_values
    Xr = U(:,1:r)*S(1:r,1:r)*V(:,1:r)' + mu;
    plot(Xr(:,10),'LineWidth',1.2);
end
xlabel('Time index');
ylabel('Traffic speed');
title('Day-10 original vs approx');
legend('Original','r=10','r=20','r=30','r=50');
grid on;

subplot(1,2,2);
plot(day20_orig,'k-','LineWidth',1.5); hold on;
for r = r_values
    Xr = U(:,1:r)*S(1:r,1:r)*V(:,1:r)' + mu;
    plot(Xr(:,20),'LineWidth',1.2);
end
xlabel('Time index');
ylabel('Traffic speed');
title('Day-20 original vs approx');
legend('Original','r=10','r=20','r=30','r=50');
grid on;

%% Q4 — Fit the last five days using r=20
clear;clc
data = load("Seattle_traffic_flow.mat").data;

Xtrain = data(:,1:360);
Xtest  = data(:,361:365);
mu = mean(Xtrain,2);
Xc = Xtrain - mu;
[U,S,V] = svd(Xc,'econ');

r = 20;
U_r = U(:,1:r);

Xtest_c = Xtest - mu;
Xtest_hat = U_r * (U_r' * Xtest_c) + mu;

figure;
for i = 1:5
    subplot(2,3,i);
    plot(Xtest(:,i),'k-','LineWidth',1.5); hold on;
    plot(Xtest_hat(:,i),'r--','LineWidth',1.5);
    xlabel('Time index');
    ylabel('Traffic speed');
    title(['Day ',num2str(360+i),' (r=20)']);
    legend('Real','Fitted');
    grid on;
end
sgtitle('Last 5 days reconstruction (r=20)');

%% Q5 — Projection onto first 4 PCs and scatter plot
clear;clc
data = load("Seattle_traffic_flow.mat").data;

X = data(:,1:360)';      % 360 x 96
Xc = X - mean(X,1);      % mean-centered
[U,S,V] = svd(Xc,'econ');
projection = Xc * V(:,1:4);
projection = projection'; % 4 x 360

figure;
i0=1;
for ii=1:3
    for jj=ii+1:4
        subplot(2,3,i0); hold on
        for kk=1:7
            scatter(projection(ii,kk:7:end),projection(jj,kk:7:end),12,'filled');
        end
        legend("weekday 1","weekday 2","weekday 3","weekday 4", ...
               "weekday 5","weekday 6","weekday 7",'Location',"best");
        xlabel("PC#"+num2str(ii));
        ylabel("PC#"+num2str(jj));
        grid on; box on;
        set(gca,'LineWidth',1.2,'FontName','Times','FontSize',12);
        hold off
        i0=i0+1;
    end
end
sgtitle('Projection onto first 4 PCs (mean-centered)');

%% Q6 — PCA with ALS (Day-10 & Day-20 reconstruction)
clear;clc
load('Seattle_traffic_flow.mat');  % data (96x365), mdata

[coeff1,~,~,~,~,~] = pca(mdata','Algorithm','als');  % ALS PCA
[coeff,~,~,~,~,~]  = pca(data');                     % Full PCA
angle_val = subspace(coeff, coeff1);
fprintf('Subspace angle (full vs ALS) = %.2e radians\n', angle_val);

train = data(:,1:360);
test  = data(:,[10 20]);
r_values = [10,20,30,50];

figure;
for i=1:length(r_values)
    r = r_values(i);

    % --- PCA reconstruction ---
    ave_data = mean(train,2);
    train_c = train - ave_data*ones(1,size(train,2));
    test_c  = test  - ave_data*ones(1,size(test,2));

    [~,~,V] = svd(train_c');
    rec_c = test_c' * V(:,1:r) * V(:,1:r)'; 
    rec   = rec_c' + ave_data*ones(1,size(test,2));

    % Plot
    subplot(2,2,i);
    plot(test(:,1),'k-','LineWidth',1.5); hold on;
    plot(rec(:,1),'r--','LineWidth',1.5);
    plot(test(:,2),'b-','LineWidth',1.5);
    plot(rec(:,2),'g--','LineWidth',1.5);
    xlabel('Time index');
    ylabel('Traffic speed');
    title(['Reconstruction with r=',num2str(r)]);
    legend('Day10 real','Day10 rec','Day20 real','Day20 rec','Location','best');
    grid on;
end
sgtitle('Day-10 & Day-20 reconstruction using truncated PCA (ALS verified)');




