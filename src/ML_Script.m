% - Senza lost_perc (11)
% - Senza lost_perc e gold (8 e 11)
% - Solo con features da 1 a 7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preprocessing ops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;
load('heroStatsNorm.mat');
hero_Features = heroesstatsnormalized_Features;
hero_Name = heroesstatsnormalized_Name;
%hero_Features(:, 1) = hero_Features(:, 1) + hero_Features(:, 4);
rho = corr(hero_Features, 'rows', 'pairwise');
figure, imagesc(rho);

% hero_Features(:, 11) = [];
% hero_Features(:, 8) = [];
hero_Features = hero_Features(:, 1:7);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K-Means MATLAB [PROVARE DIVERSE METRICHE DISTANZA] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% evalsMatlab = []; % Contiene tutte le clusterizzazioni
% for k = 2:30
%     idx = kmeans(hero_Features, k);
%     evalsMatlab = [evalsMatlab, idx];
% end
% bestK = evalclusters(hero_Features, evalsMatlab, 'silhouette');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PUNTO A: Stima del numero ottimale di cluster usando la misura di Silhouette
% K-Means Manuale e calcolo bestK [PROVARE DIVERSE METRICHE DISTANZA] 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

meansVector = []; % Contiene tutte le medie delle silhouette
for k = 3:30
    idx = kmeans(hero_Features, k, 'MaxIter', 1000);
    s = silhouette(hero_Features, idx); % Aggiungere terzo parametro METRICA DISTANZA
    meansVector = [meansVector, mean(s)];
end
meansVector = [nan, nan, meansVector];
[valueMax, bestK] = max(meansVector);
figure, plot(meansVector);
[idxFinal, centroids, intraSum] = kmeans(hero_Features, 5, 'MaxIter', 100000, ...
    'Replicates', 50);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Postprocessing ops
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp = {};
for i = 1:size(idxFinal, 1)
    temp{i} = idxFinal(i);
end
temp = temp';
clusters = [hero_Name, temp];
[clusters, indexSort] = sortrows(clusters, 2);
hero_Features_Ordered = hero_Features(indexSort, :);
el_x_cluster = zeros(1, bestK);
for i = 1:bestK
    el_x_cluster(i) = sum(idxFinal == i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temp = pdist(centroids);    
interSum = squareform(temp);
for i = 1:bestK
    intraSum(i) = intraSum(i)./el_x_cluster(i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PUNTO B: Stima della silhouette per ciascun cluster ottenuto sull'ottimo
% Calcolo Silhouette del bestK [DIVERSE METRICHE DI DISTANZA]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure, silhouette(hero_Features, idxFinal); % Aggiungere terzo parametro METRICA DISTANZA


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PUNTO C: Stima della Dissimilarity per ciascun cluster ottenuto sull'ottimo
% Calcolo Dissimilarity Matrix [DIVERSE METRICHE DI DISTANZA]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
diss_vect = pdist(hero_Features(indexSort)); % Aggiungere parametro distanza
diss_mat = squareform(diss_vect);
diss_mat_norm = diss_mat - min(diss_mat(:));
diss_mat_norm = diss_mat_norm ./ max(diss_mat_norm(:));
figure, imagesc(diss_mat_norm);
hold on;
for i = 1:5
    line([0, length(clusters)], [find(cell2mat(clusters(:, 2)) == i, 1)-1, ...
        find(cell2mat(clusters(:, 2)) == i, 1)-1], 'Color', 'k', 'LineWidth', 1.5);
    line([find(cell2mat(clusters(:, 2)) == i, 1)-1, find(cell2mat(clusters(:, 2)) == i, 1)-1], ...
        [0, length(clusters)], 'Color', 'k', 'LineWidth', 1.5);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot dei centroidi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure, h = bar(centroids);
grid on;
labels = cell(1, 7);
labels{1} = 'Kills';
labels{2} = 'Deaths';
labels{3} = 'Assists';
labels{4} = 'Last-hits';
labels{5} = 'Hero-Dmg';
labels{6} = 'Hero-Healing';
labels{7} = 'Tower-Dmg';
%labels{8} = 'Gold-Dur';
% labels{8} = 'Exp-Dur';
% labels{9} = 'Won-Perc';
% labels{10} = 'Picks';
legend(h, labels);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Garbage collector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear temp;
clear i;
clear k;
clear s;
clear idx;
clear diss_mat;
clear diss_vect;
clear herostats_Name;
clear herostats_Features;
clear indexSort;
clear valueMax;