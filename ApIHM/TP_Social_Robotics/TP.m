%% EXO 1
%% Q1 - Extraction of prosodic features

file_f0_list = dir('Kismet_data_intent/*.f0');
file_en_list = dir('Kismet_data_intent/*.en');
% Initialize an empty dataset
data_set = [];



parfor i = 1:length(file_f0_list)
    % Load the .f0 file
    file_name_f0 = file_f0_list(i).name;
    file_data_f0 = importdata(['Kismet_data_intent/',file_name_f0]);

    % Load the .en file
    file_name_en = file_en_list(i).name;
    file_data_en = importdata(['Kismet_data_intent/',file_name_en]);

    first_label = file_name_en(1:2);
    sec_label = file_name_en(7:8);

    disp([file_name_en,'-',file_name_f0])

    % Append the loaded data to the dataset
    data_set = [data_set; {first_label, sec_label, file_data_f0, file_data_en}];
end

save("Kismet_data_intent","data_set")

%% Extraction of functionals (statistics) : mean, maximum, range, variance, median, first quartile, third quartile, mean absolute of local derivate
statistics = [];

for i = 1:length(data_set)

    data = data_set(i,3);  % f0
    data = data{1}(:,2);

    mean_d = mean(data);
    max_d = max(data);
    range_d = range(data);
    var_d = var(data);
    median_d = median(data);
    Q1 = quantile(data, 0.25);
    Q3 = quantile(data, 0.75);

    local_derivative = diff(data);
    mean_absolute_local_derivative = mean(abs(local_derivative));

    f0 = [mean_d, mean_d, max_d, range_d, var_d, median_d, Q1, Q3, mean_absolute_local_derivative];

    data = data_set(i,4);  % en
    data = data{1}(:,2);

    mean_d = mean(data);
    max_d = max(data);
    range_d = range(data);
    var_d = var(data);
    median_d = median(data);
    Q1 = quantile(data, 0.25);
    Q3 = quantile(data, 0.75);

    local_derivative = diff(data);
    mean_absolute_local_derivative = mean(abs(local_derivative));

    en = [mean_d, mean_d, max_d, range_d, var_d, median_d, Q1, Q3, mean_absolute_local_derivative];

    statistics = [statistics; [f0, en]];
end


%% Q3 

% deviding the data_set to voiced and non_voiced 
clear,clc
load("Kismet_data_intent.mat")

data_set_voiced = [];
data_set_unvoiced = [];

for i = 1:length(data_set)

    data_f0 = data_set(i,3);  % f0
    data_f0 = data_f0{1}(:,2);

    data_en = data_set(i,4);  % en
    data_en = data_en{1}(:,2);

    voiced = [];
    unvoiced = [];
    for j = 1:length(data_f0)

        if data_f0(j) ~= 0 
            voiced = [voiced;[data_f0(j),data_en(j)]];
        else
            unvoiced = [unvoiced;[data_f0(j),data_en(j)]];
        end
    end

    data_set_voiced = [data_set_voiced, {voiced}];
    data_set_unvoiced = [data_set_unvoiced, {unvoiced}];
end
clear i j unvoiced voiced data_en data_f0


%%
clc
a = stat(data_set_unvoiced)
a(2,:)

            







