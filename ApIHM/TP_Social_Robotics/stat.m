function [stat] = stat(data_set)
    stat = [];
    for i = 1:length(data_set)

        data = data_set(i); 
        data = data{1}(:,1); % f0
    
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
        
        data = data_set(i); 
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
    
        stat = [stat; [f0, en]];
    end

end