%% load dataset
load('realitymining.mat')

%% find first and last times
start_time = Inf;
end_time = 0;
x = 0;
c=0;
for i = 1:106
    if isempty(s(i).locs)
        continue;
    end
    if s(i).locs(1,1) < start_time && s(i).locs(1,1) > s(102).locs(3,1)
        start_time = s(i).locs(1,1);
        c=i;
    end
    if s(i).locs(1,end) > end_time
        end_time = s(i).locs(end,1);
        x=i;
    end
end

%% make 3d matrix with a row for every minute btwn min and max times

N = int32((end_time-start_time)*24*60);

data = zeros(N, 106);

for i = 1:106
    
    if isempty(s(i).locs)
        data(:,i) = 0;
        continue;
    end
    
    time = start_time;
    for j = 1:N
        time = addtodate(time,1,'minute');
        % time is outside of range
        if time < s(i).locs(1,1) || time > s(i).locs(end,1)
            data(j, i) = 0;
        % find the index of the greatest time less than the current time
        else
            data(j, i) = s(i).locs(find(s(i).locs(:,1)<=time,1,'last'),2);
        end
    end
end

%% find contacts

[loc_l, loc_w] = size(locations);

contact_length = zeros(106,106);
adj = zeros(106,106);
for n = 1:loc_l
    
end

%% chop off first 399,000 minutes because not much goes on
locations = data(400000:end,:);

%% export locations as csv
dlmwrite('locations.csv', locations, 'delimiter', ',', 'precision', 10); 
%% save workspace
save()

%% load computed workspace
load('contact_network.mat')

