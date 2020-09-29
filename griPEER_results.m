% this script produces results for all males below 42 age old and 3 different connectivity
% matrices.

%% Clear workplace
clear

%% Set seed
rand('seed', 1);

%% Paths
%addpath('H:/Dropbox/LogisticPEER/software/Matlab')
%dirsNames  = 'H:/Dropbox/LogisticPEER/manuscript/RDA_code';  
%path_save  = 'H:/Dropbox/LogisticPEER/manuscript/RDA_code';

addpath('path to griPEER code')
dirsNames  = './data';  
path_save  = './data';


%% Objects   
connMatxNames    = {'Empty', 'SC_MfaAvg_hiv_median_modularity_masked.txt','SC_Mnf_norm_hiv_median_modularity_masked.txt'};
connMatxNamNotxt = {'Empty', 'SC_MfaAvg_hiv_median_modularity_masked','SC_Mnf_norm_hiv_median_modularity_masked'};

b_est_vec          = [];
b_idx_vec          = [];
sc_matrix_name_vec = [];
asymp_conf_lower   = [];
asymp_conf_upper   = [];
boot_conf_lower    = [];
boot_conf_upper    = [];
opt_lambR          = [];
opt_lambQ          = [];
             
%% griPEER
cd(dirsNames)

% importing data
X         = importdata('X.txt');
Z         = importdata('Z.txt');
y         = importdata('y.txt');
[n, p]    = size(Z);
m         = size(X,2);
pm        = p + m;

for ii = 1:length(connMatxNames)
    % importing connectivity matrix
    
    if strcmp(connMatxNames{ii}, 'Empty')
        Q = eye(p,p);
    else
        A      = importdata(connMatxNames{ii});
        D      = diag(sum(A,1));
        Q      = (D^(-.5))*(D - A)*(D^(-.5)); %normalized Laplacian
    end

    % analysis
    out    = griPEER(y, X, Z, Q, 'alpha', 0.05);
    b_est  = [out.beta; out.b] ;

    % storing data
    b_est_vec          = [b_est_vec; b_est];
    b_idx_vec          = [b_idx_vec; (1:pm)'];
    sc_matrix_name_vec = [sc_matrix_name_vec; cellstr(repmat(connMatxNamNotxt{ii}, pm, 1))];
    asymp_conf_lower   = [asymp_conf_lower; out.band(:,1) ];
    asymp_conf_upper   = [asymp_conf_upper; out.band(:,2) ];
    opt_lambR          = [opt_lambR; out.lambR];
    opt_lambQ          = [opt_lambQ; out.lambQ];

    % bootstrap
    out    = griPEER(y, X, Z, Q, 'ciType', 'both', 'nboot', 50000);
    boot_conf_lower    = [boot_conf_lower; out.ciboot(:,1) ];
    boot_conf_upper    = [boot_conf_upper; out.ciboot(:,2) ];
    

end


%% Saving
cd(path_save)

save b_est_vec.txt          -ascii b_est_vec
save b_idx_vec.txt          -ascii b_idx_vec
save asymp_conf_lower.txt   -ascii asymp_conf_lower
save asymp_conf_upper.txt   -ascii asymp_conf_upper
save boot_conf_lower.txt    -ascii boot_conf_lower
save boot_conf_upper.txt    -ascii boot_conf_upper
save boot_conf_upper.txt    -ascii boot_conf_upper
save opt_lambR.txt          -ascii opt_lambR
save opt_lambQ.txt          -ascii opt_lambQ

dlmcell('sc_matrix_name_vec.txt',sc_matrix_name_vec);  






