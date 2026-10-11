function startup_project()
%Ajoute les dossiers du projet au chemin MATLAB et crée
%   les dossiers de sortie (results/, models/) s'ils n'existent pas.
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'src')));
addpath(fullfile(root, 'experiments'));
addpath(fullfile(root, 'tests'));
dirs = {'results', 'models'};
for i = 1:numel(dirs)
    d = fullfile(root, dirs{i});
    if ~exist(d, 'dir'), mkdir(d); end
end
end
