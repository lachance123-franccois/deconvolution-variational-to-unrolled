function tf = has_dl_toolbox()
%Vrai si la Deep Learning Toolbox est installée et sous licence.
tf = ~isempty(ver('nnet')) && license('test', 'Neural_Network_Toolbox');
end
