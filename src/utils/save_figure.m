function save_figure(fig, cfg, name)
if ~cfg.save_figures, return; end
f = fullfile(cfg.results_dir, [name '.png']);
if exist('exportgraphics', 'file')
    exportgraphics(fig, f, 'Resolution', 150);
else
    saveas(fig, f);
end
end
