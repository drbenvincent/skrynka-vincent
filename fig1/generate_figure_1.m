addpath('~/Documents/MATLAB/export_fig')

close all
setPlotTheme

f = figure(1);
clf
f.Color = 'w';
colormap(gray)


h(1).name = 'Model 1. Trait-only';
h(1).y = [0 0 0];

h(2).name = 'Model 2: In-domain only';
h(2).y = [1 0 0];

h(3).name = 'Model 3: Monetary fungibility';
h(3).y = [1 1 0];

h(4).name = 'Model 4: Negative spillover';
h(4).y = [1 -0.5 -0.5];

h(5).name = 'Model 5: Spillover effect';
h(5).y = [1 0.5 0.5];

h(6).name = 'Model 6: State-only';
h(6).y = [1 1 1];

s = layout([1 2 3; 4 5 6]);

for n=1:numel(h)
	subplot(s(n))
	b = bar([1,2,3],h(n).y);
	title(h(n).name)
	
	% bar properties
	b.LineWidth = 2;
	b.FaceColor = [0.8 0.8 0.8];
	b.BarWidth = 0.9;
	b.LineStyle = 'none';
	
	% axis properties
	s(n).YTickLabel = [];
	s(n).XAxisLocation='origin';
	
	box off
	ylim([-0.5 1])
	xlim([0.5 3.5])
	set(gca,'XTickLabel', {'food', 'money', 'music'},...
		'TickDir', 'out')
	%xlabel('commodity')
	ylabel('change in discount rate')
end

% custom ticks for each hypothesis
s(1).YTick = [0];			s(1).YTickLabel = {'0'};
s(2).YTick = [0 1];			s(2).YTickLabel = {'0', '\alpha'};
s(3).YTick = [0 1];			s(3).YTickLabel = {'0', '\beta'};
s(4).YTick = [-0.5 0 1];	s(4).YTickLabel = {'\delta', '0', '\gamma'};
s(5).YTick = [0 0.5 1];		s(5).YTickLabel = {'0', '\zeta', '\epsilon'};
s(6).YTick = [0 1];		s(6).YTickLabel = {'0', '\eta'};

%s(5).YTick = [0 1];			s(5).YTickLabel = {'0', '\eta'};


f.Position = [0 0 1500 800]
export_fig('Figure1', '-pdf')
