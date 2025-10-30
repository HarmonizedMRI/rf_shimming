% -------------------------------------------------------------------------
%% this is a demo GRE sequence, which uses LABEL extension to produce raw
%% data reconstuctable by ICE image reconstruction on the scanner
% -------------------------------------------------------------------------

clear

Gmax = 40;
Smax = 120;

% set system limits
sys = mr.opts('MaxGrad', Gmax, 'GradUnit', 'mT/m', ...
    'MaxSlew', Smax, 'SlewUnit', 'T/m/s', 'rfRingdownTime', 20e-6, ...
    'rfDeadTime', 100e-6, 'adcDeadTime', 10e-6);

seq = mr.Sequence(sys);         % Create a new sequence object

fov = 224e-3; 
Nx = 224; 
Ny = Nx; % Define FOV and resolution

alpha = 15;                  % flip angle
thickness = 5e-3;            % slice

% Nslices = 32;                % ideally we would want to loop over PE lines in the outer loop, and slices in the inner loop for spin history
Nslices = 16;                


chan_select = 1:8;

% cp mode:
% phases = [0, -45, -90, -135, 180, 135, 90, 45];

shim1 = exp(+1j*2*pi/8*[0:7]); % anti cp
shim2 = exp(-1j*2*pi/8*[0:7]); % cp
shim3 = exp(-1j*2*pi/4*[0:7]); % tiamo 2nd mode

shim1 = shim1 / norm(shim1);
shim2 = shim2 / norm(shim2);
shim3 = shim3 / norm(shim3);


shim_vector1 = zeros(1,8);
shim_vector2 = zeros(1,8);
shim_vector3 = zeros(1,8);

shim_vector1(chan_select) = shim1(chan_select);
shim_vector2(chan_select) = shim2(chan_select);
shim_vector3(chan_select) = shim3(chan_select);


% -------------------------------------------------------------------------
%
% -------------------------------------------------------------------------


TR = 14e-3; 
TE = 7e-3;
%TE=[4.38 6.84 12]*1e-3;            % alternatively give a vector here to have multiple TEs (e.g. for field mapping)

% more in-depth parameters
rfSpoilingInc = 117;              % RF spoiling increment
roDuration = 3.2e-3;              % ADC duration

% make sure roDuration is integer multiple of block raster time
tmp = ceil(roDuration / Nx / sys.blockDurationRaster); 
roDuration = tmp * Nx * sys.blockDurationRaster;

% Create alpha-degree slice selection pulse and gradient
rf_duration = 5e-3;

[rf, gz] = mr.makeSincPulse(alpha*pi/180,'Duration',rf_duration,...
    'SliceThickness',thickness,'apodization',0.42,'timeBwProduct',4,'system',sys,...
    'use','excitation');

% Define other gradients and ADC events
deltak = 1/fov;

gx = mr.makeTrapezoid('x','FlatArea',Nx*deltak,'FlatTime',roDuration,'system',sys);

adc = mr.makeAdc(Nx,'Duration',gx.flatTime,'Delay',gx.riseTime,'system',sys);

gxPre = mr.makeTrapezoid('x','Area',-gx.area/2,'Duration',1e-3,'system',sys);
gzReph = mr.makeTrapezoid('z','Area',-gz.area/2,'Duration',1e-3,'system',sys);
phaseAreas = -((0:Ny-1)-Ny/2)*deltak; % phase area should be Kmax for clin=0 and -Kmax for clin=Ny... strange

% gradient spoiling
gxSpoil = mr.makeTrapezoid('x','Area',2*Nx*deltak,'system',sys);
gzSpoil = mr.makeTrapezoid('z','Area',4/thickness,'system',sys);

% Calculate timing
delayTE = ceil((TE - mr.calcDuration(gxPre) - gz.fallTime - gz.flatTime/2 ...
    - mr.calcDuration(gx)/2)/seq.gradRasterTime)*seq.gradRasterTime;
delayTR = ceil((TR - mr.calcDuration(gz) - mr.calcDuration(gxPre) ...
    - mr.calcDuration(gx) - delayTE)/seq.gradRasterTime)*seq.gradRasterTime;

assert(all(delayTE>=0));
assert(all(delayTR>=mr.calcDuration(gxSpoil,gzSpoil)));

% accelerate sequence calculation for objects that do not change in the loops
gz.id=seq.registerGradEvent(gz);

gxPre.id=seq.registerGradEvent(gxPre);
gzReph.id=seq.registerGradEvent(gzReph);

gx.id=seq.registerGradEvent(gx);
gxSpoil.id=seq.registerGradEvent(gxSpoil);

% gyPre.id=seq.registerGradEvent(gyPre);    % this does change in the loops
gzSpoil.id=seq.registerGradEvent(gzSpoil);

% RF spoiling
rf_phase=0;
rf_inc=0;
c = 1;      % use single echo

% all LABELS / counters an flags are automatically initialized to 0 in the beginning, no need to define initial 0's  
% so we will just increment LIN after the ADC event (e.g. during the spoiler)

seq.addBlock(mr.makeLabel('SET','REV', 1)); % left-right swap fix (needed for 1.4.0 and later)

seq.addBlock(mr.makeLabel('SET','LIN', 0), mr.makeLabel('SET','SLC', 0)); % needed to make it compatible to multiple REPs

tic
% loop over slices
for s = 1:Nslices
    disp(['slc: ', num2str(s), ' / ', num2str(Nslices)])

    rf.freqOffset=gz.amplitude*thickness*(s-1-(Nslices-1)/2);
    
    % loop over phase encodes and define sequence blocks
    for i = 1:Ny
        % acquire three Tx modes for each line succesively
        rf.phaseOffset = rf_phase/180*pi;
        adc.phaseOffset = rf_phase/180*pi;

        for t = 1:4
            rf_inc=mod(rf_inc+rfSpoilingInc, 360.0);
            rf_phase=mod(rf_phase+rf_inc, 360.0);
            
            if t == 1
                % anti cp -> dummy
                seq.addBlock(rf,gz,mr.makeRfShim(shim_vector1));                
            else
                if t== 2
                    % anti cp
                    seq.addBlock(rf,gz,mr.makeRfShim(shim_vector1));                
                else
                    if t ==3 
                        % cp
                        seq.addBlock(rf,gz,mr.makeRfShim(shim_vector2));         
                    else
                        % gradient 
                        seq.addBlock(rf,gz,mr.makeRfShim(shim_vector3));         
                    end
                end
            end

            gyPre = mr.makeTrapezoid('y','Area',phaseAreas(i),'Duration',mr.calcDuration(gxPre),'system',sys);
        
            seq.addBlock(gxPre,gyPre,gzReph);
            seq.addBlock(delayTE(c));
            seq.addBlock(gx,adc);

            gyPre.amplitude = -gyPre.amplitude;
        
            spoilBlockContents={mr.makeDelay(delayTR(c)),gxSpoil,gyPre,gzSpoil}; % here we demonstrate the technique to combine variable counter-dependent content into the same block

            if t == 4
                % if mode == 4, reset echo counter for the next line
                spoilBlockContents=[spoilBlockContents {mr.makeLabel('SET','ECO', 0)}];
    
                if i~=Ny
                    spoilBlockContents=[spoilBlockContents {mr.makeLabel('INC','LIN', 1)}];
                else
                    % last ky line in slice, reset lin and increment
                    % slice counter
                    spoilBlockContents=[spoilBlockContents {mr.makeLabel('SET','LIN', 0), mr.makeLabel('INC','SLC', 1)}];
                end

            else
                spoilBlockContents=[spoilBlockContents {mr.makeLabel('INC','ECO', 1)}];
            end

            seq.addBlock(spoilBlockContents{:});    
        end
    end
end
toc



% -------------------------------------------------------------------------
% prepare sequence export
% -------------------------------------------------------------------------


seq.setDefinition('FOV', [fov fov thickness*Nslices]);
seq.setDefinition('TE', TE) ;  
seq.setDefinition('TR', TR) ;

seq.setDefinition('ReceiverGainHigh',1); 

seq.setDefinition('Name', 'gre_rfshim_tiamo');

seq.write(['gre_lbl_rfshim_4modes.seq'])       % Write to pulseq file


% -------------------------------------------------------------------------
%% check whether the timing of the sequence is correct
% -------------------------------------------------------------------------

[ok, error_report] = seq.checkTiming;

if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end
% -------------------------------------------------------------------------
%% plot sequence and k-space diagrams
% -------------------------------------------------------------------------

seq.plot('timeRange', [0 32]*TR, 'TimeDisp', 'ms', 'Label', 'LIN,SLC'); % looks like there is a bug in the coloring of the multiple labels in the plot

% k-space trajectory calculation
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP();

% plot k-spaces
figure; plot(t_ktraj, ktraj'); % plot the entire k-space trajectory
hold; plot(t_adc,ktraj_adc(1,:),'.'); % and sampling points on the kx-axis
title('k-space components as functions of time');
figure; plot(ktraj(1,:),ktraj(2,:),'b'); % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display
hold;plot(ktraj_adc(1,:),ktraj_adc(2,:),'r.'); % plot the sampling points
title('2D k-space');


% -------------------------------------------------------------------------
%% evaluate label settings more specifically
% -------------------------------------------------------------------------

lbls=seq.evalLabels('evolution','adc');
lbl_names=fieldnames(lbls);

figure; hold on;

for n=2:length(lbl_names)-1     % skip REV and REP
    plot(lbls.(lbl_names{n}));
end

legend(lbl_names(2:end-1));
title('evolution of labels/counters/flags');
xlabel('adc number');
