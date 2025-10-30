%--------------------------------------------------------------------------
%% Dixon epi blip rewound acquisition (debra)
%--------------------------------------------------------------------------
% based on https://github.com/pulseq/pulseq/blob/master/matlab/demoSeq/writeEpiSpinEchoRS.m

% TODO, add minimal TE calculation, maybe use existing code
% TODO, readout partial fourier
% FIXME, add dTE shift for 2nd echo

clear
closeall

%--------------------------------------------------------------------------
% rf shim settings
%--------------------------------------------------------------------------

chan_select = 1:8;

shim1 = exp(+1j*2*pi/8*[0:7]); % anti cp
shim2 = exp(-1j*2*pi/8*[0:7]); % cp
shim3 = exp(-1j*2*pi/4*[0:7]); % tiamo 2nd mode

shim1 = shim1 / norm(shim1);
shim2 = shim2 / norm(shim2);
shim3 = shim3 / norm(shim3);

shim2use = 2;        % select CP mode

shim_vector1 = zeros(1,8);
shim_vector2 = zeros(1,8);
shim_vector3 = zeros(1,8);

shim_vector1(chan_select) = shim1(chan_select);
shim_vector2(chan_select) = shim2(chan_select);
shim_vector3(chan_select) = shim3(chan_select);


%--------------------------------------------------------------------------
% parameter settings
%--------------------------------------------------------------------------

isTestRun = 0; % for test only, set Nslices to 1 to accelerate script

% geometry
fov        = 224e-3;
Nx         = 224;
Nx_org     = Nx;
Ny         = Nx;   % Define FOV and resolution

thickness  = 3e-3; % slice thinckness
Nslices    = 32;

Nx_low_org = 64;
Nx_low     = Nx_low_org;
Ny_low_org = Nx_low_org;
Ny_low     = Ny_low_org;


% TR, TR, esp need to be determined manually to avoid violating timing
TR       = 4;
% if you see amplitude violation in:
% gDiff=mr.makeTrapezoid('z','amplitude',g,'riseTime',gr,'flatTime',small_delta-gr,'system',sys_diff);
% increase TE to make it fit
TE       = 60e-3;

% esp needs to be determined manually so that it can still be played given
% the Gmax and constraint to have adc dwell > 1.3us
esp               = 1.1e-3;      % the duration of gx -> needs to be set manually
isOverwriteESP    = 0;


TE_low   = TE;


R        = 3;      % net accl combined over all shots
RSegment = 1;      % number of shots
R_low    = R;


partFourierFactor = 6/8;          % partial Fourier factor: 1: full sampling 0: start with ky=0
partFourierFactor_fe  =  1;       % frequency-encoding partial Fourier factor


dTE_esp_ratio     = 2;
deltaTE           = esp*dTE_esp_ratio;
crusher_d         = 0.95e-3;
crusher_area      = 1e3;            % unit is 1/m

% debra
kqShiftPeriod        = 3;
isProlongedDebraBlip = 0; % for temp test only, should always set to 0, see Erpeng Dai's 2022 IEEE Transactions on Medical Imaging. This blip duration is equal to echo spacing, instead of blip_dur
isEchoTimeShift      = 1; % for multishot

debra_nechoes=1;


isConventionalSegmentSpacing = 0; % in conventional multi segment epi, the ky distance between 2 lines is Ry*Rsegment. If this is set to 0, then distance is Ry
isKyShiftBetweenDiffFrames = 0;

% fat saturation
isFatSat         = 1;
isSSgradReversal = 0; % slice selection gradient reversal, SSGR. 

% dummy, ref, acs
isUseGreACS  = 1; % built-in gre for sens and B0map
isRefscan    = 1; % for ghost correction
nDummyFrames = 0; % to get into steady state
is2ndEcho    = 0; % low-res 2nd spin echo

% debug options, preferred to be set to false
isWaterSat = 0; % for debug the fat not shift problem
if isWaterSat
    Nslices = 1;
    msg = 'water saturation, set #slices=1';
    warning(msg);
    warndlg(msg);
end


% diff
bvalue0_threshold    = 0; % when b-value smaller than this, crusher would be used instead of diffusion gradient
bvalues              = [bvalue0_threshold   1e3   ];


acquisition_order = {
    1, 0;           % 1st: b=1, 1 direction (crusher, no bTable)
    2, 1:6;        % 2nd: b=1000, 5 directions (bTable rows 1-6)
};


% rf
tRFex       = 3e-3;
if isSSgradReversal
    % tRFex = tRFex*1.5;
end
tRFref      = 3e-3;
spoilFactor = 1.5;             % spoiling gradient around the pi-pulse

% misc
pe_enable = 1;     % a flag to quickly disable phase encoding (1/0) as needed for the delay calibration
ro_os     = 1;     % oversampling factor (in contrast to the product sequence we don't really need it)
ro_os_low           =   1.00;           % oversampling factor (in contrast to the product sequence we don't really need it) 2.60 (if pe_pf: 2.70) (if pe_pf/TE60: 1.60)

trig_dur                 = 20e-6;
delay_after_trig         = 200e-6; % Gradient-free time for probe excitation [Unit: s] https://github.com/SkopeMagneticResonanceTechnologies/Pulseq-Sequences/blob/2576a44edb2f4123774565862472fe76ecd10435/sequences/PulseqBase.m#L102C9-L102C60
fatChemShift = 3.5e-6; % 3.5 ppm

%--------------------------------------------------------------------------
% system settings
%--------------------------------------------------------------------------

% sys_type                  = 'Connectome2'; % prisma, skyra, Connectome2, C2_simulate_prisma, trio, prisma_XA30A, premier
sys_type                  = 'terra'; % prisma, skyra, Connectome2, C2_simulate_prisma, trio, prisma_XA30A, premier


slew_safety_margin        = 0.7;
grad_safety_margin        = 0.9;

lowPNS_slew_safety_margin = 0.3;
lowPNS_grad_safety_margin = grad_safety_margin;

diff_slew_safety_margin   = 0.45; % decrease this to reduce PNS, this would not lengthen TE too much
diff_grad_safety_margin   = 0.97;

if strcmp(sys_type,'prisma') || strcmp(sys_type,'C2_simulate_prisma') || strcmp(sys_type,'prisma_XA30A')
    physical_slew_max = 200;
    physical_grad_max = 80;
    B0=2.89; % 1.5 2.89 3.0
elseif strcmp(sys_type,'premier')
    physical_slew_max = 200;
    physical_grad_max = 70;%80;
    B0=3;
elseif strcmp(sys_type,'Connectome2')
    physical_slew_max = 598.802;
    physical_grad_max = 500;
    B0=2.89;
elseif strcmp(sys_type,'skyra')
    physical_slew_max = 180;
    physical_grad_max = 43;
    B0=2.89;
elseif strcmp(sys_type,'trio')
    physical_slew_max = 170;
    physical_grad_max = 38;
    B0=2.89;
elseif strcmp(sys_type,'terra')
    physical_slew_max = 250;
    physical_grad_max = 135;
    B0=6.98;
else
    error('Undefined')
end

pislquant = 0;


% is siemens
rfDeadTime =  100e-6;
rfRingdownTime = 100e-6;
adcDeadTime = 20e-6;
%     adcRasterTime = 2e-6;
adcRasterTime = 100e-9;
rfRasterTime = 1e-6;
gradRasterTime = 10e-6;
blockDurationRaster = 10e-6;


sys = mr.opts('MaxGrad',physical_grad_max*grad_safety_margin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*slew_safety_margin,'SlewUnit','T/m/s',...
    'rfDeadTime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);

sys_lowPNS = mr.opts('MaxGrad',physical_grad_max*lowPNS_grad_safety_margin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*lowPNS_slew_safety_margin,'SlewUnit','T/m/s',...
    'rfDeadtime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);

sys_diff = mr.opts('MaxGrad',physical_grad_max*diff_grad_safety_margin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*diff_slew_safety_margin,'SlewUnit','T/m/s',...
    'rfDeadtime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);

lims = sys;

seq=mr.Sequence(sys);         % Create a new sequence object



fatOffresFreq = sys.gamma*sys.B0*fatChemShift; % Hz
% TE_ = 1/fatOffresFreq*[1 2]; % fat and water in phase for both echoes
shortest_TE_outofphase = 0.5*1/fatOffresFreq;
TE_outofphase = (1:2:500)*shortest_TE_outofphase;
[~,ind_TE_outofphase_nearCurrentTE] = min(abs(TE_outofphase-TE));
candidate_TEs = TE_outofphase(ind_TE_outofphase_nearCurrentTE-5:ind_TE_outofphase_nearCurrentTE+5);
% fprintf('TEs that will lead to out-of-phase for 1st echo (unit is millisecond): \n%s\n',num2str(1e3*candidate_TEs) );
dTE_pi_difference = 0.5*1/fatOffresFreq;


fprintf('<strong>ESP is %.2f ms\n</strong>\n',esp*1e3);


%--------------------------------------------------------------------------
% parse some inputs
%--------------------------------------------------------------------------

crusher_d = round(crusher_d/sys.gradRasterTime)*sys.gradRasterTime;

delay_time_pi_2_phase = 0;

dkyScale_debra = 1;
 
if debra_nechoes==1 % simple epi
    debra_shift_base = 1;
end


if isConventionalSegmentSpacing
    ky_dist_sameEcho_sameSeg = debra_nechoes*R*RSegment;
else
    ky_dist_sameEcho_sameSeg = debra_nechoes*R;
end


if isTestRun
    Nslices = 1;
%     num_directions_per_b = 1;
    msg = 'Test run, set Nslices to 1';
    warndlg(msg)
end


% bval and bvec
bTable      = xlsread('/autofs/cluster/berkin/xingwang/Syncbb/share/epidiff/Book_30.xlsx');
tmpStruct = load('/autofs/cluster/berkin/xingwang/Syncbb/share/epidiff/diffusion_6_directions.mat');

bTable     = tmpStruct.diffusion_directions;
bTable     = bTable.';
bTable(1,:)=[]; % delete [0 0 0]
bTable(:,3) = -bTable(:,3);
 

% Initialize output arrays
num_bvolumes = sum(cellfun(@(x) length(x), acquisition_order(:, 2))); % Total volumes
bval = zeros(num_bvolumes, 1);
bvec = zeros(num_bvolumes, 3);
current_volume = 1;

% Track volume ranges for output
volume_ranges = cell(size(acquisition_order, 1), 1); % Store [start, end] for each segment

% Generate bval and bvec based on acquisition order
for iter = 1:size(acquisition_order, 1)
    bval_idx = acquisition_order{iter, 1}; % Index into bvalues
    btable_rows = acquisition_order{iter, 2}; % bTable rows or 0 (for b=0)
    num_dirs = length(btable_rows); % Number of directions from bTable rows
    
    % Assign b-value
    bval(current_volume:current_volume+num_dirs-1) = bvalues(bval_idx);
    
    % Assign directions
    if bvalues(bval_idx) <= bvalue0_threshold
        % For b=0 (b=1), use crusher direction [0 0 1]
        v = repmat([0 0 1], num_dirs, 1);
    else
        % For b>0, use specified bTable rows
        v = bTable(btable_rows, :);
    end
    bvec(current_volume:current_volume+num_dirs-1, :) = v;
    
    % Store volume range for this segment
    volume_ranges{iter} = [current_volume, current_volume+num_dirs-1];
    
    current_volume = current_volume + num_dirs;
end

%% Validation
assert(length(bval) == num_bvolumes, 'bval size mismatch');
assert(size(bvec, 1) == num_bvolumes, 'bvec size mismatch');

% Optimized output: Print b-value and volume range in acquisition order
fprintf('Generated %d b-volumes in the following order:\n', num_bvolumes);
for iter = 1:size(acquisition_order, 1)
    bval_idx = acquisition_order{iter, 1};
    range = volume_ranges{iter};
    if range(1) == range(2)
        fprintf('  Volume %d: b=%d\n', range(1), bvalues(bval_idx));
    else
        fprintf('  Volumes %d-%d (%d volumes): b=%d\n', range(1), range(2), range(2)-range(1)+1, bvalues(bval_idx));
    end
end
% Summary of total volumes per b-value
fprintf('Summary:\n');
for b = 1:length(bvalues)
    count = sum(abs(bval - bvalues(b)) < eps); % Count volumes for each b-value
    fprintf('  b=%d: %d volumes\n', bvalues(b), count);
end


nFrames = numel(bval) + nDummyFrames;
if isRefscan
    nFrames = nFrames + 1;
end
 
nImagingFrame = numel(bval);
bval_all_frames = [bvalue0_threshold*ones(nFrames-nImagingFrame,1);bval];
% bvec = [zeros(nFrames-nImagingFrame,3);bvec];
bvec = [repmat([0 0 1],nFrames-nImagingFrame,1);bvec];
maxbval = max(bval_all_frames);
bFactor_scale = sqrt(bval_all_frames./maxbval);

assert(~ismember([0 0 0], bvec, 'rows'), 'In the current implementation, b value=0 (i.e. bvec=0 0 0) is treated as a small b value, thus its direction cannot be [0 0 0], it should be set to [0 0 1], where the last 1 means z direction diff gradient, which is used as crusher. Please change your btable accordingly.');


disp(['total number of frames: ', num2str(nFrames)])

%--------------------------------------------------------------------------
%% define RF and gradients
%--------------------------------------------------------------------------

do_rf_sim = 0;

% Create fat-sat pulse
% B0=2.89; % 1.5 2.89 3.0
sat_ppm=-3.45;
sat_freq=sat_ppm*1e-6*B0*lims.gamma;

% rf_fs = mr.makeGaussPulse(110*pi/180,'system',lims,'Duration',8e-3,...
    % 'bandwidth',abs(sat_freq),'freqOffset',sat_freq);
rf_fs = mr.makeGaussPulse(110*pi/180,'system',lims,'Duration',8e-3,...
    'bandwidth',abs(sat_freq),'freqOffset',sat_freq,'use','saturation');


rf_fs.phaseOffset=-2*pi*rf_fs.freqOffset*mr.calcRfCenter(rf_fs); % compensate for the frequency-offset induced phase
gz_fs = mr.makeTrapezoid('z',sys_lowPNS,'delay',mr.calcDuration(rf_fs),'Area',1/1e-4); % spoil up to 0.1mm


spoiler_amp = 3*8*42.58*10e2;
est_rise = 500e-6;
est_flat = 2500e-6;

gp_r = mr.makeTrapezoid('x','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gp_p = mr.makeTrapezoid('y','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gp_s = mr.makeTrapezoid('z','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);

gn_r = mr.makeTrapezoid('x','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gn_p = mr.makeTrapezoid('y','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);
gn_s = mr.makeTrapezoid('z','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',sys_lowPNS);


% Create 90 degree slice selection pulse and gradient
% this will need to be updated per slice
% [rf, gz, gzReph] = mr.makeSincPulse(pi/2,'system',sys_lowPNS,'Duration',tRFex,...
    % 'SliceThickness',thickness,'apodization',0.5,'timeBwProduct',4);
[rf, gz, gzReph] = mr.makeSincPulse(pi/2,'system',sys_lowPNS,'Duration',tRFex,...
    'SliceThickness',thickness,'apodization',0.5,'timeBwProduct',4, 'use','excitation');

% Create 90 degree slice refocusing pulse and gradients
% this will need to be updated per slice
[rf180, gz180] = mr.makeSincPulse(pi,'system',sys_lowPNS,'Duration',tRFref,...
    'SliceThickness',thickness,'apodization',0.5,'timeBwProduct',4,'PhaseOffset',pi/2,'use','refocusing');


if isSSgradReversal
    gz180 = mr.scaleGrad(gz180, -1);
end

gz180_crusher_1 = mr.makeTrapezoid('z',sys_lowPNS,'area',crusher_area);
gz180_crusher_2 = mr.makeTrapezoid('y',sys_lowPNS,'area',crusher_area);
gz180_crusher_3 = mr.makeTrapezoid('x',sys_lowPNS,'area',crusher_area);

% define the output trigger to play out with every slice excitatuion
% trig=mr.makeDigitalOutputPulse('osc0','duration', 100e-6); % possible channels: 'osc0','osc1','ext1'
% trigger for skope
trig = mr.makeDigitalOutputPulse('ext1','duration', trig_dur); % possible channels: 'osc0','osc1', ext1

% Define other gradients and ADC events
deltak  = 1/fov;
if isConventionalSegmentSpacing
    deltaky = RSegment*R*deltak*dkyScale_debra;      % Rsegement*R
    deltaky_low         =   RSegment*R_low*deltak;
else
    deltaky =          R*deltak*dkyScale_debra;
    deltaky_low         =   R_low*deltak;
end
kWidth  = Nx*partFourierFactor_fe*deltak;
kWidth_org          =   Nx_org*deltak;
kWidth_low          =   Nx_low*deltak;
kWidth_low_org      =   Nx_low_org*deltak;

% Phase blip in shortest possible time
% blip_dur = ceil(2*sqrt(deltaky/lims.maxSlew)/10e-6/2)*10e-6*2; % we round-up the duration to 2x the gradient raster time


if isConventionalSegmentSpacing
    blip_area_debra = RSegment*R*debra_shift_base*deltak;
else
    blip_area_debra = R*debra_shift_base*deltak;
end
largest_blip = max(abs(blip_area_debra),abs(deltaky));
blip_dur     = ceil( 2*sqrt( largest_blip/sys.maxSlew )/sys.gradRasterTime/2 ) *sys.gradRasterTime*2; % round-up the duration to 2x the gradient raster time

% the split code below fails if this really makes a trpezoid instead of a triangle...
gy = mr.makeTrapezoid('y',lims,'Area',-deltaky,'Duration',blip_dur); % we use negative blips to save one k-space line on our way towards the k-space center
%gy = mr.makeTrapezoid('y',lims,'amplitude',deltak/blip_dur*2,'riseTime',blip_dur/2, 'flatTime', 0);

readoutTime = esp - blip_dur;
% fprintf('<strong>Readout bandwidth per pixel = 1/readout_duration is %.1f\n</strong>',1/readoutTime); % not accurate

% readout gradient is a truncated trapezoid with dead times at the beginnig
% and at the end each equal to a half of blip_dur
% the area between the blips should be defined by kWidth
% we do a two-step calculation: we first increase the area assuming maximum
% slewrate and then scale down the amlitude to fix the area
extra_area=blip_dur/2*blip_dur/2*lims.maxSlew; % check unit!;
gx = mr.makeTrapezoid('x',lims,'Area',kWidth+extra_area,'duration',readoutTime+blip_dur);

% flat_time/total_time = (1-ab_ratio)
ramp_ratio = 1-gx.flatTime/(gx.flatTime+gx.riseTime+gx.fallTime);
fprintf('<strong>Ramp ratio = (1-flat_time/total_time)*100%% = %.1f%%\n</strong>',...
    100*ramp_ratio);
actual_area=gx.area-gx.amplitude/gx.riseTime*blip_dur/2*blip_dur/2/2-gx.amplitude/gx.fallTime*blip_dur/2*blip_dur/2/2;
gx.amplitude=gx.amplitude/actual_area*kWidth;
gx.area = gx.amplitude*(gx.flatTime + gx.riseTime/2 + gx.fallTime/2);
gx.flatArea = gx.amplitude*gx.flatTime;
fprintf('<strong>Readout Gx amplitue = %.2f mT/m\n</strong>',mr.convert(gx.amplitude,'Hz/m','mT/m'));
fprintf('<strong>Readout Gx slew     = %.2f T/m/s\n</strong>',1e-3*mr.convert(gx.amplitude,'Hz/m','mT/m')/gx.riseTime);


extra_area_org      =   blip_dur/2*blip_dur/2*lims.maxSlew; % check unit!;
%         gx_org              =   mr.makeTrapezoid('x',lims,'Area',kWidth_org+extra_area_org,'duration',readoutTime+blip_dur);
gx_org              =   mr.makeTrapezoid('x',lims,'Area',kWidth_org+extra_area_org);
actual_area_org     =   gx_org.area-gx_org.amplitude/gx_org.riseTime*blip_dur/2*blip_dur/2/2-gx_org.amplitude/gx_org.fallTime*blip_dur/2*blip_dur/2/2;
gx_org.amplitude    =   gx_org.amplitude/actual_area_org*kWidth_org;
gx_org.area         =   gx_org.amplitude*(gx_org.flatTime + gx_org.riseTime/2 + gx_org.fallTime/2);
gx_org.flatArea     =   gx_org.amplitude*gx_org.flatTime;



% calculate ADC
% we use ramp sampling, so we have to calculate the dwell time and the
% number of samples, which are will be qite different from Nx and
% readoutTime/Nx, respectively.
adcDwellNyquist=deltak/gx.amplitude/ro_os;
% round-down dwell time to 100 ns
% adcDwell=floor(adcDwellNyquist*1e7)*1e-7;
adcDwell=floor(adcDwellNyquist*(1/sys.adcRasterTime))*sys.adcRasterTime; % GE raster is not 1e-7
fprintf('<strong>adcDwell=%.2f us</strong>\n',adcDwell*1e6);
fprintf('<strong>Bandwidth per pixel = 1/adc_dwell/Nx = %.1f Hz/pixel\n</strong>',1/adcDwell/Nx);


minimalAdcDwell = 1.3;
tolerance = 1e-10; 
assert(  adcDwell*1e6-minimalAdcDwell  >=-tolerance  ,'ADC dwell too small. Currently, on Siemens scanner, the minimal adcDwell is %.2f us, while the current adc dwell is %.2f us',minimalAdcDwell,adcDwell*1e6)

adcSamples=floor(readoutTime/adcDwell/4)*4; % on Siemens the number of ADC samples need to be divisible by 4
% MZ: no idea, whether ceil,round or floor is better for the adcSamples...
adc = mr.makeAdc(adcSamples,sys,'Dwell',adcDwell,'Delay',blip_dur/2);
% realign the ADC with respect to the gradient
time_to_center=adc.dwell*((adcSamples-1)/2+0.5); % I've been told that Siemens samples in the center of the dwell period
adc.delay=round((gx.riseTime+gx.flatTime/2-time_to_center)*1e6)*1e-6; % we adjust the delay to align the trajectory with the gradient. We have to aligh the delay to 1us
adc.delay=round((gx.riseTime+gx.flatTime/2-time_to_center)*(1/sys.rfRasterTime))*sys.rfRasterTime; % Xingwang: in the mr.checkTiming, this is aligned to raster of rf
  
 
%--------------------------------------------------------------------------
% assembly blocks, #0 gre acs
%--------------------------------------------------------------------------

segmentID = 1;

if isUseGreACS
    [seq,segmentID] = fn_addacs_v0(Nx, fov, thickness, sys, sys_lowPNS, adc, seq, segmentID, pislquant, Nslices);
end
 

%--------------------------------------------------------------------------
% water sat pulses
%--------------------------------------------------------------------------

cestPrepParms.tp      = 0.1;
cestPrepParms.td      = 0.005;
cestPrepParms.shape   = 'gauss'; % gauss, block, sinc
cestPrepParms.B1cwpe  = 2; % uT
cestPrepParms.isSpoil = true;%false;
cestPrepParms.npulses = 10;
% satPulse = makeSaturationPulseFromCWPE(cestPrepParms.shape, cestPrepParms.B1cwpe, cestPrepParms.tp,cestPrepParms.td,sys);
% offsets_Hz = [192500	2000	0	32	-32	64	-64	96	-96	128	-128	192	-192	256	-256	256	-256	320	-320	320	-320	384	-384	384	-384	416	-416	416	-416	448	-448	448	-448	448	-448	448	-448	448	-448	448	-448	480	-480	480	-480	512	-512	512	-512	576	-576	640	-640	768	-768	1280	2560	3840	5120	6400	7680	8960	10240];
offsets_Hz = [	0];%	32	-32	64	-64	96	-96	128	-128	192	-192	256	-256	256	-256	320	-320	320	-320	384	-384	384	-384	416	-416	416	-416	448	-448	448	-448	448	-448	448	-448	448	-448	448	-448	480	-480	480	-480	512	-512	512	-512	576	-576	640	-640	768	-768	1280	2560	3840	5120	6400	7680	8960	10240];
seq.setDefinition('CESTfreqOffsets',offsets_Hz)

cest_prep_dur = cestPrepParms.npulses * ( cestPrepParms.tp + cestPrepParms.td );
if cestPrepParms.isSpoil
    % spoilers
    spoilRiseTime = 1e-3;
    spoilDuration = 4500e-6+ spoilRiseTime; % [s]
    % create pulseq gradient object
    [gxSpoil, gySpoil, gzSpoil] = makeSpoilerGradients(sys, spoilDuration, spoilRiseTime);
    spoil_dur = mr.calcDuration(gxSpoil, gySpoil, gzSpoil);
    
    cest_prep_dur = cest_prep_dur + spoil_dur;
end


%--------------------------------------------------------------------------
% seq blocks
%--------------------------------------------------------------------------


adc.id = seq.registerAdcEvent(adc);

segid_delayTR = segmentID + 1;
segidbase = segmentID + 2;
slice_indices = tdr_sliceorder(Nslices,1);
adc_backup = adc;

adc_dummy  = mr.makeDelay(mr.calcDuration(adc));

gx_backup  = gx;
all_kyShiftFactors_frame = [];

tic
isAlreadyExecuted = false;
isAlreadyShowDelayAfterOneSlice = false;
isAlreadyShowBvalueDueToCrusher = false;
isFirstTrigger = true;
for iterFrame = 1:nFrames
    bScale = bFactor_scale(iterFrame);
    
    isImagingFrame = true;    
    if iterFrame<=nFrames-nImagingFrame
        isImagingFrame = false;
    end
    
    if isKyShiftBetweenDiffFrames && isImagingFrame        
        kyShiftFactor_frame = R*mod(iterFrame-(nFrames-nImagingFrame),kqShiftPeriod);
        kyShiftFactor_frame = 1*mod(iterFrame-(nFrames-nImagingFrame),kqShiftPeriod); % FIXME, if only 1 seg, set this to the segment shift, to have unique and uniform shift across diffusion frame        
    else
        kyShiftFactor_frame = 0;
    end
    all_kyShiftFactors_frame(end+1) = kyShiftFactor_frame;
  
    
    % phase encoding and partial Fourier
    Ny_pre  = round((partFourierFactor-1/2)*Ny-1);  % PE steps prior to ky=0, excluding the central line
    if isConventionalSegmentSpacing
        Ny_pre  = round(Ny_pre/RSegment/R);
        Ny_post = round(Ny/2+1); % PE lines after the k-space center including the central line
        Ny_post = round(Ny_post/RSegment/R);
    else
        Ny_pre  = round(Ny_pre/   1     /R);
        Ny_post = round(Ny/2+1); % PE lines after the k-space center including the central line
        Ny_post = round(Ny_post/  1     /R);
    end
    Ny_meas = Ny_pre+Ny_post;
    
    if isImagingFrame
        number_of_shots = RSegment;
    else % dummy or ref, a single shot
        number_of_shots = 1;
    end
    
    for iterSeg = 1:number_of_shots
        gx = gx_backup;
        
        % Pre-phasing gradients
        gxPre = mr.makeTrapezoid('x',lims,'Area',-gx_org.area/2);
        if 1
            base_shift_between_seg = round(ky_dist_sameEcho_sameSeg/RSegment); % unique and uniform
            assert(base_shift_between_seg>=1,'shift too small, no shift between segments!')            
        else
            base_shift_between_seg = 1; % beneficial for the uniqueness of the ky lines            
        end
        gyPre  = mr.makeTrapezoid('y',sys,'Area',Ny_pre*deltaky/dkyScale_debra-(iterSeg-1)*base_shift_between_seg*deltak-kyShiftFactor_frame*deltak); 

        if iterFrame==1 && iterSeg==1
            gyPreFirst = gyPre;
        end
        
        
        [gxPre,gyPre]=mr.align('right',gxPre,'left',gyPre);
        % relax the PE prepahser to reduce stimulation
        gyPre = mr.makeTrapezoid('y',lims,'Area',gyPre.area,'Duration',mr.calcDuration(gxPre,gyPre));
        gyPre.amplitude=gyPre.amplitude*pe_enable;
        
        % split the blip into two halves and produnce a combined synthetic gradient
        gy_parts = mr.splitGradientAt(gy, blip_dur/2, lims);
        [gy_blipup, gy_blipdown, ~]=mr.align('right',gy_parts(1),'left',gy_parts(2),gx);
        gy_blipdownup=mr.addGradients({gy_blipdown, gy_blipup}, lims);
        
        % pe_enable support
        gy_blipup.waveform=gy_blipup.waveform*pe_enable;
        gy_blipdown.waveform=gy_blipdown.waveform*pe_enable;
        gy_blipdownup.waveform=gy_blipdownup.waveform*pe_enable;
        
        % Calculate delay times
        durationToCenter = (Ny_pre+0.5)*mr.calcDuration(gx);
        rfCenterInclDelay=rf.delay + mr.calcRfCenter(rf);
        rf180centerInclDelay=rf180.delay + mr.calcRfCenter(rf180);
        
        delayTE1=ceil((TE/2 - mr.calcDuration(rf,gz) + rfCenterInclDelay - mr.calcDuration(gzReph) -  rf180centerInclDelay)/lims.gradRasterTime)*lims.gradRasterTime;
        % we do not need to include ETS in delayTE2 calculation, since delayTE2 is defined in 1st shot, where there is not ETS
        delayTE2=ceil((TE/2 - mr.calcDuration(rf180,gz180) + rf180centerInclDelay - mr.calcDuration(trig) - delay_after_trig- durationToCenter)/lims.gradRasterTime)*lims.gradRasterTime;
        assert(delayTE1>=0);
        assert(delayTE2>=0);
        
        delayTE1_b0 = delayTE1 - mr.calcDuration(gz180_crusher_1,gz180_crusher_2,gz180_crusher_3);
        delayTE2_b0 = delayTE2 - mr.calcDuration(gz180_crusher_1,gz180_crusher_2,gz180_crusher_3);
        assert(delayTE1_b0>=0);
        assert(delayTE2_b0>=0);        
        
        [gxPre,gyPre]=mr.align('right',gxPre,'left',gyPre);       
        
        sign_gyPre = sign(gyPre.area);

        gyBlip_debra = gy;

        
        gyBlip_debra_parts = mr.splitGradientAt(gyBlip_debra, blip_dur/2, sys);
        [gyBlip_debra_up,gyBlip_debra_down,~]=mr.align('right',gyBlip_debra_parts(1),'left',gyBlip_debra_parts(2),gx);
        % now for inner echos create a special gy gradient, that will ramp down to 0, stay at 0 for a while and ramp up again
        gyBlip_debra_down_up=mr.addGradients({gyBlip_debra_down, gyBlip_debra_up}, sys);
        
        gyBlip_down_debraUp =mr.addGradients({gy_blipdown, gyBlip_debra_up}, sys);
        gyBlip_debraDown_up =mr.addGradients({gy_blipup, gyBlip_debra_down}, sys);
        
        small_delta=delayTE2-ceil(sys_diff.maxGrad/sys_diff.maxSlew/lims.gradRasterTime)*lims.gradRasterTime;
        big_delta=delayTE1+mr.calcDuration(rf180,gz180);
        
        %g=sqrt(bval(iterFrame)*1e6/bFactCalc(1,small_delta,big_delta)); % for now it looks too large!
        g=sqrt(maxbval*1e6/ fn_bFactCalc(1,small_delta,big_delta)); % for now it looks too large!
        gr=ceil(g/sys_diff.maxSlew/lims.gradRasterTime)*lims.gradRasterTime;

        gDiff=mr.makeTrapezoid('z','amplitude',g,'riseTime',gr,'flatTime',small_delta-gr,'system',sys_diff);
        
        assert(mr.calcDuration(gDiff)<=delayTE1,'TE too small');
        assert(mr.calcDuration(gDiff)<=delayTE2,'TE too small');        
        
        g_x=g.*bvec(iterFrame,1);
        g_y=g.*bvec(iterFrame,2);
        g_z=g.*bvec(iterFrame,3);
        
        if bval_all_frames(iterFrame)<=bvalue0_threshold%((sum(bvec(iterFrame,:))==0)||(sum(bvec(iterFrame,:))==1)) % b=0 or dwi with diffusion gradient on one axis we keep using the older version
            gDiff_x=gDiff; gDiff_x.channel='x';
            gDiff_y=gDiff; gDiff_y.channel='y';
            gDiff_z=gDiff; gDiff_z.channel='z';
        else
            [azimuth,elevation,r] = cart2sph(g_x,g_y,g_z);
            polar= -(pi/2-elevation);
            
            Gr=mr.rotate('z',azimuth,mr.rotate('y',polar,gDiff));
            if size(Gr,2)==3
                gDiff_x=Gr{1,2};
                gDiff_y=Gr{1,3};
                gDiff_z=Gr{1,1};
            else
                if size(Gr,2)==2
                    diffusion_blank=find( bvec(iterFrame,:)==0);
                    switch diffusion_blank
                        case 2
                            gDiff_x=Gr{1,2};
                            gDiff_z=Gr{1,1};
                            gDiff_y=gDiff; gDiff_y.channel='y'; gDiff_y.amplitude=0; gDiff_y.area=0; gDiff_y.flatArea=0;
                        case 1
                            gDiff_z=Gr{1,1};
                            gDiff_y=Gr{1,2};
                            gDiff_x=gDiff; gDiff_x.amplitude=0; gDiff_x.area=0; gDiff_x.flatArea=0;gDiff_x.channel='x';
                        case 3
                            gDiff_x=Gr{1,2};
                            gDiff_y=Gr{1,1};
                            gDiff_z=gDiff; gDiff_z.amplitude=0; gDiff_z.area=0; gDiff_z.flatArea=0;gDiff_z.channel='z';
                    end
                end
            end
        end
        
        % Calculate the echo time shift for multishot EPI (QL)
        actual_esp          = gx.riseTime + gx.flatTime + gx.fallTime;
        TEShift             = actual_esp/RSegment;
        TEShift             = round(TEShift/sys.gradRasterTime)*sys.gradRasterTime;%round(TEShift,5); % from Berkin: roundn didn't work for the latest matlab, changed to round (sign -/+) % v2
        TEShift_before_echo = (iterSeg-1)*TEShift;
 
        TEShift_after_echo  = (RSegment-(iterSeg-1))*TEShift;
 
       
        % dummy shots before turning on ADC, to reach steady state
        isDummyShot = iterFrame <= nDummyFrames;
        if isDummyShot
            adc = adc_dummy;
        else
            adc = adc_backup;
        end
        
        % First frame is EPI calibration/reference scan (blips off)
        isRefShot = (iterFrame == nDummyFrames+1 ) & isRefscan;
        
        blipsOn             = ~isDummyShot & ~isRefShot;
        blipsOn = blipsOn + (blipsOn== 0)*eps;        % non-zero scaling so that the trapezoid shape is preserved in the .seq file, for GE
        gyPre               = mr.scaleGrad(gyPre,blipsOn);
        gy_blipup           = mr.scaleGrad(gy_blipup,blipsOn);
        gy_blipdown         = mr.scaleGrad(gy_blipdown,blipsOn);
        gy_blipdownup       = mr.scaleGrad(gy_blipdownup,blipsOn);
        gyBlip_down_debraUp = mr.scaleGrad(gyBlip_down_debraUp,blipsOn);
        gyBlip_debraDown_up = mr.scaleGrad(gyBlip_debraDown_up,blipsOn);
        gyBlip_debra_down   = mr.scaleGrad(gyBlip_debra_down,blipsOn);
        
                        
        for iterSlice=slice_indices
            gx = gx_backup;
            slice_start_time = sum(seq.blockDurations);
            
            rf.freqOffset=gz.amplitude*thickness*(iterSlice-1-(Nslices-1)/2);
            rf.phaseOffset=-2*pi*rf.freqOffset*mr.calcRfCenter(rf); % compensate for the slice-offset induced phase
            rf180.freqOffset=gz180.amplitude*thickness*(iterSlice-1-(Nslices-1)/2);
            rf180.phaseOffset=pi/2-2*pi*rf180.freqOffset*mr.calcRfCenter(rf180); % compensate for the slice-offset induced phase
            
            segID_due_to_diff_encoding = 1;%~isb0 * iterFrame;
            segID_due_to_shots = iterSeg; % porbably we will not need this since we can pass pge2.validate. but for safety we just use it, since we have a bunch of available TRID
            segmentID = segidbase + isDummyShot + 2*isRefShot + segID_due_to_diff_encoding + segID_due_to_shots;
            % lblTRID = mr.makeLabel('SET', 'TRID', segmentID);
            
                        
            if isFatSat
                % seq.addBlock(gp_r,gp_p,gp_s,lblTRID);
                seq.addBlock(gp_r,gp_p,gp_s);
                seq.addBlock(rf_fs,gn_r,gn_p,gn_s);
            end
            
            if isFatSat
                if shim2use == 1
                    seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector1));
                else
                    if shim2use == 2 
                       seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector2));
                   else
                        seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector3));
                    end
                end
            else
                seq.addBlock(rf,gz,lblTRID);
            end
            seq.addBlock(gzReph);
            
            
            if bval_all_frames(iterFrame)<=bvalue0_threshold
                seq.addBlock(mr.makeDelay(delayTE1_b0));
                seq.addBlock(gz180_crusher_1,gz180_crusher_2,gz180_crusher_3);
                t_start_crusher_left = prevBlcokStartTime(seq);
                
                if shim2use == 1
                    seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector1));
                else
                    if shim2use == 2 
                       seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector2));
                   else
                        seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector3));
                    end
                end

                seq.addBlock(gz180_crusher_1,gz180_crusher_2,gz180_crusher_3);
                t_start_crusher_right = prevBlcokStartTime(seq);
                seq.addBlock(mr.makeDelay(delayTE2_b0)); 

                if isAlreadyShowBvalueDueToCrusher
                    % do nothing
                else
                    isAlreadyShowBvalueDueToCrusher = true; % only show once
                    
                    b0_small_delta = gz180_crusher_1.riseTime + gz180_crusher_1.flatTime;
                    b0_big_delta = t_start_crusher_right - t_start_crusher_left;
                    bvalue_due_to_crusher = calc_bval_trap(mr.convert(gz180_crusher_1.amplitude,'Hz/m','mT/m'),b0_small_delta,b0_big_delta,gz180_crusher_1.riseTime);
                    total_bval_crusher = 3*bvalue_due_to_crusher; % x,y,z         
                    fprintf('<strong>b-value due to crusher gradient is: %.3f s/mm^2</strong>\n',total_bval_crusher);
                end                   
            else            
                seq.addBlock(mr.makeDelay(delayTE1),mr.scaleGrad(gDiff_x,bScale),mr.scaleGrad(gDiff_y,bScale),mr.scaleGrad(gDiff_z,bScale));

                if shim2use == 1
                    seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector1));
                else
                    if shim2use == 2 
                       seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector2));
                   else
                        seq.addBlock(rf180,gz180,mr.makeRfShim(shim_vector3));
                    end
                end
                
                seq.addBlock(mr.makeDelay(delayTE2),mr.scaleGrad(gDiff_x,bScale),mr.scaleGrad(gDiff_y,bScale),mr.scaleGrad(gDiff_z,bScale));
            end
            
            
            % Store previous trigger time
            if isFirstTrigger
                isFirstTrigger = false;
            else                
                prev_trigger_time = trigger_start_time;
            end            
            trigger_start_time = sum(seq.blockDurations);            
            % Calculate time difference between triggers
            if exist('prev_trigger_time', 'var')
                trigger_time_diff = trigger_start_time - prev_trigger_time;
                assert( round(trigger_time_diff*1e6 )==round(TR*1e6/Nslices), 'Time between triggers is not equal to TR/Nslices, expected %.2f us, got %.2f us', TR*1e6/Nslices, trigger_time_diff*1e6);                
            end
            
            seq.addBlock(trig);
            seq.addBlock(mr.makeDelay(delay_after_trig));
            
            if isEchoTimeShift && TEShift_before_echo>0
                seq.addBlock(mr.makeDelay(TEShift_before_echo)); % echotimeshift for multishot
            end
            
            gyPreScale = gyPre.area / gyPreFirst.area;
            % seq.addBlock(gxPre,gyPre);
            seq.addBlock(gxPre,mr.scaleGrad(gyPreFirst,gyPreScale)); % for GE
  

            for i=1:Ny_meas
                if i==1
                    seq.addBlock(gx,gy_blipup,adc); % Read the first line of k-space with a single half-blip at the end
                elseif i==Ny_meas
                    if mod(i,debra_nechoes)==1
                        seq.addBlock(gx,gyBlip_debra_down,adc);
                    else
                        seq.addBlock(gx,gy_blipdown,adc); % Read the last line of k-space with a single half-blip at the beginning
                    end
                else
                    if mod(i,debra_nechoes)==0
                        seq.addBlock(gx,gyBlip_down_debraUp,adc);
                    elseif mod(i,debra_nechoes)==1
                        seq.addBlock(gx,gyBlip_debraDown_up,adc);
                    elseif mod(i,debra_nechoes)==2
                        seq.addBlock(gx,gy_blipdownup,adc);
                    end
                end
                gx = mr.scaleGrad(gx,-1);   % Reverse polarity of read gradient
            end
            
            main_echo_readout_end_time = sum(seq.blockDurations);
            
            
            if isEchoTimeShift && TEShift_after_echo>0
                seq.addBlock(mr.makeDelay(TEShift_after_echo))
            end 
             

            slice_end_time = sum(seq.blockDurations);
            time_1_slice = slice_end_time - slice_start_time;
            delay_after_1slice = TR/Nslices - time_1_slice;
            assert(delay_after_1slice>=0, 'TR too short, at least, you should increase TR to %.2f seconds',time_1_slice*Nslices);
            delay_after_1slice = round(delay_after_1slice/sys.blockDurationRaster)*sys.blockDurationRaster;
            if ~isAlreadyShowDelayAfterOneSlice
                isAlreadyShowDelayAfterOneSlice = true;
                fprintf('<strong>Delay after each slice is %.2f ms\n</strong>',1000*delay_after_1slice)                        
            end
            seq.addBlock(mr.makeDelay(delay_after_1slice));
            
        end % of slice loop
    end % of segment loop    
end % of frame loop, i.e. diffusion volume loop
tSeqGeneration = toc;

fprintf('Sequence generation took %.1f seconds\n', tSeqGeneration);
fprintf('<strong>Readout duration is Ny_meas*ESP = %.2f ms</strong>\n',1000*Ny_meas*esp);

nsegments_in_ref    = 1;
nsegments_in_dummy  = 1;
n_triggers_in_ref   = isRefscan*   Nslices*nsegments_in_ref;
n_triggers_in_dummy = nDummyFrames*Nslices*nsegments_in_dummy;
n_useless_triggers  = n_triggers_in_ref + n_triggers_in_dummy;
skope_msg = [];

t = sprintf('<strong>---skope: number of useless triggers=%d</strong>\n',n_useless_triggers);
skope_msg = [skope_msg, t]; 
t = sprintf('<strong>---skope: number of triggers=%d</strong>\n',Nslices*RSegment*nImagingFrame);
skope_msg = [skope_msg, t]; 
t = sprintf('<strong>---skope: time betweem trigger and end of main readout=%.2f ms</strong>\n',1e3*(main_echo_readout_end_time-trigger_start_time));
skope_msg = [skope_msg, t];
fprintf(skope_msg);
scanTime = sum(seq.blockDurations);
fprintf('<strong>Sequence duration is %02d:%02d\n</strong>', floor(scanTime/60),ceil(rem(scanTime, 60)));

% -------------------------------------------------------------------------
% estimate adc size
% -------------------------------------------------------------------------

num_imaging_blocks = numel(seq.blockEvents);
imaging_blocks = cell2mat(seq.blockEvents);
imaging_blocks = reshape(imaging_blocks,[],num_imaging_blocks);
num_adcs = nnz(  imaging_blocks(6,:)~=0  ); % these are LARGE ADCs concatenated from small adcs (atom adc)
num_adcs_expected = 32 + nFrames*Ny_meas;


% -------------------------------------------------------------------------
% check whether the timing of the sequence is correct
% -------------------------------------------------------------------------

% [ok, error_report] = seq.checkTiming;
% 
% if ok
%     fprintf('Timing check passed successfully\n');
% else
%     fprintf('Timing check failed! Error listing follows:\n');
%     fprintf([error_report{:}]);
%     fprintf('\n');
% end

% -------------------------------------------------------------------------
% prepare the sequence output for the scanner
% -------------------------------------------------------------------------

isGEscanner = 0;

seq.setDefinition('FOV', [fov fov Nslices*thickness]);
seq.setDefinition('Name', 'sedebra');
seq.setDefinition('Nx', Nx);
seq.setDefinition('Ny', Ny);
seq.setDefinition('Ry', R);
seq.setDefinition('Ry_low', R_low);
seq.setDefinition('deltaTE', deltaTE);
seq.setDefinition('Ny_meas', Ny_meas);
seq.setDefinition('partFourierFactor', partFourierFactor);
seq.setDefinition('EchoSpacing', esp);
seq.setDefinition('pislquant', pislquant);
seq.setDefinition('bvec',bvec);
seq.setDefinition('bval',bval);
seq.setDefinition('nImagingFrame',nImagingFrame);
seq.setDefinition('Nslices',Nslices);
seq.setDefinition('numSamples',adc.numSamples);
seq.setDefinition('RSegment',RSegment);
seq.setDefinition('isGEscanner',isGEscanner);
seq.setDefinition('isRefscan',isRefscan);
seq.setDefinition('debra_nechoes',debra_nechoes);
seq.setDefinition('isProlongedDebraBlip',isProlongedDebraBlip);
seq.setDefinition('isConventionalSegmentSpacing',isConventionalSegmentSpacing);
seq.setDefinition('isUseGreACS',isUseGreACS);
seq.setDefinition('dTE_esp_ratio',dTE_esp_ratio);
seq.setDefinition('kqShiftPeriod',kqShiftPeriod);
seq.setDefinition('all_kyShiftFactors_frame',all_kyShiftFactors_frame);
seq.setDefinition('bvalue0_threshold',bvalue0_threshold);
seq.setDefinition('isFatSat',isFatSat);
seq.setDefinition('isSSgradReversal',isSSgradReversal);
seq.setDefinition('Nx_low_org',Nx_low_org);
seq.setDefinition('Ny_low_org',Ny_low_org);

adc_low.numSamples = 0;
Ny_low_meas = 0;
isDebra = 0;

seq.setDefinition('numSamplesLow',adc_low.numSamples);
seq.setDefinition('Ny_low_meas',Ny_low_meas);
seq.setDefinition('Nx_low_org',Nx_low_org);
seq.setDefinition('Ny_low_org',Ny_low_org);
seq.setDefinition('dkyScale_debra',dkyScale_debra);
seq.setDefinition('isDebra',isDebra);
seq.setDefinition('base_shift_between_seg',base_shift_between_seg);
seq.setDefinition('adcDwell',adc.dwell);
seq.setDefinition('TEShift',TEShift);
seq.setDefinition('is2ndEcho',is2ndEcho);
seq.setDefinition('EchoTime',TE);
seq.setDefinition('ro_os',ro_os);

% -------------------------------------------------------------------------
% do some visualizations
% -------------------------------------------------------------------------

block_lbl = seq.evalLabels('evolution','block');


if isTestRun
    % seq.plot();             % Plot sequence waveforms
    
    % trajectory calculation
    [ktraj_adc1, t_adc1, ktraj1, t_ktraj1, t_excitation1, t_refocusing1] = seq.calculateKspacePP();
    kmax = 1/fov*Nx/2;
    
    
    % plot k-spaces
    % figure; plot(t_ktraj1, ktraj1'); % plot the entire k-space trajectory
    % hold on; plot(t_adc1,ktraj_adc1(1,:),'.'); % and sampling points on the kx-axis
    figure; plot(ktraj1(1,:),ktraj1(2,:),'b'); % a 2D plot
    axis('equal'); % enforce aspect ratio for the correct trajectory display
    hold;plot(ktraj_adc1(1,:),ktraj_adc1(2,:),'r.'); % plot the sampling points
    xline(kmax,'k--');
    xline(-kmax,'k--');
    yline(kmax,'k--');
    yline(-kmax,'k--');
end


% -------------------------------------------------------------------------
% pns check
% -------------------------------------------------------------------------

% if strcmp(sys_type,'prisma')
%     [pns,tpns]=seq.calcPNS('/autofs/cluster/berkin/xingwang/Syncbb/others_toolboxes/pulseq/matlab/idea/asc/MP_GPA_K2309_2250V_951A_AS82.asc');
% else
%     if strcmp(sys_type,'prisma_XA30A')
%         [pns,tpns]=seq.calcPNS('/autofs/cluster/berkin/xingwang/Syncbb/others_toolboxes/pulseq/matlab/idea/asc/MP_GPA_K2309_2250V_951A_AS82_XA30A_mod_twoFilesCombined.asc');
%     else
%         if strcmp(sys_type,'skyra')
%             [pns,tpns]=seq.calcPNS('/autofs/cluster/berkin/xingwang/Syncbb/others_toolboxes/pulseq/matlab/idea/asc/MP_GPA_K2309_2250V_793A_GC99.asc');
%         else
%             if strcmp(sys_type,'Connectome2') || strcmp(sys_type,'C2_simulate_prisma')
%                 [pns,tpns]=seq.calcPNS('/autofs/cluster/berkin/xingwang/Syncbb/others_toolboxes/pulseq/matlab/idea/asc/MP_GradSys_P034_c_CX600.asc');
%             else
%                 if strcmp(sys_type,'terra') 
%                     [pns,tpns]=seq.calcPNS('/autofs/cluster/berkin/xingwang/Syncbb/others_toolboxes/pulseq/matlab/idea/asc/MP_GradSys_K2298_2250V_1250A_W60_SC72CD.asc');
%                 else
%                     error('Undefined system')
%                 end
%             end
%         end
%     end
% end
% 
% 
% if ~isGEscanner && max(tpns)>0.85
%     warning('PNS=%.2f too high, the sequence may not run on the scanner',max(tpns))
% end


% -------------------------------------------------------------------------
% check forbidden frequencies
% -------------------------------------------------------------------------

% fprintf('Checking frequencies... ');
% 
% fmax2use = 3e3;
% 
% % use pre-upgrade asc file since xa60 version does not contain forbidden
% % freqs
% ascname = '/autofs/cluster/berkin/xingwang/Syncbb/others_toolboxes/pulseq/matlab/idea/asc/MP_GPA_K2298_2250V_793A_SC72CD.asc';
% 
% tic
%     gradSpectrum_bb_v1(seq,sys,ascname,fmax2use);
% toc


% -------------------------------------------------------------------------
% save sequence as .seq
% -------------------------------------------------------------------------

% directory to save seq file to:
seq_path = ['/autofs/cluster/berkin/berkin/Matlab_Code_New/PULSEQ/rf_shimming/', datestr(datetime('today')), '/' ];

if ~isfolder(seq_path)
    mkdir(seq_path)
end

% filename to use
filename = ['epise_rs_R', num2str(R), '_RSegment', num2str(RSegment), '_pf', num2str(partFourierFactor)];

if isfile([seq_path, filename, '.seq'])
    % remove existing file
    system(['rm ', seq_path, filename, '.seq'])
end

% seq.write([seq_path, filename, '_cp.seq']);
% seq.write([seq_path, filename, '_tiamo.seq']);
seq.write([seq_path, filename, '_acp.seq']);


% -------------------------------------------------------------------------
%% very optional slow step, but useful for testing during development e.g. for the real TE, TR or for staying within slewrate limits
% -------------------------------------------------------------------------

if isTestRun
    rep = seq.testReport;
    fprintf([rep{:}]); % as for January 2019 TR calculation fails for fat-sat
end

if ~isTestRun
    return
end


