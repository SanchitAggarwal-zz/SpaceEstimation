fprintf('\nAdding Directory Paths');
root = '/home/luminous/Documents/Academics/MS_IIIT/My_Projects/Research_Projects/ICPR_FINAL_23Dec/';
coderoot = [root 'Code/'];
addpath(root);
addpath(coderoot);
addpath([coderoot 'imoverlay/']);
addpath([coderoot 'mexopencv/']);
addpath([coderoot 'LabelMeToolbox/']);
addpath([coderoot 'LabelMeToolbox/LabelMeToolbox/']);
addpath([coderoot 'maxflow/']);
addpath([coderoot 'test_code/']);
addpath([coderoot 'texturefilters/']);
addpath([coderoot 'SegTool/SegToolBox/']);
addpath([root 'ICPR_Qualitative_Analysis/ICPR_Dataset/']);
addpath([root 'ICPR_Qualitative_Analysis/Input/']);
addpath([root 'ICPR_Qualitative_Analysis/Output/']);
addpath([coderoot 'kde2d/']);
addpath([coderoot 'gaussmix-v1.1/']);
addpath([coderoot 'vlfeat-0.9.14/toolbox/']);
vl_setup;

addpath([coderoot 'vlfeat-0.9.14/toolbox/']);
varshroot = [coderoot '/SpatialLayout/spatiallayoutcode/'];
addpath(varshroot);

addpath([varshroot '/GeometricContext/ms/multipleSegmentations/']);
addpath([varshroot '/GeometricContext/geomContext_src_07_02_08/src/']);
addpath([varshroot '/GeometricContext/geomContext_src_07_02_08/src/tools/misc/']);
addpath([varshroot '/GeometricContext/geomContext_src_07_02_08/src/geom/']);
addpath([varshroot '/GeometricContext/geomContext_src_07_02_08/src/mcmc/']);
addpath([varshroot '/GeometricContext/geomContext_src_07_02_08/src/textons/']);
% addpath('./GeometricContext/geomContext_src_07_02_08/src/boosting/boostDt/');
% addpath('./GeometricContext/geomContext_src_07_02_08/src/boosting/boostDt/weightedstats/');
addpath([varshroot '/GeometricContext/geomContext_src_07_02_08/src/tools/misc/']);
addpath([varshroot '/GeometricContext/segment/']);

addpath([varshroot '/GeometricContext/geomContext_src_07_02_08/src/boosting/']);

addpath([varshroot '/ComputeVP/']);
addpath([varshroot '/Geometry/']);
addpath([varshroot '/CLayouts/']);
addpath([varshroot '/Visualize/']);
% addpath('./Make3D/');
% addpath('./Cuboids/');
addpath([varshroot '/Tools/']);