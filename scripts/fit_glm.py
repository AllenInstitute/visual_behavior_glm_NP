from visual_behavior_glm_NP.glm import GLM

import argparse

parser = argparse.ArgumentParser(description='fit glm for experiment')
parser.add_argument(
    '--oeid', 
    type=int, 
    default=0,
    metavar='oeid',
    help='ophys experiment ID'
)
parser.add_argument(
    '--version', 
    type=str, 
    default='',
    metavar='glm version',
    help='model version to use'
)

def fit_experiment(oeid, version, log_results=True, log_weights=True, use_previous_fit=False):
    glm = GLM(oeid, version, log_results=log_results, log_weights=log_weights, use_previous_fit=use_previous_fit)

if __name__ == '__main__':
    args = parser.parse_args()
    fit_experiment(args.oeid, args.version)
    print('fitting finished')
