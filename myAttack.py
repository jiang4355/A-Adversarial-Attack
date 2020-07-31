from __future__ import absolute_import, division, print_function
import numpy as np
from numpy.linalg import norm

def myAttack(model, 
    sample, 
    clip_max = 1, 
    clip_min = 0, 
    constraint = 'l2', 
    num_iterations = 40, 
    gamma = 1.0, 
    target_label = None, 
    target_image = None, 
    stepsize_search = 'geometric_progression', 
    max_num_evals = 1e4,
    init_num_evals = 100,
    verbose = True,
    select_orthogonal_type = 'randomly',
    spherical_stepsize = 0.01):
    """
    Main algorithm for HopSkipJumpAttack.

    Inputs:
    model: the object that has predict method. 

    predict outputs probability scores.

    clip_max: upper bound of the image.

    clip_min: lower bound of the image.

    constraint: choose between [l2, linf].

    num_iterations: number of iterations.

    gamma: used to set binary search threshold theta. The binary search 
    threshold theta is gamma / d^{3/2} for l2 attack and gamma / d^2 for 
    linf attack.

    target_label: integer or None for nontargeted attack.

    target_image: an array with the same size as sample, or None. 

    stepsize_search: choose between 'geometric_progression', 'grid_search'.

    max_num_evals: maximum number of evaluations for estimating gradient.

    init_num_evals: initial number of evaluations for estimating gradient.

    Output:
    perturbed image.
    
    """
    # Set parameters
    original_label = np.argmax(model.predict(sample))
    params = {'clip_max': clip_max, 'clip_min': clip_min, 
                'shape': sample.shape,
                'original_label': original_label, 
                'target_label': target_label,
                'target_image': target_image, 
                'constraint': constraint,
                'num_iterations': num_iterations, 
                'gamma': gamma, 
                'd': int(np.prod(sample.shape)), 
                'stepsize_search': stepsize_search,
                'max_num_evals': max_num_evals,
                'init_num_evals': init_num_evals,
                'verbose': verbose,
                'select_orthogonal_type':select_orthogonal_type,
                'spherical_stepsize':spherical_stepsize
                }

    # Set binary search threshold.
    if params['constraint'] == 'l2':
        params['theta'] = params['gamma'] / (np.sqrt(params['d']) * params['d'])
    else:
        params['theta'] = params['gamma'] / (params['d'] ** 2)
        
    # Initialize.
    perturbed = initialize(model, sample, params)

    print(f'[params]:{params}')
    print(f'*****original_label:{original_label} ----> target_label:{target_label}*****')
    
    print(f'original_predict:{np.argmax(model.predict([sample])[0])} ---> {model.predict([sample])}')
    print(f'original_perturbed_predict:{np.argmax(model.predict([perturbed])[0])} ---> {model.predict([perturbed])}')
    
    dist = compute_distance(perturbed, sample, constraint)
    # jie debug
    if verbose:
        print('initialized distance: {:s} distance {:.4E}'.format(constraint, dist))
        print(f'perturbed_predict:{np.argmax(model.predict([perturbed])[0])} ---> {model.predict([perturbed])}')
        
    #use randomly_generate_orthogonal_step
    if select_orthogonal_type=='randomly':
        for j in range(params['num_iterations']):
            params['cur_iter'] = j + 1
            # Project the initialization to the boundary.
            perturbed, dist = binary_search_batch(sample, 
                np.expand_dims(perturbed, 0), 
                model, 
                params)
            # compute new distance.
            if verbose:
                print('iteration: {:d}, {:s} distance {:.4E}'.format(j+1, constraint, dist))
            perturbed,dist=randomly_generate_orthogonal_step(sample,perturbed,model,params)

        perturbed, dist = binary_search_batch(sample, 
                np.expand_dims(perturbed, 0), 
                model, 
                params)
        print(f'[RESULT] Final distance: {dist}')
    
    elif select_orthogonal_type=='minimum':
        perturbed, dist = binary_search_batch(sample, 
                np.expand_dims(perturbed, 0), 
                model, 
                params)
        for j in range(params['num_iterations']):
            min_perturbed=perturbed
            min_dist=dist
            if verbose:
                print('----------------------')
                print('iteration: {:d}, {:s} front_distance {:.4E}'.format(j+1, constraint, min_dist))
            
            for k in range(10):
                orthogonal_candidate,_=randomly_generate_orthogonal_step(sample,perturbed,model,params)
                if verbose:
                    print(f'success generate {k}-th candidate')
                temp_new_perturbed,temp_dist=binary_search_batch(sample, 
                    np.expand_dims(orthogonal_candidate, 0), 
                    model, 
                    params)
                if temp_dist<min_dist:
                    min_perturbed=temp_new_perturbed
                    min_dist=temp_dist
            perturbed=min_perturbed
            dist=min_dist
            if verbose:
                print('iteration: {:d}, {:s} min_distance {:.4E}'.format(j+1, constraint, dist))
        print(f'[RESULT] Final distance: {dist}')
    elif select_orthogonal_type == 'minDy':
        #TODO
        perturbed, dist = binary_search_batch(sample, 
                np.expand_dims(perturbed, 0), 
                model, 
                params)
        for j in range(params['num_iterations']):
            params['cur_iter'] = j + 1
            orthogonal_perturbation=generate_orthogonal_perturbation(sample,perturbed,params,10)
            decisions = decision_function(model,perturbed+orthogonal_perturbation,params)
            print(f'decisions:{decisions}')
            for k in range(10):
                if decisions[k]:
                    orthogonal_perturbation[k]=find_max_stepsize(perturbed,orthogonal_perturbation[k],model,dist,params,10)
                else:
                    orthogonal_perturbation[k]=find_min_stepsize(perturbed,orthogonal_perturbation[k],model,dist,params,10)
            perturbed,dist=binary_search_batch(sample,perturbed+orthogonal_perturbation, 
                    model, 
                    params)
            print(f'iter {j+1},dist:{dist}')
        print(f'[RESULT] Final distance: {dist}')
    elif select_orthogonal_type == 'spsa':
        #TODO
        perturbed, dist = binary_search_batch(sample, 
                np.expand_dims(perturbed, 0), 
                model, 
                params)
        print(f'perturbed_predict:{np.argmax(model.predict([perturbed])[0])} ---> {model.predict([perturbed])}')
        for j in range(params['num_iterations']):
            params['cur_iter'] = j + 1
            # Choose delta.
            delta = select_delta(params, dist)
            print(f"iteration {j+1},delta:{delta}")

            # Choose number of evaluations.
            num_evals = int(params['init_num_evals'] * np.sqrt(j+1))
            num_evals = int(min([num_evals, params['max_num_evals']]))

            # approximate gradient.
            gradf =  approximate_gradient_with_SPSA(model,perturbed,delta,params)
            update = gradf
            print(f"np.linalg.norm(update):{np.linalg.norm(update)}")
            # search step size.
            if params['stepsize_search'] == 'geometric_progression':
                
                # find step size.
                epsilon = geometric_progression_for_stepsize(perturbed, 
                    update, dist, model, params)
                print(f"epsilon:{epsilon}")
                print(f"update gradf:{update}")

                # Update the sample. 
                perturbed = clip_image(perturbed + epsilon * update, 
                    clip_min, clip_max)
                print(f'iter {j+1}, perturbed_predict-1:{np.argmax(model.predict([perturbed])[0])} ---> {model.predict([perturbed])}')
                # Binary search to return to the boundary. 
                perturbed, dist = binary_search_batch(sample, 
                    perturbed[None], model, params)
                print(f'iter {j+1}, perturbed_predict-2:{np.argmax(model.predict([perturbed])[0])} ---> {model.predict([perturbed])}')
            
        print(f'[RESULT] Final distance: {dist}')
    else:
        print('[WARNING]This select type is UNDO')
    return perturbed,dist

def approximate_gradient_with_SPSA(model, sample, ck, params):
    clip_max, clip_min = params['clip_max'], params['clip_min']
    shape = params['shape']
    # Generate random vectors.
    delta=2*np.round(np.random.rand(shape[0],shape[1],shape[2]))-1
    thetaplus=sample+ck*delta
    thetaminus=sample-ck*delta
    
    thetaplus = clip_image(thetaplus, clip_min, clip_max)
    thetaminus = clip_image(thetaminus, clip_min, clip_max)
    
    

    # query the model.
    score_plus = model.predict([thetaplus])[0]
    score_minus = model.predict([thetaminus])[0]
    
    index=0
    if params['target_label'] is None:
        index = np.argmax(score_plus)
    else:
        index = params['target_label']
    print("score:")
    print(score_plus,score_minus)
    print(f"index:{index}")
    score_plus = score_plus[index]
    score_minus = score_minus[index]
    print(f'score_plus:{score_plus},score_minus:{score_minus}')
    print(f'original_score:{model.predict([sample])[0][index]}')

    gradf = (score_plus-score_minus)/2*ck*delta
    print(f"np.linalg.norm(gradf):{np.linalg.norm(gradf)}")

    # Get the gradient direction.
    if np.linalg.norm(gradf)>0:
        gradf = gradf / np.linalg.norm(gradf)

    return gradf

def randomly_generate_orthogonal_step(original_image,perturbed_image,model,params):
    constraint=params['constraint']
    verbose=params['verbose']
    candidate_perturbed=generate_orthogonal_candidate(original_image,perturbed_image,params)
        
    decisions=decision_function(model,np.expand_dims(candidate_perturbed,0),params)
    i=1
    dist = compute_distance(candidate_perturbed, original_image, constraint)
    if verbose:
        print(f'random orthogonal_step:{i}-th，distance:{dist},decisions:{decisions}')

    while decisions[0]==0:
        i+=1
        candidate_perturbed=generate_orthogonal_candidate(original_image,perturbed_image,params)
        decisions=decision_function(model,np.expand_dims(candidate_perturbed,0),params)
        dist = compute_distance(candidate_perturbed, original_image, constraint)
        if verbose:
            print(f'random orthogonal_step:{i}-th，distance:{dist},decisions:{decisions}')
    return candidate_perturbed,dist


def generate_orthogonal_candidate(original_image,perturbed_image,params,N=1):
    spherical_stepsize=params['spherical_stepsize']
    clip_max, clip_min = params['clip_max'], params['clip_min']
    shape=original_image.shape
    all_perturbation=np.random.randn(N,shape[0],shape[1],shape[2])
    spherical_perturbation=[]
    for i in range(N):
        perturbation=all_perturbation[i]
        distance=compute_distance(original_image,perturbed_image,params['constraint'])
        source_direction=original_image-perturbed_image
        source_dir_norm=norm(source_direction)
        source_direction/=distance
        dot=np.vdot(perturbation,source_direction)
        perturbation-=dot*source_direction
        perturbation*=spherical_stepsize*source_dir_norm/norm(perturbation)

        perturbation=perturbed_image+perturbation
        np.clip(perturbation,clip_min,clip_max,out=perturbation)
        spherical_perturbation.append(perturbation)
    if N==1:
        return spherical_perturbation[0]
    else:
        return spherical_perturbation  #added original image
    
def generate_orthogonal_perturbation(original_image,perturbed_image,params,N=1):
    spherical_stepsize=params['spherical_stepsize']
    clip_max, clip_min = params['clip_max'], params['clip_min']
    shape=original_image.shape
    all_perturbation=np.random.randn(N,shape[0],shape[1],shape[2])
    spherical_perturbation=[]
    for i in range(N):
        perturbation=all_perturbation[i]
        distance=compute_distance(original_image,perturbed_image,params['constraint'])
        source_direction=original_image-perturbed_image
        source_dir_norm=norm(source_direction)
        source_direction/=distance
        dot=np.vdot(perturbation,source_direction)
        perturbation-=dot*source_direction
        perturbation*=spherical_stepsize*source_dir_norm/norm(perturbation)

        spherical_perturbation.append(perturbation)
    if N==1:
        return spherical_perturbation[0]
    else:
        return spherical_perturbation  #only perturbation

def decision_function(model, images, params):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    images = clip_image(images, params['clip_min'], params['clip_max'])
    prob = model.predict(images)
    if params['target_label'] is None:
        return np.argmax(prob, axis = 1) != params['original_label'] 
    else:
        return np.argmax(prob, axis = 1) == params['target_label']

def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max) 


def compute_distance(x_ori, x_pert, constraint = 'l2'):
    # Compute the distance between two images.
    if constraint == 'l2':
        return np.linalg.norm(x_ori - x_pert)
    elif constraint == 'linf':
        return np.max(abs(x_ori - x_pert))


def approximate_gradient(model, sample, num_evals, delta, params):
    clip_max, clip_min = params['clip_max'], params['clip_min']

    # Generate random vectors.
    noise_shape = [num_evals] + list(params['shape'])
    if params['constraint'] == 'l2':
        rv = np.random.randn(*noise_shape)
    elif params['constraint'] == 'linf':
        rv = np.random.uniform(low = -1, high = 1, size = noise_shape)

    rv = rv / np.sqrt(np.sum(rv ** 2, axis = (1,2,3), keepdims = True))
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(model, perturbed, params)
    decision_shape = [len(decisions)] + [1] * len(params['shape'])
    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if np.mean(fval) == 1.0: # label changes. 
        gradf = np.mean(rv, axis = 0)
    elif np.mean(fval) == -1.0: # label not change.
        gradf = - np.mean(rv, axis = 0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * rv, axis = 0) 

    # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    return gradf


def project(original_image, perturbed_images, alphas, params):
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)
    if params['constraint'] == 'l2':
        return (1-alphas) * original_image + alphas * perturbed_images
    elif params['constraint'] == 'linf':
        out_images = clip_image(
            perturbed_images, 
            original_image - alphas, 
            original_image + alphas
            )
        return out_images


def binary_search_batch(original_image, perturbed_images, model, params):
    """ Binary search to approach the boundar. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
            compute_distance(
                original_image, 
                perturbed_image, 
                params['constraint']
            ) 
            for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs based on constraint.
    if params['constraint'] == 'linf':
        highs = dists_post_update
        # Stopping criteria.
        thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
    else:
        highs = np.ones(len(perturbed_images))
        thresholds = params['theta']

    lows = np.zeros(len(perturbed_images))

    decisions = 0
    debug_i=0
    # Call recursive function. 
    #while np.max((highs - lows) / thresholds) > 1:
    while np.max((highs - lows) / thresholds) > 1 or decisions[0]==0  :
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)

        # Update highs and lows based on model decisions.
        decisions = decision_function(model, mid_images, params)
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)
        # jie debug
        dist = compute_distance(mid_images, original_image, params['constraint'])
        debug_i+=1
        print('binary search,{}-th distance: distance {:.4E}'.format(debug_i, dist))
        print('decisions:{},highs:{},lows:{}'.format(decisions,highs,lows))
    out_images = project(original_image, perturbed_images, highs, params)

    # Compute distance of the output image to select the best choice. 
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
        compute_distance(
            original_image, 
            out_image, 
            params['constraint']
        ) 
        for out_image in out_images])
    
    print(f'dists:{dists}')
    idx = np.argmin(dists)

    dist = dists[idx]
    out_image = out_images[idx]
    
    print('after {}-th binary search: distance {:.4E}'.format(debug_i,dist))
    return out_image, dist


def initialize(model, sample, params):
    """ 
    Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
    """
    success = 0
    num_evals = 0

    if params['target_image'] is None:
        # Find a misclassified random noise.
        while True:
            random_noise = np.random.uniform(params['clip_min'], 
                params['clip_max'], size = params['shape'])
            success = decision_function(model,random_noise[None], params)[0]
            num_evals += 1
            if success:
                break
            assert num_evals < 1e4,"Initialization failed! "
            "Use a misclassified image as `target_image`" 


        # Binary search to minimize l2 distance to original image.
        low = 0.0
        high = 1.0
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = (1 - mid) * sample + mid * random_noise 
            success = decision_function(model, blended[None], params)
            if success:
                high = mid
            else:
                low = mid

        initialization = (1 - high) * sample + high * random_noise 

    else:
        initialization = params['target_image']

    return initialization


def geometric_progression_for_stepsize(x, update, dist, model, params):
    """
    Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching 
    the desired side of the boundary,
    """
    epsilon = dist / np.sqrt(params['cur_iter']) 

    def phi(epsilon):
        new = x + epsilon * update
        success = decision_function(model, new[None], params)
        return success

    while not phi(epsilon):
        print(f"while epsilon:{epsilon}")
        epsilon /= 2.0

    return epsilon

def find_min_stepsize(x, perturb, model, dist, params,iter_num):
    
    epsilon = dist / np.sqrt(params['cur_iter']) 

    def phi(epsilon):
        new = x + epsilon * perturb
        success = decision_function(model, new[None], params)
        return success
    
    while not phi(epsilon) and iter_num>0:
        epsilon /= 2.0
        iter_num-=1
        
    return epsilon * perturb

def find_max_stepsize(x, perturb, model, dist, params,iter_num):
    
    epsilon = dist / np.sqrt(params['cur_iter']) 

    def phi(epsilon):
        new = x + epsilon * perturb
        success = decision_function(model, new[None], params)
        return success
    
    while phi(epsilon) and iter_num>0:
        epsilon *= 2.0
        iter_num -= 1
    return epsilon * perturb

def select_delta(params, dist_post_update):
    """ 
    Choose the delta at the scale of distance 
    between x and perturbed sample. 

    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist_post_update
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update    

    return delta


