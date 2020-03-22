import json
import os

import optimization as o
import utils
import visualizer as v

SEMINAR_GRAPH_DATA = 'timeline.json'
graph_data_path = os.path.join(utils.get_data_dir(), SEMINAR_GRAPH_DATA)


def test_graph_init():
    OUTPUT_TITLE = 'Initialization'
    OUTPUT_PICTURE = 'initial_configuration.png'
    output_picture_path = os.path.join(utils.get_output_dir(), OUTPUT_PICTURE)

    timeline = json.load(open(graph_data_path))
    optimization = o.Optimization(timeline)
    
    visualizer = v.Visualizer(grid=True)
    visualizer.update_poses(optimization.poses)
    visualizer.update_features(optimization.features)
    visualizer.show(OUTPUT_TITLE, output_picture_path)


def test_graph_optimization_without_landmarks():
    timeline = json.load(open(graph_data_path))
    optimization = o.Optimization(timeline)
    
    # Disturb vertex positions to see prior effect
    all_params = optimization._all_params
    if all_params is not None:
        all_params[::3] += 1000.5
        all_params[1::3] += 0.5
        optimization._all_params = all_params

    OUTPUT_PICTURE_BEFORE = 'before_optimization.png'
    OUTPUT_PICTURE_AFTER = 'after_optimization.png'
    OUTPUT_TITLE = 'Optimization with prior and odometric edges'

    picture_before_path = os.path.join(utils.get_output_dir(), OUTPUT_PICTURE_BEFORE)
    picture_after_path = os.path.join(utils.get_output_dir(), OUTPUT_PICTURE_AFTER)
    
    visualizer = v.Visualizer(grid=True)
    visualizer.update_poses(optimization.poses)
    visualizer.update_features(optimization.features)
    visualizer.show(OUTPUT_TITLE, picture_before_path)
    
    assert optimization.optimize(1000)

    visualizer.update_poses(optimization.poses)
    visualizer.update_features(optimization.features)
    visualizer.show(OUTPUT_TITLE, picture_after_path)


def test_graph_optimization_with_landmarks():
    N = -1 # TO_IMPLEMENT: your number in the course as listed in Anytask
    alt_graph_data_path = os.path.join(utils.get_data_dir(), 'timeline_{}.json'.format(N))
    timeline = json.load(open(alt_graph_data_path))
    optimization = o.Optimization(timeline)

    OUTPUT_PICTURE_BEFORE = 'before_lndmrk_optimization.png'
    OUTPUT_PICTURE_AFTER = 'after_lndmrk_optimization.png'
    OUTPUT_TITLE = 'Optimization with prior, odometric and landmark edges'
    picture_before_path = os.path.join(utils.get_output_dir(), OUTPUT_PICTURE_BEFORE)
    picture_after_path = os.path.join(utils.get_output_dir(), OUTPUT_PICTURE_AFTER)
    
    visualizer = v.Visualizer(grid=True)
    visualizer.update_poses(optimization.poses)
    visualizer.update_features(optimization.features)
    visualizer.show(OUTPUT_TITLE, picture_before_path)
    
    assert optimization.optimize(30)
    visualizer.update_poses(optimization.poses)
    visualizer.update_features(optimization.features)
    visualizer.show(OUTPUT_TITLE, picture_after_path)
