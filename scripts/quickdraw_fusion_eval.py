import os.path
import pickle
import sys
import warnings

_project_folder_ = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from scripts.base_eval import SketchFusionEval, write_row

if __name__ == '__main__':
    app = SketchFusionEval()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        accuracies, stats = app.run()

        log_dir = app.config['log_dir']
        model_fn = app.config['model_fn']
        with open(os.path.join(log_dir, 'progressive_accuracies_fusion_{}.csv'.format(model_fn)), 'w') as fh:
            write_row(fh, app.drawing_ratios)
            write_row(fh, accuracies)

        if len(stats) > 0:
            # Save collected stats
            with open(os.path.join(log_dir, app.dataset.get_name_prefix() + '.pkl'), 'wb') as fh:
                to_save = {'stats': stats}
                pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)
