import numpy as np
import os.path
import pickle
import sys
import warnings

_project_folder_ = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)

from scripts.base_eval import SketchCNNEval, write_row


class TUBerlinSketchCNNEval(SketchCNNEval):

    def __init__(self):
        super().__init__()
        self.chosen_fold = 0
        self.ckpt_step = 0

    def set_run_params(self, chosen_fold, ckpt_step):
        self.chosen_fold = chosen_fold
        self.ckpt_step = ckpt_step

    def prepare_dataset(self, dataset):
        super().prepare_dataset(dataset)
        dataset.set_fold(self.chosen_fold)

    def checkpoint_prefix(self):
        ckpt = self.config['checkpoint']
        return ckpt.format(self.chosen_fold, self.ckpt_step)


if __name__ == '__main__':
    app = TUBerlinSketchCNNEval()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        log_dir = app.config['log_dir']
        model_fn = app.config['model_fn']
        accuracies_list = list()

        ckpt_step = app.config['min_ckpt_step']
        while ckpt_step <= app.config['max_ckpt_step']:

            xvalid_accus = list()
            for fidx in range(3):  # For each fold
                app.set_run_params(fidx, ckpt_step)
                accuracies, stats = app.run()
                xvalid_accus.append(accuracies)
                if len(stats) > 0:
                    # Save collected stats
                    with open(os.path.join(log_dir, app.dataset.get_name_prefix() + '-{}.pkl'.format(ckpt_step)), 'wb') as fh:
                        to_save = {'stats': stats}
                        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)
            avg_xvalid_accus = np.mean(np.array(xvalid_accus, dtype=np.float32), axis=0)
            accuracies_list.append((ckpt_step, avg_xvalid_accus.tolist()))

            ckpt_step += app.config['ckpt_step_freq']

        with open(os.path.join(log_dir, 'progressive_accuracies_{}.csv'.format(model_fn)), 'w') as fh:
            write_row(fh, ['ckpt_step', ] + app.drawing_ratios)
            for accu_tup in accuracies_list:
                write_row(fh, [accu_tup[0], ] + accu_tup[1])

        accus_sort = sorted(accuracies_list, key=lambda tup: tup[1][-1])
        print('Best cross validation accuracy: {} at epoch {}'.format(accus_sort[-1][1][-1],
                                                                      accus_sort[-1][0]))
