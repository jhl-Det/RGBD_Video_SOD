import argparse
import os.path as osp
from Code.utils.evaluator import Eval_thread
from Code.utils.dataloader import EvalDataset

def main(cfg):
    root_dir = cfg.root_dir
    gt_dir   = cfg.gt_dir
    method_names  = cfg.methods
        
    threads = []
    for method in method_names:
        
        test_res = []
        loader = EvalDataset(osp.join(root_dir), osp.join(gt_dir))
        thread = Eval_thread(loader, method, device=cfg.device)
        threads.append(thread)

        ##
        print(['start evaluation'])
        mae,s,max_f,max_e= thread.run()    ## only compute MAE and s_measure
        
        tmp_str = f'MAE: {mae}----- Smeansure: {s} ------ max_f: {max_f} ----- max_e: {max_e}\n'
        print(tmp_str)
        
        test_res.append([mae,s,max_f,max_e])
        with open("./cal_results.txt", "a") as f:
            f.write(root_dir + "\n")
            f.write(tmp_str)
            f.write("=====" * 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    gt_path       = '/Users/junhaolin/Downloads/workspace/dataset_vidsod_100/test'
    sal_path      = './Predict_maps/'
    test_datasets = ['NJU2K','NLPR', 'DES', 'SSD','SIP', 'STERE'] 
    
    parser.add_argument('--methods',  type=str,  default=['ATF-Net'])
    parser.add_argument('--datasets', type=str,  default=test_datasets)
    parser.add_argument('--gt_dir',   type=str,  default=gt_path)
    parser.add_argument('--root_dir', type=str,  default=sal_path)
    parser.add_argument('--save_dir', type=str,  default=None)
    parser.add_argument('--device',   type=str, default='cpu')
    cfg = parser.parse_args()
    main(cfg)