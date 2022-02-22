import argparse
import yaml
def get_config():    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--yaml")
    # argparser.add_argument("--data_folder_path")
    # argparser.add_argument("--class_list")
    # argparser.add_argument("--method")
    # argparser.add_argument("--feature_path",type=str,default='/data/jiashi/')
    # argparser.add_argument("--data_train_path",type=str,default='')
    # argparser.add_argument("--data_test_path",type=str,default='')
    # argparser.add_argument("--split",default='default_path')
    # argparser.add_argument("--restart",default='0')
    # argparser.add_argument("--nepoch",type=int,default=70)
    # argparser.add_argument("--step_schedular_decay",type=int,default=30)
    # argparser.add_argument("--schedular_step",type=float,default=0.1)
    # argparser.add_argument("--batch_size",type=int,default=64)
    # argparser.add_argument("--start_lr",type=float,default=0.01)
    # argparser.add_argument("--weight_decay",type=float,default=1e-5)
    # argparser.add_argument("--momentum",type=float,default=0.9)
    # argparser.add_argument("--timestamp",type=int,default=10)
    # argparser.add_argument("--num_classes",type=int,default=11)
    # argparser.add_argument("--num_instance_each_class",type=int,default=300)
    # argparser.add_argument("--num_instance_each_class_test",type=int,default=150)
    # argparser.add_argument("--random_seed",type=int,default=1111)
    # argparser.add_argument("--test_split",type=float,default=0.3)
    # argparser.add_argument("--pretrain_feature",type=str,default='None')
    # argparser.add_argument("--max_memory_size",type=int,default=3000)
    args = argparser.parse_args()
    yaml_file_path=args.yaml
    cfg = yaml.load(open(yaml_file_path, 'r'), Loader=yaml.Loader)
    for key in cfg.keys():
        sub_dict=cfg[key]
        for keyy in sub_dict.keys():
            vars(args)[keyy]=sub_dict[keyy]
    return args