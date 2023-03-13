import torch
from datasets.dataset import UVOSDataset


def create_dataset(args, dataset_names, is_train):
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    elif isinstance(dataset_names, list):
        dataset_names = dataset_names
    else:
        raise NotImplementedError
    dataset = UVOSDataset(args, is_train=is_train, datasets=dataset_names)

    return dataset
  
  
  def create_model(args):

    if args.model == "segformer_b0_ade":
        from models.segformer.segformer import segformer_b0_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b0_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b0_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b1_ade":
        from models.segformer.segformer import segformer_b1_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b1_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b1_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b2_ade":
        from models.segformer.segformer import segformer_b2_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b2_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b2_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b3_ade":
        from models.segformer.segformer import segformer_b3_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b3_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b3_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b4_ade":
        from models.segformer.segformer import segformer_b4_ade
        if args.pretrained:
            pretrained = "your_pretrained/path"
            model = segformer_b4_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b4_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b5_ade":
        from models.segformer.segformer import segformer_b5_ade
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b5_ade(pretrained=pretrained, num_classes=1,
                                     uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b5_ade(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b0_city":
        from models.segformer.segformer import segformer_b0_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b0_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b0_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b1_city":
        from models.segformer.segformer import segformer_b1_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b1_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b1_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b2_city":
        from models.segformer.segformer import segformer_b2_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b2_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b2_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b3_city":
        from models.segformer.segformer import segformer_b3_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b3_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b3_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b4_city":
        from models.segformer.segformer import segformer_b4_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b4_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b4_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b5_city":
        from models.segformer.segformer import segformer_b5_city
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b5_city(pretrained=pretrained, num_classes=1,
                                      uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b5_city(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b0":
        from models.segformer.segformer import segformer_b0
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b0(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b0(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b1":
        from models.segformer.segformer import segformer_b1
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b1(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b1(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b2":
        from models.segformer.segformer import segformer_b2
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b2(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b2(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b3":
        from models.segformer.segformer import segformer_b3
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b3(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b3(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b4":
        from models.segformer.segformer import segformer_b4
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b4(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b4(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    elif args.model == "segformer_b5":
        from models.segformer.segformer import segformer_b5
        if args.pretrained:
            pretrained = "your/pretrained/path"
            model = segformer_b5(pretrained=pretrained, num_classes=1,
                                 uncertainty_probability=args.uncertainty_probability)
        else:
            model = segformer_b5(num_classes=1, uncertainty_probability=args.uncertainty_probability)
            print("No use pretrained weights.")

    else:
        raise ValueError(f"Not support this model: {args.model}")

    return model
  
  
  class Ensemble(object):
    def __init__(self,model, weights, map_location="cpu"):
        self.model = model
        self.weights = weights if isinstance(weights, list) else [weights]
        self.map_location = map_location

    def __call__(self):

        use_ema = False
        state_dicts = []
        final_state_dict = {}
        for weight in self.weights:
            checkpoint = torch.load(weight, map_location=self.map_location)
            if checkpoint.get("model_ema"):
                model_ema = ModelEma(
                    self.model,
                    decay=0.99996)
                use_ema = True
            state_dict = checkpoint["model_ema" if checkpoint.get("model_ema") else "model"]
            state_dicts.append(state_dict)
        for key in state_dicts[0].keys():
            temp_value = []
            for i in range(len(state_dicts)):
                temp_value.append(state_dicts[i][key].float())
            final_state_dict[key] = torch.stack(temp_value).mean(0)
            # final_state_dict[key] = torch.stack(temp_value).max(0)[0]
        if use_ema:
            final_state_dict = {"state_dict_ema": final_state_dict}
            _load_checkpoint_for_ema(model_ema, final_state_dict)
            self.model = model_ema.ema
        else:
            self.model.load_state_dict(final_state_dict, strict=True)
        return self.model
