class DatasetConstants:
    def __init__(self, base_path, dataset_name):
        self.base_path = base_path
        self.dataset_name = dataset_name.lower()

        self.DATA_PATH = {
            "btad": f"{base_path}/btad",
            "mpdd": f"{base_path}/MPDD",
            "mvtec": f"{base_path}/mvtec",
            "visa": f"{base_path}/visa",
        }

        self.CLASS_NAMES = {
            "mvtec": [
                "bottle",
                "cable",
                "capsule",
                "carpet",
                "grid",
                "hazelnut",
                "leather",
                "metal_nut",
                "pill",
                "screw",
                "tile",
                "transistor",
                "toothbrush",
                "wood",
                "zipper",
            ],
            "visa": [
                "candle",
                "pcb3",
                "capsules",
                "pipe_fryum",
                "pcb4",
                "macaroni2",
                "pcb2",
                "chewinggum",
                "macaroni1",
                "cashew",
                "fryum",
                "pcb1",
            ],
            "mpdd": [
                "connector",
                "tubes",
                "metal_plate",
                "bracket_white",
                "bracket_brown",
                "bracket_black",
            ],
            "btad": ["01", "02", "03"],
        }

        self.PROMPTS = {
            "prompt_normal": ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', 
                              '{} without flaw', '{} without defect', '{} without damage'],
            "prompt_abnormal": ['damaged {}', 'broken {}', '{} with flaw', 
                                '{} with defect', '{} with damage'],
            "prompt_templates": ['a bad photo of a {}.', 
                                 'a low resolution photo of the {}.', 
                                 'a bad photo of the {}.', 
                                 'a cropped photo of the {}.', 
                                 'a bright photo of a {}.', 
                                 'a dark photo of the {}.', 
                                 'a photo of my {}.', 
                                 'a photo of the cool {}.', 
                                 'a close-up photo of a {}.', 
                                 'a black and white photo of the {}.', 
                                 'a bright photo of the {}.', 
                                 'a cropped photo of a {}.', 
                                 'a jpeg corrupted photo of a {}.', 
                                 'a blurry photo of the {}.', 
                                 'a photo of the {}.', 
                                 'a good photo of the {}.', 
                                 'a photo of one {}.', 
                                 'a close-up photo of the {}.', 
                                 'a photo of a {}.', 
                                 'a low resolution photo of a {}.', 
                                 'a photo of a large {}.', 
                                 'a blurry photo of a {}.', 
                                 'a jpeg corrupted photo of the {}.', 
                                 'a good photo of a {}.', 
                                 'a photo of the small {}.', 
                                 'a photo of the large {}.', 
                                 'a black and white photo of a {}.', 
                                 'a dark photo of a {}.', 
                                 'a photo of a cool {}.',
                                'a photo of a small {}.', 
                                'there is a {} in the scene.', 
                                'there is the {} in the scene.', 
                                'this is a {} in the scene.', 
                                'this is the {} in the scene.', 
                                'this is one {} in the scene.']
        }

    def get_data_path(self):
        return self.DATA_PATH[self.dataset_name]

    def get_class_names(self):
        return self.CLASS_NAMES[self.dataset_name]

    def get_domain(self):
        return self.DOMAINS[self.dataset_name]
    
    def get_prompts(self):
        return self.PROMPTS


if __name__ == "__main__":
    base_path = "/path/to/datasets"
    dataset_name = "MVTec"   
    constants = DatasetConstants(base_path, dataset_name)

    print(constants.get_data_path())
    print(constants.get_class_names())
    print(constants.get_domain())
    print(constants.get_prompts())
