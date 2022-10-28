import os
import sys
import torch
import torch.utils.data as data
from parsing_generation_segm_attr_dataset import ParsingGenerationDeepFashionAttrSegmDataset
#from pose_attr_dataset import DeepFashionAttrPoseDataset
from pose_encoder import ShapeAttrEmbedding, PoseEncoder


def main():
    train_dataset = ParsingGenerationDeepFashionAttrSegmDataset(
        segm_dir = sys.argv[0],
        pose_dir = sys.argv[1],
        ann_file = sys.argv[2])

    train_loader = data. DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    my_item_dict = train_dataset.__getitem__(6969)

    my_pose_encoder = PoseEncoder(size=512)

    my_attr_embedder = ShapeAttrEmbedding(dim=8, out_dim=128, cls_num_list=[2, 4, 6, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2, 2, 2])

    my_attr_embedding = my_attr_embedder(my_item_dict["attr"])

    x = my_pose_encoder(input = my_item_dict["densepose"], attr_embedding=my_attr_embedding)

    print(x)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
