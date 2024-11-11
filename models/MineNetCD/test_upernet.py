import torch
import transformers
import numpy as np
import evaluate
from torch.utils import data
from torch import nn
import os
from PIL import Image

from datasets import load_dataset, load_from_disk
import torchvision.transforms as tfs

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

from minenetcd import UperNetForSemanticSegmentation

miou_list=[]
f1_list=[]


ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class ChangeDetectionDataset(data.Dataset):
    def __init__(self,dataset,transform=None) -> None:
        super().__init__()
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return(len(self.dataset))
    def __getitem__(self, index):
        imageA=self.transform(self.dataset[index]["imageA"])
        imageB=self.transform(self.dataset[index]["imageB"])
        label=tfs.ToTensor()(self.dataset[index]["label"])
        label=torch.cat([label],dim=0)
        return imageA,imageB,label,index


def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def main(model_name,dataset_name,model_id,num_classes):
    print(f'testing {model_id}')
    print(transformers.__file__)

    logger = get_logger(__name__)
    accelerator=Accelerator()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=accelerator.device
    batch_size=10
    # preprocessor=Mask2FormerImageProcessor(ignore_index=-1,reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
    # dataset=load_dataset("ericyu/LEVIRCD_Cropped_256")
    # dataset=load_dataset("ericyu/LEVIRCD_Cropped_256")

    dataset=load_dataset(dataset_name)
    # dataset=load_from_disk("/home/yu34/CD/datasets/MineNetCD_Cropped_256")
    logger.info(dataset,main_process_only=True)
    # train_ds=dataset["train"]
    test_ds=dataset["test"]
    # val_ds=dataset["val"]
    transform=tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=ADE_MEAN,std=ADE_STD),
    ])

    # train_dataset=ChangeDetectionDataset(train_ds,transform=transform)
    # val_dataset=ChangeDetectionDataset(val_ds,transform=transform)
    test_dataset=ChangeDetectionDataset(test_ds,transform=transform)

    # train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # test_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id,ignore_mismatched_sizes=True)
    model = UperNetForSemanticSegmentation.from_pretrained(model_id, ignore_mismatched_sizes=True)
    model = model.to(device)

    model, test_dataloader=accelerator.prepare(model,test_dataloader)
    model.eval()
    # metric=evaluate.combine(["accuracy", "f1", "precision", "recall"])
    # mean_iou=evaluate.load("mean_iou")
    
    TP,TN,FP,FN=0,0,0,0 
    os.makedirs(f"./test_predictions/{model_name}/", exist_ok=True)
    for i, batch in enumerate(tqdm(test_dataloader,disable=not accelerator.is_local_main_process, miniters=20)):
        with torch.no_grad():

            imageA,imageB, labels, index=batch

            outputs = model(
                    x1=imageA, x2=imageB
                )
            predicted_segmentation_maps=torch.nn.Softmax(dim=1)(outputs.logits)
            predicted_segmentation_maps=torch.argmax(predicted_segmentation_maps,dim=1)
            tp,fp,tn,fn=confusion(predicted_segmentation_maps,labels.squeeze())
            TP+=tp
            TN+=tn
            FP+=fp
            FN+=fn
            print(TP,TN,FP,FN)
            # print(predicted_segmentation_maps.shape)

            # print(predicted_segmentation_maps.shape)

            # original_images = batch["original_images"]
            # target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
            # predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            
            img_idx=index

            for i in range(len(predicted_segmentation_maps)):
                segmentation_map=predicted_segmentation_maps[i]
                # print(segmentation_map.shape)
                # probs = F.softmax(segmentation_map,dim=1)
                # prediction=probs.max(dim=1)[1]

                # segmentation_map=Image.fromarray(segmentation_map.cpu().numpy())
                # segmentation_map=tfs.ToPILImage()(segmentation_map)

                # print(segmentation_map.flatten().shape,batch["original_segmentation_maps"][i].type(torch.int32).flatten().shape)
                # references=accelerator.gather(batch["original_segmentation_maps"][i])
                # predictions=accelerator.gather(segmentation_map)

                # predictions, references=accelerator.gather_for_metrics((segmentation_map.flatten(),batch["original_segmentation_maps"][i].type(torch.int32).flatten()))
                # metric.add_batch(references=references.type(torch.int32).flatten(),predictions=predictions.flatten())

                predictions=segmentation_map.squeeze().unsqueeze(0)
                # references=labels.squeeze()[i].unsqueeze(0)

                # print("before_gather", predictions.shape,references.shape)

                # predictions, references = accelerator.gather_for_metrics((predictions, references))

                # print("after_gather", predictions.shape,references.shape)
                # metric.add_batch(references=references.type(torch.int32).flatten(),predictions=predictions.flatten())
                # mean_iou.add_batch(references=references.type(torch.int32),predictions=predictions)

                segmentation_map = Image.fromarray((255*segmentation_map).cpu().numpy().astype(np.uint8))
                
                segmentation_map.save(os.path.join(f"./test_predictions/{model_name}/"+str(int(img_idx[i]))+".png"))

    # eval_metric=metric.compute()
    # miou=mean_iou.compute(num_labels=num_classes,ignore_index=255)
    OA=(TP+TN)/(TP+TN+FP+FN)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    f1=2*TP/(2*TP+FP+FN)
    f1_each_device=f1
    cIoU=TP/(TP+FP+FN)
    # miou_each_device=miou["mean_iou"]

    ts_metrics_list=torch.FloatTensor([OA,f1,precision,recall, cIoU]).cuda().unsqueeze(0)
    
    # print(eval_metric)
    # print(miou)

    # ts_eval_metric=convert_dict_to_tensor_dict(eval_metric)
    # ts_miou=convert_dict_to_tensor_dict(miou)
    # ts_eval_metric.update({"mean_iou": miou_each_device})

    # print("gathering")

    ts_eval_metric_gathered=accelerator.gather(ts_metrics_list)


    # print(ts_eval_metric_gathered)
    # print(torch.mean(ts_eval_metric_gathered, dim=0))

    final_metric=torch.mean(ts_eval_metric_gathered, dim=0)
    accelerator.print(f'Accuracy={final_metric[0]}, mF1={final_metric[1]}, Precision={final_metric[2]}, Recall={final_metric[3]}, cIoU={final_metric[4]}')
    # ts_miou_gathered=accelerator.gather(ts_miou)

    # print(ts_eval_metric_gathered)
    # print(ts_miou_gathered)
    # f1_gathered=accelerator.gather(f1_each_device)
    # print(f"f1_gathered:{f1_gathered}")

    # print(eval_metric)
    # print(miou)

    # miou_list.append(miou["mean_iou"])
    # f1_list.append(eval_metric["f1"])
    # print(miou_list,f1_list)
    # accelerator.print(eval_metric)
    # accelerator.print(miou)

    # print(metric.compute())


    # model=model.to(device)
    # batch = next(iter(train_dataloader))
    # # print(batch["pixel_values"])
    # outputs=model(batch["pixel_values"].to(device),
    #             class_labels=[labels.to(device) for labels in batch["class_labels"][0:batch_size]],
    #             mask_labels=[labels.to(device) for labels in batch["mask_labels"][0:batch_size]])
    # print(outputs.loss)
    

    # metric=evaluate.combine(["accuracy", "f1", "precision", "recall"])

    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-7, last_epoch=-1, verbose=False)
    # running_loss = 0.0
    # num_samples = 0

    # model.to(device)

    # trainingargs=TrainingArguments(output_dir="./exp/",overwrite_output_dir=True,do_train=True, do_eval=True,learning_rate=1e-4,num_train_epochs=50,fp16=True)
    # trainer=CDTrainer(model=model,args=trainingargs,data_collator=collate_fn,train_dataset=train_dataset,eval_dataset=val_dataset,compute_metrics=metric)

    # model, optimizer, train_dataloader, scheduler=accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    # trainer=accelerator.prepare(trainer)
    # trainer.train()

    

    # for epoch in range(100):
    #     logger.info(f'Epoch:{epoch}',main_process_only=True)
    #     model.train()
    #     for idx, batch in enumerate(tqdm(train_dataloader)):
    #         # Reset the parameter gradients
    #         optimizer.zero_grad()

    #         # Forward pass
            
    #         pixel_values=batch["pixel_values"]
    #         mask_labels=[labels for labels in batch["mask_labels"]][0:batch_size]
    #         class_labels=[labels for labels in batch["class_labels"]][0:batch_size]
    #         outputs = model(
    #                 pixel_values=pixel_values,
    #                 mask_labels=mask_labels,
    #                 class_labels=class_labels,
    #             )
    #         # Backward propagation
    #         loss = outputs.loss
    #         accelerator.backward(loss)

    #         batch_size = batch["pixel_values"].size(0)
    #         running_loss += loss.item()
    #         num_samples += batch_size

    #         if idx % 100 == 0:
    #             print("Loss:", running_loss/num_samples)

    #         # Optimization
    #         optimizer.step()
    #         scheduler.step()
        # if epoch%5==0:
            # model.eval()
        #     for idx, batch in enumerate(tqdm(test_dataloader)):
        #         if idx > 5:
        #             break
        #         pixel_values = batch["pixel_values"]
        
        #         # Forward pass
        #         with torch.no_grad():
        #             outputs = model(pixel_values=pixel_values.to(device))

        #         # get original images
        #         original_images = batch["original_images"]
        #         target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
        #         # print(f'trg_sizes={target_sizes}')
        #         # predict segmentation maps
        #         predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
        #                                                                                     target_sizes=target_sizes)

        #         # get ground truth segmentation maps
        #         ground_truth_segmentation_maps = [labels.type(torch.int32) for labels in batch["original_segmentation_maps"]]
        #         predicted_segmentation_maps=[labels.type(torch.int32) for labels in predicted_segmentation_maps]

        #         # print(torch.cat(ground_truth_segmentation_maps).shape,torch.stack(predicted_segmentation_maps).shape)
        #         metric.add_batch(references=torch.cat(ground_truth_segmentation_maps).flatten(), predictions=torch.stack(predicted_segmentation_maps).flatten())
        
        #     print(f'F1-Score:{metric.compute()["f1"]},accuracy:{metric.compute()["accuracy"]},precision:{metric.compute()["precision"]},recall:{metric.compute()["recall"]}')
        # save_pretrained_path=f"./exp/{epoch}"
        # os.makedirs(save_pretrained_path,exist_ok=True)
        # model.module.save_pretrained(save_pretrained_path)

def convert_dict_to_tensor_dict(ori_dict):
    tensor_dict={}
    for key, values in ori_dict.items():
        if isinstance(values,list):
            tensor_dict.update({key,torch.FloatTensor(values)})
        else:
            tensor_dict.update({key,torch.FloatTensor([values])})
    return tensor_dict


if __name__=="__main__":
    # main(model_name="gvlm_concat",dataset_name="ericyu/CLCD_Cropped_256",model_id="exp/clcd_concat/95",num_classes=2)
    # main(model_name="sysu_concat",dataset_name="ericyu/SYSU_CD",model_id="exp/sysu_concat/95",num_classes=2)
    main(model_name=f"minenetcd_upernet_ResNet_Diff_18_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"ericyu/minenetcd_upernet_ResNet_Diff_18_Pretrained",num_classes=2)
    

    # main(model_name="exp/minenetcd_upernet_VSSM_B_ST_ChannelMixing/95",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_B_ST_ChannelMixing/95",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_S_ST_ChannelMixing/95",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_S_ST_ChannelMixing/95",num_classes=2)
    # for i in range(0,102,3):
    #     main(model_name=f"exp/minenetcd_upernet_VSSM_T_ST_Diff_Pretrained_ChannelMixing_Dropout_Run2/{i}",dataset_name="ericyu/MineNetCD256",model_id=f"exp/minenetcd_upernet_VSSM_T_ST_Diff_Pretrained_ChannelMixing_Dropout_Run2/{i}",num_classes=2)
    
    # main(model_name=f"minenetcd_upernet_ResNet_Diff_18_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_ResNet_Diff_18_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_ResNet_Diff_18_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_ResNet_Diff_18_Pretrained_ChannelMixing_Dropout",num_classes=2)
    # main(model_name=f"minenetcd_upernet_ResNet_Diff_50_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_ResNet_Diff_50_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_ResNet_Diff_50_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_ResNet_Diff_50_Pretrained_ChannelMixing_Dropout",num_classes=2)
    # main(model_name=f"minenetcd_upernet_ResNet_Diff_101_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_ResNet_Diff_101_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_ResNet_Diff_101_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_ResNet_Diff_101_Pretrained_ChannelMixing_Dropout",num_classes=2)

    # main(model_name=f"minenetcd_upernet_Swin_Diff_B_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_Swin_Diff_B_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_Swin_Diff_B_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_Swin_Diff_B_Pretrained_ChannelMixing_Dropout",num_classes=2)
    # main(model_name=f"minenetcd_upernet_Swin_Diff_S_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_Swin_Diff_S_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_Swin_Diff_S_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_Swin_Diff_S_Pretrained_ChannelMixing_Dropout",num_classes=2)
    # main(model_name=f"minenetcd_upernet_Swin_Diff_T_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_Swin_Diff_T_Pretrained",num_classes=2)
    # # main(model_name=f"minenetcd_upernet_Swin_Diff_T_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_Swin_Diff_T_Pretrained_ChannelMixing_Dropout",num_classes=2)

    # main(model_name=f"minenetcd_upernet_VSSM_B_ST_Diff_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_VSSM_B_ST_Diff_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_VSSM_B_ST_Diff_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_VSSM_B_ST_Diff_Pretrained_ChannelMixing_Dropout",num_classes=2)
    # main(model_name=f"minenetcd_upernet_VSSM_S_ST_Diff_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_VSSM_S_ST_Diff_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_VSSM_S_ST_Diff_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_VSSM_S_ST_Diff_Pretrained_ChannelMixing_Dropout",num_classes=2)
    # main(model_name=f"minenetcd_upernet_VSSM_T_ST_Diff_Pretrained",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_VSSM_T_ST_Diff_Pretrained",num_classes=2)
    # main(model_name=f"minenetcd_upernet_VSSM_T_ST_Diff_Pretrained_ChannelMixing_Dropout",dataset_name="ericyu/MCD_Test",model_id=f"archived/minenetcd_upernet_VSSM_T_ST_Diff_Pretrained_ChannelMixing_Dropout",num_classes=2)




    # main(model_name=f"exp/minenetcd_upernet_VSSM_S_ST_Pretrained_ChannelMixing/5",dataset_name="ericyu/MCD_Test",model_id=f"exp/minenetcd_upernet_VSSM_S_ST_Pretrained_ChannelMixing/5",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/95",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/95",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/90",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/90",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/85",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/85",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/80",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/80",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/75",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/75",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/70",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/70",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/65",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/65",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/60",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/60",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/55",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/55",num_classes=2)
    # main(model_name="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/50",dataset_name="ericyu/MineNetCD256",model_id="exp/minenetcd_upernet_VSSM_T_ST_Pretrained_ChannelMixing/50",num_classes=2)


    
    # main(model_name="clcd",dataset_name="ericyu/LEVIRCD_Cropped256",model_id="exp/levircd_75_ct/90",num_classes=2)
    # main(model_name="clcd",dataset_name="ericyu/LEVIRCD_Cropped256",model_id="exp/levircd_75_ct/85",num_classes=2)
    # main(model_name="clcd",dataset_name="ericyu/LEVIRCD_Cropped256",model_id="exp/levircd_75_ct/80",num_classes=2)
    # main(model_name="clcd",dataset_name="ericyu/LEVIRCD_Cropped256",model_id="exp/levircd_75_ct/75",num_classes=2)
    # main(model_name="clcd",dataset_name="ericyu/LEVIRCD_Cropped256",model_id="exp/levircd_75_ct/70",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/45",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/40",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/35",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/30",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/25",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/20",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/15",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/10",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/5",num_classes=2)
    # main(model_name="MNCD_concat_256",dataset_name="",model_id="exp/MNCD_concat_256/50",num_classes=2)

    # main(model_id="exp/clcd_concat/95",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/90",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/85",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/80",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/75",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/70",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/65",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/60",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/55",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/50",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)
    # main(model_id="exp/clcd_concat/45",dataset_name="ericyu/CLCD_Cropped_256",model_name="clcd_concat",num_classes=2)

