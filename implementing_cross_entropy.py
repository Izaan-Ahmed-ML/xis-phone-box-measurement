#need a function that implements cross entropy loss
import torch
import numpy as np
import torch.nn as nn

def cross_entropy_loss(input,output_GT,ignore):
    gt = output_GT.squeeze(0)  # [H, W]
    print(gt.shape)
    probs = torch.softmax(input, dim=0)
    # probs_2=probs[]  # [C, H, W]
    print(probs.shape)
    C = probs.shape[0]  # number of classes
    index = gt.clamp(min=0, max=C-1).unsqueeze(0)
    print(index.shape)
    print(index)
    gathered = torch.gather(probs, dim=0, index=index).squeeze(0)
    loss_per_pixel = -torch.log(gathered.clamp(min=1e-12))
    mask = (gt != ignore).float()
    result = (loss_per_pixel * mask).sum() / mask.sum().clamp(min=1)
    print("loss_per_pixel:", loss_per_pixel)
    print("mask:", mask)
    print("result:", result)
    return result
    # sum_output=0
    # ignore_pixel=output_GT==ignore
    # print("ignore_pixel:", ignore_pixel)
    # i=0
    # j=0
    # k=0
    # classes_=input.shape[0]
    # # total_unqiue_classes=torch.unique(output_GT)
    # # print("total_unqiue_classes:", total_unqiue_classes)
    # # d_array=torch.zeros(len(total_unqiue_classes))
    # # print("d_array:", d_array)
    # print("class_:", classes_)
    # c1=0
    # total_number_of_pixels=output_GT.shape[1]*output_GT.shape[2]
    # print("total_number_of_pixels:", total_number_of_pixels)
    # counted_pixels=0
    # height = output_GT.shape[1]
    # width = output_GT.shape[2]
    # for j in range(height):
    #     for k in range(width):

    #         if ignore_pixel[0][j][k] != True:

    #             counted_pixels += 1

    #             ground_truth_class = output_GT[0][j][k]

    #             print("Pixel:", j, k)
    #             print("Ground truth:", ground_truth_class)

    #             pixel_logits = input[:, j, k]
    #             print("pixel_logits:", pixel_logits)

    #             softmax_var = nn.Softmax(dim=0)

    #             probabilities = softmax_var(pixel_logits)
    #             print("probabilities:", probabilities)

    #             output = probabilities[ground_truth_class]

    #             log_output = -torch.log(output)

    #             sum_output += log_output

    #             print("selected probability:", output)
    #             print("pixel loss:", log_output)

    # # for pixel in range(total_number_of_pixels):

    # #     if(ignore_pixel[0][j][k]!=True):
    # #         counted_pixels+=1
    # #         ground_truth_class=output_GT[0][j][k]
    # #         print("ground_truth_class:", ground_truth_class)
    # #         print("executing pixel:", pixel)
                
    # #         #print("ignore_pixel[j][k]:", ignore_pixel[0][j][k])
    # #         pixel_logits = input[:, j, k]
    # #         print("pixel_logits:", pixel_logits)

    # #         softmax_var= nn.Softmax(dim=0)
    # #         probabilities = softmax_var(pixel_logits)
    # #         print("probabilities:", probabilities)
    # #         output = probabilities[ground_truth_class]
                            
    # #         log_output=-(torch.log(output))
    # #         sum_output=sum_output+log_output
    # #         print("softmax_output:", output)
    # #         print("log_output:", log_output)
    # #         print("sum_output:", sum_output)
        
    # #         c1=c1+1
    # #         if(c1<width):
    # #             k=k+1
                
    # #         else:
    # #             k=0
    # #             if(j<height):
    # #                 j=j+1
    # #                 c1=0
    # #             else:
    # #                 j=0
    # #                 k=0
    # #     else:
    # #         c1=c1+1
    # #         if(c1<width):
    # #             k=k+1
                
    # #         else:
    # #             k=0
    # #             if(j<height):
    # #                 j=j+1
    # #                 c1=0
    # #             else:
    # #                 j=0
    # #                 k=0

    # result=sum_output/counted_pixels
    # print("result:", result)
        
    # # print(input)
    # # print(output_GT)
    return result







input = torch.tensor([
    [
        [2., 1.],   # Class 0 logits
        [4., 1.]
    ],
    [
        [1., 3.],   # Class 1 logits
        [2., 2.]
    ]
])
output_GT= target = torch.tensor([
    [0, 1],
    [255, 1]
]).unsqueeze(0)
  # This is the pixel value to ignore, which is 255 in this case
cross_entropy_loss(input, output_GT,ignore=255)