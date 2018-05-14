clear
close all
clc
load dataset.mat


root_path_images = 'D:\datasets\BoxCars116k\BoxCars116k\images\';

id_sample = 1;
num_instance = numel(boxcars_dataset.samples{id_sample}.instances)
for i_instance = 1:num_instance
    file_funame = [root_path_images boxcars_dataset.samples{id_sample}.instances{i_instance}.path];
    im =imread(file_funame);
    figure
    imshow(im)
    hold on
    instance = boxcars_dataset.samples{id_sample}.instances{i_instance}
    bb_2d = instance.x0x32_DBB;
    bb_3d = instance.x0x33_DBB;%See Fig. 2 of ITS 2018 for ordering of those points
    offset = instance.x0x33_DBB_offset;
    %% draw 2d bounding box
%     rectangle('Position', bb_2d,...
%         'EdgeColor','r', 'LineWidth', 3)
    %% draw 3d bounding box
   bb3d_cropped = bb_3d - repmat(offset,[8,1]);% A,B,C,...,H
    for i = 1:8
        text(bb3d_cropped(i,1), bb3d_cropped(i,2),num2str(i));
%         text(bb3d_cropped(i,1), bb3d_cropped(i,2),'o','color','r');
    end
    %red
     line([bb3d_cropped(1,1),bb3d_cropped(2,1)],[bb3d_cropped(1,2),bb3d_cropped(2,2)],'Color','red','LineWidth',2);
     line([bb3d_cropped(3,1),bb3d_cropped(4,1)],[bb3d_cropped(3,2),bb3d_cropped(4,2)],'Color','red','LineWidth',2);
     line([bb3d_cropped(5,1),bb3d_cropped(6,1)],[bb3d_cropped(5,2),bb3d_cropped(6,2)],'Color','red','LineWidth',2);
     line([bb3d_cropped(7,1),bb3d_cropped(8,1)],[bb3d_cropped(7,2),bb3d_cropped(8,2)],'Color','red','LineWidth',2);
    %blue
     line([bb3d_cropped(1,1),bb3d_cropped(4,1)],[bb3d_cropped(1,2),bb3d_cropped(4,2)],'Color',[0, 1, 1],'LineWidth',2);
     line([bb3d_cropped(2,1),bb3d_cropped(3,1)],[bb3d_cropped(2,2),bb3d_cropped(3,2)],'Color',[0, 1, 1],'LineWidth',2);
     line([bb3d_cropped(5,1),bb3d_cropped(8,1)],[bb3d_cropped(5,2),bb3d_cropped(8,2)],'Color',[0, 1, 1],'LineWidth',2);
     line([bb3d_cropped(6,1),bb3d_cropped(7,1)],[bb3d_cropped(6,2),bb3d_cropped(7,2)],'Color',[0, 1, 1],'LineWidth',2);
    %yellow
     line([bb3d_cropped(1,1),bb3d_cropped(5,1)],[bb3d_cropped(1,2),bb3d_cropped(5,2)],'Color',[1, 0.8431, 0],'LineWidth',2);
     line([bb3d_cropped(2,1),bb3d_cropped(6,1)],[bb3d_cropped(2,2),bb3d_cropped(6,2)],'Color',[1, 0.8431, 0],'LineWidth',2);
     line([bb3d_cropped(3,1),bb3d_cropped(7,1)],[bb3d_cropped(3,2),bb3d_cropped(7,2)],'Color',[1, 0.8431, 0],'LineWidth',2);
     line([bb3d_cropped(4,1),bb3d_cropped(8,1)],[bb3d_cropped(4,2),bb3d_cropped(8,2)],'Color',[1, 0.8431, 0],'LineWidth',2);
     %% to obtain the center of 3d bounding box
     [cx,cy] = polyxpoly([bb3d_cropped(1,1),bb3d_cropped(7,1)]', [bb3d_cropped(1,2), bb3d_cropped(7,2)]',...
         [bb3d_cropped(4,1),bb3d_cropped(6,1)]',[bb3d_cropped(4,2), bb3d_cropped(6,2)]')
     text(cx, cy,'o','color','r');
     [cx2,cy2] = polyxpoly([bb3d_cropped(3,1),bb3d_cropped(5,1)]', [bb3d_cropped(3,2), bb3d_cropped(5,2)]',...
     [bb3d_cropped(2,1),bb3d_cropped(8,1)]',[bb3d_cropped(2,2), bb3d_cropped(8,2)]')
     text(cx2, cy2,'+','color','b');
end
