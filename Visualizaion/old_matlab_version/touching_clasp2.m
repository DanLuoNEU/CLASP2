%% VISUAL_CLASP
clear all;

load('./cam09exp2_bin_person.mat');% bins/person
load('./cam09exp2_joints.mat');% joint
impath = '/home/ubuntu/Demo/CLASP-Project/Person-bin Detection/faster_rcnn_pytorch/examples/cam09exp2/';
save_path = '/home/eubuntu/Demo/CLASP-Project/Person-bin Detection/faster_rcnn_pytorch/examples/cam09exp2_clasp1/';
% impath = '/Users/yuexizhang/Documents/CLASP/alert-data/04_11_2018/exp5aC9';
imlist = dir(fullfile(impath,'*.jpg'));
[~,I1] = sort(bin(:,5));
bin = bin(I1,:);
Id_bin = zeros(length(bin),1);

[~,I2] = sort(person(:,5));
person = person(I2,:);
Id_person = zeros(length(person),1);

%%

for i = 1 : length(person)
    %id = num2str(bin(i,5),'%04d');
    id = person(i,5);
    
    %Bin(i).bbx = [bin(i,1:2) bin(i,3)-bin(i,1) bin(i,4)-bin(i,2)];
     Person(i).bbx = [person(i,1:2) person(i,4)-person(i,2) person(i,3)-person(i,1)];
     Person(i).imId = id;
     
    Id_person(i,1) = person(i,5);
end

for i = 1 : length(bin)
    %id = num2str(bin(i,5),'%04d');
    id = bin(i,5);
    
    %Bin(i).bbx = [bin(i,1:2) bin(i,3)-bin(i,1) bin(i,4)-bin(i,2)];
    Bin(i).bbx = [bin(i,1:2) bin(i,4)-bin(i,2) bin(i,3)-bin(i,1)];
    Bin(i).imId = id;
    Bin(i).act = 0;
    Bin(i).change = 0;
    Id_bin(i,1) = bin(i,5);
end

%%
clear Result
for i = 1 : length(people)
    name1 = strsplit(image_name(i,:),'/');
    name2 = strsplit(name1{1,end},'.jpg');
    Result(i).imId = str2num(name2{1,1});
    % transform people joint information to candi form
    % build a field contains all left/right hand information in form (x,y,confidence, 3 for right hand/18 for left hand)
    candi = [];
    candi_index = 1;
    dim_persons = size(people{1,i});
    for j = 1 : dim_persons(1)
        candi(candi_index, 1:3) = people{1,i}(j,1,:);
        candi(candi_index, 4) = 3;
        candi_index = candi_index + 1;
        candi(candi_index, 1:3) = people{1,i}(j,2,:);
        candi(candi_index, 4) = 18;
        candi_index = candi_index + 1;
    end
    Result(i).candi = candi;
    Result(i).imPath =[imlist(i).folder,'/',imlist(i).name];
  
end

%%
for fr = length(people)
    Img = imread(Result(fr).imPath);
    ind = find(Result(fr).imId == Id_bin);
%     figure(1)
    %imshow(Img)
%     for idx = 1 : length(ind)
%         bx = Bin(ind(idx)).bbx;
%         rectangle('Position',[bx(:,1:2), bx(1,4) bx(1,3)],'EdgeColor','b','LineWidth',3);
%         
% %         hold on
% %         pause(0.05);
%         %pause;
%         
%     end
%     imwrite(Img, fr+'.jpg', 'jpg')
end

%% TOUCHING

disp('Action Detection')
close all

for ind = 1 : length(people) %1 : 1150
    ind
    candidates = Result(ind).candi;
    Img = imread(Result(ind).imPath);
    Indx = find(Result(ind).imId == Id_bin);
    
    
    %bxx1 = [bx(1,1)+bx(1,3),bx(1,2);bx(1,1)+bx(1,3), bx(1,2)+bx(1,4)];%[up,low]
    %bxx2 = [bx(1,1), bx(1,2);bx(1,1), bx(1,2)+bx(1,4)];
    idx = candidates(:,4);
%%
    if ~isempty(Indx)
        for numBx = 1 : length(Indx)
        
            bx = Bin(Indx(numBx)).bbx; % box 
            Img = insertShape(Img,'Rectangle',[bx(:,1:2), bx(1,4) bx(1,3)],'Color','blue','LineWidth',3);
            center_bx = [bx(1,1)+0.5*bx(1,4), bx(1,2)+0.5*bx(1,3)];
            w_bx = 170;
            h_bx = 190;
            center_bx_region = [center_bx(1,1)-0.5*w_bx, center_bx(1,2)-0.5*h_bx, w_bx,h_bx];
        %Img = insertShape(Img,'Rectangle',center_bx_region,'Color','green','LineWidth',2);
            rightHand = candidates(find(idx==3),1:2);
            leftHand = candidates(find(idx==18),1:2);
            if ~isempty(rightHand)% make sure there are hand detection
                for jt = 1 : size(rightHand,1)
  
                    center = rightHand(jt,:);
                    w = 80;
                    h = 90;
                    handAeara = [center(1,1)-0.5*w, center(1,2)-0.5*h, w,h];
                    %center_bx_region = [center_bx(1,1)-0.5*80, center_bx(1,2)-0.5*80, 80,80];
                    overlapRatio = bboxOverlapRatio(bx,handAeara);
                    overlapRatio_act  = bboxOverlapRatio(center_bx_region,handAeara);
                    if overlapRatio > 0.01
                        %imshow(Img)
                        %Img1 = insertShape(Img,'Rectangle',bx,'Color','blue','LineWidth',3);
                        Bin(Indx(numBx)).act = 1;
                        Img = insertShape(Img,'FilledRectangle',handAeara,'Color','yellow');
                        if overlapRatio_act > 0.05
                            Img = insertShape(Img,'FilledRectangle',[bx(:,1:2),450,400],'Color','red','LineWidth',3);
                        end
                        %imshow(OverlapImg);
                    end
                
                end
         
            end
        
            if ~isempty(leftHand)
                for jt = 1 : size(leftHand,1)
  
                    center = leftHand(jt,:);
                    w = 60;
                    h = 60;
                    handAeara = [center(1,1)-0.5*w, center(1,2)-0.5*h, w,h];
                    overlapRatio = bboxOverlapRatio(bx,handAeara);
                    overlapRatio_act  = bboxOverlapRatio(center_bx_region,handAeara);
                    if overlapRatio > 0.01
                        Bin(Indx(numBx)).act = 1;
                        %imshow(Img)
                        %Img1 = insertShape(Img,'Rectangle',bx,'Color','blue','LineWidth',3);
                        Img = insertShape(Img,'FilledRectangle',handAeara,'Color','yellow');
                        if overlapRatio_act > 0.05
                            Img = insertShape(Img,'FilledRectangle',[bx(:,1:2),450,400],'Color','red','LineWidth',3);
                        end
                        %imshow(OverlapImg);
                    end
                
                end
        
        
        
            end
%      imshow(Img)
     imwrite(Img, fullfile(save_path,[num2str(ind),'.jpg']))
     %rectangle('Position',bx,'EdgeColor','b','LineWidth',3);
     %hold on
     end
    else
%         imshow(Img);
        imwrite(Img, fullfile(save_path,[num2str(ind),'.jpg']))
    end
    
        
    pause(0.01);
end