%% Demo demonstrating the algorithm result formats for COCO

%% select results type for demo
type = {'segmentation','bbox','caption'};
type = type{1}; % specify type here

%% set appropriate files for given type of results
dataDir='../';
switch type
  case 'segmentation'
    annFile = 'instances_val2014';
    resFile = 'instances_val2014_fakeseg_results';
  case 'bbox'
    annFile = 'instances_val2014';
    resFile = 'instances_val2014_fakebox_results';
  case 'caption'
    annFile = 'captions_val2014';
    resFile = 'captions_val2014_fakecap_results';
end
annFile=sprintf('%s/annotations/%s.json',dataDir,annFile);
resFile=sprintf('%s/results/%s.json',dataDir,resFile);

%% initialize COCO ground truth and results api's
fprintf('Running demo for %s results.\n',type);
coco = CocoApi(annFile);
cocoRes = coco.loadRes(resFile);

%% visialuze ground truth and results side by side
imgIds = [cocoRes.data.annotations.image_id]; nImgs = length(imgIds);
imgId = imgIds(randi(nImgs)); img = coco.loadImgs(imgId);
I = imread(sprintf('%s/images/val2014/%s',dataDir,img.file_name));
figure(1); subplot(1,2,1); imagesc(I); axis('image'); axis off;
annIds = coco.getAnnIds('imgIds',imgId); title('ground truth')
anns = coco.loadAnns(annIds); coco.showAnns(anns);
figure(1); subplot(1,2,2); imagesc(I); axis('image'); axis off;
annIds = cocoRes.getAnnIds('imgIds',imgId); title('results')
anns = cocoRes.loadAnns(annIds); cocoRes.showAnns(anns);

%% load raw JSON and show exact format for results
res = gason(fileread(resFile));
fprintf('results structure have the following format:\n'); disp(res)

%% the following command can be used to save the results back to disk
if(0), f=fopen(resFile,'w'); fwrite(f,gason(res)); fclose(f); end
