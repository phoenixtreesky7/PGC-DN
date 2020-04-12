### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import time
import torch
from torch.autograd import Variable
opt = TestOptions().parse(save=False)
opt.nThreads = 1   # mytest code only supports nThreads = 1
opt.batchSize = 1  # mytest code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

model = create_model(opt)
model = model.cuda()
#model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
save_fake = 1
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# mytest
t0 = time.time()
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    
    generated = model.inference(data['label'].cuda(), data['image'].cuda())
    #losses, generated, dehazed, coarse = model(Variable(data['label'].cuda()), Variable(data['inst'].cuda()), 
    #        Variable(data['image']), Variable(data['feat']), infer=save_fake)
    visuals = OrderedDict([
                           #('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           #('fake', util.tensor2im(generated[2].data[0])),
                           ('fake', util.tensor2im(generated[1].data[0]))
                           ])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)
t1 = time.time()
print(str((t1-t0)/500))
webpage.save()
