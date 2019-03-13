require 'torch'
require 'nn'
require 'nnx'
require 'image'
require 'ffmpeg'
require 'qttorch'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'


local cmd = torch.CmdLine()

-- Model options
cmd:option('-models', 'models/instance_norm/candy.t7')
cmd:option('-height', 480)
cmd:option('-width', 640)

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)

-- Video options
cmd:option('-seconds', 6)
cmd:option('-fps', 30)
cmd:option('-video', './samples/test.mp4')


local function main()
  local opt = cmd:parse(arg)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local models = {}

  local preprocess_method = nil
  for _, checkpoint_path in ipairs(opt.models:split(',')) do
    print('loading model from ', checkpoint_path)
    local checkpoint = torch.load(checkpoint_path)
    local model = checkpoint.model
    model:evaluate()
    model:type(dtype)
    if use_cudnn then
      cudnn.convert(model, cudnn)
    end
    table.insert(models, model)
    local this_preprocess_method = checkpoint.opt.preprocessing or 'vgg'
    if not preprocess_method then
      print('got here')
      preprocess_method = this_preprocess_method
      print(preprocess_method)
    else
      if this_preprocess_method ~= preprocess_method then
        error('All models must use the same preprocessing')
      end
    end
  end

  local preprocess = preprocess[preprocess_method]

  video = ffmpeg.Video{path=opt.video, width=opt.width, height=opt.height, 
                        fps=opt.fps, length=opt.seconds, delete=false}

  local win = nil
  local frame_count = 0
  while true do
    -- Grab a frame from the video
    local img = video:forward()

    -- Preprocess the frame
    local H, W = img:size(2), img:size(3)
    img = img:view(1, 3, H, W)
    local img_pre = preprocess.preprocess(img):type(dtype)

    -- Run the models
    local imgs_out = {}
    for i, model in ipairs(models) do
      local img_out_pre = model:forward(img_pre)

      -- Deprocess the frame and show the image
      local img_out = preprocess.deprocess(img_out_pre)[1]:float()
      table.insert(imgs_out, img_out)
    end
    local img_disp = image.toDisplayTensor{
      input = imgs_out,
      min = 0,
      max = 1,
      nrow = math.floor(math.sqrt(#imgs_out)),
    }
  end
end


main()

