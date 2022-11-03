# encoding:utf-8
import visdom, time, numpy as np, torch, random, cv2, json
from PIL import Image

from torchvision import transforms
from copy import deepcopy

goboard_backgrand_image = cv2.imread('utils/goboard_empty.png')

def show_board(go_array, pred_point, gt_point):
    if isinstance(go_array, torch.Tensor):
        go_array = go_array.cpu().numpy()
        pred_point = pred_point.cpu().numpy()
        gt_point = gt_point.cpu().numpy()
        
    def int2point(num):
        x = num % 19
        y = num // 19
        return y, x
    goboard = deepcopy(goboard_backgrand_image)
    # bsize = goboard.shape[0]
    # print(goboard.shape)
    radius = 25
    point_radius = 62
    shift = 35
    last_point_tickness = 5
    color_list = [(0, 0, 0), (255, 255, 225)]
    gt_color = (0, 255, 0)
    black_points = np.where(go_array[0]==1.)
    black_points = (black_points[0].tolist(), black_points[1].tolist())

    white_points = np.where(go_array[1]==1.)
    white_points = (white_points[0].tolist(), white_points[1].tolist())
    # print(black_points)
    # print(white_points)
    for x, y in zip(black_points[0], black_points[1]):
        # print(((point_radius+1 if x >= 3 else point_radius) * x + shift, (point_radius+1 if y >= 3 else point_radius) * y + shift), radius, color_list[0])
        cv2.circle(goboard, ((point_radius+1 if x >= 3 else point_radius) * x + shift, (point_radius+1 if y >= 3 else point_radius) * y + shift), radius=radius, color=color_list[0], thickness=-1)
    for x, y in zip(white_points[0], white_points[1]):
        cv2.circle(goboard, ((point_radius+1 if x >= 3 else point_radius) * x + shift, (point_radius+1 if y >= 3 else point_radius) * y + shift), radius=radius, color=color_list[1], thickness=-1)
    
    x, y = int2point(gt_point)
    # print(x, y)
    # print(((point_radius+1 if x >= 3 else point_radius) * x + shift, (point_radius+1 if y >= 3 else point_radius) * y+shift), radius+5, gt_color)
    cv2.circle(goboard, ((point_radius+1 if x >= 3 else point_radius) * x + shift, (point_radius+1 if y >= 3 else point_radius) * y+shift), radius=radius+5, color=gt_color, thickness=-1)
    x, y = int2point(pred_point)
    cv2.circle(goboard, ((point_radius+1 if x >= 3 else point_radius) * x + shift, (point_radius+1 if y >= 3 else point_radius) * y+shift), radius=radius, color=color_list[(len(black_points) + len(white_points) + 1) % 2], thickness=-1)
    cv2.circle(goboard, ((point_radius+1 if x >= 3 else point_radius) * x + shift, (point_radius+1 if y >= 3 else point_radius) * y+shift), radius=last_point_tickness, color=color_list[(len(black_points) + len(white_points)) % 2], thickness=last_point_tickness-2)
    
    # cv2.imshow('goboard', goboard)
    # if cv2.waitKey(1000000)&0xFF == ord('q'):
    #     return
    return goboard

class Visual(object):

   def __init__(self, env='default', port=8097, log_to_file=None, **kwargs):
       self.vis = visdom.Visdom(env=env, port=port, log_to_filename=log_to_file, **kwargs)
    #    if log_to_file:
    #        self.create_log_at(log_to_file, env)
    #    else:
       self.index = {} 
       self.log_text = ''

   def reinit(self, env='default', **kwargs):
       self.vis = visdom.Visdom(env=env, **kwargs)
       return self
    
   def create_log_at(self, file_path, current_env, new_env=None):
      new_env = current_env if new_env is None else new_env
      vis = visdom.Visdom(env=current_env)

      data = json.loads(vis.get_window_data())
      self.index = data
      if len(data) == 0:
         print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
         return

    

   def plot_many(self, d):
       
       for k, v in d.iteritems():
           self.plot(k, v)
    
   def multi_cls_bar(self, name, x, legend, rowname):
       
       self.vis.bar(
            win=name,
            X=x,
            opts=dict(
                stacked=True,
                legend=legend,
                rownames=rowname
            )
        )

   def img_many(self, d):
       for k, v in d.iteritems():
           self.img(k, v)

   def plot(self, name, y, **kwargs):
       
       x = self.index.get(name, 0)
    #    print(x)
    #    print(x.keys())
       self.vis.line(Y=np.array([y]), X=np.array([x]),
                     win=name,
                     opts=dict(title=name),
                     update='append' if x > 0 else None,
                     **kwargs)
       self.index[name] = x + 1

   def img(self, name, img_, **kwargs):
       def cv2PIL(img):
           return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

       if isinstance(img_, np.ndarray):
           img_ = cv2PIL(img_)
           img_ = transforms.ToTensor()(img_)
       self.vis.images(img_,
                      win=name,
                      opts=dict(title=name),
                      **kwargs)

   def log(self, info, win='log_text'):
       
       self.log_text += ('[{time}] {info} <br>'.format(
                           time=time.strftime('%m%d_%H%M%S'),
                           info=info))
       self.vis.text(self.log_text, win)

   def __getattr__(self, name):
       
       return getattr(self.vis, name)

if __name__ == '__main__':
    go_points = np.zeros((3, 19, 19), np.float32)
    go_points[0, 1, 2] = 1.0
    go_points[1, 2, 3] = 1.0
    go_points[0, 18, 12] = 1.0
    go_points[1, 12, 18] = 1.0
    pred_point = 360
    gt_point = 7 + 19 * 8
    # pred_point = gt_point
    show_board(go_points, pred_point, gt_point)