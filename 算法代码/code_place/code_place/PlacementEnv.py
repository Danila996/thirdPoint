
import gymnasium
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



"""
    obs[0]:父模块的布局位置
    obs[1]:污染情况，包含布线
    action = pos
    reward:1、父操作的重叠，越多越好，2、与其他布局的重叠，越少越好，3、发生的清洗代价，越少越好，4、ASR的面积，越小越好
    如果用图卷积:
        节点:模块的信息：大小，在这个拓扑结构中的清洗代价，位置，芯片单元潜在清洗代价表，布局顺序
        边：流体流动的信息：起始模块和目标模块的编号，清洗代价
        图：当前的布局情况
    
    为了使用向量化环境，统一观察空间的大小，然后设置一个限制区域来模拟芯片
    两种方案：
    1、直接布局
    2、生成初始布局，对布局进行调整
"""
class PlacementEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,ModuleAttribute,OperationDependence,FluidContamination,chip_params,env_params,PlacementSequence,
                 max_timestep,render_mode,
                 ):
        super(PlacementEnv, self).__init__()
        #初始化一些固定的参数
        self.BiochipLength=chip_params['BiochipLength'] #长度，有几列
        self.BiochipWidth=chip_params['BiochipWidth']#宽度，有几行
        self.NumModule=chip_params['NumModule']#有多少个模块
        self.reward_overbound = env_params['reward_overbound']
        self.reward_fail = env_params['reward_fail']
        self.reward_finish = env_params['reward_finish']
        self.weight_ASPR = env_params['weight_ASPR']
        self.weight_surdist = env_params['weight_surdist']
        self.weight_WP = 1 - self.weight_ASPR - self.weight_surdist
        #初始化一些动态变化的参数
        self.finishPlacement  =0  # 是否完成布线子任务
        self.record_agent = [] #记录上个布局的位置，不应该连续选择同一个位置
        self.bufferWP = 0
        self.PlacementScheme = np.full(shape=(self.NumModule,2),fill_value=-1,dtype=int)#记录已布局的模块的位置
        #一些关于违反时间步长的参数
        self.timestep=0
        self.max_timestep=max_timestep
        #任务信息
        self.FluidContamination=FluidContamination#每个模块的清洗代价,规则:下标是模块编号，值为WP
        self.ModuleAttribute=ModuleAttribute#每个模块的大小
        self.OperationDependence =  OperationDependence
        self.PlacementSequence = PlacementSequence#布局顺序
        self.ptr = 0 #在布局顺序表中的指针
        #辅助可视化
        self.render_mode=render_mode
        self.washPrice=0
        self.washtime=0
        #初始化状态空间；当开启新一轮的时候，需要先把第一个布线任务更新到状态里
        self.obs = np.zeros(shape=(3,13,13),dtype=float)#状态空间的具体实现
        for row in range(self.BiochipWidth):#将芯片置为0.1
            for col in range(self.BiochipLength):
                self.obs[0][self.map_chip2obs(row)][self.map_chip2obs(col)] = 0.00001
                self.obs[1][self.map_chip2obs(row)][self.map_chip2obs(col)] = 0.00001
                self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)] = 0.00001

        for row in range(self.ModuleAttribute[self.PlacementSequence[0]][0]):
            for col in range(self.ModuleAttribute[self.PlacementSequence[0]][1]):
                self.obs[0][self.map_chip2obs(row)][self.map_chip2obs(col)] = self.FluidContamination[self.PlacementSequence[0]]
        self.record_ASPR = 0
        self.record_surdist = 0

        #--------------------------------------------------
        #下面是封装动作空间和状态空间
        self.action_space = spaces.Discrete(169)#第一维是行，第二维是列
        self.observation_space = spaces.Box(low=0,high=1,shape=(3,13,13),dtype=float)#第一层模块大小及清洗代价，第二层父模块位置，第三层污染感知
#下面是一些可视化相关函数---------------------------------------------------------------------------------------------------
    #获取图像
    def map_chip2obs(self,coop):
        return coop

    # def render(self, mode='human'):#默认可视化子问题的结果，不然就是可视化当前状态
    #     FPVA = self.obs[2].copy()
    #     # 创建一个颜色映射
    #     cmap = mcolors.ListedColormap(['white','yellow', 'green','blue'])
    #     bounds = [0, 0.099,0.2,0.9, 1]
    #     norm = mcolors.BoundaryNorm(bounds, cmap.N)
    #     plt.title(f'ptr: {self.ptr}')
    #
    #     # 使用颜色映射来显示数据
    #     plt.imshow(FPVA, cmap=cmap, norm=norm)
    #
    #     # 显示图像
    #     plt.show()

    def cal_ASR(self):#计算ASR的面积
        lefttop_x = min(x for x in self.PlacementScheme[:,0] if x != -1)
        lefttop_y = min(y for y in self.PlacementScheme[:,1] if y != -1)
        #下面先记录所有模块的右下角单元坐标
        rightbottom = np.full(shape=(self.NumModule,2),fill_value=-1,dtype=int)
        for i in range(self.NumModule):
            if self.PlacementScheme[i][0] != -1:
                rightbottom[i][0] = self.PlacementScheme[i][0] + self.ModuleAttribute[i][0] - 1
                rightbottom[i][1] = self.PlacementScheme[i][1] + self.ModuleAttribute[i][1] - 1
        #下面在rightbottom中找到最右下角的单元坐标
        rightbottom_x = max(x for x in rightbottom[:,0] if x != -1)
        rightbottom_y = max(y for y in rightbottom[:,1] if y != -1)
        limarea = (rightbottom_x - lefttop_x + 1) * (rightbottom_y - lefttop_y + 1)
        outputdist = self.BiochipWidth-1 - rightbottom_x + self.BiochipLength-1 - rightbottom_y
        surdist = abs(self.BiochipWidth-1-rightbottom_x - lefttop_x) + abs(self.BiochipLength-1-rightbottom_y - lefttop_y)
        return limarea,outputdist,surdist

    def cal_reward(self,sit):#根据情况计算奖励:0->布局越过边界;1->按step结算;2->布局失败;3->布局任务完成
        if sit == 0:
            return self.reward_overbound
        elif sit == 1:#每布局一块，就要对同族和异族之间的重叠进行奖励
            limarea,outputdist,surdist = self.cal_ASR()
            sub_ASPR = limarea - self.record_ASPR
            sub_surdist = surdist - self.record_surdist
            return (self.reward_overlap_father -
                    (self.weight_WP*self.tmpWP+self.weight_ASPR*sub_ASPR/1000+self.weight_surdist*sub_surdist/1000))
        elif sit == 2:
            return self.reward_fail
        elif sit == 3:
            return self.reward_finish

    def action2pos(self,action):
        posx = action // 13
        posy = action % 13
        return [posx,posy]
    # def get_mask(self):
    #     mask = np.ones(shape=(2,13),dtype=int)
    #     for row in range(self.BiochipWidth,13):
    #         mask[0][row] = 0
    #     for col in range(self.BiochipLength,13):
    #         mask[1][col] = 0
    #     curmod_sz = self.ModuleAttribute[self.PlacementSequence[self.ptr]]
    #     for row in range(self.BiochipWidth - curmod_sz[0] + 1,self.BiochipWidth):
    #         mask[0][row] = 0
    #     for col in range(self.BiochipLength - curmod_sz[1] + 1,self.BiochipLength):
    #         mask[1][col] = 0
    #     action_mask = mask.astype(np.float32)
    #     return action_mask
    def get_mask(self):
        mask = np.ones(shape=(13,13),dtype=bool)
        for row in range(self.BiochipWidth,13):
            for col in range(13):
                mask[row][col] = 0
        for col in range(self.BiochipLength,13):
            for row in range(13):
                mask[row][col] = 0
        curmod_sz = self.ModuleAttribute[self.PlacementSequence[self.ptr]]
        for row in range(self.BiochipWidth - curmod_sz[0] + 1,self.BiochipWidth):
            for col in range(13):
                mask[row][col] = 0
        for col in range(self.BiochipLength - curmod_sz[1] + 1,self.BiochipLength):
            for row in range(13):
                mask[row][col] = 0
        if self.ptr > 0:
            lastmod = self.PlacementSequence[self.ptr-1]
            lastmod_posx = self.PlacementScheme[lastmod][0]
            lastmod_posy = self.PlacementScheme[lastmod][1]
            mask[lastmod_posx][lastmod_posy] = 0
        action_masks = mask.flatten()
        return action_masks

    def reset(self,seed=None,options=None):
        self.finishPlacement  =0  # 是否完成布线子任务
        self.record_agent = [] #记录上个布局的位置，不应该连续选择同一个位置
        self.bufferWP = 0
        self.PlacementScheme = np.full(shape=(self.NumModule,2),fill_value=-1,dtype=int)#记录已布局的模块的位置
        #一些关于违反时间步长的参数
        self.timestep=0
        self.ptr = 0 #在布局顺序表中的指针
        #辅助可视化
        self.washPrice=0
        #初始化状态空间；当开启新一轮的时候，需要先把第一个布线任务更新到状态里
        self.obs = np.zeros(shape=(3,13,13),dtype=float)#状态空间的具体实现
        for row in range(self.BiochipWidth):#将芯片置为0.1
            for col in range(self.BiochipLength):
                self.obs[0][self.map_chip2obs(row)][self.map_chip2obs(col)] = 0.00001
                self.obs[1][self.map_chip2obs(row)][self.map_chip2obs(col)] = 0.00001
                self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)] = 0.00001
        for row in range(self.ModuleAttribute[self.PlacementSequence[0]][0]):
            for col in range(self.ModuleAttribute[self.PlacementSequence[0]][1]):
                self.obs[0][self.map_chip2obs(row)][self.map_chip2obs(col)] = self.FluidContamination[self.PlacementSequence[0]]
        self.washtime = 0
        self.record_ASPR = 0
        self.record_surdist = 0
        info = {}
        return self.obs,info
#---------------------------------------------------------------------------------------------------上面是一些reset相关的函数

#下面是step函数-----------------------------------------------------------------------------------------------------------
    def step(self, action):#由三部分组成:1、根据action布局当前模块;2、更新要布局的模块;3、随机布线/或者Astar布线
        action = self.action2pos(action)
        terminated=False
        info={}
        truncated=False
        reward=0
        obs = self.obs.copy()
    #----上面初始化一些必须返回的参数--------
        self.timestep += 1
        if self.timestep > self.max_timestep:
            truncated = True
            reward += self.cal_reward(2)
    #-------上面判断一下是否truncate----------
        modnum = self.PlacementSequence[self.ptr]
        posx = action[0]
        posy = action[1]
        szx = self.ModuleAttribute[modnum][0]
        szy = self.ModuleAttribute[modnum][1]
        self.tmpWP = 0
        obscopy = self.obs.copy()
        self.cnt_overlap_father = 0
        for row in range(posx,posx+szx):#布局模块
            for col in range(posy,posy+szy):
                if (row >= self.BiochipWidth or col >=self.BiochipLength or
                        row < 0 or col < 0):#布局越界
                    reward += self.cal_reward(0)
                    self.obs = obscopy.copy()
                    return  self.obs, reward, terminated, truncated, info
                if self.obs[1][self.map_chip2obs(row)][self.map_chip2obs(col)] == 1 :  # 该单元上个使用者确实是父组件
                    flag_overlap = 0
                    for f in self.OperationDependence[modnum]:#枚举所有父组件的所有单元
                        if f == -1:
                            continue
                        if self.PlacementScheme[f][0] == -1:
                            continue
                        if self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)] == self.FluidContamination[f]:
                            self.cnt_overlap_father += 1
                            self.reward_overlap_father = 0.01
                            flag_overlap = 1
                            break
                    if flag_overlap == 0:  # 虽然父子，但是有插足者
                        self.reward_overlap_father = 0.002
                        if self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)] > 0.00001:  # 非重叠单元，计算清洗代价
                            self.washtime += 1
                            self.tmpWP += self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)]
                else:#父子非重叠单元
                    self.reward_overlap_father = 0.002
                    if self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)] > 0.00001:#非重叠单元，计算清洗代价
                        self.washtime += 1
                        self.tmpWP += self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)]
                self.obs[2][self.map_chip2obs(row)][self.map_chip2obs(col)] = self.FluidContamination[modnum]
        self.PlacementScheme[modnum][0] = action[0]
        self.PlacementScheme[modnum][1] = action[1]
        if self.ptr == len(self.PlacementSequence)-1:#全部布局任务完成
            reward += self.cal_reward(3)
            obs = self.obs.copy()
            terminated = True
            return obs, reward, terminated, truncated, info
#-----------上面进行布局和更新obs[2]------------------------------------------

#-------------下面更新obs[1]及相关变量-----------------------------------------

        self.ptr += 1
        #------------更新下个布局任务的父模块的位置----------------
        for row in range(self.BiochipWidth):
            for col in range(self.BiochipLength):
                self.obs[1][self.map_chip2obs(row)][self.map_chip2obs(col)] = 0.00001
        #-----------更新obs[1]
        nextmod = self.PlacementSequence[self.ptr]
        fathermod = self.OperationDependence[nextmod]
        for mem in fathermod:
            if mem == -1:
                continue
            father_posx = self.PlacementScheme[mem][0]
            father_posy = self.PlacementScheme[mem][1]
            father_szx = self.ModuleAttribute[mem][0]
            father_szy = self.ModuleAttribute[mem][1]
            if father_posx == -1:
                continue
            for row in range(father_posx, father_posx + father_szx):
                for col in range(father_posy, father_posy + father_szy):
                    self.obs[1][self.map_chip2obs(row)][self.map_chip2obs(col)] = 1
        for row in range(self.ModuleAttribute[nextmod][0]):
            for col in range(self.ModuleAttribute[nextmod][1]):
                self.obs[0][self.map_chip2obs(row)][self.map_chip2obs(col)] = self.FluidContamination[nextmod]
        #----------------------------------------------------------
        #------------进行奖励结算----------------
        self.washPrice += self.tmpWP
        reward += self.cal_reward(1)
        #-----------记录这次step后的一些信息-----------------
        self.record_ASPR, useless, self.record_surdist = self.cal_ASR()
        obs = self.obs.copy()

        return obs, reward, terminated, truncated, info

    #结束该实例
    def close(self):
        plt.close()

#下面是一些待选函数--------------------------------------------------------------------------------------------------------


