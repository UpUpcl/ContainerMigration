import numpy as np
from mealpy.evolutionary_based.CRO import BaseCRO
from mealpy.swarm_based.PSO import BasePSO
from Migration.read_cpu import read_preediction_value
from sko.PSO import PSO
np.random.seed(54321)


class Placement:
    def __init__(self):
        self.method = "FF"
        self.Container_Num = 80
        self.init_Node_Num = 10
        self.Node_Num = self.getNodeNum()
        self.resource_total_EN = np.array(
            [[100, 32 * np.random.randint(1, 3), np.random.randint(300, 500)] for i in
             range(self.Node_Num)])

        self.resource_Container = np.array(
            [[np.random.randint(1, 3), np.random.randint(1, 8), np.random.randint(5, 15)] for i in range(self.Container_Num)])
        predictionValue = np.round(read_preediction_value("../data/prediction_cpu_15min.csv", self.Container_Num))
        for i in range(self.Container_Num):
            self.resource_Container[i][0] = predictionValue[i]
        summen = 0
        for i in range(self.Container_Num):
            summen = summen + self.resource_Container[i][1]
        self.MenMean = summen/self.Container_Num
        # np.random.shuffle(self.resource_Container)
        self.PM = self.getLoadMean()


    def placement(self):
        # 节点数根据15分钟的预测值

        # 每个节点可以提供的资源

        """
            根据资源以及每个容器在T需要的资源做出部署决策 X
        """
        # 决策矩阵
        # PSO
        # if self.method == "PSO":
        #     max_iter = 1000
        #     pso = PSO(func=self.LoadBalance, n_dim=self.Container_Num, pop=self.Container_Num+20, max_iter=max_iter,
        #               w=0.75, c1=0.65, c2=0.65, lb=0, ub=self.Node_Num-1)
        #     pso.run(max_iter=max_iter)
        #     print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        #     place_load = pso.gbest_y
        #     X = np.zeros([self.Container_Num, self.Node_Num], dtype=int)
        #     for i in range(self.Container_Num):
        #         j = int(pso.gbest_x[i])
        #         X[i][j] = 1
        #     return X, self.Node_Num, self.Container_Num, self.resource_total_EN, self.resource_Container, self.PM, place_load, self.method

        if self.method == "PSO":
            obj_func = self.LoadBalance
            verbose = False
            epoch = 15000
            pop_size = 100
            problemSize = self.Container_Num
            lb2 = 0
            ub2 = self.Node_Num - 1
            md2 = BasePSO(obj_func, lb2, ub2, verbose, epoch, pop_size,
                          problem_size=problemSize)  # Remember the keyword "problem_size"
            best_pos1, best_fit1, list_loss1 = md2.train()
            print(best_pos1, best_fit1)
            X = np.zeros([self.Container_Num, self.Node_Num], dtype=int)
            for i in range(self.Container_Num):
                j = int(best_pos1[i])
                X[i][j] = 1
            place_load = best_fit1
            return X, self.Node_Num, self.Container_Num, self.resource_total_EN, self.resource_Container, self.PM, place_load, self.method

        # CRO
        if self.method == "CRO":
                obj_func = self.LoadBalance
                verbose = False
                epoch = 1000
                pop_size = 100
                problemSize = self.Container_Num
                lb2 = 0
                ub2 = self.Node_Num - 1
                md2 = BaseCRO(obj_func, lb2, ub2, verbose, epoch, pop_size,
                              problem_size=problemSize)  # Remember the keyword "problem_size"
                best_pos1, best_fit1, list_loss1 = md2.train()
                print(best_pos1, best_fit1)
                X = np.zeros([self.Container_Num, self.Node_Num], dtype=int)
                for i in range(self.Container_Num):
                    j = int(best_pos1[i])
                    X[i][j] = 1
                place_load = best_fit1
                return X, self.Node_Num, self.Container_Num, self.resource_total_EN, self.resource_Container, self.PM, place_load, self.method
        # First fit
        if self.method == "FF":
            w_1, w_2 = 0.7, 0.3
            Load_ = w_1 * self.PM + w_2 * self.MenMean
            NodeLoad = np.zeros([self.Node_Num], dtype=int)
            NodeCPU = np.array([100] * self.Node_Num)
            Nodemen = np.zeros([self.Node_Num], dtype=int)
            X = np.zeros([self.Container_Num, self.Node_Num], dtype=int)
            for c in range(self.Container_Num):
                put = False
                tempLoad = w_1 * self.resource_Container[c][0] + w_2 * self.resource_Container[c][1]
                tempNodeLoad = np.array(NodeLoad.copy())
                for n in range(self.Node_Num):
                    if NodeCPU[n] - self.resource_Container[c][0] > 0 and self.resource_total_EN[n][1] > Nodemen[n]:
                        if NodeLoad[n] + tempLoad < Load_:
                            NodeCPU[n] = NodeCPU[n] - self.resource_Container[c][0]
                            Nodemen[n] = Nodemen[n] + self.resource_Container[c][1]
                            NodeLoad[n] = NodeLoad[n] + tempLoad
                            X[c][n] = 1
                            put = True
                            break

                if not put:
                    for i in range(self.Node_Num):
                        min = np.argmin(tempNodeLoad)
                        print("min", min)
                        if NodeCPU[min] - self.resource_Container[c][0] > 0 and Nodemen[min] < self.resource_total_EN[n][1]:
                            NodeCPU[min] = NodeCPU[min] - self.resource_Container[c][0]
                            Nodemen[min] = Nodemen[min] + self.resource_Container[c][1]
                            NodeLoad[min] = NodeLoad[min] + tempLoad
                            X[c][min] = 1
                            print("容器放置在：", c, min)
                            break
                        else:
                            tempNodeLoad[min] = np.inf

            temp = 0
            for i in range(self.Node_Num):
                temp = temp + (NodeLoad[i] - Load_) ** 2
            place_load = temp / self.Node_Num
            return X, self.Node_Num, self.Container_Num, self.resource_total_EN, self.resource_Container, self.PM, place_load, self.method



    # make the CPU utilization fall into 40 to 60 (15min)
    def getNodeNum(self):

        NodeNum = self.init_Node_Num
        path_15min = "../data/prediction_cpu_15min.csv"
        prediction = read_preediction_value(path_15min, self.Container_Num)
        predictionMean = np.round(sum(prediction) / NodeNum)
        print("开始节点数为：", NodeNum, "\t预测的均值为：", predictionMean)
        num = 0
        while predictionMean > 80 or predictionMean < 20:
            num = num + 1
            if predictionMean > 100:
                newNodeNum = NodeNum * 2
            elif predictionMean < 40:
                newNodeNum = np.math.ceil(NodeNum / int(np.round(50 / predictionMean)))
                if newNodeNum == NodeNum:
                    newNodeNum = int(0.67 * NodeNum)
            elif 60 < predictionMean < 100:
                newNodeNum = int(np.round(predictionMean / 50)) * NodeNum
                if newNodeNum == NodeNum:
                    newNodeNum = int(1.33 * NodeNum)
            NodeNum = newNodeNum
            predictionMean = np.round(sum(prediction) / NodeNum)
        print("结束节点数为：", NodeNum, "\t预测的均值为：", predictionMean)
        return NodeNum

    # get PredictionModel value of 15 min at slot 0
    def getLoadMean(self):
        path_5min = "../data/prediction_cpu.csv"
        predictionValue = read_preediction_value(path_5min, self.Container_Num)
        predictionMeanValue = np.round(sum(np.round(predictionValue)) / self.Node_Num)
        print("放置阶段的CPU均值为：", predictionMeanValue)
        return predictionMeanValue

    # calculate the load of service
    def LoadBalance(self, x):
        """
        ServerLoadList: 存放每个服务器的LoadBalance
        ServerLoadCpuList：存在每个服务器的CPU load
        ServerLoadMenList：存放每个服务器的Men load
        ServerLoadDiskList：存放每个服务器的Disk load
        """
        ServerLoadList = []
        ServerLoadCpuList = []
        ServerLoadMenList = []
        ServerLoadDiskList = []
        x_i = np.zeros([self.Container_Num, self.Node_Num], dtype=int)
        for i in range(self.Container_Num):
            j = int(x[i])
            x_i[i][j] = 1

        w_1, w_2 = 0.7, 0.3
        for i in range(self.Node_Num):
            load_i = 0
            load_Cpu_i, load_Men_i, load_Disk_i = 0, 0, 0
            for j in range(self.Container_Num):
                if x_i[j][i] == 1:
                    containerLoad_i = w_1 * self.resource_Container[j][0] + w_2 * self.resource_Container[j][1]
                    load_i = load_i + containerLoad_i
                    load_Cpu_i = load_Cpu_i + self.resource_Container[j][0]
                    load_Men_i = load_Men_i + self.resource_Container[j][1]
                    load_Disk_i = load_Disk_i + self.resource_Container[j][2]
            ServerLoadList.append(load_i)
            ServerLoadCpuList.append(load_Cpu_i)
            ServerLoadMenList.append(load_Men_i)
            ServerLoadDiskList.append(load_Disk_i)
        temp = 0
        Load_ = w_1 * self.PM + w_2 * self.MenMean
        # print(Load_)
        for i in range(self.Node_Num):
            temp = temp + (ServerLoadList[i] - Load_) ** 2
        L_balance = np.round(temp/self.Node_Num)
        flag = True
        '''
        placement 放置检测约束 cpu, mem, disk
        '''
        for idx in range(self.Node_Num):
            if ServerLoadCpuList[idx] > 100 or ServerLoadMenList[idx] > self.resource_total_EN[idx][1] \
                    or ServerLoadDiskList[idx] > self.resource_total_EN[idx][2]:
                flag = False
                break
        if flag:
            return L_balance
        else:
            return np.inf

