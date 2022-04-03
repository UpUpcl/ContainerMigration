import numpy as np
from mealpy.evolutionary_based.CRO import BaseCRO
from sko.PSO import PSO
from sko.GA import GA

from Migration.Placement import Placement
from Migration.read_cpu import *

np.random.seed(54321)


class Migration:
    def __init__(self):
        replacement = Placement()
        '''
            X 上一个阶段部署矩阵， Node_Num 节点数量， Container_Num容器数量，resource_total_EN 节点能够提供的资源能力
            before_resource_Container 迁移之前的容器资源，before_load_mean迁移之前的cpu均值，place_load放置之后集群的负载
            self.method使用的方法
        '''

        self.X, self.Node_Num, self.Container_Num, self.resource_total_EN, self.before_resource_Container, \
        self.before_load_mean, self.place_load, self.method = replacement.placement()
        self.path = "../data/prediction_cpu_1.csv"
        self.path_real_0 = "../data/real_cpu_15min.csv"
        self.path_real_1 = "../data/real_cpu_1.csv"
        # 资源类型为 CUP MEMORY DISK, resource_Container 为下一个slot时container需要的资源数
        self.resource_Container = np.array(
            [[np.random.randint(1, 3), np.random.randint(1, 8), np.random.randint(5, 15)] for i in
             range(self.Container_Num)])
        # container迁移时的挂在数据大小
        self.data_Container = np.array([[np.random.randint(100, 2000)] for i in range(self.Container_Num)])
        # 将预测值拼接到容器初始化中
        predictionValue = np.round(read_preediction_value(self.path, self.Container_Num))
        for i in range(self.Container_Num):
            self.resource_Container[i][0] = predictionValue[i]

        # 放置阶段真实数据
        self.real_0_contianer = self.resource_Container.copy()
        real_value0 = np.round(read_preediction_value(self.path_real_0, self.Container_Num))
        for i in range(self.Container_Num):
            self.real_0_contianer[i][0] = real_value0[i]

        # 迁移阶段真实数据
        self.real_1_contianer = self.resource_Container.copy()
        real_value1 = np.round(read_preediction_value(self.path_real_1, self.Container_Num))
        for i in range(self.Container_Num):
            self.real_0_contianer[i][0] = real_value1[i]

        # 内存均值
        summen = 0
        for i in range(self.Container_Num):
            summen = summen + self.resource_Container[i][1]
        self.MenMean = summen / self.Container_Num

        self.data_trans_Container = np.random.randint(100, 500, self.Container_Num)

        # 节点之间的带宽   对称矩阵
        self.Band = np.triu(
            np.random.randint(100, 300, self.Node_Num * self.Node_Num).reshape(self.Node_Num, self.Node_Num))
        self.Band += self.Band.T - np.diag(self.Band.diagonal())
        self.Band = self.Band - np.diag(np.diag(self.Band))

        # 初始化每个节点各类资源的使用情况
        self.resource_used_EN = np.zeros([self.Node_Num, 3], dtype=int)
        # 初始化每个节点剩余资源
        self.resource_remaining_EN = np.zeros([self.Node_Num, 3], dtype=int)
        # 节点中资源的利用率
        self.resource_utilization_EN = np.zeros([self.Node_Num, 3], dtype=int)

        self.predictionVM = self.get_prediction_MeanValue()

        self.next_resource_used_EN = np.zeros([self.Node_Num, 3], dtype=int)
        self.next_resource_remaining_EN = np.zeros([self.Node_Num, 3], dtype=int)
        # 节点中资源的利用率
        self.next_resource_utilization_EN = np.zeros([self.Node_Num, 3], dtype=int)

        self.migration_container = []

        self.Y = self.X.copy()

    def init(self):
        '''
        利用上一个slot的部署方案，判断下一个slot的container_resource是否会过载
        :return:
        '''
        for i in range(self.Node_Num):
            for j in range(self.Container_Num):
                if self.X[j][i] == 1:
                    self.resource_used_EN[i][0] += self.before_resource_Container[j][0]
                    self.resource_used_EN[i][1] += self.before_resource_Container[j][1]
                    self.resource_used_EN[i][2] += self.before_resource_Container[j][2]
            self.resource_remaining_EN[i] = self.resource_total_EN[i] - self.resource_used_EN[i]
        self.resource_utilization_EN = self.resource_used_EN / self.resource_total_EN
        print("目前节点资源使用情况：\n", self.resource_used_EN)
        print("目前节点资源剩余使用情况：\n", self.resource_remaining_EN)
        print("目前节点资源使用率：\n", self.resource_utilization_EN)

    def next_slot_(self):
        """
        根据当前X， 计算一下一个slot的资源状况
        """
        for i in range(self.Node_Num):
            for j in range(self.Container_Num):
                if self.X[j][i] == 1:
                    self.next_resource_used_EN[i][0] += self.resource_Container[j][0]
                    self.next_resource_used_EN[i][1] += self.resource_Container[j][1]
                    self.next_resource_used_EN[i][2] += self.resource_Container[j][2]
            self.next_resource_remaining_EN[i] = self.resource_total_EN[i] - self.next_resource_used_EN[i]
        self.next_resource_utilization_EN = self.next_resource_used_EN / self.resource_total_EN
        print("next slot节点资源使用情况：\n", self.next_resource_used_EN)
        print("next slot节点资源剩余使用情况：\n", self.next_resource_remaining_EN)
        print("next slot节点资源使用率：\n", self.next_resource_utilization_EN)

    # calculate the mean value of PredictionModel CPU utilization
    def get_prediction_MeanValue(self):

        predictionValue = read_preediction_value(self.path, self.Container_Num)
        predictionValueMean = np.round(sum(np.round(predictionValue)) / self.Node_Num)
        print("预测的下一个slot服务器cpu均值为：", predictionValueMean)
        return predictionValueMean

    #  select the CPU overload of service according to get_prediction_MeanValue() 过载服务器列表
    def select_Overload_Service(self):
        """
            根据下一个slot的cpu均值计算下一个时刻的服务器是否按照当前的部署决策是否是过载的
            :return: 过载的服务器列表
        """
        overload = []
        w_1, w_2 = 0.7, 0.3
        cpu_mean = w_1 * self.predictionVM + w_2 * self.MenMean
        for i in range(self.Node_Num):
            deployedCpu = 0
            # 只衡量了一个cpu，后续在思考是否考虑memory？
            for j in range(self.Container_Num):
                if self.X[j][i] == 1:
                    deployedCpu = deployedCpu + w_1 * self.resource_Container[j][0] + w_2 * self.resource_Container[j][
                        1]
            if deployedCpu > cpu_mean:
                contaninerlist = self.get_deployed_server(i)
                if len(contaninerlist) > 1:
                    overload.append(i)
            # if self.resource_utilization_EN[i][0] > cpu_mean:
            #     overload.append(i)
        print("过载的服务器为", overload)
        return overload

    # 计算传输时间
    def transTime(self, c):
        """
            这里需要修改时间
        """
        # volumn = self.resource_Container[c][2] * 1000
        volumn = self.resource_Container[c][2] * 1000 + self.data_Container[c]
        c_local = 0
        for i in range(self.Node_Num):
            if self.X[c][i] == 1:
                c_local = i
                break
        cost = []
        for i in range(self.Node_Num):
            if i == c_local:
                cost.append(0)
            else:
                cost.append(volumn / self.Band[c_local][i])
        return cost

    # 得到部署在节点i的容器
    def get_deployed_server(self, i):
        ContainerList = []
        for j in range(self.Container_Num):
            if self.X[j][i] == 1:
                ContainerList.append(j)
        return ContainerList

    #  get migrated containers from overload service
    def select_Migration_Containers(self):

        overload_server = self.select_Overload_Service()

        for i in overload_server:
            select_container = []
            contaninerlist = self.get_deployed_server(i)
            # select containers from containerlist, which make the overload of server near to cpu mean
            '''
                这一部分将描述如何选择出待迁移的容器 (都只考虑了cpu)
                1. 按降序排列，依次将CPU使用量最少的容器标记为待迁移，直到所剩下的使用量最接近下一个slot的均值。
            '''
            containerNum = len(contaninerlist)
            containerResList = np.zeros([containerNum, 3], dtype=int)
            c = 0
            for cid in contaninerlist:
                containerResList[c][0] = cid
                containerResList[c][1] = self.resource_Container[cid][1]
                containerResList[c][2] = self.resource_Container[cid][0]
                c += 1
            containerResList = containerResList[np.lexsort(containerResList.T)]
            print("部署在节点", i, "容器为", contaninerlist, "\n 各个容器资源为：\n", containerResList)
            k = 0
            w_1, w_2 = 0.7, 0.3
            Load_ = w_1 * self.predictionVM + w_2 * self.MenMean
            tempLoad = w_1 * self.next_resource_used_EN[i][0] + w_2 * self.next_resource_used_EN[i][1]
            while containerNum > 1:
                temp = tempLoad
                tempLoad = tempLoad - w_1 * containerResList[k][2] - w_2 * containerResList[k][1]
                if abs(tempLoad - w_1 * self.predictionVM - w_2 * self.MenMean) < abs(
                        temp - w_1 * self.predictionVM - w_2 * self.MenMean):
                    select_container.append(containerResList[k][0])
                    k += 1
                    containerNum -= 1
                else:
                    break

            # update the resources of all nodes according to migration_container
            # resource_used_EN[i]  (-)
            for k in range(len(select_container)):
                container_id = select_container[k]
                self.migration_container.append(container_id)
                self.next_resource_used_EN[i][0] -= self.resource_Container[container_id][0]
                self.next_resource_used_EN[i][1] -= self.resource_Container[container_id][1]
                self.next_resource_used_EN[i][2] -= self.resource_Container[container_id][2]
            self.next_resource_remaining_EN[i] = self.resource_total_EN[i] - self.next_resource_used_EN[i]
            self.next_resource_utilization_EN[i] = self.next_resource_used_EN[i] / self.resource_total_EN[i]
        print("需要迁移的容器为：\n", self.migration_container)
        print("确定待迁移容器之后节点资源利用率为: \n", self.next_resource_utilization_EN)
        print("确定待迁移容器之后节点剩余资源为: \n", self.next_resource_remaining_EN)
        # return migration_container

    # 计算w * cost + v * load  (eq 13)
    def MigrationCostAndBalance(self, Z):
        cost = 0
        # print("Z:", Z)
        if self.method == "CRO":
            Z = list(np.trunc(Z))
            if len(self.migration_container) == 1:
                for i in self.migration_container:
                    for j in range(self.Node_Num):
                        if self.X[i][j] == 1:
                            if j in Z:
                                Z.remove(j)
                            else:
                                Z.remove(Z[np.random.randint(0, 2)])

        for i in self.migration_container:
            for j in range(self.Node_Num):
                self.Y[i][j] = 0

        for m in range(len(Z)):
            self.Y[self.migration_container[m]][int(Z[m])] = 1

        '''
            修改
        '''
        # 计算迁移时间
        for cid in self.migration_container:
            for y_i in range(self.Node_Num):
                if self.Y[cid][y_i] == 1:
                    for x_i in range(self.Node_Num):
                        if self.X[cid][x_i] == 1:
                            if x_i == y_i:
                                cost = cost + 0
                            else:
                                temp = ((self.resource_Container[cid][2] * 1000 + self.data_Container[cid]) /
                                        self.Band[x_i][y_i])
                                if cost < temp:
                                    cost = temp
                                # # cost = cost + (self.resource_Container[cid][2] * 1000 / self.Band[x_i][y_i])
                                # cost = cost + ((self.resource_Container[cid][2] * 1000 + self.data_Container[cid]) /
                                #                self.Band[x_i][y_i])

        ServerLoadList = []
        w_1, w_2 = 0.7, 0.3
        for i in range(self.Node_Num):
            load_i = 0
            for j in range(self.Container_Num):
                if self.Y[j][i] == 1:
                    containerLoad_i = w_1 * self.resource_Container[j][0] + w_2 * self.resource_Container[j][1]
                    load_i = load_i + containerLoad_i
            ServerLoadList.append(load_i)
        temp = 0
        Load_ = w_1 * self.predictionVM + w_2 * self.MenMean
        for i in range(self.Node_Num):
            temp = temp + (ServerLoadList[i] - Load_) ** 2
        LB = np.round(temp / self.Node_Num)
        a, b = 0.4, 0.6
        return a * cost + b * LB

    # 计算迁移代价
    def MigrationCost(self):
        if len(self.migration_container) == 0:
            print("Cost is 0, load is xxx")
            return 0
        cost = 0
        for cid in self.migration_container:
            for y_i in range(self.Node_Num):
                if self.Y[cid][y_i] == 1:
                    for x_i in range(self.Node_Num):
                        if self.X[cid][x_i] == 1:
                            if x_i == y_i:
                                cost = cost + 0
                                break
                            else:
                                temp = ((self.resource_Container[cid][2] * 1000 + self.data_Container[cid]) /
                                        self.Band[x_i][y_i])
                                if cost < temp:
                                    cost = temp
                                # cost = cost + ((self.resource_Container[cid][2] * 1000 + self.data_Container[cid]) / self.Band[x_i][y_i])
                                break

        ServerLoadList = []
        w_1, w_2 = 0.7, 0.3
        for i in range(self.Node_Num):
            load_i = 0
            for j in range(self.Container_Num):
                if self.Y[j][i] == 1:
                    containerLoad_i = w_1 * self.real_1_contianer[j][0] + w_2 * self.real_1_contianer[j][1]
                    load_i = load_i + containerLoad_i
            ServerLoadList.append(load_i)
        temp = 0
        cpusum = 0
        mensum = 0
        for i in range(self.Container_Num):
            cpusum = cpusum + self.real_0_contianer[i][0]
            mensum = mensum + self.real_0_contianer[i][1]

        CVM = cpusum / self.Node_Num
        MVM = mensum / self.Node_Num
        Load_ = w_1 * CVM + w_2 * MVM

        for i in range(self.Node_Num):
            temp = temp + (ServerLoadList[i] - Load_) ** 2
        LB = np.round(temp / self.Node_Num)
        a, b = 0.4, 0.6
        print("Cost", cost, "LB", LB)
        return a * cost + b * LB

    def LoadBalanceDegree(self):

        ServerLoadList_1 = []
        w_1, w_2 = 0.7, 0.3
        cpusum = 0
        mensum = 0
        for i in range(self.Container_Num):
            cpusum = cpusum + self.real_0_contianer[i][0]
            mensum = mensum + self.real_0_contianer[i][1]

        CVM = cpusum / self.Node_Num
        MVM = mensum / self.Node_Num
        Load_ = w_1 * CVM + w_2 * MVM
        for i in range(self.Node_Num):
            load_ii = 0
            for j in range(self.Container_Num):
                if self.X[j][i] == 1:
                    containerLoad_ii = w_1 * self.real_0_contianer[j][0] + w_2 * self.real_0_contianer[j][1]
                    load_ii = load_ii + containerLoad_ii
            ServerLoadList_1.append(load_ii)
        temp = 0
        for i in range(self.Node_Num):
            temp = temp + (ServerLoadList_1[i] - Load_) ** 2
        LB_before = np.round(temp / self.Node_Num)
        print("放置阶段负载：", LB_before)
        return LB_before

    def CompareLoad(self):
        # after migrateion
        cpusum = 0
        mensum = 0
        for i in range(self.Container_Num):
            cpusum = cpusum + self.real_1_contianer[i][0]
            mensum = mensum + self.real_1_contianer[i][1]

        CVM = cpusum / self.Node_Num
        MVM = mensum / self.Node_Num

        ServerLoadList = []
        w_1, w_2 = 0.7, 0.3
        for i in range(self.Node_Num):
            load_i = 0
            for j in range(self.Container_Num):
                if self.Y[j][i] == 1:
                    containerLoad_i = w_1 * self.real_1_contianer[j][0] + w_2 * self.real_1_contianer[j][1]
                    load_i = load_i + containerLoad_i
            ServerLoadList.append(load_i)
        temp = 0
        Load_ = w_1 * CVM + w_2 * MVM
        for i in range(self.Node_Num):
            temp = temp + (ServerLoadList[i] - Load_) ** 2
        LB_after = np.round(temp / self.Node_Num)
        print("迁移之后的负载", ServerLoadList)
        # before
        ServerLoadList_1 = []
        w_1, w_2 = 0.7, 0.3
        for i in range(self.Node_Num):
            load_ii = 0
            for j in range(self.Container_Num):
                if self.X[j][i] == 1:
                    containerLoad_ii = w_1 * self.real_1_contianer[j][0] + w_2 * self.real_1_contianer[j][1]
                    load_ii = load_ii + containerLoad_ii
            ServerLoadList_1.append(load_ii)
        print("迁移之前的负载", ServerLoadList_1)
        temp = 0
        for i in range(self.Node_Num):
            temp = temp + (ServerLoadList_1[i] - Load_) ** 2
        LB_before = np.round(temp / self.Node_Num)
        print("迁移之前的负载为：", LB_before, "迁移之后的负载为：", LB_after)
        return LB_before, LB_after

    #  get Migration strategy according to the constraint condition and updated resources table of services
    def get_Migration_Strategy(self):
        # adopt Heuristics methods
        """

        :param migration_container: 需要进行迁移的容器
        :return: 返回待迁移容器的部署策略
        """
        # 复制X是为了可以比较迁移前后的负载
        # 将待迁移的容器部署方案全部设为0

        if len(self.migration_container) == 0:
            return 0
        else:
            # PSO
            if self.method == "PSO":
                print("待迁移容器个数为：", len(self.migration_container))
                max_iter = 100
                pso = PSO(func=self.MigrationCostAndBalance, n_dim=len(self.migration_container),
                          pop=100, max_iter=max_iter, w=0.65,
                          c1=0.65, c2=0.55, lb=0, ub=self.Node_Num - 1)
                pso.run(max_iter=max_iter)
                print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
                return pso.gbest_y

            # First Fit
            if self.method == "FF":

                w_1, w_2 = 0.7, 0.3
                a, b = 0.4, 0.6
                NodeLoad = np.zeros([self.Node_Num], dtype=int)

                Load_ = w_1 * self.predictionVM + w_2 * self.MenMean
                z = np.zeros([len(self.migration_container)], dtype=int)
                for i in range(self.Node_Num):
                    NodeLoad[i] = w_1 * self.next_resource_used_EN[i][0] + w_2 * self.next_resource_used_EN[i][1]

                for c in range(len(self.migration_container)):
                    min_migration_cost = np.inf
                    best_node = 0
                    transtime = self.transTime(self.migration_container[c])
                    tempLoad = w_1 * self.resource_Container[self.migration_container[c]][0] + w_2 * \
                               self.resource_Container[self.migration_container[c]][1]
                    for n in range(self.Node_Num):
                        if self.next_resource_remaining_EN[n][0] < self.resource_Container[self.migration_container[c]][
                            0] and self.next_resource_remaining_EN[n][1] < \
                                self.resource_Container[self.migration_container[c]][1]:
                            continue
                        migration_cost = a * transtime[n] + b * (tempLoad + NodeLoad[n] - Load_) ** 2
                        if migration_cost < min_migration_cost:
                            min_migration_cost = migration_cost
                            best_node = n
                    z[c] = best_node
                    NodeLoad[best_node] = NodeLoad[best_node] + tempLoad

                cost = self.MigrationCostAndBalance(Z=z)
                return cost

            # CRO
            if self.method == "CRO":

                print("待迁移容器个数为：", len(self.migration_container))
                obj_func = self.MigrationCostAndBalance
                verbose = False
                epoch = 200
                pop_size = 100
                if len(self.migration_container) == 1:
                    problemSize = len(self.migration_container) + 1
                else:
                    problemSize = len(self.migration_container)
                # problemSize = len(self.migration_container)
                lb2 = 0
                ub2 = self.Node_Num - 1
                md = BaseCRO(obj_func, lb2, ub2, verbose, epoch, pop_size,
                             problem_size=problemSize)  # Remember the keyword "problem_size"
                best_pos1, best_fit1, list_loss1 = md.train()
                print("最佳迁移方案为：", best_pos1, "最佳迁移代价：", best_fit1)
                return best_fit1

    def run(self):
        # slot 控制是否出去T周的开始
        slot = 1

        if slot % 3 == 1:
            print("部署的容器数为：", self.Container_Num, "\t部署的节点数为", self.Node_Num)
            print("节点配置的资源情况:\n", self.resource_total_EN)
            print("下一时刻容器需要的资源数：\n", self.resource_Container)
            print("目前部署矩阵为：\n", self.X)
            # 初始化当前节点资源情况
            self.init()
            # 计算安装replacement阶段部署方案X的节点资源情况
            self.next_slot_()
            # 选择出待迁移的容器
            self.select_Migration_Containers()
            # 获得迁移矩阵
            MCOST = self.get_Migration_Strategy()
            # 为了统计结果的功能函数
            MigrationCost = self.MigrationCost()
            LB_load, AFTER_laod = self.CompareLoad()
            PlaceLoad = self.LoadBalanceDegree()

            # 保存数据
            filename = '../res_3_18/result_' + self.method + '_ConNum_' + str(self.Container_Num) + '.txt'
            f = open(filename, mode='w')
            f.write("\n节点数" + str(self.Node_Num) + "\t容器数:" + str(self.Container_Num) + "\t放置阶段load均值\n" + str(
                self.before_load_mean))
            f.write("节点资源情况\n")
            f.write(str(self.resource_total_EN))
            f.write("\n放置阶段容器资源需求\n")
            f.write(str(self.before_resource_Container))
            f.write("\n放置阶段容器部署方案\n")
            for i in range(self.Container_Num):
                for j in range(self.Node_Num):
                    f.write(str(self.X[i][j]))
                f.write("\n")
            f.write("\n放置阶段的负载均衡：" + str(self.place_load))
            f.write("\n放置阶段的负载均衡_real：" + str(PlaceLoad))
            f.write("\n放置阶段资源利用率\n")
            f.write(str(self.resource_utilization_EN))
            f.write("\n放置阶段资源剩余情况\n")
            f.write(str(self.resource_remaining_EN))
            f.write("\n放置阶段资源使用情况\n")
            f.write(str(self.resource_used_EN))
            f.write("\n迁移阶段容器资源需求\n")
            f.write(str(self.resource_Container))
            f.write("\n迁移阶段容器部署方案\n")
            for i in range(self.Container_Num):
                for j in range(self.Node_Num):
                    f.write(str(self.Y[i][j]))
                f.write("\n")
            f.write("\n需要迁移的容器为:\t" + str(self.migration_container))
            f.write("\n迁移代价：\t" + str(MigrationCost))
            f.write("\n迁移之前的负载" + str(LB_load) + "迁移之后的负载" + str(AFTER_laod))
            f.close()


migration = Migration()
migration.run()
