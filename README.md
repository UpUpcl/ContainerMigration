# ContainerMigration
A Two-Stage Container Management in the Cloud for Optimizing the Load Balancing and Migration Cost
## Folder Migration:
### 1. File placement.py presents Algorithms that is applied to place containers
1. **func placement**: we can choose placement algorithms in function placement (e.g. FF, PSO, CRO). Our proposed algorithm BACP is based on FF, which is compared with PSO and CRO. 
2. **func getNodeNum**: we can adopt this function to get the number of servers according to the prediction workload of containers
3. **func getLoadMean**: this function can get the mean value of prediction workload of long stage
4. **func LoadBalance**: this function is used to present the objective equation for container placement, and it  can get load imbalance degree according to Deployment matrix.
### 2. File migration.py presents Algorithms that is used to migrate containers (These functions constitute the ATCM Algorithm for Container Migration)
at file migration, we will get placement matrix from Placement stage, then, we will migrate containers based on this matrix. In addition, if we choose CRO at Placement stage, then we also use it at Migration stage.
1. **func init**: this function calculates resource_used, resource_remaining and resource_utilization for each server node according to Placement matrix
2. **func next_slot**: this function calculates resource_used, resource_remaining and resource_utilization for each server node based on Placement matrix and the resources that the container needs at the next slot
3. **func select_Overload_Service**: this function can select servers  that are overloaded, with load degree > mean value (load degree) based on Placement matrix and return the resources that the container needs at the next slot
4. **func select_Migration_Containers**: his function will select the containers from overloaded server list for container migration. In addition, this function will update the resource_used, resource_remaining and resource_utilization when these containers are migrated.
5. **func MigrationCostAndBalance**: this function is used to present objective function for container migration. It can obtain the load degree according to Deployment matrix. 
6. **func get_Migration_Strategy**: this function performs container migration. 
## Folder Prediction
In this folder, we present the machine learning models to predict workload for containers and leverage prediction results to manage containers (e.g., placement and migration). We also conduct algorithm comparison to evaluate the performance of the proposed algorithm. 
