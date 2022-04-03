# ContainerMigration
A Two-Stage Container Management in the Cloud for Optimizing the Load Balancing and Migration Cost
## Folder Migration:
### 1. File placement is Algorithms that is applied to place containers
1. **func placement**: we can choose placement algorithms in function placement (e.g. FF(our proposed), PSO, CRO)
FF is our present algorithm, that is algorithms 1 BACP Algorithm for container placement in this paper
PSO and CRO are Contrast method
2. **func getNodeNum**: we can adopt the func to get the number of servers according to our prediction workload of containers
3. **func getLoadMean**: this function can get the mean value of prediction workload of long stage
4. **func LoadBalance**: this function is used to as objective Equation for Heuristic Algorithms, and this function can get load degree according to Deployment matrix. This function is optimized eq (7)
###2. File migration is Algorithms that is used to migrate containers (These functions constitute the ATCM Algorithm for Container Migration in the paper)
at file migration, we will get placement matrix from Placement stage, then, we will migrate containers based on this matrix. In addition, if we choose CRO at Placement stage, then we also use it at Migration stage.
1. **func init**: this function calculate resource_used, resource_remaining and resource_utilization for each server node according to Placement matrix
2. **func next_slot**: this function calculate resource_used, resource_remaining and resource_utilization for each server node based on Placement matrix and the resources that the container needs at the next slot
3. **func select_Overload_Service**: this function can select server of overload which load degree > mean value (load degree) based on Placement matrix and the resources that the container needs at the next slot
4. **func select_Migration_Containers**: this function will select migration containers from overload server list, which can make these servers in overload list <= mean value(load degree) In addition, this can calculate the resource_used, resource_remaining and resource_utilization when these containers are migrated. This algorithm is Algorithm 2 selecting containers for migration in this paper.
5. **func MigrationCostAndBalance**: this function is used to as objective Equation for Heuristic Algorithms, and this function can get load degree according to Deployment matrix. This function is optimized eq (13)
6. **func get_Migration_Strategy**: there are three algorithms, FF is our proposed, others is contrast methods. This function is Algorithm 3 Algorithm of container migration in this paper.
Others methods are designed to Statistical experimental results.
## Folder Prediction
In this folder, there are some models to predict workload for containers, this is our previous work. In this paper, we employ models to prediction workload of containers, and use prediction results to management containers(e.g., placement and migration) At the beginning of the experiment， we predict workload of containers, these results will be adopted all algorithms, including comparison algorithm (e.g., PSO, CRO).