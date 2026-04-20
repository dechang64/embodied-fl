syntax = "proto3";
package embodiedfl;

// ============================================================
// Embodied-FL: Federated Learning Platform for Embodied AI
// ============================================================
// 核心设计理念：
// 1. 数据不出域 — 每个工厂/机器人本地训练，只共享模型梯度
// 2. 异构任务联邦 — 不同任务（抓取/导航/装配）可以协作
// 3. 任务匹配 — HNSW 向量搜索找到相似任务，智能聚合
// 4. 贡献追踪 — 区块链审计链记录每方贡献，为数据定价
// ============================================================

// 联邦学习服务 — 核心训练循环
service FederatedService {
  // 客户端注册（工厂/机器人上线）
  rpc RegisterClient(RegisterRequest) returns (RegisterResponse);
  // 客户端注销
  rpc UnregisterClient(UnregisterRequest) returns (UnregisterResponse);
  // 上报本地任务描述（用于任务匹配）
  rpc ReportTask(TaskReport) returns (TaskMatchResult);
  // 下载全局模型
  rpc GetGlobalModel(ModelRequest) returns (ModelResponse);
  // 上传本地梯度/模型更新
  rpc UploadUpdate(UpdateRequest) returns (UpdateResponse);
  // 获取联邦训练轮次状态
  rpc GetRoundStatus(RoundStatusRequest) returns (RoundStatusResponse);
  // 触发新一轮聚合
  rpc TriggerAggregation(AggregationRequest) returns (AggregationResponse);
}

// 任务注册服务 — 管理异构任务
service TaskRegistry {
  // 注册新任务
  rpc RegisterTask(NewTaskRequest) returns (TaskInfo);
  // 搜索相似任务（HNSW）
  rpc FindSimilarTasks(SimilarTaskRequest) returns (SimilarTaskResponse);
  // 列出所有任务
  rpc ListTasks(ListTasksRequest) returns (ListTasksResponse);
  // 更新任务状态
  rpc UpdateTaskStatus(UpdateTaskStatusRequest) returns (TaskInfo);
}

// 贡献追踪服务 — 数据价值量化
service ContributionService {
  // 查询贡献记录
  rpc GetContributions(ContributionQuery) returns (ContributionResponse);
  // 查询贡献排行榜
  rpc GetLeaderboard(LeaderboardRequest) returns (LeaderboardResponse);
  // 验证审计链完整性
  rpc VerifyAuditChain(VerifyRequest) returns (VerifyResponse);
}

// 监控服务 — 训练过程可视化
service MonitorService {
  // 获取训练统计
  rpc GetTrainingStats(StatsRequest) returns (TrainingStatsResponse);
  // 获取客户端状态
  rpc GetClientStatus(ClientStatusRequest) returns (ClientStatusResponse);
  // 获取系统健康
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
}

// ============================================================
// 消息定义
// ============================================================

// --- 客户端注册 ---

message RegisterRequest {
  string client_id = 1;
  string client_name = 2;        // e.g., "Factory-A-SMT-Line-3"
  string task_type = 3;          // e.g., "grasping", "navigation", "assembly"
  string robot_type = 4;         // e.g., "ur5e", "franka_panda", "custom"
  string environment = 5;        // e.g., "factory", "warehouse", "home"
  map<string, string> metadata = 6;
}

message RegisterResponse {
  bool success = 1;
  string message = 2;
  int64 round_id = 3;            // 当前训练轮次
  string global_model_version = 4;
}

message UnregisterRequest {
  string client_id = 1;
}

message UnregisterResponse {
  bool success = 1;
}

// --- 任务匹配 ---

message TaskReport {
  string client_id = 1;
  string task_type = 2;
  repeated float task_embedding = 3;  // 任务特征向量（由客户端计算）
  map<string, string> metadata = 4;
}

message TaskMatchResult {
  bool matched = 1;
  repeated TaskSimilarity similar_tasks = 2;
  float aggregation_weight = 3;       // 基于相似度的聚合权重
}

message TaskSimilarity {
  string client_id = 1;
  string task_type = 2;
  float similarity = 3;               // 0-1
  string model_compatible = 4;        // "full", "partial", "none"
}

// --- 模型交换 ---

message ModelRequest {
  string client_id = 1;
  string model_type = 2;              // "policy", "value", "world_model"
  string version = 3;                 // 留空=最新
}

message ModelResponse {
  bool success = 1;
  bytes model_weights = 2;
  string version = 3;
  int64 round_id = 4;
  int32 num_contributors = 5;
  map<string, float> contributor_weights = 6;  // 各贡献者权重
}

message UpdateRequest {
  string client_id = 1;
  bytes local_weights = 2;            // 本地训练后的模型权重
  int32 num_local_epochs = 3;
  int32 num_local_samples = 4;
  float local_loss = 5;
  float local_accuracy = 6;
  map<string, float> metrics = 7;     // 额外指标
  string model_type = 8;
}

message UpdateResponse {
  bool accepted = 1;
  string message = 2;
  float contribution_score = 3;       // 本次贡献分数
  float total_contribution = 4;       // 累计贡献分数
}

// --- 训练轮次 ---

message RoundStatusRequest {
  int64 round_id = 1;                 // 留空=当前轮
}

message RoundStatusResponse {
  int64 round_id = 1;
  string status = 2;                  // "waiting", "aggregating", "complete"
  int32 total_clients = 3;
  int32 reported_clients = 4;
  float global_loss = 5;
  float global_accuracy = 6;
  int64 started_at = 7;
  int64 completed_at = 8;
}

message AggregationRequest {
  string requester_id = 1;
  string strategy = 2;                // "fedavg", "task_weighted", "contribution_weighted"
}

message AggregationResponse {
  bool triggered = 1;
  int64 new_round_id = 2;
  string new_model_version = 3;
  int32 num_updates_aggregated = 4;
  float new_global_loss = 5;
}

// --- 任务注册 ---

message NewTaskRequest {
  string task_id = 1;
  string client_id = 2;
  string task_type = 3;               // "grasping", "navigation", "assembly", "inspection"
  string description = 4;
  repeated float task_embedding = 5;  // 任务特征向量
  map<string, string> config = 6;     // 任务配置参数
}

message TaskInfo {
  string task_id = 1;
  string client_id = 2;
  string task_type = 3;
  string description = 4;
  string status = 5;                  // "active", "paused", "completed"
  int64 created_at = 6;
  int64 updated_at = 7;
  int32 rounds_participated = 8;
  float total_contribution = 9;
  map<string, string> config = 10;
}

message SimilarTaskRequest {
  repeated float query_embedding = 1;
  int32 top_k = 2;                    // 默认5
  string task_type_filter = 3;        // 可选：按任务类型过滤
}

message SimilarTaskResponse {
  repeated TaskMatch matches = 1;
}

message TaskMatch {
  string task_id = 1;
  string client_id = 2;
  string task_type = 3;
  float similarity = 4;
  string model_compatible = 5;
}

message ListTasksRequest {
  string task_type = 1;               // 可选过滤
  string status = 2;                  // 可选过滤
}

message ListTasksResponse {
  repeated TaskInfo tasks = 1;
  int32 total = 2;
}

message UpdateTaskStatusRequest {
  string task_id = 1;
  string status = 2;
}

// --- 贡献追踪 ---

message ContributionQuery {
  string client_id = 1;
  int32 limit = 2;
}

message ContributionResponse {
  repeated ContributionEntry entries = 1;
  float total_score = 2;
}

message ContributionEntry {
  string client_id = 1;
  int64 round_id = 2;
  float score = 3;
  int32 num_samples = 4;
  float loss_improvement = 5;
  string timestamp = 6;
  string audit_hash = 7;
}

message LeaderboardRequest {
  int32 top_k = 1;                    // 默认10
  string task_type = 2;               // 可选过滤
}

message LeaderboardResponse {
  repeated LeaderboardEntry entries = 1;
}

message LeaderboardEntry {
  string client_id = 1;
  string client_name = 2;
  float total_contribution = 3;
  int32 rounds_participated = 4;
  int32 total_samples = 5;
  float avg_loss_improvement = 6;
}

message VerifyRequest {}

message VerifyResponse {
  bool valid = 1;
  int64 chain_length = 2;
  string latest_hash = 3;
}

// --- 监控 ---

message StatsRequest {}

message TrainingStatsResponse {
  int64 current_round = 1;
  int32 total_rounds_completed = 2;
  int32 active_clients = 3;
  int32 total_tasks = 4;
  float best_global_loss = 5;
  float best_global_accuracy = 6;
  repeated RoundHistory round_history = 7;
}

message RoundHistory {
  int64 round_id = 1;
  float loss = 2;
  float accuracy = 3;
  int32 participants = 4;
  int64 timestamp = 5;
}

message ClientStatusRequest {
  string client_id = 1;               // 留空=所有客户端
}

message ClientStatusResponse {
  repeated ClientStatus clients = 1;
}

message ClientStatus {
  string client_id = 1;
  string client_name = 2;
  string task_type = 3;
  string status = 4;                  // "online", "training", "offline"
  int32 rounds_participated = 5;
  float last_reported_loss = 6;
  string last_active = 7;
}

message HealthCheckRequest {}

message HealthCheckResponse {
  bool healthy = 1;
  string version = 2;
  int64 uptime_seconds = 3;
  int32 active_connections = 4;
  string message = 5;
}
