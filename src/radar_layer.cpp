#include "radar_layer/radar_layer.hpp"

#include "nav2_costmap_2d/costmap_math.hpp"
#include "nav2_costmap_2d/footprint.hpp"
#include "nav2_util/node_utils.hpp"
#include "rclcpp/parameter_events_filter.hpp"

namespace radar_layer
{
RadarLayer::RadarLayer()
{
  costmap_ = NULL;
}

RadarLayer::~RadarLayer()
{
}

void RadarLayer::onInitialize()
{
  double transform_tolerance;

  min_bound = std::numeric_limits<double>::lowest();
  max_bound = std::numeric_limits<double>::max();

  std::string topics_string;

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("combination_method", rclcpp::ParameterValue(1));
  declareParameter("observation_sources", rclcpp::ParameterValue(std::string("")));
  declareParameter("minimum_probability", rclcpp::ParameterValue(0.01));
  declareParameter("number_of_time_steps", rclcpp::ParameterValue(1));
  declareParameter("sample_time", rclcpp::ParameterValue(0.1));
  declareParameter("max_prediction_speed", rclcpp::ParameterValue(2.0));  // m/s
  declareParameter("stamp_footprint", rclcpp::ParameterValue(true));
  declareParameter("covariance_scaling_factor", rclcpp::ParameterValue(1.0));
  // ★ 추가: 입력 무소식 시 버퍼 비우는 TTL(초)
  declareParameter("observation_ttl", rclcpp::ParameterValue(0.8));  // 0.5~1.0 권장

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node"};
  }

  node->get_parameter(name_ + "." + "enabled", enabled_);
  node->get_parameter(name_ + "." + "combination_method", combination_method_);
  node->get_parameter(name_ + "." + "observation_sources", topics_string);
  node->get_parameter(name_ + "." + "minimum_probability", min_probability_);
  node->get_parameter(name_ + "." + "number_of_time_steps", number_of_time_steps_);
  node->get_parameter(name_ + "." + "sample_time", sample_time_);
  node->get_parameter(name_ + "." + "stamp_footprint", stamp_footprint_);
  node->get_parameter(name_ + "." + "covariance_scaling_factor", covariance_scaling_factor_);
  node->get_parameter("transform_tolerance", transform_tolerance);
  transform_tolerance_ = tf2::durationFromSec(transform_tolerance);

  rolling_window_ = layered_costmap_->isRolling();
  global_frame_ = layered_costmap_->getGlobalFrameID();
  default_value_ = nav2_costmap_2d::FREE_SPACE;
  current_ = true;

  RadarLayer::matchSize();

  auto sub_opt = rclcpp::SubscriptionOptions();
  sub_opt.callback_group = callback_group_;

  std::stringstream ss(topics_string);
  std::string source;

  while (ss >> source) {
    std::string topic;
    double qos_deadline_hz;

    declareParameter(source + "." + "topic", rclcpp::ParameterValue(source));
    declareParameter(source + "." + "qos_deadline_hz", rclcpp::ParameterValue(1.0));

    node->get_parameter(name_ + "." + source + "." + "topic", topic);
    node->get_parameter(name_ + "." + source + "." + "qos_deadline_hz", qos_deadline_hz);

    rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_sensor_data;
    custom_qos_profile.depth = 100;
    custom_qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;

    detection_buffers_.push_back(std::make_shared<nav2_dynamic_msgs::msg::ObstacleArray>());
    obstacle_buffers_.push_back(std::make_shared<nav2_dynamic_msgs::msg::ObstacleArray>());

    auto sub = std::make_shared<
      message_filters::Subscriber<nav2_dynamic_msgs::msg::ObstacleArray,
                                  rclcpp_lifecycle::LifecycleNode>>(
      node, topic, custom_qos_profile, sub_opt);
    sub->unsubscribe();

    auto filter =
      std::make_shared<tf2_ros::MessageFilter<nav2_dynamic_msgs::msg::ObstacleArray>>(
        *sub, *tf_, global_frame_, 200,
        node->get_node_logging_interface(),
        node->get_node_clock_interface(),
        transform_tolerance_);

    filter->registerCallback(
      std::bind(&RadarLayer::obstacleCallback, this,
                std::placeholders::_1, detection_buffers_.back(), obstacle_buffers_.back()));

    observation_subscribers_.push_back(sub);
    observation_notifiers_.push_back(filter);
    observation_notifiers_.back()->setTolerance(rclcpp::Duration::from_seconds(0.3));
  }

  RCLCPP_INFO(logger_, "RadarLayer::global_frame_ %s", global_frame_.c_str());
  dyn_params_handler_ = node->add_on_set_parameters_callback(
    std::bind(&RadarLayer::dynamicParametersCallback, this, std::placeholders::_1));
}



void RadarLayer::updateBounds(
  double robot_x, double robot_y, double robot_yaw,
  double* min_x, double* min_y, double* max_x, double* max_y)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  resetMaps();

  if (rolling_window_) {
    updateOrigin(robot_x - getSizeInMetersX() / 2, robot_y - getSizeInMetersY() / 2);
  }

  touch(max_bound, max_bound, min_x, min_y, max_x, max_y);
  touch(min_bound, min_bound, min_x, min_y, max_x, max_y);

  // TTL
  double ttl_sec = 0.8;
  if (auto node = node_.lock()) node->get_parameter(name_ + "." + "observation_ttl", ttl_sec);
  const rclcpp::Duration ttl = rclcpp::Duration::from_seconds(std::max(0.05, ttl_sec));

  const auto now   = clock_->now();
  const auto ctype = clock_->get_clock_type();

  // ★ 검출/트래킹 버퍼를 같은 인덱스로 함께 순회
  for (size_t s = 0; s < obstacle_buffers_.size(); ++s) {
    auto& obs_arr = obstacle_buffers_[s];
    auto& det_arr = detection_buffers_[s];

    // 유효성(TTL) 체크: 트래킹 기준으로 우선 판단
    const rclcpp::Time obs_stamp(obs_arr->header.stamp, ctype);
    const bool obs_empty  = (obs_arr->obstacles.empty());
    const bool obs_stale  = (obs_arr->header.stamp.sec == 0 && obs_arr->header.stamp.nanosec == 0)
                            || ((now - obs_stamp) > ttl);

    if (obs_empty || obs_stale) {
      obs_arr->obstacles.clear();
      continue;
    }

    const int n_objs = static_cast<int>(obs_arr->obstacles.size());
    if (n_objs <= 0) continue;

    if (stamp_footprint_) {
      stampFootprint(obs_arr, n_objs);
    } else {
      // ★ 씨앗은 "검출", 퍼짐은 "트래킹"을 사용
      predictiveCostSeeded(obs_arr, det_arr, n_objs);
    }
  }
}

void RadarLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i,
  int min_j,
  int max_i,
  int max_j)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());

  if (!enabled_) {
    return;
  }

  switch (combination_method_) {
    case 0: // Overwrite
      updateWithOverwrite(master_grid, min_i, min_j, max_i, max_j);
      break;
    case 1: // Maximum
      updateWithMax(master_grid, min_i, min_j, max_i, max_j);
      break;
    default: // Nothing
      break;
  }

  current_ = true;
}

void RadarLayer::activate()
{
  for (auto & notifier : observation_notifiers_) {
    notifier->clear();
  }

  // if we're stopped we need to re-subscribe to topics
  for (unsigned int i = 0; i < observation_subscribers_.size(); ++i) {
    if (observation_subscribers_[i] != NULL) {
      observation_subscribers_[i]->subscribe();
    }
  }
}

void RadarLayer::deactivate()
{
  for (unsigned int i = 0; i < observation_subscribers_.size(); ++i) {
    if (observation_subscribers_[i] != NULL) {
      observation_subscribers_[i]->unsubscribe();
    }
  }
}

void RadarLayer::reset()
{
  resetMaps();
}

rcl_interfaces::msg::SetParametersResult RadarLayer::dynamicParametersCallback(
  std::vector<rclcpp::Parameter> parameters)
{
  std::lock_guard<Costmap2D::mutex_t> guard(*getMutex());
  rcl_interfaces::msg::SetParametersResult result;

  for (auto parameter : parameters) {
    const auto & param_type = parameter.get_type();
    const auto & param_name = parameter.get_name();

    if (param_type == ParameterType::PARAMETER_INTEGER) {
      if (param_name == name_ + "." + "combination_method" &&
        combination_method_ != parameter.as_int())
      {
        combination_method_ = parameter.as_int();
      }
      if (param_name == name_ + "." + "number_of_time_steps" &&
        number_of_time_steps_ != parameter.as_int())
      {
        number_of_time_steps_ = parameter.as_int();
      }
    }
    if (param_type == ParameterType::PARAMETER_DOUBLE) {
      if (param_name == name_ + "." + "covariance_scaling_factor" &&
        covariance_scaling_factor_ != parameter.as_double())
      {
        covariance_scaling_factor_ = parameter.as_double();
      }
      if (param_name == name_ + "." + "sample_time" &&
        sample_time_ != parameter.as_double())
      {
        sample_time_ = parameter.as_double();
      }
      if (param_name == name_ + "." + "minimum_probability" &&
        min_probability_ != parameter.as_double())
      {
        min_probability_ = parameter.as_double();
      }
    }
  }

  result.successful = true;
  return result;
}

void RadarLayer::populateGrid(
  double x_0,
  double y_0,
  double length,
  double width,
  std::vector<geometry_msgs::msg::PointStamped> & points_in_obstacle_frame,
  Eigen::MatrixXd & xs,
  Eigen::MatrixXd & ys,
  std::vector<int> & x_index,
  std::vector<int> & y_index,
  nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr obstacle_array)
{

  int length_in_grid = int(length / resolution_);
  int width_in_grid = int(width / resolution_);

  unsigned int point_in_obstacle_frame_index = 0;
  for (int x_i = 0; x_i < length_in_grid; x_i++) {
    for (int y_i = 0; y_i < width_in_grid; y_i++, point_in_obstacle_frame_index++) {

      double dx = -length / 2 + x_i * resolution_;
      double dy = -width / 2 + y_i * resolution_;

      xs(x_i, y_i) = x_0 + dx;
      ys(x_i, y_i) = y_0 + dy;

      points_in_obstacle_frame[point_in_obstacle_frame_index].header.stamp =
        obstacle_array->header.stamp;
      points_in_obstacle_frame[point_in_obstacle_frame_index].header.frame_id =
        obstacle_array->header.frame_id;
      points_in_obstacle_frame[point_in_obstacle_frame_index].point.x = xs(x_i, y_i);
      points_in_obstacle_frame[point_in_obstacle_frame_index].point.y = ys(x_i, y_i);
      points_in_obstacle_frame[point_in_obstacle_frame_index].point.z = 0;
      x_index[point_in_obstacle_frame_index] = x_i;
      y_index[point_in_obstacle_frame_index] = y_i;
    }
  }
}


void RadarLayer::predictiveCost(
  nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr obstacle_array,
  int number_of_objects)
{
  auto clk = clock_->get_clock_type();
  rclcpp::Time stamp(obstacle_array->header.stamp, clk);

  // 지연 보정 → 너무 큰 값은 억제
  double latency = (clock_->now() - stamp).seconds();
  latency = std::clamp(latency, 0.0, 0.5);
  const int latency_steps =
      static_cast<int>(std::round(latency / std::max(0.01, sample_time_)));

  const double eps = 1e-9;

  // 좌표 변환 필요 여부(예측 모드에선 직접 처리)
  const bool need_tf = (obstacle_array->header.frame_id != global_frame_);
  double dx = 0, dy = 0, x_x = 1, x_y = 0, y_x = 0, y_y = 1;
  double inv_x_x, inv_x_y, inv_y_x, inv_y_y; // 사용 안함(계산만 일관성 유지)
  if (need_tf) {
    getTransformCoefficients(
      /*source_frame=*/global_frame_, /*target_frame=*/obstacle_array->header.frame_id,
      stamp, dx, dy, x_x, x_y, y_x, y_y, inv_x_x, inv_x_y, inv_y_x, inv_y_y);
  }

  // 디버그 한번만
  RCLCPP_INFO_ONCE(
      logger_, "predictiveCost(): input frame=%s, global=%s",
      obstacle_array->header.frame_id.c_str(), global_frame_.c_str());

  for (size_t i = 0; i < static_cast<size_t>(number_of_objects); i++) {
    nav2_dynamic_msgs::msg::Obstacle obstacle_0 = obstacle_array->obstacles[i];

    // 초기 분산(너무 작거나 나쁘면 skip)
    double cov0_x = obstacle_0.position_covariance[0] + std::max(0.0, obstacle_0.size.x / 2.0);
    double cov0_y = obstacle_0.position_covariance[4] + std::max(0.0, obstacle_0.size.y / 2.0);
    cov0_x = std::clamp(cov0_x, eps, 10.0);
    cov0_y = std::clamp(cov0_y, eps, 10.0);

    double sqrt_2_pi_det_covariance_0 = std::sqrt(2.0 * M_PI * cov0_x * cov0_y);
    if (!(std::isfinite(sqrt_2_pi_det_covariance_0) && sqrt_2_pi_det_covariance_0 > eps)) {
      continue;
    }

    // 최대 예측 폭 기준 공분산(캐시 크기 산정용)
    Eigen::Matrix2d covariance = projectCovariance(
        obstacle_0, sample_time_, number_of_time_steps_ + latency_steps);
    covariance(0, 0) = std::clamp(covariance(0, 0) + obstacle_0.size.x / 2.0, eps, 20.0);
    covariance(1, 1) = std::clamp(covariance(1, 1) + obstacle_0.size.y / 2.0, eps, 20.0);

    Eigen::Matrix2d inv_covariance = Eigen::Matrix2d::Zero();
    inv_covariance(0, 0) = 1.0 / covariance(0, 0);
    inv_covariance(1, 1) = 1.0 / covariance(1, 1);

    double sqrt_2_pi_det_covariance =
        std::sqrt(2.0 * M_PI * covariance(0, 0) * covariance(1, 1));
    if (!(std::isfinite(sqrt_2_pi_det_covariance) && sqrt_2_pi_det_covariance > eps)) {
      continue;
    }
    double inv_sqrt_2_pi_det_covariance = 1.0 / sqrt_2_pi_det_covariance;
    double covariance_ratio = sqrt_2_pi_det_covariance_0 / sqrt_2_pi_det_covariance;

    // 반경 계산 (문턱값)
    const double safe_min_prob = std::min(std::max(min_probability_, 1e-6), 0.999999);
    const double safe_ratio = std::max(covariance_ratio, eps);
    auto axis_extent_m = [&](double cov_ii) -> double {
      double inner = -2.0 * (std::log(safe_min_prob) - std::log(safe_ratio)) * cov_ii;
      if (!std::isfinite(inner) || inner <= 0.0) return 0.0;
      return 2.0 * std::sqrt(inner);
    };
    double length = axis_extent_m(covariance(0, 0));
    double width  = axis_extent_m(covariance(1, 1));

    int length_in_grid = static_cast<int>(std::floor(length / resolution_));
    int width_in_grid  = static_cast<int>(std::floor(width  / resolution_));
    const int max_r = std::max(1, static_cast<int>(std::min(size_x_, size_y_) / 4) - 2);
    length_in_grid = std::clamp(length_in_grid, 0, max_r);
    width_in_grid  = std::clamp(width_in_grid,  0, max_r);
    mean_inflation_radius_ = std::max(length_in_grid, width_in_grid);
    int max_dist = generateIntegerDistances();

    // 시간 전개
    for (int k = 0; k < number_of_time_steps_; ++k) {
      if (seen_.size() != size_x_ * size_y_) {
        seen_ = std::vector<bool>(size_x_ * size_y_, false);
      }
      std::fill(begin(seen_), end(seen_), false);

      // 예측된 평균 (여기선 '검출 위치'를 요구사항대로 그대로 사용)
      Eigen::Vector2d mean = projectMean(obstacle_0, sample_time_, k);

      // 필요시 2D 변환(검출 → global_frame_)
      double mean_x = mean(0), mean_y = mean(1);
      if (need_tf) {
        const double tx = mean_x * x_x + mean_y * y_x + dx;
        const double ty = mean_x * x_y + mean_y * y_y + dy;
        mean_x = tx; mean_y = ty;
      }

      // 각 step의 공분산/캐시 갱신
      covariance = projectCovariance(obstacle_0, sample_time_, k);
      covariance(0, 0) = std::clamp(covariance(0, 0) + obstacle_0.size.x / 2.0, eps, 20.0);
      covariance(1, 1) = std::clamp(covariance(1, 1) + obstacle_0.size.y / 2.0, eps, 20.0);
      inv_covariance(0, 0) = 1.0 / covariance(0, 0);
      inv_covariance(1, 1) = 1.0 / covariance(1, 1);

      sqrt_2_pi_det_covariance =
          std::sqrt(2.0 * M_PI * covariance(0, 0) * covariance(1, 1));
      if (!(std::isfinite(sqrt_2_pi_det_covariance) && sqrt_2_pi_det_covariance > eps)) {
        continue;
      }
      inv_sqrt_2_pi_det_covariance = 1.0 / sqrt_2_pi_det_covariance;

      const double scaled_s0 =
          sqrt_2_pi_det_covariance_0 * covariance_scaling_factor_;

      inflation_cells_.clear();
      inflation_cells_.resize(max_dist + 1);
      computeCacheCosts(inv_covariance, inv_sqrt_2_pi_det_covariance, const_cast<double&>(sqrt_2_pi_det_covariance_0));

      // ── ★ 씨앗(검출 좌표) 강제 스탬프 ─────────────────────────────
      unsigned int cx, cy;
      if (!worldToMap(mean_x, mean_y, cx, cy)) {
        RCLCPP_DEBUG(logger_, "worldToMap failed at (%.3f, %.3f) → skip", mean_x, mean_y);
        continue;
      }
      const unsigned int cidx = getIndex(cx, cy);
      costmap_[cidx] = std::max<unsigned char>(costmap_[cidx], nav2_costmap_2d::LETHAL_OBSTACLE);

      // 전파 시작점 큐에 넣기
      auto & bin0 = inflation_cells_[0];
      bin0.reserve(16);
      bin0.emplace_back(cx, cy, cx, cy);

      const int r_guard = mean_inflation_radius_ + 2;

      // ── 인플레이션 전파 ─────────────────────────────────────────
      for (auto & dist_bin : inflation_cells_) {
        for (std::size_t idx = 0; idx < dist_bin.size(); ++idx) {
          const CellData & cell = dist_bin[idx];
          const unsigned int mx = cell.x_;
          const unsigned int my = cell.y_;
          const unsigned int sx = cell.src_x_;
          const unsigned int sy = cell.src_y_;
          const unsigned int map_index = getIndex(mx, my);

          if (seen_[map_index]) continue;
          seen_[map_index] = true;

          const int dxg = static_cast<int>(mx) - static_cast<int>(sx);
          const int dyg = static_cast<int>(my) - static_cast<int>(sy);
          if (std::abs(dxg) > r_guard || std::abs(dyg) > r_guard ||
              (dxg*dxg + dyg*dyg > r_guard*r_guard)) {
            continue;
          }

          const bool is_center = (mx == sx) && (my == sy);

          // 중심은 무조건 유지, 주변은 확률 문턱 적용
          unsigned char cell_cost = is_center
              ? nav2_costmap_2d::LETHAL_OBSTACLE
              : cost_matrix_[std::abs(dxg)][std::abs(dyg)];
          const double prob = is_center
              ? 1.0
              : static_cast<double>(cell_cost) / (252.0 * std::max(scaled_s0, eps));

          if (is_center || prob > safe_min_prob) {
            costmap_[map_index] = std::max(costmap_[map_index], cell_cost);
            // 이웃 enqueue
            if (mx > 0)           enqueue(map_index - 1,        mx - 1, my,     sx, sy);
            if (my > 0)           enqueue(map_index - size_x_,  mx,     my - 1, sx, sy);
            if (mx < size_x_ - 1) enqueue(map_index + 1,        mx + 1, my,     sx, sy);
            if (my < size_y_ - 1) enqueue(map_index + size_x_,  mx,     my + 1, sx, sy);
          }
        }
        dist_bin = std::vector<CellData>(); // 메모리 회수
      }
    }
  }
}



int RadarLayer::generateIntegerDistances()
{
  const int r = mean_inflation_radius_ + 2;
  const int size = r * 2 + 1;

  std::vector<std::pair<int, int>> points;

  for (int y = -r; y <= r; y++) {
    for (int x = -r; x <= r; x++) {
      if (x * x + y * y <= r * r) {
        points.emplace_back(x, y);
      }
    }
  }

  std::sort(
    points.begin(), points.end(),
    [](const std::pair<int, int> & a, const std::pair<int, int> & b) -> bool
    {
      return a.first * a.first + a.second * a.second < b.first * b.first + b.second * b.second;
    });

  std::vector<std::vector<int>> distance_matrix(size, std::vector<int>(size, 0));
  std::pair<int, int> last = {0, 0};
  int level = 0;
  for (auto const & p : points) {

    if (p.first * p.first + p.second * p.second !=
      last.first * last.first + last.second * last.second)
    {
      level++;
    }

    distance_matrix[p.first + r][p.second + r] = level;
    last = p;
  }

  distance_matrix_ = distance_matrix;
  return level;
}

void RadarLayer::computeCacheCosts(
  Eigen::MatrixXd inv_covariance,
  double & inv_sqrt_2_pi_det_covariance,
  double & sqrt_2_pi_det_covariance_0)
{
  const int r = mean_inflation_radius_ + 2;

  std::vector<std::pair<int, int>> points;

  for (int y = 0; y <= r; y++) {
    for (int x = 0; x <= r; x++) {
      if (x * x + y * y <= r * r) {
        points.emplace_back(x, y);
      }
    }
  }

  std::vector<std::vector<unsigned char>> cost_matrix(r + 1, std::vector<unsigned char>(r + 1, 0));
  Eigen::Vector2d mean(0.0, 0.0);
  Eigen::VectorXd xs(points.size());
  Eigen::VectorXd ys(points.size());

  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto & p = points[i];
    double x = p.first * resolution_;
    double y = p.second * resolution_;

    xs(i) = x;
    ys(i) = y;
  }

  Eigen::MatrixXd probabilities = getProbabilityBatch(
    mean, inv_covariance,
    inv_sqrt_2_pi_det_covariance, xs, ys);

  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto & p = points[i];
    cost_matrix[p.first][p.second] = uint8_t(252 * sqrt_2_pi_det_covariance_0 * probabilities(i));
  }

  cost_matrix_ = cost_matrix;

  return;
}

// void RadarLayer::enqueue(
//   unsigned int index, unsigned int mx, unsigned int my,
//   unsigned int src_x, unsigned int src_y)
// {
//   const unsigned int r = mean_inflation_radius_ + 2;
//   const auto dist = distance_matrix_[mx - src_x + r][my - src_y + r];
//   inflation_cells_[dist].emplace_back(mx, my, src_x, src_y);
// }

void RadarLayer::enqueue(
  unsigned int index, unsigned int mx, unsigned int my,
  unsigned int src_x, unsigned int src_y)
{
  const int r = static_cast<int>(mean_inflation_radius_) + 2;

  // 상대 오프셋 (signed)
  const int dx = static_cast<int>(mx) - static_cast<int>(src_x);
  const int dy = static_cast<int>(my) - static_cast<int>(src_y);

  // ▶ 사전 가드: 반경 원판 밖이면 큐에 넣지 않음 (OOB + 논리 오류 방지)
  if ((std::abs(dx) > r) || (std::abs(dy) > r) || (dx*dx + dy*dy > r*r)) {
    return;
  }

  const unsigned int ix = static_cast<unsigned int>(dx + r);
  const unsigned int iy = static_cast<unsigned int>(dy + r);

  // ▶ distance_matrix_도 범위 가드
  if (ix >= distance_matrix_.size() || iy >= distance_matrix_[ix].size()) {
    return;
  }

  const auto dist = distance_matrix_[ix][iy];
  if (static_cast<size_t>(dist) >= inflation_cells_.size()) {
    return;
  }

  inflation_cells_[dist].emplace_back(mx, my, src_x, src_y);
}


void RadarLayer::stampFootprint(
  nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr obstacle_array, int number_of_objects)
{

  geometry_msgs::msg::PointStamped point_in_global_frame;
  geometry_msgs::msg::TransformStamped radar_to_global_transform;
  double dx;
  double dy;
  double x_x;
  double x_y;
  double y_x;
  double y_y;
  double inv_x_x;
  double inv_x_y;
  double inv_y_x;
  double inv_y_y;

  auto clk = clock_->get_clock_type();
  rclcpp::Time stamp(obstacle_array->header.stamp, clk);
  getTransformCoefficients(
    global_frame_, obstacle_array->header.frame_id, stamp,
    dx, dy, x_x, x_y, y_x, y_y, inv_x_x, inv_x_y, inv_y_x, inv_y_y);

  for (size_t i = 0; i < number_of_objects; i++) {
    double length = obstacle_array->obstacles[i].size.x;
    double width = obstacle_array->obstacles[i].size.y;

    int length_in_grid = int(length / resolution_);
    int width_in_grid = int(width / resolution_);

    Eigen::MatrixXd xs(length_in_grid, width_in_grid);
    Eigen::MatrixXd ys(length_in_grid, width_in_grid);
    std::vector<geometry_msgs::msg::PointStamped> points_in_obstacle_frame(length_in_grid *
      width_in_grid);
    std::vector<geometry_msgs::msg::PointStamped> points_in_global_frame(length_in_grid *
      width_in_grid);
    std::vector<int> x_index(length_in_grid * width_in_grid);
    std::vector<int> y_index(length_in_grid * width_in_grid);

    populateGrid(
      obstacle_array->obstacles[i].position.x, obstacle_array->obstacles[i].position.y,
      length, width, points_in_obstacle_frame, xs, ys, x_index, y_index,
      obstacle_array);

    bool batch_transform_success = batchTransform2DPoints(
      x_x, x_y, y_x, y_y, dx, dy,
      points_in_obstacle_frame,
      points_in_global_frame, global_frame_,
      transform_tolerance_);

    std::vector<int> cells_to_inflate_x;
    std::vector<int> cells_to_inflate_y;

    if (batch_transform_success) {
      for (size_t i = 0; i < points_in_global_frame.size(); i++) {
        unsigned int mx, my;

        if (worldToMap(
            points_in_global_frame[i].point.x, points_in_global_frame[i].point.y, mx,
            my))
        {
          unsigned int index = getIndex(mx, my);
          costmap_[index] = LETHAL_OBSTACLE;
        }
      }
    }
  }
}

void RadarLayer::getTransformCoefficients(
  std::string source_frame,
  std::string target_frame,
  const rclcpp::Time & stamp,
  double & dx, double & dy, double & x_x, double & x_y, double & y_x,
  double & y_y, double & inv_x_x, double & inv_x_y, double & inv_y_x, double & inv_y_y)
{
  // 동일 프레임이면 항등
  if (source_frame == target_frame) {
    dx = 0.0; dy = 0.0;
    x_x = 1.0; x_y = 0.0;
    y_x = 0.0; y_y = 1.0;
    inv_x_x = 1.0; inv_x_y = 0.0;
    inv_y_x = 0.0; inv_y_y = 1.0;
    return;
  }

  geometry_msgs::msg::TransformStamped transform;

  try {
    // ★ 메시지 시각으로 TF 조회 (최대 0.3s 대기)
    transform = tf_->lookupTransform(
      source_frame, target_frame, stamp, rclcpp::Duration::from_seconds(0.3));
  } catch (tf2::TransformException & ex) {
    // 마지막 수단: 최신 TF로라도 시도 (짧게 대기)
    try {
      transform = tf_->lookupTransform(
        source_frame, target_frame, rclcpp::Time(0, 0, stamp.get_clock_type()),
        rclcpp::Duration::from_seconds(0.1));
    } catch (tf2::TransformException & ex2) {
      RCLCPP_ERROR(logger_, "TF lookup failed (%s->%s): %s / fallback: %s",
                   source_frame.c_str(), target_frame.c_str(), ex.what(), ex2.what());
      // 실패 시 항등으로 반환 (보수적)
      dx = 0.0; dy = 0.0;
      x_x = 1.0; x_y = 0.0;
      y_x = 0.0; y_y = 1.0;
      inv_x_x = 1.0; inv_x_y = 0.0;
      inv_y_x = 0.0; inv_y_y = 1.0;
      return;
    }
  }

  tf2::Quaternion q(
    transform.transform.rotation.x,
    transform.transform.rotation.y,
    transform.transform.rotation.z,
    transform.transform.rotation.w);

  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  const double cos_yaw = std::cos(yaw);
  const double sin_yaw = std::sin(yaw);
  const double cos_pitch = std::cos(pitch);
  const double sin_pitch = std::sin(pitch);
  const double cos_roll = std::cos(roll);
  const double sin_roll = std::sin(roll);

  dx = transform.transform.translation.x;
  dy = transform.transform.translation.y;

  // 원래 코드의 2D 투영 계수 유지
  x_x = cos_yaw * cos_pitch;
  x_y = sin_yaw * cos_pitch;

  y_x = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
  y_y = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;

  Eigen::MatrixXd rotation_matrix = Eigen::MatrixXd::Zero(2, 2);
  rotation_matrix(0, 0) = x_x;
  rotation_matrix(0, 1) = x_y;
  rotation_matrix(1, 0) = y_x;
  rotation_matrix(1, 1) = y_y;

  Eigen::MatrixXd inverse_rotation_matrix = rotation_matrix.inverse();
  inv_x_x = inverse_rotation_matrix(0, 0);
  inv_x_y = inverse_rotation_matrix(0, 1);
  inv_y_x = inverse_rotation_matrix(1, 0);
  inv_y_y = inverse_rotation_matrix(1, 1);
}

void RadarLayer::obstacleCallback(
  nav2_dynamic_msgs::msg::ObstacleArray::ConstSharedPtr message,
  const std::shared_ptr<nav2_dynamic_msgs::msg::ObstacleArray> & detection_buffer,
  const std::shared_ptr<nav2_dynamic_msgs::msg::ObstacleArray> & obstacle_buffer)
{
  *detection_buffer = *message;
  if (stamp_footprint_) {
    *obstacle_buffer = *detection_buffer;
  } else {
    findUuid(obstacle_buffer, detection_buffer);
  }
}


void RadarLayer::findUuid(
  const nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr obstacles,
  const nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr detections)
{
  int number_of_obstacles = obstacles->obstacles.size();
  int number_of_detections = detections->obstacles.size();
  std::vector<std::pair<size_t, size_t>> matched_indices;

  // ★ costmap 노드 clock 타입으로 통일
  auto clk = clock_->get_clock_type();
  rclcpp::Time detection_time(detections->header.stamp, clk);
  rclcpp::Time obstacle_time(obstacles->header.stamp,  clk);
  rclcpp::Duration duration = detection_time - obstacle_time;
  double dt = duration.seconds() + duration.nanoseconds() / 1e9;

  if (number_of_detections > 0) {
    if (number_of_obstacles == 0) {
      *obstacles = *detections;
    } else {
      obstacles->header.stamp = detections->header.stamp;
      for (size_t i = 0; i < static_cast<size_t>(number_of_detections); i++) {
        for (size_t j = 0; j < static_cast<size_t>(number_of_obstacles); j++) {
          if (memcmp(detections->obstacles[i].uuid.uuid.data(),
                     obstacles->obstacles[j].uuid.uuid.data(), 16) == 0)
          {
            RCLCPP_DEBUG(logger_, "Matching UUIDs Found");
            updateGaussian(obstacles->obstacles[j], detections->obstacles[i], dt);
            matched_indices.push_back({i, j});
            break;
          }
        }
      }
      removeUnmatchedObstacles(number_of_obstacles, matched_indices, obstacles);
      addUnmatchedDetections(number_of_detections, matched_indices, obstacles, detections);
    }
  }
}


void RadarLayer::removeUnmatchedObstacles(
  int number_of_obstacles,
  std::vector<std::pair<size_t, size_t>> matched_indices,
  const nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr obstacles)
{
  // ★ 연속 미검출 허용 프레임 수
  static const int kMaxMisses = 3;

  // ★ uuid별 누락 카운터(헤더 수정 없이 function-static 사용)
  static std::unordered_map<std::string, int> miss_counts;

  // 매칭된 obstacle index 모음
  std::vector<bool> matched(number_of_obstacles, false);
  for (auto &p : matched_indices) {
    size_t j = p.second; // obstacles 쪽 인덱스
    if (j < matched.size()) matched[j] = true;
  }

  // 지울 인덱스 모으기
  std::vector<size_t> to_erase;

  for (int j = 0; j < number_of_obstacles; ++j) {
    const auto &uuid = obstacles->obstacles[j].uuid.uuid;
    const std::string key(reinterpret_cast<const char*>(uuid.data()), uuid.size());

    if (matched[j]) {
      miss_counts[key] = 0;  // 보였으면 리셋
    } else {
      int &cnt = miss_counts[key];
      cnt += 1;
      if (cnt >= kMaxMisses) {
        to_erase.push_back(static_cast<size_t>(j));
        miss_counts.erase(key);
      }
    }
  }

  if (!to_erase.empty()) {
    std::sort(to_erase.begin(), to_erase.end(), std::greater<size_t>());
    for (size_t idx : to_erase) {
      obstacles->obstacles.erase(obstacles->obstacles.begin() + idx);
    }
  }
}


void RadarLayer::addUnmatchedDetections(
  int number_of_detections,
  std::vector<std::pair<size_t, size_t>> matched_indices,
  const nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr obstacles,
  const nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr detections)
{

  std::vector<size_t> unmatched_detections = findUnmatchedIndices(
    number_of_detections,
    matched_indices, true);

  if (unmatched_detections.size() > 0) {
    RCLCPP_DEBUG(logger_, "Unmatched Detections Found");
    for (size_t i = 0; i < unmatched_detections.size(); i++) {
      obstacles->obstacles.push_back(detections->obstacles[unmatched_detections[i]]);
    }
  }
}

void RadarLayer::updateGaussian(
  nav2_dynamic_msgs::msg::Obstacle & obstacle,
  const nav2_dynamic_msgs::msg::Obstacle & detection,
  double dt)
{

  Eigen::Vector2d obstacle_position_mean(obstacle.position.x, obstacle.position.y);
  Eigen::Vector2d obstacle_velocity_mean(obstacle.velocity.x, obstacle.velocity.y);

  Eigen::Vector2d detection_position_mean(detection.position.x, detection.position.y);
  Eigen::Vector2d detection_velocity_mean(detection.velocity.x, detection.velocity.y);

  Eigen::Matrix2d obstacle_position_covariance = Eigen::MatrixXd::Zero(2, 2);
  obstacle_position_covariance(0, 0) = obstacle.position_covariance[0];   //*covariance_scaling_factor_;
  obstacle_position_covariance(1, 1) = obstacle.position_covariance[4];   //*covariance_scaling_factor_;

  Eigen::Matrix2d obstacle_position_covariance_inv = Eigen::MatrixXd::Zero(2, 2);
  obstacle_position_covariance_inv(0, 0) = 1 / obstacle.position_covariance[0];   //*covariance_scaling_factor_;
  obstacle_position_covariance_inv(1, 1) = 1 / obstacle.position_covariance[4];   //*covariance_scaling_factor_;

  Eigen::Matrix2d obstacle_velocity_covariance = Eigen::MatrixXd::Zero(2, 2);
  obstacle_velocity_covariance(0, 0) = obstacle.velocity_covariance[0];
  obstacle_velocity_covariance(1, 1) = obstacle.velocity_covariance[4];

  Eigen::Matrix2d obstacle_velocity_covariance_inv = Eigen::MatrixXd::Zero(2, 2);
  obstacle_velocity_covariance_inv(0, 0) = 1 / obstacle.velocity_covariance[0];
  obstacle_velocity_covariance_inv(1, 1) = 1 / obstacle.velocity_covariance[4];

  Eigen::Matrix2d detection_position_covariance = Eigen::MatrixXd::Zero(2, 2);
  detection_position_covariance(0, 0) = detection.position_covariance[0];   //*covariance_scaling_factor_;
  detection_position_covariance(1, 1) = detection.position_covariance[4];   // #*covariance_scaling_factor_;

  Eigen::Matrix2d detection_position_covariance_inv = Eigen::MatrixXd::Zero(2, 2);
  detection_position_covariance_inv(0, 0) = 1 / detection_position_covariance(0, 0);
  detection_position_covariance_inv(1, 1) = 1 / detection_position_covariance(1, 1);

  Eigen::Matrix2d detection_velocity_covariance = Eigen::MatrixXd::Zero(2, 2);
  detection_velocity_covariance(0, 0) = detection.velocity_covariance[0];
  detection_velocity_covariance(1, 1) = detection.velocity_covariance[4];

  Eigen::MatrixXd detection_velocity_covariance_inv = Eigen::MatrixXd::Zero(2, 2);
  detection_velocity_covariance_inv(0, 0) = 1 / detection_velocity_covariance(0, 0);
  detection_velocity_covariance_inv(1, 1) = 1 / detection_velocity_covariance(1, 1);

  Eigen::Matrix2d R;
  R(
    0,
    0) = 1.0 /
    (obstacle_position_covariance(0, 0) + dt * dt * obstacle_velocity_covariance(0, 0));
  R(
    1,
    1) = 1.0 /
    (obstacle_position_covariance(1, 1) + dt * dt * obstacle_velocity_covariance(1, 1));
  // R(0,0) =
  //   (obstacle_position_covariance + dt * dt * obstacle_velocity_covariance).inverse();
  Eigen::Matrix2d new_position_covariance = (R + detection_position_covariance_inv).inverse();
  Eigen::Vector2d new_position_mean = new_position_covariance *
    (R * (obstacle_position_mean + dt * obstacle_velocity_mean) +
    detection_position_covariance_inv * detection_position_mean);

  Eigen::Matrix2d new_velocity_covariance =
    (obstacle_velocity_covariance_inv + detection_velocity_covariance_inv).inverse();
  Eigen::Vector2d new_velocity_mean = new_velocity_covariance *
    (obstacle_velocity_covariance_inv * obstacle_velocity_mean +
    detection_velocity_covariance_inv * detection_velocity_mean);

  obstacle.position.x = new_position_mean(0);
  obstacle.position.y = new_position_mean(1);
  obstacle.velocity.x = new_velocity_mean(0);
  obstacle.velocity.y = new_velocity_mean(1);
  obstacle.position_covariance[0] = new_position_covariance(0, 0);
  obstacle.position_covariance[4] = new_position_covariance(1, 1);
  obstacle.velocity_covariance[0] = new_velocity_covariance(0, 0);
  obstacle.velocity_covariance[4] = new_velocity_covariance(1, 1);
}

Eigen::MatrixXd RadarLayer::getProbabilityBatch(
  const Eigen::Vector2d & mean,
  const Eigen::Matrix2d & inv_covariance,
  double inv_sqrt_2_pi_det_covariance,
  const Eigen::VectorXd & xs,
  const Eigen::VectorXd & ys)
{
  // Stack these vectors into a single 2-row matrix (inputs)
  Eigen::MatrixXd inputs(2, xs.size());
  inputs.row(0) = xs;
  inputs.row(1) = ys;

  // Subtract the mean from each column (broadcasting)
  Eigen::MatrixXd differences = inputs.colwise() - mean;

  // Calculate the Mahalanobis distance for each point
  Eigen::MatrixXd mahalanobis =
    (inv_covariance * differences).cwiseProduct(differences).colwise().sum();

  // Compute probabilities using the Gaussian formula
  Eigen::MatrixXd probabilities = (inv_sqrt_2_pi_det_covariance) *
    (-0.5 * mahalanobis.array()).exp();

  return probabilities;
}

inline double RadarLayer::getProbability(
  const Eigen::Matrix2d & inv_covariance,
  double & inv_sqrt_2_pi_det_covariance,
  const Eigen::Vector2d & mean,
  double & x, double & y)
{
  Eigen::Vector2d input(x, y);
  Eigen::Vector2d difference = input - mean;
  double b = inv_covariance(0, 0) * difference(0) * difference(0) +
    inv_covariance(1, 1) * difference(1) * difference(1) +
    2 * inv_covariance(0, 1) * difference(0) * difference(1);

  double probability = (inv_sqrt_2_pi_det_covariance) * exp(-b / 2);

  return probability;
}

std::vector<size_t> RadarLayer::findUnmatchedIndices(
  size_t number_of_elements,
  const std::vector<std::pair<size_t, size_t>> & matched_indices,
  bool check_first_index)
{

  std::set<size_t> matched_set;
  for (const auto & pair : matched_indices) {
    matched_set.insert(check_first_index ? pair.first : pair.second);
  }

  std::vector<size_t> unmatched_indices;
  for (size_t index = 0; index < number_of_elements; ++index) {
    if (matched_set.find(index) == matched_set.end()) {
      unmatched_indices.push_back(index);
    }
  }

  return unmatched_indices;
}

bool RadarLayer::batchTransform2DPoints(
  double x_x,
  double x_y,
  double y_x,
  double y_y,
  double dx,
  double dy,
  const std::vector<geometry_msgs::msg::PointStamped> & input_points,
  std::vector<geometry_msgs::msg::PointStamped> & output_points,
  const std::string & target_frame,
  const tf2::Duration & timeout) const
{
  bool all_transformed = true;

  for (size_t i = 0; i < input_points.size(); i++) {
    output_points[i].header.stamp = input_points[i].header.stamp;
    output_points[i].header.frame_id = target_frame;
    output_points[i].point.x = input_points[i].point.x * x_x + input_points[i].point.y * y_x + dx;
    output_points[i].point.y = input_points[i].point.x * x_y + input_points[i].point.y * y_y + dy;
    output_points[i].point.z = 0;
  }

  return all_transformed;
}

bool RadarLayer::batchTransformPoints(
  const std::vector<geometry_msgs::msg::PointStamped> & input_points,
  std::vector<geometry_msgs::msg::PointStamped> & output_points,
  const std::string & target_frame,
  const tf2::Duration & timeout) const
{
  bool all_transformed = true;
  for (size_t i = 0; i < input_points.size(); i++) {
    try {
      tf_->transform(input_points[i], output_points[i], target_frame, timeout);
      output_points[i].header.stamp = input_points[i].header.stamp;
    } catch (tf2::TransformException & ex) {
      RCLCPP_ERROR(logger_, "Exception in batchTransformPoints: %s", ex.what());
      all_transformed = false;
    }
  }
  return all_transformed;
}

Eigen::Vector2d RadarLayer::projectMean(
  nav2_dynamic_msgs::msg::Obstacle obstacle,
  double sample_time,
  int time_steps)
{
  Eigen::VectorXd position(2);

  // costmap 시작점은 항상 디텍션된 위치에서 시작
  position(0) = obstacle.position.x;
  position(1) = obstacle.position.y;

  return position;
}


Eigen::Matrix2d RadarLayer::projectCovariance(
  nav2_dynamic_msgs::msg::Obstacle obstacle,
  double sample_time,
  int time_steps)
{
  Eigen::MatrixXd position_covariance(2, 2);
  Eigen::MatrixXd velocity_covariance(2, 2);
  Eigen::MatrixXd position_covariance_projected(2, 2);

  position_covariance(0, 0) = obstacle.position_covariance[0];
  position_covariance(1, 1) = obstacle.position_covariance[4];
  velocity_covariance(0, 0) = obstacle.velocity_covariance[0];
  velocity_covariance(1, 1) = obstacle.velocity_covariance[4];

  if (velocity_covariance(0, 0) < 0 || velocity_covariance(1, 1) < 0) {
    RCLCPP_INFO(logger_, "NEGATIVE VELOCITY COVARIANCE");
  }

  position_covariance_projected = position_covariance +
    pow(time_steps * sample_time, 2) * velocity_covariance;

  return position_covariance_projected;
}

rmw_time_t RadarLayer::convertHzToRmwTimeS(double qos_deadline_hz)
{
  if (qos_deadline_hz <= 0.0) {
    throw std::invalid_argument("qos_deadline_hz must be greater than 0");
  }

  // Convert Hz to seconds
  double total_seconds = 1.0 / qos_deadline_hz;

  // Populate rmw_time_t
  rmw_time_t rmw_time;
  rmw_time.sec = static_cast<uint64_t>(floor(total_seconds));                    // Whole seconds
  rmw_time.nsec = static_cast<uint64_t>((total_seconds - rmw_time.sec) * 1e9);   // Remaining nanoseconds

  return rmw_time;
}


// 검출(Detection)을 씨앗으로 사용하고, 각 시간 스텝마다
// 공분산 + 최소확률 문턱으로 퍼짐 반경을 산정하여 전파.
// (이 함수가 생성하는 이방 퍼짐은 레이어 목록의 "마지막"에서 MAX로 얹힌다.)
void RadarLayer::predictiveCostSeeded(
  nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr obstacles,     // tracking
  nav2_dynamic_msgs::msg::ObstacleArray::SharedPtr detections,    // detections (seed)
  int number_of_objects)
{
  const auto ctype = clock_->get_clock_type();
  const rclcpp::Time obs_stamp(obstacles->header.stamp, ctype);
  const double eps = 1e-9;

  // ── 검출 UUID → seed point (검출 좌표, 검출 시각)
  struct Seed { geometry_msgs::msg::Point p; rclcpp::Time st; };
  std::unordered_map<std::string, Seed> seed_map;
  if (detections) {
    const rclcpp::Time det_stamp(detections->header.stamp, ctype);
    for (const auto& d : detections->obstacles) {
      const std::string key(reinterpret_cast<const char*>(d.uuid.uuid.data()), d.uuid.uuid.size());
      seed_map[key] = Seed{d.position, det_stamp};
    }
  }

  // ── 프레임 변환 계수 준비 (source → global_frame_)
  auto make_tf = [&](const std::string& src_frame,
                     double& dx,double&dy,double& x_x,double&x_y,double& y_x,double& y_y){
    double inv_x_x, inv_x_y, inv_y_x, inv_y_y;
    if (src_frame != global_frame_) {
      getTransformCoefficients(
        /*source_frame=*/global_frame_, /*target_frame=*/src_frame, obs_stamp,
        dx, dy, x_x, x_y, y_x, y_y, inv_x_x, inv_x_y, inv_y_x, inv_y_y);
      return true;
    }
    dx=dy=0.0; x_x=1.0; x_y=0.0; y_x=0.0; y_y=1.0; return false;
  };

  double odx, ody, oxx, oxy, oyx, oyy;  // obstacles frame → map
  make_tf(obstacles->header.frame_id, odx, ody, oxx, oxy, oyx, oyy);

  double ddx, ddy, dxx, dxy, dyx, dyy;  // detections frame → map
  if (detections) make_tf(detections->header.frame_id, ddx, ddy, dxx, dxy, dyx, dyy);
  else            { ddx=ddy=0; dxx=1; dxy=0; dyx=0; dyy=1; }

  // ── 객체 루프
  for (int i = 0; i < number_of_objects; ++i) {
    nav2_dynamic_msgs::msg::Obstacle ob = obstacles->obstacles[i];

    // 1) seed 결정 (검출 우선, 없으면 tracking 좌표)
    const std::string key(reinterpret_cast<const char*>(ob.uuid.uuid.data()), ob.uuid.uuid.size());
    double seed_x, seed_y; // map 좌표
    if (auto it = seed_map.find(key); it != seed_map.end()) {
      const double sx = it->second.p.x, sy = it->second.p.y;
      seed_x = sx * dxx + sy * dyx + ddx;
      seed_y = sx * dxy + sy * dyy + ddy;
    } else {
      const double ox = ob.position.x, oy = ob.position.y;
      seed_x = ox * oxx + oy * oyx + odx;
      seed_y = ox * oxy + oy * oyy + ody;
    }

    // 2) 초기 공분산/스케일(확률 정규화 항)
    double cov0_x = std::clamp(ob.position_covariance[0] + std::max(0.0, ob.size.x / 2.0), eps, 10.0);
    double cov0_y = std::clamp(ob.position_covariance[4] + std::max(0.0, ob.size.y / 2.0), eps, 10.0);
    double s0 = std::sqrt(2.0 * M_PI * cov0_x * cov0_y);
    if (!(std::isfinite(s0) && s0 > eps)) continue;
    const double s0_scaled = s0 * covariance_scaling_factor_;
    const double p_min = std::min(std::max(min_probability_, 1e-6), 0.999999);

    // 3) k=0에서 물체 크기 직사각형 스탬핑(축 정렬)
    auto stamp_box_map = [&](double cx, double cy, double L, double W) {
      const double hx = std::max(0.0, L * 0.5);
      const double hy = std::max(0.0, W * 0.5);
      for (double x = cx - hx; x <= cx + hx; x += resolution_) {
        for (double y = cy - hy; y <= cy + hy; y += resolution_) {
          unsigned int mx, my;
          if (worldToMap(x, y, mx, my)) {
            const unsigned int idx = getIndex(mx, my);
            costmap_[idx] = std::max<unsigned char>(costmap_[idx], nav2_costmap_2d::LETHAL_OBSTACLE);
          }
        }
      }
    };

    // 4) 시간 전개
    for (int k = 0; k < number_of_time_steps_; ++k) {
      // (a) 스텝별 평균 위치 (seed 기준 + v*dt*k)
      double mean_x = seed_x + ob.velocity.x * sample_time_ * k;
      double mean_y = seed_y + ob.velocity.y * sample_time_ * k;

      // k=0일 때 LETHAL + 직사각형 스탬프(물체 크기 보장)
      if (k == 0) {
        unsigned int cx0, cy0;
        if (worldToMap(mean_x, mean_y, cx0, cy0)) {
          const unsigned int cidx0 = getIndex(cx0, cy0);
          costmap_[cidx0] = std::max<unsigned char>(costmap_[cidx0], nav2_costmap_2d::LETHAL_OBSTACLE);
        }
        stamp_box_map(mean_x, mean_y, ob.size.x, ob.size.y);
      }

      // (b) 공분산 갱신 및 반경 산정
      Eigen::Matrix2d cov = projectCovariance(ob, sample_time_, k);
      cov(0,0) = std::clamp(cov(0,0) + ob.size.x / 2.0, eps, 20.0);
      cov(1,1) = std::clamp(cov(1,1) + ob.size.y / 2.0, eps, 20.0);

      Eigen::Matrix2d inv_cov = Eigen::Matrix2d::Zero();
      inv_cov(0,0) = 1.0 / cov(0,0);
      inv_cov(1,1) = 1.0 / cov(1,1);

      double s = std::sqrt(2.0 * M_PI * cov(0,0) * cov(1,1));
      if (!(std::isfinite(s) && s > eps)) continue;
      double inv_s = 1.0 / s;
      double ratio = s0 / std::max(s, 1e-9);

      // 축 방향 지름(미터) → 셀 반경으로
      double Lm = axis_extent_m(cov(0,0), p_min, ratio);
      double Wm = axis_extent_m(cov(1,1), p_min, ratio);
      int radius_cells = static_cast<int>(std::floor(std::max(Lm, Wm) / std::max(1e-6, resolution_)));
      radius_cells = std::max(radius_cells, 1);

      // 물체 크기(반지름) 이상 보장
      int min_by_size = static_cast<int>(
        std::ceil(0.5 * std::max(ob.size.x, ob.size.y) / std::max(1e-6, resolution_))
      ) + 1;
      mean_inflation_radius_ = std::max(radius_cells, min_by_size);

      // 거리/코스트 캐시 재생성
      int max_dist = generateIntegerDistances();
      inflation_cells_.clear();
      inflation_cells_.resize(max_dist + 1);
      computeCacheCosts(inv_cov, inv_s, s0);

      // (c) 중심 셀에서 BFS 전파
      unsigned int cx, cy;
      if (!worldToMap(mean_x, mean_y, cx, cy)) continue;
      const unsigned int cidx = getIndex(cx, cy);
      costmap_[cidx] = std::max<unsigned char>(costmap_[cidx], nav2_costmap_2d::LETHAL_OBSTACLE);

      auto& bin0 = inflation_cells_[0];
      bin0.clear();
      bin0.emplace_back(cx, cy, cx, cy);

      const int r_guard = mean_inflation_radius_ + 2;

      if (seen_.size() != size_x_ * size_y_) seen_.assign(size_x_ * size_y_, false);
      std::fill(seen_.begin(), seen_.end(), false);

      for (auto& dist_bin : inflation_cells_) {
        for (size_t idx = 0; idx < dist_bin.size(); ++idx) {
          const CellData& cell = dist_bin[idx];
          const unsigned int mx = cell.x_;
          const unsigned int my = cell.y_;
          const unsigned int sx = cell.src_x_;
          const unsigned int sy = cell.src_y_;
          const unsigned int mindex = getIndex(mx, my);

          if (seen_[mindex]) continue;
          seen_[mindex] = true;

          const int dx = static_cast<int>(mx) - static_cast<int>(sx);
          const int dy = static_cast<int>(my) - static_cast<int>(sy);
          if (std::abs(dx) > r_guard || std::abs(dy) > r_guard || (dx*dx + dy*dy > r_guard*r_guard))
            continue;

          const bool center = (mx == sx && my == sy);
          unsigned char cost = center ? nav2_costmap_2d::LETHAL_OBSTACLE
                                      : cost_matrix_[std::abs(dx)][std::abs(dy)];
          const double prob = center ? 1.0 : static_cast<double>(cost) / (252.0 * std::max(s0_scaled, eps));

          if (center || prob > p_min) {
            costmap_[mindex] = std::max(costmap_[mindex], cost);
            if (mx > 0)           enqueue(mindex - 1,        mx - 1, my,     sx, sy);
            if (my > 0)           enqueue(mindex - size_x_,  mx,     my - 1, sx, sy);
            if (mx < size_x_-1)   enqueue(mindex + 1,        mx + 1, my,     sx, sy);
            if (my < size_y_-1)   enqueue(mindex + size_x_,  mx,     my + 1, sx, sy);
          }
        }
        dist_bin = std::vector<CellData>(); // 메모리 반환
      }

      // (선택) 디버그 로그
      RCLCPP_DEBUG(logger_,
        "[radar infl] k=%d radius=%d (~%.2fm) p_min=%.3f cov=(%.3f,%.3f) s0=%.3f scale=%.2f",
        k, mean_inflation_radius_, mean_inflation_radius_ * resolution_,
        p_min, cov(0,0), cov(1,1), s0, covariance_scaling_factor_);
    }
  }
}


} // namespace radar_layer

// This is the macro allowing a radar_layer::RadarLayer class
// to be registered in order to be dynamically loadable of base type
// nav2_costmap_2d::CostmapLayer.
// Usually places in the end of cpp-file where the loadable class written.
#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(radar_layer::RadarLayer, nav2_costmap_2d::Layer)
