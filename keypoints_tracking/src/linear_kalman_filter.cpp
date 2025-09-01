/* TU Delft Swarm Lab - Human Swarm Interaction Project
 * Description: Simple linear Kalman filter implementation.
 * Author: Alexander James Becoy
 * Date: 2024-10-28
 */

#include <keypoints_tracking/linear_kalman_filter.hpp>

#include <iostream>

using namespace HumanSwarmInteraction;

LinearKalmanFilter::LinearKalmanFilter(const std::array<double, STATE_SIZE>& initial_state,
  const std::array<double, STATE_SIZE>& covariances, const std::array<double, STATE_SIZE>& process_noises, 
  const std::array<double, MEASUREMENT_SIZE>& observation_noises, const double& dt)
{
  initializeStateTransitionMatrix(dt);
  initializeObservationMatrix();
  initializeState(initial_state);
  initializeCovarianceMatrix(covariances);
  initializeProcessNoiseMatrix(process_noises);
  initializeObservationNoiseMatrix(observation_noises);
}

LinearKalmanFilter::LinearKalmanFilter(const std::vector<double>& initial_state, 
  const std::vector<double>& covariances, std::vector<double>& process_noises,
  const std::vector<double>& observation_noises, const double& dt)
{
  initializeStateTransitionMatrix(dt);
  initializeObservationMatrix();
  initializeState(initial_state);
  initializeCovarianceMatrix(covariances);
  initializeProcessNoiseMatrix(process_noises);
  initializeObservationNoiseMatrix(observation_noises);
}

std::array<double, OUTPUT_SIZE> LinearKalmanFilter::getCentroidCoordinate() const
{
  std::array<double, OUTPUT_SIZE> centroid;
  centroid[0] = m_x(STATE_HORIZONTAL_CENTROID_COORDINATE);
  centroid[1] = m_x(STATE_VERTICAL_CENTROID_COORDINATE);
  return centroid;
}

void LinearKalmanFilter::predict()
{
  // Predict next state and covariance
  m_x.noalias() = m_A * m_x;
  m_P.noalias() = m_A * m_P * m_A.transpose() + m_Q;
}

void LinearKalmanFilter::update(const double& x, const double& y, const double& w, const double& h)
{
  // Create measurement vector
  Eigen::VectorXd z(MEASUREMENT_SIZE);
  z << x, y, w, h;

  // Calculate Kalman gain
  Eigen::MatrixXd S = m_H * m_P * m_H.transpose() + m_R;
  Eigen::MatrixXd K = m_P * m_H.transpose() * S.inverse();

  // Update state and covariance
  m_x.noalias() += K * (z - m_H * m_x);
  m_P.noalias() -= K * m_H * m_P;
}

void LinearKalmanFilter::setInitialState(const double& x, const double& y, const double& w, const double& h,
  const double& vx, const double& vy, const double& vw, const double& vh)
{
  m_x(STATE_HORIZONTAL_CENTROID_COORDINATE) = x;
  m_x(STATE_VERTICAL_CENTROID_COORDINATE) = y;
  m_x(STATE_TRACKING_WINDOW_WIDTH) = w;
  m_x(STATE_TRACKING_WINDOW_HEIGHT) = h;
  m_x(STATE_HORIZONTAL_CENTROID_VELOCITY) = vx;
  m_x(STATE_VERTICAL_CENTROID_VELOCITY) = vy;
  m_x(STATE_TRACKING_WINDOW_WIDTH_VELOCITY) = vw;
  m_x(STATE_TRACKING_WINDOW_HEIGHT_VELOCITY) = vh;
  m_P = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE) * 1e-3;
  m_initialized = true;
}

bool LinearKalmanFilter::isInitialized() const
{
  return m_initialized;
}

void LinearKalmanFilter::initializeStateTransitionMatrix(const double& dt)
{
  m_A = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
  for (int i = 0; i < MEASUREMENT_SIZE; i++) {
    m_A(i, i + MEASUREMENT_SIZE) = dt;
  }
}

void LinearKalmanFilter::initializeObservationMatrix()
{
  m_H = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, STATE_SIZE);
  for (int i = 0; i < MEASUREMENT_SIZE; i++) {
    m_H(i, i) = 1.0;
  }
}

void LinearKalmanFilter::initializeState(const std::array<double, STATE_SIZE>& initial_state)
{
  m_x = Eigen::Map<const Eigen::VectorXd>(initial_state.data(), STATE_SIZE);
}

void LinearKalmanFilter::initializeCovarianceMatrix(const std::array<double, STATE_SIZE>& covariances)
{
  m_P = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
  m_P.diagonal() = Eigen::Map<const Eigen::VectorXd>(covariances.data(), STATE_SIZE);
}

void LinearKalmanFilter::initializeProcessNoiseMatrix(const std::array<double, STATE_SIZE>& process_noises)
{
  m_Q = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
  m_Q.diagonal() = Eigen::Map<const Eigen::VectorXd>(process_noises.data(), STATE_SIZE);
}

void LinearKalmanFilter::initializeObservationNoiseMatrix(const std::array<double, MEASUREMENT_SIZE>& observation_noises)
{
  m_R = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, MEASUREMENT_SIZE);
  m_R.diagonal() = Eigen::Map<const Eigen::VectorXd>(observation_noises.data(), MEASUREMENT_SIZE);
}

void LinearKalmanFilter::initializeState(const std::vector<double>& initial_state)
{
  m_x = Eigen::Map<const Eigen::VectorXd>(initial_state.data(), STATE_SIZE);
}

void LinearKalmanFilter::initializeCovarianceMatrix(const std::vector<double>& covariances)
{
  m_P = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
  m_P.diagonal() = Eigen::Map<const Eigen::VectorXd>(covariances.data(), STATE_SIZE);
}

void LinearKalmanFilter::initializeProcessNoiseMatrix(const std::vector<double>& process_noises)
{
  m_Q = Eigen::MatrixXd::Zero(STATE_SIZE, STATE_SIZE);
  m_Q.diagonal() = Eigen::Map<const Eigen::VectorXd>(process_noises.data(), STATE_SIZE);
}

void LinearKalmanFilter::initializeObservationNoiseMatrix(const std::vector<double>& observation_noises)
{
  m_R = Eigen::MatrixXd::Zero(MEASUREMENT_SIZE, MEASUREMENT_SIZE);
  m_R.diagonal() = Eigen::Map<const Eigen::VectorXd>(observation_noises.data(), MEASUREMENT_SIZE);
}
