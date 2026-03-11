"""
AirSim 

:
1. : 0
2. : 

:
1.  AirSim 
2. : python -m pytest tests/test_airsim_dynamics.py -v -s
   : python tests/test_airsim_dynamics.py
"""

import time
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from pathlib import Path

try:
    import airsim
except ImportError:
    raise ImportError(" airsim: pip install airsim")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, ThreeDConfig


def load_dynamics_config_from_yaml() -> dict:
    """ config.yaml """
    cfg = load_config()
    return {
        "ip": cfg.air_sim.ip,
        "port": cfg.air_sim.port,
        "max_vx": cfg.control.action_bounds["vx"],
        "max_vy": cfg.control.action_bounds["vy"],
        "max_vz": cfg.control.action_bounds["vz"],
        "max_yaw_rate": cfg.control.action_bounds["yaw_rate"],
        "smoothing_alpha": cfg.control.smoothing_alpha,
        "step_hz": cfg.environment.step_hz,
    }


@dataclass
class DynamicsTestConfig:
    """"""
  
    ip: str = "127.0.0.1"
    port: int = 41451
    
  
    max_vx: float = 8.0   # 
    max_vy: float = 4.0   # 
    max_vz: float = 3.0   # 
    max_yaw_rate: float = 0.8  #  (rad/s)
    
  
    smoothing_alpha: float = 0.7
    step_hz: int = 10
    
  
    acceleration_time: float = 5.0  # 
    stabilize_time: float = 2.0     # 
    brake_timeout: float = 15.0     # 
    turn_timeout: float = 30.0      # 
    
  
    speed_threshold: float = 0.1    # 
    position_sample_rate: float = 0.05  #  ()
    
  
    test_altitude: float = -30.0    # NED
    
    @classmethod
    def from_yaml(cls) -> "DynamicsTestConfig":
        """ config.yaml """
        yaml_cfg = load_dynamics_config_from_yaml()
        return cls(
            ip=yaml_cfg["ip"],
            port=yaml_cfg["port"],
            max_vx=yaml_cfg["max_vx"],
            max_vy=yaml_cfg["max_vy"],
            max_vz=yaml_cfg["max_vz"],
            max_yaw_rate=yaml_cfg["max_yaw_rate"],
            smoothing_alpha=yaml_cfg["smoothing_alpha"],
            step_hz=yaml_cfg["step_hz"],
        )
    
    @property
    def max_speed(self) -> float:
        """"""
        return self.max_vx
    
    @property
    def dt(self) -> float:
        """"""
        return 1.0 / self.step_hz


class AirSimDynamicsTester:
    """AirSim """
    
    def __init__(self, config: DynamicsTestConfig = None):
        self.config = config or DynamicsTestConfig()
        self.client: Optional[airsim.MultirotorClient] = None
        
    def connect(self) -> bool:
        """ AirSim"""
        try:
            self.client = airsim.MultirotorClient(
                ip=self.config.ip, 
                port=self.config.port
            )
            self.client.confirmConnection()
            print(f"[] AirSim @ {self.config.ip}:{self.config.port}")
            return True
        except Exception as e:
            print(f"[] {e}")
            return False
    
    def setup_drone(self) -> bool:
        """"""
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
          
            print(f"[] : {-self.config.test_altitude}m")
            self.client.takeoffAsync().join()
            
          
            self.client.moveToZAsync(
                self.config.test_altitude, 
                velocity=3.0
            ).join()
            
            time.sleep(1.0)
            print("[] ")
            return True
        except Exception as e:
            print(f"[] {e}")
            return False
    
    def get_position(self) -> np.ndarray:
        """ (NED)"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])
    
    def get_velocity(self) -> np.ndarray:
        """"""
        state = self.client.getMultirotorState()
        vel = state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])
    
    def get_speed(self) -> float:
        """ ()"""
        vel = self.get_velocity()
        return np.linalg.norm(vel[:2])  # 
    
    def get_yaw(self) -> float:
        """ ()"""
        state = self.client.getMultirotorState()
        q = state.kinematics_estimated.orientation
      
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val * q.y_val + q.z_val * q.z_val)
        return math.atan2(siny_cosp, cosy_cosp)

    def test_braking_distance(self) -> Tuple[float, dict]:
        """
        
        
        :
            braking_distance:  ()
            details: 
        """
        print("\n" + "="*60)
        print("")
        print("="*60)
        
        cfg = self.config
        
      
        print("[1] ...")
        self.client.hoverAsync().join()
        time.sleep(1.0)
        
      
        start_pos = self.get_position()
        print(f"  : {start_pos}")
        
      
        print(f"[2]  {cfg.max_speed} m/s...")
        self.client.moveByVelocityAsync(
            vx=cfg.max_speed,
            vy=0,
            vz=0,
            duration=cfg.acceleration_time
        ).join()
        
      
        print(f"[3]  {cfg.stabilize_time}s...")
        self.client.moveByVelocityAsync(
            vx=cfg.max_speed,
            vy=0,
            vz=0,
            duration=cfg.stabilize_time
        ).join()
        
      
        brake_start_pos = self.get_position()
        brake_start_speed = self.get_speed()
        print(f"  : {brake_start_pos}")
        print(f"  : {brake_start_speed:.2f} m/s")
        
      
        print("[4] ...")
        brake_start_time = time.time()
        
      
        self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
        
      
        positions = [brake_start_pos.copy()]
        speeds = [brake_start_speed]
        timestamps = [0.0]
        
        while True:
            time.sleep(cfg.position_sample_rate)
            elapsed = time.time() - brake_start_time
            
            current_pos = self.get_position()
            current_speed = self.get_speed()
            
            positions.append(current_pos.copy())
            speeds.append(current_speed)
            timestamps.append(elapsed)
            
          
            self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
            
            if current_speed < cfg.speed_threshold:
                print(f"  ! : {elapsed:.2f}s")
                break
            
            if elapsed > cfg.brake_timeout:
                print(f"  [] !")
                break
        
      
        brake_end_pos = self.get_position()
        braking_distance = np.linalg.norm(brake_end_pos[:2] - brake_start_pos[:2])
        braking_time = timestamps[-1]
        
      
        if braking_time > 0:
            avg_deceleration = brake_start_speed / braking_time
        else:
            avg_deceleration = 0
        
        print(f"\n[]")
        print(f"  : {brake_start_speed:.2f} m/s")
        print(f"  : {braking_distance:.2f} m")
        print(f"  : {braking_time:.2f} s")
        print(f"  : {avg_deceleration:.2f} m/s")
        
        details = {
            "initial_speed": brake_start_speed,
            "braking_distance": braking_distance,
            "braking_time": braking_time,
            "avg_deceleration": avg_deceleration,
            "positions": positions,
            "speeds": speeds,
            "timestamps": timestamps,
        }
        
        return braking_distance, details


    def test_minimum_turn_radius(self) -> Tuple[float, dict]:
        """
        
        
        : 
        
        :
            min_turn_radius:  ()
            details: 
        """
        print("\n" + "="*60)
        print("")
        print("="*60)
        
        cfg = self.config
        
      
        print("[1] ...")
        self.client.hoverAsync().join()
        time.sleep(1.0)
        
      
        start_pos = self.get_position()
        start_yaw = self.get_yaw()
        print(f"  : {start_pos}")
        print(f"  : {math.degrees(start_yaw):.1f}")
        
      
        print(f"[2]  {cfg.max_speed} m/s...")
        self.client.moveByVelocityAsync(
            vx=cfg.max_speed,
            vy=0,
            vz=0,
            duration=cfg.acceleration_time
        ).join()
        
        current_speed = self.get_speed()
        print(f"  : {current_speed:.2f} m/s")
        
      
        print("[3] ...")
        turn_start_time = time.time()
        turn_start_pos = self.get_position()
        turn_start_yaw = self.get_yaw()
        target_altitude = turn_start_pos[2]  #  (NED)
        
      
        positions = [turn_start_pos.copy()]
        yaws = [turn_start_yaw]
        timestamps = [0.0]
        
      
        target_yaw_change = math.pi  # 180
        
      
        max_yaw_rate = cfg.max_yaw_rate
        print(f"  : {math.degrees(max_yaw_rate):.1f}/s ( config.yaml)")
        print(f"  : {-target_altitude:.1f}m ()")
        
      
        altitude_kp = 2.0  # 
        
        while True:
            time.sleep(cfg.position_sample_rate)
            elapsed = time.time() - turn_start_time
            
          
            current_pos = self.get_position()
            altitude_error = target_altitude - current_pos[2]  # NED: 
            
          
            vz_correction = altitude_kp * altitude_error
            vz_correction = max(-cfg.max_vz, min(cfg.max_vz, vz_correction))  # 
            
          
            self.client.moveByVelocityBodyFrameAsync(
                vx=cfg.max_vx,  # 
                vy=0,
                vz=vz_correction,  # 
                duration=0.1,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(max_yaw_rate))
            )
            
            current_pos = self.get_position()
            current_yaw = self.get_yaw()
            current_speed = self.get_speed()
            
            positions.append(current_pos.copy())
            yaws.append(current_yaw)
            timestamps.append(elapsed)
            
          
            altitude_drift = current_pos[2] - target_altitude
            if abs(altitude_drift) > 1.0:
                print(f"  [] {-altitude_drift:.2f}m, vz={vz_correction:.2f}")
            
          
            yaw_change = abs(current_yaw - turn_start_yaw)
          
            if yaw_change > math.pi:
                yaw_change = 2 * math.pi - yaw_change
            
            if yaw_change >= target_yaw_change:
                print(f"  180! : {elapsed:.2f}s")
                break
            
            if elapsed > cfg.turn_timeout:
                print(f"  [] ! : {math.degrees(yaw_change):.1f}")
                break
        
      
        positions = np.array(positions)
        
      
        turn_radius_fitted = self._fit_circle_radius(positions[:, :2])
        
      
        avg_speed = np.mean([self.get_speed() for _ in range(5)])
        turn_radius_theoretical = avg_speed / cfg.max_yaw_rate
        
      
        chord_length = np.linalg.norm(positions[-1, :2] - positions[0, :2])
        total_yaw_change = abs(yaws[-1] - yaws[0])
        if total_yaw_change > math.pi:
            total_yaw_change = 2 * math.pi - total_yaw_change
        if total_yaw_change > 0.1:
            turn_radius_chord = chord_length / (2 * math.sin(total_yaw_change / 2))
        else:
            turn_radius_chord = float('inf')
        
      
        min_turn_radius = turn_radius_fitted
        
        print(f"\n[]")
        print(f"  : {avg_speed:.2f} m/s ( max_vx={cfg.max_vx})")
        print(f"  : {math.degrees(cfg.max_yaw_rate):.1f}/s")
        print(f"   (): {turn_radius_fitted:.2f} m")
        print(f"   ( v/): {turn_radius_theoretical:.2f} m")
        print(f"   (): {turn_radius_chord:.2f} m")
        print(f"  : {math.degrees(total_yaw_change):.1f}")
        
      
        z_values = positions[:, 2]
        z_drift = z_values[-1] - z_values[0]
        z_max_drift = np.max(np.abs(z_values - z_values[0]))
        print(f"  : ={-z_drift:.2f}m, ={z_max_drift:.2f}m")
        
        details = {
            "avg_speed": avg_speed,
            "max_yaw_rate": cfg.max_yaw_rate,
            "turn_radius_fitted": turn_radius_fitted,
            "turn_radius_theoretical": turn_radius_theoretical,
            "turn_radius_chord": turn_radius_chord,
            "total_yaw_change_deg": math.degrees(total_yaw_change),
            "positions": positions.tolist(),
            "yaws": yaws,
            "timestamps": timestamps,
            "config": {
                "max_vx": cfg.max_vx,
                "max_vy": cfg.max_vy,
                "max_vz": cfg.max_vz,
                "max_yaw_rate": cfg.max_yaw_rate,
            }
        }
        
        return min_turn_radius, details
    
    def _fit_circle_radius(self, points: np.ndarray) -> float:
        """
        
        
        points: (N, 2) 2D
        """
        if len(points) < 3:
            return float('inf')
        
      
      
      
        
        x = points[:, 0]
        y = points[:, 1]
        
      
        A = np.column_stack([2*x, 2*y, np.ones(len(x))])
        b = x**2 + y**2
        
        try:
          
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            a, b_val, c = result
            
          
            r_squared = c + a**2 + b_val**2
            if r_squared > 0:
                return math.sqrt(r_squared)
            else:
                return float('inf')
        except Exception:
            return float('inf')
    
    def cleanup(self):
        """"""
        if self.client:
            print("\n[] ...")
            try:
              
                self.client.reset()
                time.sleep(0.5)
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except Exception as e:
                print(f"  [] : {e}")
            print("[] ")


def run_all_tests():
    """"""
  
    config = DynamicsTestConfig.from_yaml()
    
    print("="*60)
    print(" config.yaml :")
    print(f"  AirSim: {config.ip}:{config.port}")
    print(f"  max_vx: {config.max_vx} m/s")
    print(f"  max_vy: {config.max_vy} m/s")
    print(f"  max_vz: {config.max_vz} m/s")
    print(f"  max_yaw_rate: {config.max_yaw_rate} rad/s ({math.degrees(config.max_yaw_rate):.1f}/s)")
    print(f"  smoothing_alpha: {config.smoothing_alpha}")
    print(f"  step_hz: {config.step_hz}")
    print("="*60)
    
    tester = AirSimDynamicsTester(config)
    
    if not tester.connect():
        return None
    
    if not tester.setup_drone():
        return None
    
    results = {}
    
    try:
      
        braking_dist, braking_details = tester.test_braking_distance()
        results["braking"] = {
            "distance_m": braking_dist,
            "details": braking_details
        }
        
      
        time.sleep(2.0)
        tester.client.hoverAsync().join()
        time.sleep(1.0)
        
      
        turn_radius, turn_details = tester.test_minimum_turn_radius()
        results["turn_radius"] = {
            "radius_m": turn_radius,
            "details": turn_details
        }
        
    finally:
        tester.cleanup()
    
  
    print("\n" + "="*60)
    print("")
    print("="*60)
    print(f": config.yaml")
    print(f" (max_vx): {config.max_vx} m/s")
    print(f" (max_vy): {config.max_vy} m/s")
    print(f" (max_vz): {config.max_vz} m/s")
    print(f": {config.max_yaw_rate} rad/s ({math.degrees(config.max_yaw_rate):.1f}/s)")
    print(f": {results['braking']['distance_m']:.2f} m")
    print(f": {results['turn_radius']['radius_m']:.2f} m")
    print(f" (v/): {config.max_vx / config.max_yaw_rate:.2f} m")
    print("="*60)
    
    return results


# Comment translated to English.
class TestAirSimDynamics:
    """AirSim  (pytest)"""
    
    @classmethod
    def setup_class(cls):
        """"""
        cls.config = DynamicsTestConfig.from_yaml()
        cls.tester = AirSimDynamicsTester(cls.config)
        
        if not cls.tester.connect():
            raise RuntimeError(" AirSim")
        
        if not cls.tester.setup_drone():
            raise RuntimeError("")
    
    @classmethod
    def teardown_class(cls):
        """"""
        if hasattr(cls, 'tester'):
            cls.tester.cleanup()
    
    def test_braking_distance(self):
        """"""
        braking_dist, details = self.tester.test_braking_distance()
        
      
        assert braking_dist > 0, "0"
        assert braking_dist < 100, " (<100m)"
        assert details["braking_time"] > 0, "0"
        
        print(f"\n: {braking_dist:.2f}m (max_vx={self.config.max_vx}m/s)")
    
    def test_minimum_turn_radius(self):
        """"""
      
        self.tester.client.hoverAsync().join()
        time.sleep(2.0)
        
        turn_radius, details = self.tester.test_minimum_turn_radius()
        
      
        assert turn_radius > 0, "0"
        assert turn_radius < 50, " (<50m)"
        
        theoretical = self.config.max_vx / self.config.max_yaw_rate
        print(f"\n: {turn_radius:.2f}m (: {theoretical:.2f}m)")


if __name__ == "__main__":
    results = run_all_tests()



def test_filter_response_comparison():
    """
     vs AirSim 
    
     0 
    """
    print("\n" + "="*60)
    print("")
    print("="*60)
    
  
    config = DynamicsTestConfig.from_yaml()
    max_speed = config.max_vx
    dt = config.dt
    current_alpha = config.smoothing_alpha
    test_steps = 30  # 3
    
    print(f": config.yaml")
    print(f"  max_vx: {max_speed} m/s")
    print(f"  step_hz: {config.step_hz} Hz (dt={dt}s)")
    print(f"   smoothing_alpha: {current_alpha}")
    
  
    alphas = [0.5, 0.55, 0.6, current_alpha, 0.8]
    alphas = sorted(set(alphas))  # 
    
    print("\n[]  EMA :")
    print("-" * 50)
    
    for alpha in alphas:
      
        velocities = []
        v = 0.0
        target = max_speed
        
        for step in range(test_steps):
            v = alpha * target + (1 - alpha) * v
            velocities.append(v)
        
      
        time_to_90 = next((i * dt for i, v in enumerate(velocities) if v >= 0.9 * target), None)
        time_to_95 = next((i * dt for i, v in enumerate(velocities) if v >= 0.95 * target), None)
        time_to_99 = next((i * dt for i, v in enumerate(velocities) if v >= 0.99 * target), None)
        
        marker = " <-- " if alpha == current_alpha else ""
        print(f"  alpha={alpha}: 90%={time_to_90:.2f}s, 95%={time_to_95:.2f}s, 99%={time_to_99:.2f}s{marker}")
    
  
    print("\n[AirSim] :")
    print("-" * 50)
    
    tester = AirSimDynamicsTester(config)
    
    if not tester.connect():
        print("   AirSim")
        return
    
    if not tester.setup_drone():
        print("  ")
        return
    
    try:
      
        tester.client.hoverAsync().join()
        time.sleep(1.0)
        
      
        airsim_velocities = []
        airsim_times = []
        
        start_time = time.time()
        
      
        tester.client.moveByVelocityAsync(vx=max_speed, vy=0, vz=0, duration=5.0)
        
        for _ in range(test_steps):
            elapsed = time.time() - start_time
            speed = tester.get_speed()
            airsim_velocities.append(speed)
            airsim_times.append(elapsed)
            time.sleep(dt)
        
      
        time_to_90_real = next((t for t, v in zip(airsim_times, airsim_velocities) if v >= 0.9 * max_speed), None)
        time_to_95_real = next((t for t, v in zip(airsim_times, airsim_velocities) if v >= 0.95 * max_speed), None)
        
        print(f"  AirSim: 90%={time_to_90_real:.2f}s, 95%={time_to_95_real:.2f}s" if time_to_90_real else "  90%")
        
      
        print("\n[]  alpha:")
        print("-" * 50)
        
        best_alpha = None
        best_error = float('inf')
        alpha_errors = []
        
        for alpha in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
          
            sim_v = 0.0
            error = 0.0
            for i, real_v in enumerate(airsim_velocities):
                sim_v = alpha * max_speed + (1 - alpha) * sim_v
                error += (sim_v - real_v) ** 2
            
            rmse = math.sqrt(error / len(airsim_velocities))
            alpha_errors.append((alpha, rmse))
            if rmse < best_error:
                best_error = rmse
                best_alpha = alpha
        
        print(f"   smoothing_alpha = {best_alpha} (RMSE={best_error:.3f})")
        print(f"   smoothing_alpha = {current_alpha}")
        
        if abs(best_alpha - current_alpha) > 0.1:
            print(f"  :  config.yaml  smoothing_alpha  {current_alpha}  {best_alpha}")
        
      
        print("\n[]  (10):")
        print(f"  Step | Time(s) | AirSim | alpha={current_alpha} | alpha={best_alpha}")
        print("  " + "-"*55)
        
        sim_current, sim_best = 0.0, 0.0
        for i in range(min(10, len(airsim_velocities))):
            sim_current = current_alpha * max_speed + (1 - current_alpha) * sim_current
            sim_best = best_alpha * max_speed + (1 - best_alpha) * sim_best
            print(f"  {i+1:4d} | {airsim_times[i]:6.2f}  | {airsim_velocities[i]:6.2f} | {sim_current:12.2f} | {sim_best:11.2f}")
        
    finally:
        tester.cleanup()


def test_lidar_warning_braking():
    """
    
    
    :
    1. 
    2. 
    3. 
    4. 
    5. 
    
    :
        python tests/test_airsim_dynamics.py lidar
    """
    print("\n" + "="*60)
    print("")
    print("="*60)
    
  
    config = DynamicsTestConfig.from_yaml()
    
    print(f": config.yaml")
    print(f"  max_vx: {config.max_vx} m/s")
    print(f"  step_hz: {config.step_hz} Hz")
    
    tester = AirSimDynamicsTester(config)
    
    if not tester.connect():
        print(" AirSim")
        return None
    
    if not tester.setup_drone():
        print("")
        return None
    
  
    print("\n[] ...")
    lidar_available, lidar_type = check_lidar_availability(tester.client)
    if not lidar_available:
        print("   !")
        print("   AirSim settings.json  Distance  Lidar")
        print("  ")
    else:
        print(f"   : {lidar_type}")
    
  
    warning_distances = [15.0, 12.0, 10.0, 8.0, 7.0, 6.0, 5.0]
    results = []
    
    try:
        for warning_dist in warning_distances:
            print(f"\n{'='*50}")
            print(f": {warning_dist}m")
            print(f"{'='*50}")
            
            result = run_single_lidar_braking_test(tester, config, warning_dist)
            results.append(result)
            
            if result["collision"]:
                print(f"   !  {warning_dist}m ")
            else:
                print(f"   : {result['min_distance']:.2f}m")
            
          
            print("  ...")
            
          
            tester.client.reset()
            time.sleep(0.5)
            tester.client.enableApiControl(True)
            tester.client.armDisarm(True)
            tester.client.takeoffAsync().join()
            tester.client.moveToZAsync(config.test_altitude, velocity=3.0).join()
            time.sleep(1.0)
        
      
        print("\n" + "="*60)
        print("")
        print("="*60)
        print(f"{'(m)':<12} {'(m)':<12} {'(m)':<12} {'(m)':<12} {'':<6} {'':<6}")
        print("-" * 72)
        
        min_safe_warning = None
        for r in results:
            collision_mark = "" if r["collision"] else ""
          
            is_safe = not r["collision"] and r["min_distance"] > 1.0
            safe_mark = "" if is_safe else ""
            trigger_dist_str = f"{r['trigger_distance']:.2f}" if r['trigger_distance'] else "N/A"
            print(f"{r['warning_distance']:<12.1f} {trigger_dist_str:<12} {r['min_distance']:<12.2f} {r['braking_distance']:<12.2f} {collision_mark:<6} {safe_mark:<6}")
            
            if is_safe:
                min_safe_warning = r["warning_distance"]
        
        print("-" * 72)
        if min_safe_warning:
            print(f": {min_safe_warning}m")
        else:
            print(": !")
        
        return results
        
    finally:
        tester.cleanup()

# Comment translated to English.
LIDAR_SENSOR_NAME = "LidarSensor1"  # 
DISTANCE_SENSOR_NAME = "Distance"   # 
VEHICLE_NAME = ""                   # 


def check_lidar_availability(client: airsim.MultirotorClient) -> Tuple[bool, str]:
    """"""
  
    try:
        lidar_data = client.getLidarData(lidar_name=LIDAR_SENSOR_NAME, vehicle_name=VEHICLE_NAME)
        if len(lidar_data.point_cloud) >= 3:
            return True, f"Lidar ({LIDAR_SENSOR_NAME})"
    except Exception as e:
        print(f"  [] : {e}")
    
  
    try:
        distance_data = client.getDistanceSensorData(
            distance_sensor_name=DISTANCE_SENSOR_NAME, 
            vehicle_name=VEHICLE_NAME
        )
        if distance_data.distance > 0 and distance_data.distance < 100:
            return True, f"Distance Sensor ({DISTANCE_SENSOR_NAME})"
    except Exception as e:
        print(f"  [] : {e}")
    
    return False, "None"


def run_single_lidar_braking_test(
    tester: AirSimDynamicsTester,
    config: DynamicsTestConfig,
    warning_distance: float,
) -> dict:
    """
    
    
    Args:
        tester: AirSim
        config: 
        warning_distance:  ()
    
    Returns:
        
    """
  
    tester.client.hoverAsync().join()
    time.sleep(0.5)
    
    start_pos = tester.get_position()
    
  
    print("  [1] ...")
    tester.client.moveByVelocityAsync(
        vx=config.max_vx,
        vy=0,
        vz=0,
        duration=config.acceleration_time
    ).join()
    
  
    print(f"  [2]  (={warning_distance}m)...")
    
    lidar_triggered = False
    trigger_pos = None
    trigger_distance = None
    min_distance = float('inf')
    collision = False
    
    positions = []
    distances = []
    speeds = []
    timestamps = []
    
    start_time = time.time()
    timeout = 30.0  # 
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print("  [] ")
            break
        
      
        collision_info = tester.client.simGetCollisionInfo()
        if collision_info.has_collided:
            collision = True
            current_pos = tester.get_position()
            print(f"  [!] AirSim: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")
          
            tester.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
            break
        
      
        forward_distance = get_forward_lidar_distance(tester.client)
        
        current_pos = tester.get_position()
        current_speed = tester.get_speed()
        
        positions.append(current_pos.copy())
        distances.append(forward_distance)
        speeds.append(current_speed)
        timestamps.append(elapsed)
        
      
        if forward_distance > 0 and forward_distance < min_distance:
            min_distance = forward_distance
        
      
        if not lidar_triggered and forward_distance > 0 and forward_distance <= warning_distance:
            lidar_triggered = True
            trigger_pos = current_pos.copy()
            trigger_distance = forward_distance
            print(f"  [] ={forward_distance:.2f}m, ={current_speed:.2f}m/s")
            
          
            tester.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
        
        if lidar_triggered:
          
            tester.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.1)
            
          
            if current_speed < config.speed_threshold:
                print(f"  [] ={current_speed:.2f}m/s, ={min_distance:.2f}m")
                break
        else:
          
            tester.client.moveByVelocityAsync(
                vx=config.max_vx,
                vy=0,
                vz=0,
                duration=0.1
            )
        
        time.sleep(config.position_sample_rate)
    
  
    braking_distance = 0.0
    if trigger_pos is not None:
        final_pos = tester.get_position()
        braking_distance = np.linalg.norm(final_pos[:2] - trigger_pos[:2])
    
  
    if collision and not lidar_triggered:
        print("  [] ")
        min_distance = 0.0
    
    return {
        "warning_distance": warning_distance,
        "trigger_distance": trigger_distance,
        "min_distance": min_distance if min_distance != float('inf') else 0.0,
        "braking_distance": braking_distance,
        "collision": collision,
        "positions": positions,
        "distances": distances,
        "speeds": speeds,
        "timestamps": timestamps,
    }


def get_forward_lidar_distance(client: airsim.MultirotorClient) -> float:
    """
    
    
     AirSim 
    
    Returns:
         () -1
    """
  
    try:
        lidar_data = client.getLidarData(lidar_name=LIDAR_SENSOR_NAME, vehicle_name=VEHICLE_NAME)
        if len(lidar_data.point_cloud) >= 3:
          
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape(-1, 3)
            
          
          
          
            forward_mask = (points[:, 0] > 0.5) & (np.abs(points[:, 1]) < points[:, 0]) & (np.abs(points[:, 2]) < 3.0)
            forward_points = points[forward_mask]
            
            if len(forward_points) > 0:
              
                distances = np.linalg.norm(forward_points, axis=1)
                return float(np.min(distances))
    except Exception:
        pass
    
  
    try:
        distance_data = client.getDistanceSensorData(
            distance_sensor_name=DISTANCE_SENSOR_NAME, 
            vehicle_name=VEHICLE_NAME
        )
        if distance_data.distance > 0 and distance_data.distance < 100:
            return distance_data.distance
    except Exception:
        pass
    
  
    return -1.0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "filter":
        test_filter_response_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "lidar":
        test_lidar_warning_braking()
    else:
        results = run_all_tests()
