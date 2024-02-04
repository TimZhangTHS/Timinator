import tinyik
import numpy as np

print('start')


'''
Coordinates are meassured in millimeters
'''
ARM_BASE_COORDINATE = (10,20,30)


def map_camera_to_arm(target_coordinate):
  x = target_coordinate[0] - ARM_BASE_COORDINATE[0]
  y = target_coordinate[1] - ARM_BASE_COORDINATE[1]
  z = target_coordinate[2] - ARM_BASE_COORDINATE[2]
  return (x,y,z)

target_cam_coordinate = (30,30,50)
target_arm_coord = map_camera_to_arm( target_cam_coordinate )
print(target_arm_coord)

arm = tinyik.Actuator([
  'y', [0., 20., 0.], 
  'z', [0., 60., 0.],
  [0., 0., -11.],
  # 'z', [-54., 0, 0.],
  'z', [13.97,-52.16,0], # 15 Degree
#  'y', [54,0,0],
  [0., 0., -5.],
#  'x', [0., 1., 0.],
  'z', [11.0, 0.0, 0.],
])



arm.ee = target_arm_coord

print(np.rad2deg(arm.angles))


print("ANG:", arm.angles)
print("POS:", arm.ee)

tinyik.visualize(arm, radius=2.5)

print('done')
exit()

# arm.ee = [50,20,0]
arm.angles = [0,0,0,0]
arm.angles = [7.5896, 1.7592, -9.6865, -1.3091]

print(np.rad2deg(arm.angles))


print("ANG:", arm.angles)
print("POS:", arm.ee)

tinyik.visualize(arm, radius=2.5)

print("ANG:", arm.angles)
print("POS:", arm.ee)

print('done')

'''
pip install tinyik -U
pip install open3d


'''

# import numpy as np

# import tinyik


# tokens = [
#     [.3, .0, .0], 'z', [.3, .0, .0], 'x', [.0, -.5, .0], 'x', [.0, -.5, .0]]


# def visualize():
#     leg = tinyik.Actuator(tokens)
#     leg.angles = np.deg2rad([30, 45, -90])
#     tinyik.visualize(leg)


# def visualize_with_target():
#     leg = tinyik.Actuator(tokens)
#     leg.angles = np.deg2rad([30, 45, -90])
#     tinyik.visualize(leg, target=[.8, .0, .8])


# large_tokens = [
#     [85., 80., 0.],
#     'z',
#     [500., 0., 0.],
#     'z',
#     [0., -500., 0.],
# ]


# def large_visualize():
#     arm = tinyik.Actuator(large_tokens)
#     tinyik.visualize(arm, radius=15.)


# def large_visualize_with_target():
#     arm = tinyik.Actuator(large_tokens)
#     tinyik.visualize(arm, target=[400., -300., 0.], radius=15.)


# def visualize_with_z_axis():
#     arm = tinyik.Actuator([
#         'z', [0, 0, 180.7], 'y', [-612.7, 0, 0], 'y', [-571.55, 0, 0],
#         'y', [0, -174.15, 0], 'z', [0, 0, -119.85], 'y', [0, -116.55, 0]])
#     tinyik.visualize(arm, radius=10.)


# if __name__ == '__main__':
#     visualize()
#     visualize_with_target()
#     large_visualize()
#     large_visualize_with_target()
#     visualize_with_z_axis()
