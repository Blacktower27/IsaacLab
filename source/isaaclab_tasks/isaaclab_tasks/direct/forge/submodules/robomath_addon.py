import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), '..')))
print(parent_dir)
sys.path.append(parent_dir)
# import robomath as rm
# import submodules.robomath as rm
import moveit_motion.diffusion_policy_cam.submodules.robomath as rm
import numpy as np
from typing import Union

# def quaternion_2_euler(qin : np.ndarray | list) -> np.ndarray:
#     # output is in radians
#     _pose = rm.quaternion_2_pose(qin)
#     _eul = rm.pose_2_xyzrpw(_pose)
#     _eul = np.array(_eul[3:])
#     _eul = _eul*np.pi/180 #convert to radians
    
#     # _eul = convert_theta_to_pi_range(_eul) #normalize the angles from -pi to pi # TODO: comment out
#     return _eul

def TxyzQwxyz_2_Pose(XYZwxyz: Union[list, np.ndarray]) -> rm.Mat:
    '''
    XYZwxyz --> Pose (Homegeneous 4X4 Matrix)
    this converts XYZ Quat to corresponding Homogeneous Matrix
    '''
    _pose_Mat = rm.quaternion_2_pose(XYZwxyz[3:]) #Add Quaternions
    for i in range(3):
        _pose_Mat[i,3] = XYZwxyz[i] #Add XYZ to the last column of the matrix
    return _pose_Mat


def Pose_2_TxyzQwxyz(pose: rm.Mat) -> np.ndarray:
    wxyz=rm.pose_2_quaternion(pose)
    XYZ = [pose[i,3] for i in range(3)]
    return np.array([*XYZ, *wxyz])


def TxyzRxyz_2_TxyzQwxyz(XYZrpy: Union[list, np.ndarray]) -> np.ndarray:
    '''**Rxyz in radians**
    param xyzrpw: [x,y,z,rx,ry,rz] in m/mm and radians
    return: [x,y,z,w,x,y,z] i.e. position and quaternion
    '''
    return Pose_2_TxyzQwxyz(rm.TxyzRxyz_2_Pose(XYZrpy))


def TxyzQwxyz_2_TxyzRxyz(XYZwxyz: Union[list, np.ndarray]) -> np.ndarray:
    '''
    **Rxyz in radians**
    param pose: [x,y,z,w,x,y,z] i.e. position and quaternion
    return: [x,y,z,rx,ry,rz] in m/mm and radians
    '''
    return rm.Pose_2_TxyzRxyz(TxyzQwxyz_2_Pose(XYZwxyz))


def normalize_theta(theta: float) -> float:
    """
    Normalize theta (in radian) to the range [-pi, pi).
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def normalize_eulers_method2(angles: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of euler angles (in range -pi to pi) continuous and differentiable to ensure smooth transitions.
    Note: The angles can exceed the range [-pi, pi) after smoothing.
    """
    smooth_angles = angles.copy()
    for i in range(1, len(angles)):
        delta = smooth_angles[i] - smooth_angles[i - 1]
        if np.abs(delta) > np.pi:
            if smooth_angles[i] > smooth_angles[i-1]:
                smooth_angles[i] -= 2 * np.pi
            else:
                smooth_angles[i] += 2 * np.pi
    return smooth_angles


def normalize_eulers(angles: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of euler angles (in range -pi to pi) continuous and differentiable to ensure smooth transitions.
    Note: The angles can exceed the range [-pi, pi) after smoothing.
    """
    rows, cols = angles.shape

    if cols != 3:
        raise ValueError("The input array should have 3 columns for euler angles.")
    for col in range(cols):
        angles[:, col] = np.unwrap(angles[:, col])
    
    return angles
    

''''can stop using these now   
    added new function which is more dynamic then these on in data_filter.py 
'''

def motive_2_robodk_rigidbody(XYZwxyz : Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert motive frame to RoboDK frame.
    """
    X = XYZwxyz[0]; Y = XYZwxyz[1]; Z = XYZwxyz[2]; w = XYZwxyz[3]; x = XYZwxyz[4]; y = XYZwxyz[5]; z = XYZwxyz[6]
    return [Z/1000, X/1000, Y/1000, w, z, x, y]

def motive_2_robodk_marker(XYZ : Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert motive frame to RoboDK frame.
    """
    X = XYZ[0]; Y = XYZ[1]; Z = XYZ[2]
    return [Z/1000, X/1000, Y/1000]

def robodk_2_motive(XYZwxyz : Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert RoboDK quaternions to motive quaternions.
    """
    X = XYZwxyz[0]; Y = XYZwxyz[1]; Z = XYZwxyz[2]; w = XYZwxyz[3]; x = XYZwxyz[4]; y = XYZwxyz[5]; z = XYZwxyz[6]

    return [Y, Z, X, -z, x, -y, w]


def Vxyz_wrt_Pose(Txyz: Union[list, np.ndarray], Pose: rm.Mat) -> list:
    """
    from a vector to a point in the base frame, get a vector to the point from tool frame (in the tool frame)
    """
    W_M = list(Txyz) # vector in the base frame
    W_T_G = Pose # base to tool transformation matrix

    W_R_G = W_T_G.rotationPose()
    W_G = W_T_G.Pos()
    
    W_V = rm.transl(W_M) - rm.transl(W_G)
    W_V = W_V[:,3]
    G_V = rm.invH(W_R_G) * W_V
    G_V = G_V.tolist()[:3]
    return G_V 

def Vxyz_wrt_TxyzQwxyz(Txyz: Union[list, np.ndarray], TxyzQwxyz: Union[list, np.ndarray]) -> np.ndarray:
    """
    from a vector to a point in the base frame, get a vector to the point from tool frame (in the tool frame)
    """
    _Pose = TxyzQwxyz_2_Pose(TxyzQwxyz) # base to tool transformation matrix
    return Vxyz_wrt_Pose(Txyz, _Pose)


def Vxyz_wrt_TxyzRxyz(Txyz: Union[list, np.ndarray], TxyzRxyz: Union[list, np.ndarray]) -> np.ndarray:
    """
    from a vector to a point in the base frame, get a vector to the point from tool frame (in the tool frame)
    """
    _Pose = rm.TxyzRxyz_2_Pose(TxyzRxyz) # base to tool transformation matrix
    return Vxyz_wrt_Pose(Txyz, _Pose)

def BxyzQwxyz_wrt_AxyzQwxyz(AxyzQwxyz: Union[list, np.ndarray], BxyzQwxyz: Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert from base frame to the tool frame.
    """
    W_T_A = TxyzQwxyz_2_Pose(AxyzQwxyz) # base to tool transformation matrix
    W_T_B = TxyzQwxyz_2_Pose(BxyzQwxyz) # base to tool transformation matrix
    A_T_B = rm.invH(W_T_A) * W_T_B
    return Pose_2_TxyzQwxyz(A_T_B)

def BxyzRxyz_wrt_AxyzRxyz(AxyzRxyz: Union[list, np.ndarray], BxyzRxyz: Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert a point in the base frame to the tool frame.
    """
    W_T_A = rm.TxyzRxyz_2_Pose(AxyzRxyz) # base to tool transformation matrix
    W_T_B = rm.TxyzRxyz_2_Pose(BxyzRxyz) # base to tool transformation matrix
    A_T_B = rm.invH(W_T_A) * W_T_B
    return rm.Pose_2_TxyzRxyz(A_T_B)

def first_derivative(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Compute the first derivative of the data.
    """
    return np.gradient(data, times, edge_order=2)


def empathy_transform(W_T_R: rm.Mat, R_O: rm.Mat, W_T_Fin: rm.Mat) -> rm.Mat:
    ''' 
    Reference Frame -> W_T_R
    Matrix Operation -> R_O
    Frame In -> W_T_Fin
    ---
    Frame Out -> W_T_Fout
    '''
    return W_T_R * R_O * rm.invH(W_T_R) * W_T_Fin

def empathy_transform_TxyzRxyz(W_TxyzRxyz_R: list[6], R_OxyzRxyz: list[6], W_TxyzRxyz_Fin: list[6]) -> list[6]:
    ''' 
    Reference Frame -> W_TxyzRxyz_R
    Matrix Operation -> R_OxyzRxyz
    Frame In -> W_TxyzRxyz_Fin
    ---
    Frame Out -> W_TxyzRxyz_Fout
    '''
    W_T_R = rm.TxyzRxyz_2_Pose(W_TxyzRxyz_R)
    R_O = rm.TxyzRxyz_2_Pose(R_OxyzRxyz)
    W_T_Fin = rm.TxyzRxyz_2_Pose(W_TxyzRxyz_Fin)
    W_T_Fout = empathy_transform(W_T_R, R_O, W_T_Fin)
    W_TxyzRxyz_Fout = rm.Pose_2_TxyzRxyz(W_T_Fout)
    return W_TxyzRxyz_Fout

def empathy_transform_TxyzQwxyz(W_TxyzQwxyz_R: list[7], R_OxyzQwxyz: list[7], W_TxyzQwxyz_Fin: list[7]) -> list[7]:
    ''' 
    Reference Frame -> W_TxyzQwxyz_R
    Matrix Operation -> R_OxyzQwxyz
    Frame In -> W_TxyzQwxyz_Fin
    ---
    Frame Out -> W_TxyzQwxyz_Fout
    '''
    W_T_R = TxyzQwxyz_2_Pose(W_TxyzQwxyz_R)
    R_O = TxyzQwxyz_2_Pose(R_OxyzQwxyz)
    W_T_Fin = TxyzQwxyz_2_Pose(W_TxyzQwxyz_Fin)
    W_T_Fout = empathy_transform(W_T_R, R_O, W_T_Fin)
    W_TxyzQwxyz_Fout = Pose_2_TxyzQwxyz(W_T_Fout)
    return W_TxyzQwxyz_Fout

