import os
from math import cos, sin
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from source.config.config import Config
from is_msgs.image_pb2 import HumanKeypoints as HKP

config = Config()

DEFAULT_ML_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "logisticRegression.sav"
)


class Gesture:
    def __init__(self, model_path: str = DEFAULT_ML_MODEL_PATH):
        self.ml_model = joblib.load(model_path)
        self.TO_COCO_IDX = {
            HKP.Value("NOSE"): 0,
            HKP.Value("LEFT_EYE"): 1,
            HKP.Value("RIGHT_EYE"): 2,
            HKP.Value("LEFT_EAR"): 3,
            HKP.Value("RIGHT_EAR"): 4,
            HKP.Value("LEFT_SHOULDER"): 5,
            HKP.Value("RIGHT_SHOULDER"): 6,
            HKP.Value("LEFT_ELBOW"): 7,
            HKP.Value("RIGHT_ELBOW"): 8,
            HKP.Value("LEFT_WRIST"): 9,
            HKP.Value("RIGHT_WRIST"): 10,
            HKP.Value("LEFT_HIP"): 11,
            HKP.Value("RIGHT_HIP"): 12,
            HKP.Value("LEFT_KNEE"): 13,
            HKP.Value("RIGHT_KNEE"): 14,
            HKP.Value("LEFT_ANKLE"): 15,
            HKP.Value("RIGHT_ANKLE"): 16,
        }

        self._LEFT_SHOULDER_IDX = self.TO_COCO_IDX[HKP.Value("LEFT_SHOULDER")]
        self._LEFT_WRIST_IDX = self.TO_COCO_IDX[HKP.Value("LEFT_WRIST")]
        self._LEFT_ELBOW_IDX = self.TO_COCO_IDX[HKP.Value("LEFT_ELBOW")]
        self._RIGHT_SHOULDER_IDX = self.TO_COCO_IDX[
            HKP.Value("RIGHT_SHOULDER")
        ]
        self._RIGHT_WRIST_IDX = self.TO_COCO_IDX[HKP.Value("RIGHT_WRIST")]
        self._RIGHT_ELBOW_IDX = self.TO_COCO_IDX[HKP.Value("RIGHT_ELBOW")]
        # print(self._LEFT_SHOULDER_IDX)

        # Define the exact order of indices for vectorized extraction
        # This matches the order of p0_L, p1_L, p2_L, p0_R, p1_R, p2_R in the return
        self._TARGET_INDICES = np.array(
            [
                self._LEFT_SHOULDER_IDX,  # p0_L
                self._LEFT_ELBOW_IDX,  # p1_L
                self._LEFT_WRIST_IDX,  # p2_L
                self._RIGHT_SHOULDER_IDX,  # p0_R
                self._RIGHT_ELBOW_IDX,  # p1_R
                self._RIGHT_WRIST_IDX,  # p2_R
            ]
        )

    # def extract_points(self, parts: np.ndarray):
    #     # # p = np.zeros(3)

    #     p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = (0, 0, 0, 0, 0, 0)

    #     if parts.shape[0] > 6:
    #         # # p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

    #         p0_L = np.array(
    #             parts[self.TO_COCO_IDX[HKP.Value("LEFT_SHOULDER")]]
    #         )  # Ombro
    #         p2_L = np.array(parts[self.TO_COCO_IDX[HKP.Value("LEFT_WRIST")]])  # pulso
    #         p1_L = np.array(
    #             parts[self.TO_COCO_IDX[HKP.Value("LEFT_ELBOW")]]
    #         )  # cotovelo

    #         p0_R = np.array(parts[self.TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]])
    #         p2_R = np.array(parts[self.TO_COCO_IDX[HKP.Value("RIGHT_WRIST")]])
    #         p1_R = np.array(parts[self.TO_COCO_IDX[HKP.Value("RIGHT_ELBOW")]])

    #     else:
    #         p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = (
    #             parts[0],
    #             parts[1],
    #             parts[2],
    #             parts[3],
    #             parts[4],
    #             parts[5],
    #         )

    #     # # if parts.all():
    #     # #     p0_L = p
    #     # #     p2_L = p
    #     # #     p1_L = p
    #     # #     p0_R = p
    #     # #     p2_R = p
    #     # #     p1_R = p

    #     # # else:
    #     # p0_L = np.array(parts[self.TO_COCO_IDX[HKP.Value("LEFT_SHOULDER")]])  # Ombro
    #     # p2_L = np.array(parts[self.TO_COCO_IDX[HKP.Value("LEFT_WRIST")]])  # pulso
    #     # p1_L = np.array(parts[self.TO_COCO_IDX[HKP.Value("LEFT_ELBOW")]])  # cotovelo

    #     # p0_R = np.array(parts[self.TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]])
    #     # p2_R = np.array(parts[self.TO_COCO_IDX[HKP.Value("RIGHT_WRIST")]])
    #     # p1_R = np.array(parts[self.TO_COCO_IDX[HKP.Value("RIGHT_ELBOW")]])

    #     return p0_L, p1_L, p2_L, p0_R, p1_R, p2_R

    def extract_points(self, parts):

        # Check if 'parts' is valid (e.g., not empty, has enough dimensions)
        if not isinstance(parts, np.ndarray) or parts.ndim < 2:
            raise ValueError("Input 'parts' must be a 2D NumPy array.")

        # Determine the source of the points
        if parts.shape[0] > 6:
            extracted_points = parts[self._TARGET_INDICES]

            # Unpack the points
            p0_L = extracted_points[0]
            p1_L = extracted_points[1]
            p2_L = extracted_points[2]
            p0_R = extracted_points[3]
            p1_R = extracted_points[4]
            p2_R = extracted_points[5]

        elif parts.shape[0] == 6:
            p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = parts

        else:
            p0_L, p1_L, p2_L = np.zeros(3), np.zeros(3), np.zeros(3)
            p0_R, p1_R, p2_R = np.zeros(3), np.zeros(3), np.zeros(3)

        return p0_L, p1_L, p2_L, p0_R, p1_R, p2_R

    def list_to_dataframe(self, parts):
        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)
        all_coords = np.concatenate(
            [p0_L, p0_R, p1_L, p1_R, p2_L, p2_R]
        ).flatten()

        # Define the feature names in the EXACT order they appear in all_coords
        feature_names = [
            "xLEFT_SHOULDER",
            "yLEFT_SHOULDER",
            "zLEFT_SHOULDER",
            "xRIGHT_SHOULDER",
            "yRIGHT_SHOULDER",
            "zRIGHT_SHOULDER",
            "xLEFT_ELBOW",
            "yLEFT_ELBOW",
            "zLEFT_ELBOW",
            "xRIGHT_ELBOW",
            "yRIGHT_ELBOW",
            "zRIGHT_ELBOW",
            "xLEFT_WRIST",
            "yLEFT_WRIST",
            "zLEFT_WRIST",
            "xRIGHT_WRIST",
            "yRIGHT_WRIST",
            "zRIGHT_WRIST",
        ]
        df = pd.DataFrame(all_coords.reshape(1, -1), columns=feature_names)
        return df

    def distancia_euclidiana_2d(self, p1, p2):
        """
        Computes the Euclidean distance between two 2D points.

        Parameters:
        p1 -- array or list with coordinates (x1, y1) for point 1
        p2 -- array or list with coordinates (x2, y2) for point 2

        Returns:
        The Euclidean distance between the points.
        """
        p1 = np.array(p1)
        p2 = np.array(p2)

        distancia = np.linalg.norm(
            p2 - p1
        )  # Compute the norm (distance) between the points

        return distancia

    def colineares(self, p0, p1, p2):

        vetor1 = np.subtract(p1, p0)
        vetor2 = np.subtract(p2, p0)

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        if norm_vetor1 != 0 and norm_vetor2 != 0:
            sim = prod_interno / (norm_vetor1 * norm_vetor2)
        else:
            return 0

        if sim > 1:
            sim = 1
        if sim < -1:
            sim = -1

        teta = (180 * np.arccos(sim)) / np.pi

        return teta

    def angulo_entre_vetores(self, vetor1, vetor2):

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        if norm_vetor1 != 0 and norm_vetor2 != 0:
            sim = prod_interno / (norm_vetor1 * norm_vetor2)
        else:
            return 0

        if sim > 1:
            sim = 1
        if sim < -1:
            sim = -1

        teta = (180 * np.arccos(sim)) / np.pi

        return teta

    def angulos_internos(self, parts):

        (
            left_ombro,
            left_cotovelo,
            left_pulso,
            right_ombro,
            right_cotovelo,
            right_pulso,
        ) = self.extract_points(parts)

        # left

        vetor1 = np.subtract(left_ombro, left_cotovelo)
        vetor2 = np.subtract(left_ombro, left_pulso)
        vetor3 = np.subtract(left_cotovelo, left_pulso)
        vetor4 = np.subtract(left_cotovelo, left_ombro)
        vetor5 = np.subtract(left_pulso, left_ombro)
        vetor6 = np.subtract(left_pulso, left_cotovelo)

        left_alpha = self.angulo_entre_vetores(vetor4, vetor3)
        left_theta = self.angulo_entre_vetores(vetor1, vetor2)
        left_beta = self.angulo_entre_vetores(vetor5, vetor6)

        # right

        vetor1 = np.subtract(right_ombro, right_cotovelo)
        vetor2 = np.subtract(right_ombro, right_pulso)
        vetor3 = np.subtract(right_cotovelo, right_pulso)
        vetor4 = np.subtract(right_cotovelo, right_ombro)
        vetor5 = np.subtract(right_pulso, right_ombro)
        vetor6 = np.subtract(right_pulso, right_cotovelo)

        right_alpha = self.angulo_entre_vetores(vetor4, vetor3)
        right_theta = self.angulo_entre_vetores(vetor1, vetor2)
        right_beta = self.angulo_entre_vetores(vetor5, vetor6)

        return (
            left_alpha,
            left_theta,
            left_beta,
            right_alpha,
            right_theta,
            right_beta,
        )

    def comparacao_normas(self, parts):

        (
            left_ombro,
            left_cotovelo,
            left_pulso,
            right_ombro,
            right_cotovelo,
            right_pulso,
        ) = self.extract_points(parts)

        # left
        vetor1 = np.subtract(left_ombro, left_cotovelo)
        vetor2 = np.subtract(left_ombro, left_pulso)

        left_norm_vetor1 = np.linalg.norm(vetor1)
        left_norm_vetor2 = np.linalg.norm(vetor2)

        # right
        vetor1 = np.subtract(right_ombro, right_cotovelo)
        vetor2 = np.subtract(right_ombro, right_pulso)

        right_norm_vetor1 = np.linalg.norm(vetor1)
        right_norm_vetor2 = np.linalg.norm(vetor2)

        return (
            left_norm_vetor1,
            left_norm_vetor2,
            right_norm_vetor1,
            right_norm_vetor2,
        )

    def calcular_verticalidade_reta_3d(self, ponto1, ponto2, em_graus=True):
        """
        Computes the angle a 3D line makes with the Z axis (the normal to the XY plane),
        indicating how vertical it is relative to the XY plane.

        Args:
            ponto1 (tuple): Tuple (x1, y1, z1) for the first point.
            ponto2 (tuple): Tuple (x2, y2, z2) for the second point.
            em_graus (bool): If True, returns the angle in degrees; otherwise radians.
                Default is True.

        Returns:
            float: The angle in degrees or radians.
            str: An error message if the points are coincident.
        """
        x1, y1, z1 = ponto1
        x2, y2, z2 = ponto2

        # 1. Compute the line direction vector
        vx = x2 - x1
        vy = y2 - y1
        vz = z2 - z1
        vetor_reta = (vx, vy, vz)

        # 2. Compute the magnitude of the direction vector
        modulo_vetor_reta = np.sqrt(vx**2 + vy**2 + vz**2)

        # Check if the points are coincident
        if modulo_vetor_reta == 0:
            return 0

        # 3. Define the Z-axis vector (normal to the XY plane)
        vetor_eixo_z = (0, 0, 1)

        # 4. Compute the dot product between the line vector and Z axis
        produto_escalar = (
            vetor_reta[0] * vetor_eixo_z[0]
            + vetor_reta[1] * vetor_eixo_z[1]
            + vetor_reta[2] * vetor_eixo_z[2]
        )  # Simplifies to 'vz' because (0,0,1)

        # 5. Z-axis vector has magnitude 1, so no need to multiply by it
        # cos_theta = produto_escalar / (modulo_vetor_reta * modulo_vetor_eixo_z)
        cos_theta = produto_escalar / modulo_vetor_reta

        # Keep value inside arccos domain [-1, 1] for floating point stability
        cos_theta = max(-1.0, min(1.0, cos_theta))

        # 6. Compute angle using arccos
        angulo_radianos = np.arccos(cos_theta)

        if em_graus:
            return np.degrees(angulo_radianos)
        else:
            return angulo_radianos

    def verticalidade(self, parts):

        (
            left_ombro,
            left_cotovelo,
            left_pulso,
            right_ombro,
            right_cotovelo,
            right_pulso,
        ) = self.extract_points(parts)

        ponto1 = left_ombro
        ponto2 = left_cotovelo
        left = self.calcular_verticalidade_reta_3d(
            ponto1, ponto2, em_graus=True
        )

        ponto1 = right_ombro
        ponto2 = right_cotovelo
        right = self.calcular_verticalidade_reta_3d(
            ponto1, ponto2, em_graus=True
        )

        return left, right

    def colinearidade1(self, parts):

        (
            left_ombro,
            left_cotovelo,
            left_pulso,
            right_ombro,
            right_cotovelo,
            right_pulso,
        ) = self.extract_points(parts)

        vetor1 = np.subtract(left_ombro, left_cotovelo)
        vetor2 = np.subtract(left_ombro, left_pulso)

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        if norm_vetor1 != 0 and norm_vetor2 != 0:
            sim = prod_interno / (norm_vetor1 * norm_vetor2)
        else:
            return 0

        if sim > 1:
            sim = 1
        if sim < -1:
            sim = -1

        left_theta = (180 * np.arccos(sim)) / np.pi

        vetor1 = np.subtract(right_ombro, right_cotovelo)
        vetor2 = np.subtract(right_ombro, right_pulso)

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        if norm_vetor1 != 0 and norm_vetor2 != 0:
            sim = prod_interno / (norm_vetor1 * norm_vetor2)
        else:
            return 0

        if sim > 1:
            sim = 1
        if sim < -1:
            sim = -1

        right_theta = (180 * np.arccos(sim)) / np.pi

        return left_theta, right_theta

    def colinearidade2(self, parts):

        (
            left_ombro,
            left_cotovelo,
            left_pulso,
            right_ombro,
            right_cotovelo,
            right_pulso,
        ) = self.extract_points(parts)

        vetor1 = np.subtract(left_cotovelo, left_ombro)
        vetor2 = np.subtract(left_cotovelo, left_pulso)

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        if norm_vetor1 != 0 and norm_vetor2 != 0:
            sim = prod_interno / (norm_vetor1 * norm_vetor2)
        else:
            return 0

        if sim > 1:
            sim = 1
        if sim < -1:
            sim = -1

        left_alpha = (180 * np.arccos(sim)) / np.pi

        vetor1 = np.subtract(right_cotovelo, right_ombro)
        vetor2 = np.subtract(right_cotovelo, right_pulso)

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        if norm_vetor1 != 0 and norm_vetor2 != 0:
            sim = prod_interno / (norm_vetor1 * norm_vetor2)
        else:
            return 0

        if sim > 1:
            sim = 1
        if sim < -1:
            sim = -1

        right_alpha = (180 * np.arccos(sim)) / np.pi

        return left_alpha, right_alpha

    def distancia(self, parts):
        (
            left_ombro,
            left_cotovelo,
            left_pulso,
            right_ombro,
            right_cotovelo,
            right_pulso,
        ) = self.extract_points(parts)

        left_distancia = np.linalg.norm(left_pulso - left_ombro)

        right_distancia = np.linalg.norm(right_pulso - right_ombro)

        return left_distancia, right_distancia

    def classificador(
        self,
        parts: np.ndarray,
        dist: float,
        colinear: int,
        option: bool = False,
    ):
        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

        # limiar_distancia: float = 0.25
        # limiar_colinear: int = 30
        limiar_distancia = dist
        limiar_colinear = colinear

        pt1_L = p0_L[:2]  # Ombro
        pt2_L = p2_L[:2]  # pulso

        pt1_R = p0_R[:2]
        pt2_R = p2_R[:2]

        left_distancia = self.distancia_euclidiana_2d(pt1_L, pt2_L)
        right_distancia = self.distancia_euclidiana_2d(pt1_R, pt2_R)

        left_teta = self.colineares(p0_L, p1_L, p2_L)
        right_teta = self.colineares(p0_R, p1_R, p2_R)

        # Decision tree
        if (
            left_distancia > limiar_distancia
            and right_distancia > limiar_distancia
        ):
            if left_teta < limiar_colinear and right_teta < limiar_colinear:
                y = 3
            elif left_teta < limiar_colinear:
                y = 1
            elif right_teta < limiar_colinear:
                y = 2
            else:
                y = 0
        elif left_distancia > limiar_distancia:
            if left_teta < limiar_colinear:
                y = 1
            else:
                y = 0
        elif right_distancia > limiar_distancia:
            if right_teta < limiar_colinear:
                y = 2
            else:
                y = 0
        else:
            y = 0

        if not option:
            if y != 0:
                y = 1

        return (
            y,
            left_distancia,
            left_teta,
            right_distancia,
            right_teta,
        )

    def classificador_ml(
        self,
        parts: np.ndarray,
        dist: float = 0.18,
        colinear: int = 45,
        dz: float = -0.35,
    ):
        """
        ML-based pointing gesture classifier.

        Pipeline: skeleton -> normalizacao() -> list_to_dataframe() (18 features)
        -> LogisticRegression.predict() -> binary 0/1. When binary == 1, the
        side (left/right/both) is decided by three per-arm conditions on the
        normalized skeleton:
          - 2D (XY) shoulder-wrist distance > dist  (pointing ~0.25m, rest ~0.11m)
          - shoulder-elbow-wrist collinearity angle  < colinear
          - wrist.z - shoulder.z > dz  (pointing ~-0.17, rest ~-0.49)

        Returns the same tuple shape as classificador(): (y, left_dist,
        left_teta, right_dist, right_teta), where y in {0, 1, 2, 3}.
        """
        skM = self.normalizacao(parts)
        df = self.list_to_dataframe(skM)
        predict_binario = int(self.ml_model.predict(df.loc[[0]])[0])

        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(skM)
        left_distancia = self.distancia_euclidiana_2d(p0_L[:2], p2_L[:2])
        right_distancia = self.distancia_euclidiana_2d(p0_R[:2], p2_R[:2])
        left_teta = self.colineares(p0_L, p1_L, p2_L)
        right_teta = self.colineares(p0_R, p1_R, p2_R)

        # dZ = wrist.z - shoulder.z on normalized skeleton.
        # Pointing arm: wrist near shoulder height (dZ > -0.35).
        # Resting arm: wrist well below shoulder (dZ ≈ -0.49).
        dZ_left  = p2_L[2] - p0_L[2]
        dZ_right = p2_R[2] - p0_R[2]

        if predict_binario == 0:
            y = 0
        else:
            left_ok  = left_distancia  > dist and left_teta  < colinear and dZ_left  > dz
            right_ok = right_distancia > dist and right_teta < colinear and dZ_right > dz
            if left_ok and right_ok:
                y = 3
            elif left_ok:
                y = 1
            elif right_ok:
                y = 2
            else:
                # ML says pointing but heuristic finds no clear arm — use the
                # arm with the highest wrist elevation as a last resort.
                y = 0

        return (
            y,
            left_distancia,
            left_teta,
            right_distancia,
            right_teta,
        )

    def classificador2(self, parts: np.ndarray, dist: float):
        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

        # limiar_distancia: float = 0.25
        # limiar_colinear: int = 30
        limiar_distancia = dist

        pt1_L = p0_L[:2]  # Ombro
        pt2_L = p2_L[:2]  # pulso

        pt1_R = p0_R[:2]
        pt2_R = p2_R[:2]

        left_distancia = self.distancia_euclidiana_2d(pt1_L, pt2_L)
        right_distancia = self.distancia_euclidiana_2d(pt1_R, pt2_R)

        # Decision tree
        if (
            left_distancia > limiar_distancia
            and right_distancia > limiar_distancia
        ):
            y = 3
        elif left_distancia > limiar_distancia:
            y = 2
        elif right_distancia > limiar_distancia:
            y = 1
        else:
            y = 0

        return (
            y,
            left_distancia,
            right_distancia,
        )

    def classificador3(
        self,
        parts: np.ndarray,
        dist: float,
        colinear: int,
        option: bool = False,
    ):
        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

        limiar_distancia = dist
        limiar_colinear = colinear

        pt1_L = p0_L[:2]  # Ombro
        pt2_L = p2_L[:2]  # pulso

        pt1_R = p0_R[:2]
        pt2_R = p2_R[:2]

        left_distancia = self.distancia_euclidiana_2d(pt1_L, pt2_L)
        right_distancia = self.distancia_euclidiana_2d(pt1_R, pt2_R)

        # Decision tree
        if (
            left_distancia > limiar_distancia
            and right_distancia > limiar_distancia
        ):
            y = 3
        elif left_distancia > limiar_distancia:
            y = 2
        elif right_distancia > limiar_distancia:
            y = 1
        else:
            y = 0

        return y

    def reta_para_plot(self, parts: np.ndarray, length=0.5):
        p0_L, p1_L, p2_L, p0_R, p1_R, p2_R = self.extract_points(parts)

        left_vector = p2_L - p0_L
        right_vector = p2_R - p0_R

        left_vector = left_vector / np.linalg.norm(left_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)

        # Line start and end (fixed length)
        left_start = p2_L
        left_end = p2_L + left_vector * length

        right_start = p2_R
        right_end = p2_R + right_vector * length

        # Return ready-to-plot coordinates
        return (
            np.array([right_start[0], right_end[0]]),
            np.array([right_start[1], right_end[1]]),
            np.array([right_start[2], right_end[2]]),
            np.array([left_start[0], left_end[0]]),
            np.array([left_start[1], left_end[1]]),
            np.array([left_start[2], left_end[2]]),
            left_end,
            right_end,
        )

    def detectar_gesto(
        self,
        dfX: pd.DataFrame,
        dist: float,
        colinear: int,
        option: bool = False,
    ) -> pd.Series:
        """
        Classifies DataFrame rows based on collinearity and Euclidean distance.

        The function analyzes anatomical points extracted from the DataFrame (shoulders,
        elbows, and wrists) and applies a decision tree to classify them.

        Parameters:
        -----------
        dfX : pd.DataFrame
            DataFrame containing (x, y, z) coordinates for anatomical points.
        option : bool, optional
            If False (default), simplifies classification to a binary output (0 or 1).

        Returns:
        --------
        pd.Series
            Series with the classifications for each DataFrame row.

        Notes:
        ------
        - Uses helper functions `distancia_euclidiana_2d` and `colineares`.
        - Expected DataFrame columns:
          ["xLEFT_SHOULDER", "xRIGHT_SHOULDER", "xLEFT_WRIST", "xRIGHT_WRIST", "xLEFT_ELBOW", "xRIGHT_ELBOW"].
        - Classification is based on wrist-to-shoulder Euclidean distance and the
          angle formed by shoulder, elbow, and wrist.
        """

        colunas = [
            "xLEFT_SHOULDER",
            "xRIGHT_SHOULDER",
            "xLEFT_WRIST",
            "xRIGHT_WRIST",
            "xLEFT_ELBOW",
            "xRIGHT_ELBOW",
        ]

        y = pd.Series(dtype=int)

        # limiar_distancia, limiar_colinear = 0.25, 30
        limiar_distancia, limiar_colinear = dist, colinear

        for x in dfX.index:

            # xLEFT_SHOULDER, yLEFT_SHOULDER, zLEFT_SHOULDER
            idx0 = dfX.columns.get_loc(colunas[0])
            p0_L = dfX.iloc[:, idx0 : idx0 + 3].loc[x]
            p0_L = p0_L.to_numpy()

            # xRIGHT_SHOULDER, yRIGHT_SHOULDER, zRIGHT_SHOULDER
            idx1 = dfX.columns.get_loc(colunas[1])
            p0_R = dfX.iloc[:, idx1 : idx1 + 3].loc[x]
            p0_R = p0_R.to_numpy()

            # xLEFT_WRIST, yLEFT_WRIST, zLEFT_WRIST
            idx2 = dfX.columns.get_loc(colunas[2])
            p2_L = dfX.iloc[:, idx2 : idx2 + 3].loc[x]
            p2_L = p2_L.to_numpy()

            # xRIGHT_WRIST, yRIGHT_WRIST, zRIGHT_WRIST
            idx3 = dfX.columns.get_loc(colunas[3])
            p2_R = dfX.iloc[:, idx3 : idx3 + 3].loc[x]
            p2_R = p2_R.to_numpy()

            # xLEFT_ELBOW, yLEFT_ELBOW, zLEFT_ELBOW
            idx4 = dfX.columns.get_loc(colunas[4])
            p1_L = dfX.iloc[:, idx4 : idx4 + 3].loc[x]
            p1_L = p1_L.to_numpy()

            # xRIGHT_ELBOW, yRIGHT_ELBOW, zRIGHT_ELBOW
            idx5 = dfX.columns.get_loc(colunas[5])
            p1_R = dfX.iloc[:, idx5 : idx5 + 3].loc[x]
            p1_R = p1_R.to_numpy()

            pt1_L = p0_L[0:2]
            pt1_R = p0_R[0:2]

            pt2_L = p2_L[0:2]
            pt2_R = p2_R[0:2]

            # p0, pt1 -> shoulder
            # p2, pt2 -> wrist
            # p1 -> elbow

            left_distancia = self.distancia_euclidiana_2d(pt1_L, pt2_L)
            right_distancia = self.distancia_euclidiana_2d(pt1_R, pt2_R)

            left_teta = self.colineares(p0_L, p1_L, p2_L)
            right_teta = self.colineares(p0_R, p1_R, p2_R)

            # Decision tree
            if (
                left_distancia > limiar_distancia
                and right_distancia > limiar_distancia
            ):
                if (
                    left_teta < limiar_colinear
                    and right_teta < limiar_colinear
                ):
                    y.loc[x] = 3
                elif left_teta < limiar_colinear:
                    y.loc[x] = 1
                elif right_teta < limiar_colinear:
                    y.loc[x] = 2
                else:
                    y.loc[x] = 0
            elif left_distancia > limiar_distancia:
                if left_teta < limiar_colinear:
                    y.loc[x] = 1
                else:
                    y.loc[x] = 0
            elif right_distancia > limiar_distancia:
                if right_teta < limiar_colinear:
                    y.loc[x] = 2
                else:
                    y.loc[x] = 0
            else:
                y.loc[x] = 0

            # if x == 50:
            #     break

        if not option:
            for i in y.index:
                if y.loc[i] != 0:
                    y.loc[i] = 1

        return y

    def plot_dataframe(self, dfX: pd.DataFrame) -> np.ndarray:
        dfX = dfX.reset_index(drop=True)

        colunas = [
            "xLEFT_SHOULDER",
            "xRIGHT_SHOULDER",
            "xLEFT_WRIST",
            "xRIGHT_WRIST",
            "xLEFT_ELBOW",
            "xRIGHT_ELBOW",
        ]

        # xLEFT_SHOULDER, yLEFT_SHOULDER, zLEFT_SHOULDER
        idx0 = dfX.columns.get_loc(colunas[0])
        p0_L = dfX.iloc[:, idx0 : idx0 + 3].loc[0]
        p0_L = p0_L.to_numpy()

        # xRIGHT_SHOULDER, yRIGHT_SHOULDER, zRIGHT_SHOULDER
        idx1 = dfX.columns.get_loc(colunas[1])
        p0_R = dfX.iloc[:, idx1 : idx1 + 3].loc[0]
        p0_R = p0_R.to_numpy()

        # xLEFT_WRIST, yLEFT_WRIST, zLEFT_WRIST
        idx2 = dfX.columns.get_loc(colunas[2])
        p2_L = dfX.iloc[:, idx2 : idx2 + 3].loc[0]
        p2_L = p2_L.to_numpy()

        # xRIGHT_WRIST, yRIGHT_WRIST, zRIGHT_WRIST
        idx3 = dfX.columns.get_loc(colunas[3])
        p2_R = dfX.iloc[:, idx3 : idx3 + 3].loc[0]
        p2_R = p2_R.to_numpy()

        # xLEFT_ELBOW, yLEFT_ELBOW, zLEFT_ELBOW
        idx4 = dfX.columns.get_loc(colunas[4])
        p1_L = dfX.iloc[:, idx4 : idx4 + 3].loc[0]
        p1_L = p1_L.to_numpy()

        # xRIGHT_ELBOW, yRIGHT_ELBOW, zRIGHT_ELBOW
        idx5 = dfX.columns.get_loc(colunas[5])
        p1_R = dfX.iloc[:, idx5 : idx5 + 3].loc[0]
        p1_R = p1_R.to_numpy()

        # p0, pt1 -> shoulder
        # p2, pt2 -> wrist
        # p1 -> elbow

        parts = np.array([p0_L, p1_L, p2_L, p0_R, p1_R, p2_R])

        return parts

    def normalizacao(self, sk):
        def z_rotation(angle):
            rotation_matrix = np.array(
                [
                    [cos(angle), -sin(angle), 0, 0],
                    [sin(angle), cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            return rotation_matrix

        def move(dx, dy, dz):
            T = np.eye(4)
            T[0, -1] = dx
            T[1, -1] = dy
            T[2, -1] = dz
            return T

        skNorm = np.transpose(sk)
        skNorm = np.vstack([skNorm, np.ones(np.size(skNorm, 1))])

        dx = -sk[self.TO_COCO_IDX[HKP.Value("NOSE")]][0]
        dy = -sk[self.TO_COCO_IDX[HKP.Value("NOSE")]][1]
        dz = 0

        T = move(dx, dy, dz)

        # print("T: ", T)

        skNorm = np.transpose(sk)
        num_columns = np.size(skNorm, 1)
        ones_line = np.ones(num_columns)
        skNorm = np.vstack([skNorm, ones_line])

        # print("skNorm: ", skNorm)

        skT = np.dot(T, skNorm)
        skT = np.transpose(skT).reshape(17, 4, 1)

        # print("skT: ", skT[0:2])
        p0 = [
            [0],
            [0],
            skT[self.TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][2],
            [1.0],
        ]
        p1 = [
            [1],
            [0],
            skT[self.TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][2],
            [1.0],
        ]
        p2 = skT[self.TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]]

        direcao = skT[self.TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][
            1
        ] / np.abs(skT[self.TO_COCO_IDX[HKP.Value("RIGHT_SHOULDER")]][1])

        # print("p0: ", p0, "p2: ", p2[0], "p2: ", p2, "p3: ", p3)

        vetor1 = np.subtract(p1, p0).reshape(1, 4)[0][0:3]
        vetor2 = np.subtract(p2, p0).reshape(1, 4)[0][0:3]

        norm_vetor1 = np.linalg.norm(vetor1)
        norm_vetor2 = np.linalg.norm(vetor2)

        prod_interno = np.dot(vetor1, vetor2)

        if norm_vetor1 != 0 and norm_vetor2 != 0:
            sim = prod_interno / (norm_vetor1 * norm_vetor2)
        else:
            sim = 0

        # angle = (180 * np.arccos(sim)) / np.pi

        if direcao < 0:
            angle = np.arccos(sim)
        else:
            angle = -np.arccos(sim)

        # print("Angle: ", angle)

        Rz = z_rotation(angle)

        # print("Rz: ", Rz)

        M = Rz @ T

        # print("M: ", M)

        skM = np.dot(M, skNorm)
        skM = np.transpose(skM).reshape(17, 4, 1)

        # print("skM: ", skM[0:2])

        skM = skM.reshape(17, 4)[:, 0:3]

        # print("skM: ", skM)

        return skM
