import cv2
import numpy as np
import math
from itertools import combinations

# 각도 계산 함수
def calculate_angle(p1, p2, p3):
    """
    주어진 세 점에서 p2를 기준으로 각도를 계산.
    :param p1: 첫 번째 점 (x1, y1)
    :param p2: 두 번째 점 (기준점)
    :param p3: 세 번째 점 (x3, y3)
    :return: 각도 (degree)
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

# 이미지 읽기
image = cv2.imread("IMG_3119.jpg")
image_triangle = image.copy()
image_washer = image.copy()

# 그레이스케일 및 블러 처리
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.bitwise_not(gray)

# 엣지 검출 및 이진화
edges = cv2.Canny(gray, 100, 200)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# 컨투어 찾기
contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# 내부 원 중심점 및 와셔 분석
inner_centers = []
for contour in contours:
    if cv2.contourArea(contour) < 500:
        continue
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter = 2 * radius
    if 400 > diameter or 500 < diameter: 
        inner_centers.append((int(x), int(y)))

    # 내부 원 추출
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked_image = cv2.bitwise_and(gray, gray, mask=mask)
    circles = cv2.HoughCircles(masked_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=radius/2,
                               param1=50, param2=30, minRadius=5, maxRadius=int(radius/2))
    num_inner_circles = circles.shape[1] if circles is not None else 0

    # 와셔 결과 시각화
    color = (0, 0, 255) if diameter >= 500 else (0, 255, 0)
    cv2.circle(image_washer, (int(x), int(y)), 5, color, -1)
    cv2.putText(image_washer, f"Diameter: {int(diameter)}", (int(x - radius), int(y - radius - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 2)
    cv2.putText(image_washer, f"Inner: {num_inner_circles}", (int(x - radius), int(y + radius + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 2)

print(inner_centers)
# 삼각형 및 각도 계산
valid_triangles = []
radius_max_threshold = 400
radius_min_threshold = 300

for start, end in combinations(inner_centers, 2):
    # 두 점 사이 거리 조건
    distance = math.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
    if radius_min_threshold <= distance <= radius_max_threshold:
        cv2.line(image_triangle, start, end, (0, 255, 0), 1)

for triangle in combinations(inner_centers, 3):
    p1, p2, p3 = triangle
    d1 = math.dist(p1, p2)
    d2 = math.dist(p2, p3)
    d3 = math.dist(p1, p3)
    if d1 + d2 > d3 and d2 + d3 > d1 and d1 + d3 > d2:
        valid_triangles.append(triangle)

for p1, p2, p3 in valid_triangles:
    angle1 = calculate_angle(p2, p1, p3)
    angle2 = calculate_angle(p1, p2, p3)
    angle3 = calculate_angle(p1, p3, p2)

    # 각도가 45 < 각도 < 50인 경우에만 출력
    if 45 < angle1 < 50:
        # 선의 중앙 좌표 계산
        mid_point1 = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(image_triangle, f"{int(angle1)}", mid_point1,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 1)

    if 45 < angle2 < 50:
        mid_point2 = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)
        cv2.putText(image_triangle, f"{int(angle2)}", mid_point2,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 1)

    if 45 < angle3 < 50:
        mid_point3 = ((p1[0] + p3[0]) // 2, (p1[1] + p3[1]) // 2)
        cv2.putText(image_triangle, f"{int(angle3)}", mid_point3,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 1)

# 두 이미지를 동시에 출력
combined = np.hstack((image_triangle, image_washer))
cv2.imshow("Triangles and Washer Analysis", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
