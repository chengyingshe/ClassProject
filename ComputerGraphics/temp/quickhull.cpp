#include <iostream>
#include <vector>
#include <cmath>
#include <set>
#include <tuple>

using namespace std;

struct Point {
    double x, y, z;

    Point(double _x = 0, double _y = 0, double _z = 0) : x(_x), y(_y), z(_z) {}

    // 重载比较运算符，以便可以在set中使用
    bool operator<(const Point &p) const {
        if (x != p.x) return x < p.x;
        if (y != p.y) return y < p.y;
        return z < p.z;
    }
};

struct Face {
    Point a, b, c;
    
    Face(Point _a, Point _b, Point _c) : a(_a), b(_b), c(_c) {}
};

// 叉积，计算向量AB和AC的法向量（二维、三维空间均可）
Point crossProduct(const Point& A, const Point& B, const Point& C) {
    Point AB(B.x - A.x, B.y - A.y, B.z - A.z);
    Point AC(C.x - A.x, C.y - A.y, C.z - A.z);
    return Point(AB.y * AC.z - AB.z * AC.y, AB.z * AC.x - AB.x * AC.z, AB.x * AC.y - AB.y * AC.x);
}

// 点到平面的距离公式（点P到平面ABC）
double pointToPlaneDistance(const Point& A, const Point& B, const Point& C, const Point& P) {
    Point normal = crossProduct(A, B, C);
    return fabs(normal.x * (P.x - A.x) + normal.y * (P.y - A.y) + normal.z * (P.z - A.z)) /
           sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
}

// 判断点P是否在平面的同侧（用于排除内点）
double signedDistanceToPlane(const Point& A, const Point& B, const Point& C, const Point& P) {
    Point normal = crossProduct(A, B, C);
    return normal.x * (P.x - A.x) + normal.y * (P.y - A.y) + normal.z * (P.z - A.z);
}

// 从给定面的外部点集中找到距离该面最远的点
Point findFurthestPoint(const vector<Point>& points, const Face& face) {
    double maxDistance = -1;
    Point furthestPoint;
    
    for (const Point& p : points) {
        double distance = pointToPlaneDistance(face.a, face.b, face.c, p);
        if (distance > maxDistance) {
            maxDistance = distance;
            furthestPoint = p;
        }
    }
    
    return furthestPoint;
}

// QuickHull 核心递归函数
void quickHullRecursive(set<Point>& hull, const vector<Point>& points, const Face& face) {
    // 找到距离当前面最远的点
    Point furthestPoint = findFurthestPoint(points, face);
    
    // 如果最远的点距离为0（即所有点都在当前凸包内），停止递归
    if (pointToPlaneDistance(face.a, face.b, face.c, furthestPoint) < 1e-9) {
        hull.insert(face.a);
        hull.insert(face.b);
        hull.insert(face.c);
        return;
    }

    // 分治：将外部点分成新生成的面的点集，并递归处理
    vector<Point> leftOfAB, leftOfBC, leftOfCA;
    
    for (const Point& p : points) {
        if (signedDistanceToPlane(face.a, face.b, furthestPoint, p) > 0) {
            cout << "add to leftOfAB" << endl;
            leftOfAB.push_back(p);
        }
        if (signedDistanceToPlane(face.b, face.c, furthestPoint, p) > 0) {
            cout << "add to leftOfBC" << endl;

            leftOfBC.push_back(p);
        }
        if (signedDistanceToPlane(face.c, face.a, furthestPoint, p) > 0) {
            cout << "add to leftOfCA" << endl;

            leftOfCA.push_back(p);
        }
    }

    // 递归处理每个新的子问题
    quickHullRecursive(hull, leftOfAB, Face(face.a, face.b, furthestPoint));
    quickHullRecursive(hull, leftOfBC, Face(face.b, face.c, furthestPoint));
    quickHullRecursive(hull, leftOfCA, Face(face.c, face.a, furthestPoint));
}

// QuickHull 算法的入口函数
set<Point> quickHull(const vector<Point>& points) {
    set<Point> hull;

    // 1. 寻找极点来初始化初始四面体
    Point minX = points[0], maxX = points[0];
    for (const Point& p : points) {
        if (p.x < minX.x) minX = p;
        if (p.x > maxX.x) maxX = p;
    }

    Point minY = points[0], maxY = points[0];
    for (const Point& p : points) {
        if (p.y < minY.y) minY = p;
        if (p.y > maxY.y) maxY = p;
    }

    Point minZ = points[0], maxZ = points[0];
    for (const Point& p : points) {
        if (p.z < minZ.z) minZ = p;
        if (p.z > maxZ.z) maxZ = p;
    }

    // 2. 形成初始四面体
    Face initialFace(minX, maxX, maxY);

    // 3. 递归求解凸包
    quickHullRecursive(hull, points, initialFace);

    return hull;
}

// 主函数
int main() {
    vector<Point> points = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 4}, {1, 1, 1}, {2, 0, 0}, {0, 2, 0}, {2, 3, 0}, {0, 2, 4}    
    };

    set<Point> convexHull = quickHull(points);

    cout << "Convex Hull points:\n";
    for (const auto& p : convexHull) {
        cout << "(" << p.x << ", " << p.y << ", " << p.z << ")\n";
    }

    return 0;
}
