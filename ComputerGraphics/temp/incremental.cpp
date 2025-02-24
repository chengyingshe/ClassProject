#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <cassert>
#include <algorithm>

using namespace std;
#define EPSOLON 1e-6


struct Point3D {
    double x, y, z;
    Point3D() {}
    Point3D(double x, double y, double z) : x(x), y(y), z(z) {}
    bool operator==(const Point3D& other) const {
        return abs(x - other.x) < EPSOLON && abs(y - other.y) < EPSOLON && abs(z - other.z) < EPSOLON;
    }
};

// 辅助函数，计算向量叉积
Point3D cross(const Point3D& u, const Point3D& v) {
    return Point3D(
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x
    );
}

// 辅助函数，计算向量点积
double dot(const Point3D& u, const Point3D& v) {
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

// 判断点 p 是否在三角形 abc 上方
bool isAbove(const Point3D& a, const Point3D& b, const Point3D& c, const Point3D& p) {
    Point3D n = cross(Point3D(b.x - a.x, b.y - a.y, b.z - a.z), Point3D(c.x - a.x, c.y - a.y, c.z - a.z));
    double d = dot(n, Point3D(p.x - a.x, p.y - a.y, p.z - a.z));
    return d > 0;
}

// 三角形结构体
struct Triangle {
    Point3D a, b, c;
    Triangle(const Point3D& a, const Point3D& b, const Point3D& c) : a(a), b(b), c(c) {}
    bool operator==(const Triangle& other) const {
        return (a == other.a && b == other.b && c == other.c) ||
               (a == other.b && b == other.a && c == other.c) ||
               (a == other.c && b == other.a && c == other.b);
    }

};

// 主体算法
class ConvexHull3D {
public:
    ConvexHull3D(const vector<Point3D>& points) {
        assert(points.size() >= 4);
        buildHull(points);
    }

    const vector<Triangle>& getTriangles() const {
        return triangles;
    }

private:
    vector<Triangle> triangles;  // 保存凸包三角形
    vector<Point3D> points;
    void buildHull(const vector<Point3D>& points) {
        // 初始化，先用前四个点构建一个简单的四面体
        addInitialTetrahedron(points);

        // 逐点插入
        for (size_t i = 4; i < points.size(); ++i) {
            addPointToHull(points[i]);
        }
    }

    void addInitialTetrahedron(const vector<Point3D>& points) {
        // 初始四面体，用前四个点构建
        triangles.push_back(Triangle(points[0], points[1], points[2]));
        triangles.push_back(Triangle(points[0], points[1], points[3]));
        triangles.push_back(Triangle(points[0], points[2], points[3]));
        triangles.push_back(Triangle(points[1], points[2], points[3]));
        this->points.push_back(points[0]);
        this->points.push_back(points[1]);
        this->points.push_back(points[2]);
        this->points.push_back(points[3]);
    }

    void addPointToHull(const Point3D& p) {
        // 找出所有可以看见新点的三角形面
        vector<Triangle> visibleFaces;
        for (const Triangle& t : triangles) {
            if (isAbove(t.a, t.b, t.c, p)) {
                visibleFaces.push_back(t);
            }
        }

        // 删除这些可见的三角形
        for (const Triangle& face : visibleFaces) {
            auto it = find(triangles.begin(), triangles.end(), face);
            if (it != triangles.end()) {
                triangles.erase(it);
            }
        }

        // 生成新三角形，将新点与可见面的边缘构成新的面
        for (const Triangle& face : visibleFaces) {
            triangles.push_back(Triangle(face.a, face.b, p));
            triangles.push_back(Triangle(face.b, face.c, p));
            triangles.push_back(Triangle(face.c, face.a, p));
        }
    }
};

int main() {
    // 输入3D点集
    vector<Point3D> points = {
        Point3D(0, 0, 0),
        Point3D(1, 0, 0),
        Point3D(0, 1, 0),
        Point3D(0, 0, 1),
        Point3D(1, 1, 1),
        Point3D(0.5, 0.5, 2)
    };

    // 构建凸包
    ConvexHull3D hull(points);

    // 输出凸包上的三角形面
    const vector<Triangle>& triangles = hull.getTriangles();
    for (const Triangle& t : triangles) {
        cout << "Triangle: (" << t.a.x << ", " << t.a.y << ", " << t.a.z << ") -> ("
             << t.b.x << ", " << t.b.y << ", " << t.b.z << ") -> ("
             << t.c.x << ", " << t.c.y << ", " << t.c.z << ")" << endl;
    }

    return 0;
}
