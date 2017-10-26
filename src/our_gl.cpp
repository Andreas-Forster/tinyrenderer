#include <cmath>
#include <limits>
#include <cstdlib>
#include "our_gl.h"

Matrix ModelView;
Matrix Viewport;
Matrix Projection;

IShader::~IShader() {}

void viewport(int x, int y, int w, int h) {
    Viewport = Matrix::Identity();
    Viewport(0,3) = x+w/2.f;
    Viewport(1,3) = y+h/2.f;
    Viewport(2,3) = 1.f;
    Viewport(0,0) = w/2.f;
    Viewport(1,1) = h/2.f;
    Viewport(2,2) = 0;
}

void projection(float coeff) {
    Projection = Matrix::Identity();
    Projection(3,2) = coeff;
}

void lookat(Vec3f eye, Vec3f center, Vec3f up) {
    Vec3f z = eye-center;
    z.normalize();
    Vec3f x = up.cross(z);
    x.normalize();
    Vec3f y = z.cross(x);
    y.normalize();
    Matrix Minv = Matrix::Identity();
    Matrix Tr   = Matrix::Identity();
    for (int i=0; i<3; i++) {
        Minv(0,i) = x(i);
        Minv(1,i) = y(i);
        Minv(2,i) = z(i);
        Tr(i,3) = -center(i);
    }
    ModelView = Minv*Tr;
}

Vec3f barycentric(Vec2f A, Vec2f B, Vec2f C, Vec2f P) {
    Eigen::Matrix<float,2,3> s;
    for (int i=2; i--; ) {
        s(i,0) = C(i)-A(i);
        s(i,1) = B(i)-A(i);
        s(i,2) = A(i)-P(i);
    }
    Vec3f u = s.row(0).cross(s.row(1));
    if (std::abs(u(2))>1e-2) // dont forget that u(2) is integer. If it is zero then triangle ABC is degenerate
        return Vec3f(1.f-(u(0)+u(1))/u(2), u(1)/u(2), u(0)/u(2));
    return Vec3f(-1,1,1); // in this case generate negative coordinates, it will be thrown away by the rasterizator
}

void triangle(Eigen::Matrix<float,4,3> &clipc, IShader &shader, TGAImage &image, float *zbuffer) {
    Eigen::Matrix<float,3,4> pts  = (Viewport*clipc).transpose();
    Eigen::Matrix<float,3,2> pts2;
    for (int i=0; i<3; i++) pts2.row(i) = (pts.row(i)/pts(i,3)).head(2);

    Vec2f bboxmin( std::numeric_limits<float>::max(),  std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    Vec2f clamp(image.get_width()-1, image.get_height()-1);
    for (int i=0; i<3; i++) {
        for (int j=0; j<2; j++) {
            bboxmin(j) = std::max(0.f,      std::min(bboxmin(j), pts2(i,j)));
            bboxmax(j) = std::min(clamp(j), std::max(bboxmax(j), pts2(i,j)));
        }
    }
    Vec2i P;
    TGAColor color;
    for (P(0)=bboxmin(0); P(0)<=bboxmax(0); P(0)++) {
        for (P(1)=bboxmin(1); P(1)<=bboxmax(1); P(1)++) {
          //std::cout << P(0) << "/" << P(1) << std::endl;
            Vec3f bc_screen  = barycentric(pts2.row(0), pts2.row(1), pts2.row(2), P.cast<float>());
            Vec3f bc_clip = Vec3f(bc_screen(0)/pts(0,3), bc_screen(1)/pts(1,3), bc_screen(2)/pts(2,3));
            bc_clip = bc_clip/(bc_clip(0)+bc_clip(1)+bc_clip(2));
            float frag_depth = clipc.row(2).dot(bc_clip);
            if (bc_screen(0)<0 || bc_screen(1)<0 || bc_screen(2)<0 || zbuffer[P(0)+P(1)*image.get_width()]>frag_depth) continue;
            bool discard = shader.fragment(bc_clip, color);
            if (!discard) {
                zbuffer[P(0)+P(1)*image.get_width()] = frag_depth;
                image.set(P(0), P(1), color);
            }
        }
    }
}

