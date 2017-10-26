#include <vector>
#include <limits>
#include <iostream>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"

Model *model = NULL;

const int width  = 800;
const int height = 800;

Vec3f light_dir(1,1,1);
Vec3f       eye(1,1,3);
Vec3f    center(0,0,0);
Vec3f        up(0,1,0);

struct Shader : public IShader {
    Eigen::Matrix<float,2,3> varying_uv;  // triangle uv coordinates, written by the vertex shader, read by the fragment shader
    Eigen::Matrix<float,4,3> varying_tri; // triangle coordinates (clip coordinates), written by VS, read by FS
    Eigen::Matrix<float,3,3> varying_nrm; // normal per vertex to be interpolated by FS
    Eigen::Matrix<float,3,3> ndc_tri;     // triangle in normalized device coordinates

    virtual Vec4f vertex(int iface, int nthvert) {
        varying_uv.col(nthvert) = model->uv(iface, nthvert);
        Eigen::Matrix<float,4,4> pm = Projection*ModelView;
        Vec4f gl_Vertex = pm*embed(model->vert(iface, nthvert), 1.f);
        varying_nrm.col(nthvert) = (pm.transpose().inverse()*embed(model->normal(iface, nthvert), 0.f)).head(3);
        varying_tri.col(nthvert) = gl_Vertex;
        ndc_tri.col(nthvert) = (gl_Vertex/gl_Vertex[3]).head(3);
        return gl_Vertex;
    }

    virtual bool fragment(Vec3f bar, TGAColor &color) {
        Vec3f bn = varying_nrm*bar;
        bn.normalize();
        Vec2f uv = varying_uv*bar;

        Eigen::Matrix<float,3,3> A;
        A.row(0) = ndc_tri.col(1) - ndc_tri.col(0);
        A.row(1) = ndc_tri.col(2) - ndc_tri.col(0);
        A.row(2) = bn;

        Eigen::Matrix<float,3,3> AI = A.inverse();

        Vec3f i = AI * Vec3f(varying_uv(0,1) - varying_uv(0,0), varying_uv(0,2) - varying_uv(0,0), 0);
        i.normalize();

        Vec3f j = AI * Vec3f(varying_uv(1,1) - varying_uv(1,0), varying_uv(1,2) - varying_uv(1,0), 0);
        j.normalize();

        Eigen::Matrix<float,3,3> B;
        B.col(0) = i;
        B.col(1) = j;
        B.col(2) = bn;

        Vec3f n = B*model->normal(uv);
        n.normalize();

        float diff = std::max(0.f, n.dot(light_dir));
        color = model->diffuse(uv)*diff;

        return false;
    }
};

int main(int argc, char** argv) {
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
        return 1;
    }

    float *zbuffer = new float[width*height];
    for (int i=width*height; i--; zbuffer[i] = -std::numeric_limits<float>::max());

    TGAImage frame(width, height, TGAImage::RGB);
    lookat(eye, center, up);
    viewport(width/8, height/8, width*3/4, height*3/4);
    projection(-1.f/(eye-center).norm());
    light_dir = (Projection*ModelView*embed(light_dir, 0.f)).head(3);
    light_dir.normalize();

    for (int m=1; m<argc; m++) {
        model = new Model(argv[m]);
        Shader shader;
        for (int i=0; i<model->nfaces(); i++) {
            for (int j=0; j<3; j++) {
                shader.vertex(i, j);
            }
            triangle(shader.varying_tri, shader, frame, zbuffer);
        }
        delete model;
    }
    frame.flip_vertically(); // to place the origin in the bottom left corner of the image
    frame.write_tga_file("framebuffer.tga");

    delete [] zbuffer;
    return 0;
}

