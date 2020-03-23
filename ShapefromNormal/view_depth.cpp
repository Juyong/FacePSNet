#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Eigen>
#include <unistd.h>

int main(int argc, char *argv[])
{
    std::ifstream fin(argv[1]);
    std::string path;
    while (fin >> path)
    {
        chdir(path.c_str());
        cv::Mat normal_map = cv::imread("normal.png");
        int height = normal_map.rows, width = normal_map.cols;

        cv::Mat depth_show(height, width, CV_8UC1);
        depth_show.setTo(0);
        std::ifstream finuv("uvs.txt");
        std::ifstream finz("opt_z.txt");

        std::ifstream fincam("pca_pose_cam.txt");
        float f, cx, cy;
        for (int i = 0; i < 185; i++)
        {
            fincam >> f;
        }
        fincam >> f >> cx >> cy;

        int u, v;
        float z;
        std::ofstream fout("depth.obj");
        Eigen::MatrixXi pixel_ids(height, width);
        pixel_ids.setOnes();
        pixel_ids *= -1;
        int cur_id = 0;
        while (finuv >> u >> v)
        {
            finz >> z;
            depth_show.at<char>(v, u) = int(z + 700) * 2;

            float X = -(u - cx) * z / f;
            float Y = (v - cy) * z / f;
            fout << "v " << X << " " << Y << " " << z << std::endl;
            pixel_ids(v, u) = cur_id;
            cur_id++;
        }
        fincam.close();
        finuv.close();
        finz.close();

        for (int y = 0; y < height - 1; y++)
        {
            for (int x = 0; x < width - 1; x++)
            {
                int id_c = pixel_ids(y, x);
                int id_r = pixel_ids(y, x + 1);
                int id_b = pixel_ids(y + 1, x);
                if (id_c != -1 && id_r != -1 && id_b != -1)
                {
                    fout << "f " << id_c + 1 << " " << id_b + 1 << " " << id_r + 1 << std::endl;
                }
            }
        }

        for (int y = 1; y < height; y++)
        {
            for (int x = 1; x < width; x++)
            {
                int id_c = pixel_ids(y, x);
                int id_l = pixel_ids(y, x - 1);
                int id_t = pixel_ids(y - 1, x);
                if (id_c != -1 && id_l != -1 && id_t != -1)
                {
                    fout << "f " << id_c + 1 << " " << id_t + 1 << " " << id_l + 1 << std::endl;
                }
            }
        }

        fout.close();
        std::cout << path + " view_depth done" << std::endl;
    }
    fin.close();
    return 0;
}