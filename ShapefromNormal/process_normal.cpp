#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <unistd.h>

using namespace Eigen;

template <typename dtype>
void save_vec(std::vector<dtype> &vec, std::string path)
{
    std::ofstream fout(path);
    for (int i = 0; i < vec.size(); i++)
    {
        fout << vec[i] << std::endl;
    }
    fout.close();
}

template <typename dtype>
void save_vec_bin(std::vector<dtype> &vec, std::string path)
{
    FILE *pfile = fopen(path.c_str(), "wb");
    fwrite(vec.data(), sizeof(dtype), vec.size(), pfile);
    fclose(pfile);
}

template <typename dtype>
void save_mat(Matrix<dtype, -1, -1> &mat, std::string path)
{
    std::ofstream fout(path);
    for (int y = 0; y < mat.rows(); y++)
    {
        for (int x = 0; x < mat.cols(); x++)
        {
            fout << mat(y, x) << " ";
        }
        fout << std::endl;
    }
    fout.close();
}

template <typename dtype>
void save_mat_T_bin(Matrix<dtype, -1, -1> &mat, std::string path)
{
    Matrix<dtype, -1, -1> mat_t = mat.transpose();
    FILE *pfile = fopen(path.c_str(), "wb");
    fwrite(mat_t.data(), sizeof(dtype), mat_t.size(), pfile);
    fclose(pfile);
}

int main(int argc, char *argv[])
{
    std::ifstream fin(argv[1]);
    int bd_l = -1;
    int bd_t = -1;
    int bd_r = -1;
    int bd_b = -1;
    // if (argc > 2)
    // {
    //     bd_l = atoi(argv[2]);
    //     bd_t = atoi(argv[3]);
    //     bd_r = atoi(argv[4]);
    //     bd_b = atoi(argv[5]);
    // }
    std::string path;
    while (fin >> path)
    {
        chdir(path.c_str());
        cv::Mat normal_map = cv::imread("normal.png", CV_LOAD_IMAGE_UNCHANGED);
        int height = normal_map.rows, width = normal_map.cols;

        cv::Mat init_depth(height, width, CV_32FC1);
        FILE *pfile = fopen("init_depth.bin", "rb");
        fread(init_depth.data, sizeof(float), width * height, pfile);
        fclose(pfile);

        MatrixXi pixel_inds(height, width);
        pixel_inds.setOnes();
        pixel_inds *= -1;

        MatrixXf normals(3, height * width);
        MatrixXi pixel_poses(2, height * width);
        int valid_normal_id = 0;

        std::vector<float> init_deps;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                cv::Vec3w normal_w = normal_map.at<cv::Vec3w>(y, x);
                Vector3f normal;
                for (int i = 0; i < 3; i++)
                {
                    normal[i] = float(normal_w[2 - i]);
                }
                if (normal.norm() < 0.1)
                {
                    continue;
                }
                float init_dep = init_depth.at<float>(y, x);
                if (init_dep < -2000)
                {
                    continue;
                }
                init_deps.push_back(init_dep);
                normal = normal / 30000.0f - Vector3f::Ones();
                normals.col(valid_normal_id) = normal;
                pixel_inds(y, x) = valid_normal_id;
                pixel_poses(0, valid_normal_id) = x;
                pixel_poses(1, valid_normal_id) = y;
                valid_normal_id++;
            }
        }

        MatrixXf valid_normals = normals.leftCols(valid_normal_id).transpose();
        MatrixXi valid_poses = pixel_poses.leftCols(valid_normal_id).transpose();

        // for normal computation
        std::vector<int> valid_normal_pixels;
        std::vector<std::vector<int>> for_compute_normals;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int id_c = pixel_inds(y, x);
                if (id_c == -1)
                {
                    continue;
                }

                int id_l = -1, id_r = -1, id_t = -1, id_b = -1;
                if (x > 0)
                {
                    id_l = pixel_inds(y, x - 1);
                }
                if (x < width - 1)
                {
                    id_r = pixel_inds(y, x + 1);
                }
                if (y > 0)
                {
                    id_t = pixel_inds(y - 1, x);
                }
                if (y < height - 1)
                {
                    id_b = pixel_inds(y + 1, x);
                }
                std::vector<int> pixel_tris;
                if (id_r != -1 && id_t != -1)
                {
                    pixel_tris.push_back(id_c);
                    pixel_tris.push_back(id_r);
                    pixel_tris.push_back(id_t);
                }
                if (id_t != -1 && id_l != -1)
                {
                    pixel_tris.push_back(id_c);
                    pixel_tris.push_back(id_t);
                    pixel_tris.push_back(id_l);
                }
                if (id_l != -1 && id_b != -1)
                {
                    pixel_tris.push_back(id_c);
                    pixel_tris.push_back(id_l);
                    pixel_tris.push_back(id_b);
                }
                if (id_b != -1 && id_r != -1)
                {
                    pixel_tris.push_back(id_c);
                    pixel_tris.push_back(id_b);
                    pixel_tris.push_back(id_r);
                }

                if (pixel_tris.empty())
                {
                    continue;
                }

                valid_normal_pixels.push_back(id_c);
                int cur_tri_num = pixel_tris.size() / 3;
                if (cur_tri_num < 4)
                {
                    for (int i = cur_tri_num; i < 4; i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            pixel_tris.push_back(pixel_tris[3 * (i % cur_tri_num) + j]);
                        }
                    }
                }
                for_compute_normals.push_back(pixel_tris);
            }
        }

        // for laplacian computation
        std::vector<int> valid_lap_pixels;
        std::vector<std::vector<int>> for_compute_laps;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int id_c = pixel_inds(y, x);
                if (id_c == -1)
                {
                    continue;
                }

                std::vector<int> neighbors;
                if (x > 0)
                {
                    if (pixel_inds(y, x - 1) != -1)
                    {
                        neighbors.push_back(pixel_inds(y, x - 1));
                    }
                }
                if (x < width - 1)
                {
                    if (pixel_inds(y, x + 1) != -1)
                    {
                        neighbors.push_back(pixel_inds(y, x + 1));
                    }
                }
                if (y > 0)
                {
                    if (pixel_inds(y - 1, x) != -1)
                    {
                        neighbors.push_back(pixel_inds(y - 1, x));
                    }
                }
                if (y < height - 1)
                {
                    if (pixel_inds(y + 1, x) != -1)
                    {
                        neighbors.push_back(pixel_inds(y + 1, x));
                    }
                }

                if (neighbors.empty())
                {
                    continue;
                }

                valid_lap_pixels.push_back(id_c);
                int cur_neibor_num = neighbors.size();
                if (cur_neibor_num < 4)
                {
                    for (int i = cur_neibor_num; i < 4; i++)
                    {
                        neighbors.push_back(neighbors[i % cur_neibor_num]);
                    }
                }
                for_compute_laps.push_back(neighbors);
            }
        }

        // save normal lap info
        int valid_normals_num = valid_normal_pixels.size();
        MatrixXi for_normals(3, 4 * valid_normals_num);
        for (int j = 0; j < 4; j++)
        {
            for (int i = 0; i < valid_normals_num; i++)
            {
                for (int k = 0; k < 3; k++)
                {
                    for_normals(k, i * 4 + j) = for_compute_normals[i][3 * j + k];
                }
            }
        }
        int valid_laps_num = valid_lap_pixels.size();
        MatrixXi for_laps(4, valid_laps_num);
        for (int j = 0; j < 4; j++)
        {
            for (int i = 0; i < valid_laps_num; i++)
            {
                for_laps(j, i) = for_compute_laps[i][j];
            }
        }

        // if (argc > 2)
        // {
        //     std::cout << bd_l << " " << bd_t << " " << bd_r << " " << bd_b << std::endl;

        //     MatrixXi for_region_laps(5, valid_laps_num);
        //     int region_num = 0;
        //     for (int i = 0; i < valid_laps_num; i++)
        //     {
        //         int lap_id = valid_lap_pixels[i];
        //         int u = valid_poses(i, 0), v = valid_poses(i, 1);
        //         if (u >= bd_l && u <= bd_r && v >= bd_t && v <= bd_b)
        //         {
        //             for_region_laps(0, region_num) = lap_id;
        //             for (int j = 0; j < 4; j++)
        //             {
        //                 for_region_laps(j + 1, region_num) = for_compute_laps[i][j];
        //             }
        //             region_num++;
        //         }
                
        //     }
        //     std::cout << region_num << std::endl;
        //     for_region_laps = for_region_laps.leftCols(region_num);
        //     MatrixXi for_region_laps_t = for_region_laps.transpose();
        //     save_mat_T_bin(for_region_laps_t, "for_region_laps.bin");
        // }

        save_vec_bin(valid_normal_pixels, "valid_normals.bin");
        save_vec_bin(valid_lap_pixels, "valid_laps.bin");
        MatrixXi for_normals_t = for_normals.transpose();
        save_mat_T_bin(for_normals_t, "for_normals.bin");
        MatrixXi for_laps_t = for_laps.transpose();
        save_mat_T_bin(for_laps_t, "for_laps.bin");
        save_mat_T_bin(valid_normals, "normals.bin");
        save_vec_bin(init_deps, "init_deps.bin");
        save_mat(valid_poses, "uvs.txt");
        std::cout << path + " process_normal done" << std::endl;
    }
    fin.close();

    return 0;
}