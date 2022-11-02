#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp> //Thanks to Alessandro
#include <fstream>

using namespace cv;
using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;
//    cv::String img1  = "/home/user/CLionProjects/3d-mapping/cmake-build-debug/origin_1.jpg";
//    cv::String img2 = "/home/user/CLionProjects/3d-mapping/cmake-build-debug/origin_2.jpg";
//
//    cv::String images[2] = {img1, img2};
//
//    for (int i=0; i < 2; i++) {
//        cv::Mat gray = cv::imread(images[i], cv::IMREAD_GRAYSCALE);
//        gray = 255 - gray;
//        cv::Mat bin;
//        cv::threshold(gray, bin, 1, 255, cv::THRESH_BINARY);
//        cv::imwrite("gray.jpg", gray);
//
//        const cv::Mat input = cv::imread("gray.jpg", 0); //Load as grayscale
//
//        cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
//        std::vector<cv::KeyPoint> keypoints;
//        detector->detect(input, keypoints);
//
//        // Add results to image and save.
//        cv::Mat output;
//        cv::drawKeypoints(input, keypoints, output);
//        if( i == 0) {
//            cv::imwrite("sift_result0.jpg", output);
//        } else{
//            cv::imwrite("sift_result1.jpg", output);
//        }
//
//        std::cout << "done" << std::endl;
//
//    }

    // ---------- MATCHES --------------
    double ratio = 0.9;

    cv::Mat image1 = cv::imread("/home/user/CLionProjects/3d-mapping/data/2019_03_13_Nadir_g401b40071_f020_043.JPG");
    cv::Mat image2 = cv::imread("/home/user/CLionProjects/3d-mapping/data/2019_03_13_Nadir_g401b40071_f020_044.JPG");

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(10000);
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create(10000);

    vector<KeyPoint> keypoints1, keypoints2;
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    cout << "# keypoints of image1 :" << keypoints1.size() << endl;
    cout << "# keypoints of image2 :" << keypoints2.size() << endl;

    Mat descriptors1,descriptors2;
    extractor->compute(image1,keypoints1,descriptors1);
    extractor->compute(image2,keypoints2,descriptors2);

    cout << "Descriptors size :" << descriptors1.cols << ":"<< descriptors1.rows << endl;

    vector< vector<DMatch> > matches12, matches21;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->knnMatch( descriptors1, descriptors2, matches12, 2);
    matcher->knnMatch( descriptors2, descriptors1, matches21, 2);


    std::cout << "after matcher" << std::endl;
    BFMatcher bfmatcher(NORM_L2, true);
    vector<DMatch> matches;
    bfmatcher.match(descriptors1, descriptors2, matches);



    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < descriptors1.rows; i++)
    {
        double dist = matches12[i].data()->distance;
        if(dist < min_dist)
            min_dist = dist;
        if(dist > max_dist)
            max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    cout << "Matches1-2:" << matches12.size() << endl;
    cout << "Matches2-1:" << matches21.size() << endl;

    std::vector<DMatch> good_matches1, good_matches2;
    for(auto & i : matches12) {
        if(i[0].distance < ratio * i[1].distance) {
            good_matches1.push_back(i[0]);
//            std::cout<< i[0].distance << " " << i[1].distance << std::endl;
            std::cout<< i[0].queryIdx << " " << i[1].trainIdx << std::endl;
        }
    }

    for(auto & i : matches21) {
        if(i[0].distance < ratio * i[1].distance)
            good_matches2.push_back(i[0]);
    }

    cout << "Good matches1:" << good_matches1.size() << endl;
    cout << "Good matches2:" << good_matches2.size() << endl;

    // Symmetric Test
    std::vector<DMatch> better_matches;
    for(auto & i : good_matches1) {
        for(auto & j : good_matches2) {
            if(i.queryIdx == j.trainIdx && j.queryIdx == i.trainIdx) {
                better_matches.push_back(DMatch(i.queryIdx, i.trainIdx, i.distance));

                break;
            }
        }
    }

    cout << "Better matches:" << better_matches.size() << endl;

    // show it on an image
    Mat output;
    drawMatches(image1, keypoints1, image2, keypoints2, better_matches, output);
//    for(int i=0; i< better_matches.size(); i++) {
//        std::cout << better_matches[i] << std::endl;
//    }
    //imshow("Matches result",output);
   // cv::imwrite("matches_drone.jpg", output);

    std::vector< Point2f > points_list_1;
    std::vector< Point2f > points_list_2;

   for (int i=0; i<better_matches.size(); i++) {
       // Get the matching keypoints for each of the images
       int img1_index = better_matches[i].queryIdx;
       int img2_index = better_matches[i].trainIdx;

       // x - columns
       // y - rows
       // get the coordinates
       points_list_1.push_back(keypoints1[img1_index].pt);
       points_list_2.push_back(keypoints2[img2_index].pt);

   }

    std::cout<< points_list_1.size() << " " << points_list_2.size() << std::endl;

    ofstream file1, file2;
    file1.open ("/home/user/CLionProjects/3d-mapping/coordinates/file1.txt");
    file2.open ("/home/user/CLionProjects/3d-mapping/coordinates/file2.txt");
    for(int i=0; i<points_list_1.size(); i++) {
        file1 << points_list_1[i].x << " " << points_list_1[i].y << "\n";
        file2 << points_list_2[i].x << " " << points_list_2[i].y << "\n";

    }
    file1.close();
    file2.close();

    waitKey(0);


    return 0;
}
