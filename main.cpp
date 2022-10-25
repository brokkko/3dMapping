#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp> //Thanks to Alessandro

using namespace cv;
using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;
//    cv::String img1  = "/home/user/PycharmProjects/3d-mapping/data/2020_07_03_PhotoCamera_g401b40179_f001_033.JPG";
//    cv::String img2 = "/home/user/PycharmProjects/3d-mapping/data/2020_07_03_PhotoCamera_g401b40179_f001_034.JPG";
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

    cv::Mat image1 = cv::imread("/home/user/PycharmProjects/3d-mapping/data/2020_07_03_PhotoCamera_g401b40179_f001_033.JPG");
    cv::Mat image2 = cv::imread("/home/user/PycharmProjects/3d-mapping/data/2020_07_03_PhotoCamera_g401b40179_f001_034.JPG");

    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    cv::Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create();

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
    //matcher->knnMatch( descriptors2, descriptors1, matches21, 2);

    std::cout << "after matcher" << std::endl;
    //BFMatcher bfmatcher(NORM_L2, true);
    //vector<DMatch> matches;
    //bfmatcher.match(descriptors1, descriptors2, matches);
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
    for(int i=0; i < matches12.size(); i++)
    {
        if(matches12[i][0].distance < ratio * matches12[i][1].distance)
            good_matches1.push_back(matches12[i][0]);
    }

    for(int i=0; i < matches21.size(); i++)
    {
        if(matches21[i][0].distance < ratio * matches21[i][1].distance)
            good_matches2.push_back(matches21[i][0]);
    }

    cout << "Good matches1:" << good_matches1.size() << endl;
    cout << "Good matches2:" << good_matches2.size() << endl;

    // Symmetric Test
    std::vector<DMatch> better_matches;
    for(int i=0; i<good_matches1.size(); i++)
    {
        for(int j=0; j<good_matches2.size(); j++)
        {
            if(good_matches1[i].queryIdx == good_matches2[j].trainIdx && good_matches2[j].queryIdx == good_matches1[i].trainIdx)
            {
                better_matches.push_back(DMatch(good_matches1[i].queryIdx, good_matches1[i].trainIdx, good_matches1[i].distance));
                break;
            }
        }
    }

    cout << "Better matches:" << better_matches.size() << endl;

    // show it on an image
    Mat output;
    drawMatches(image1, keypoints1, image2, keypoints2, better_matches, output);
    imshow("Matches result",output);
    waitKey(0);


    return 0;
}
