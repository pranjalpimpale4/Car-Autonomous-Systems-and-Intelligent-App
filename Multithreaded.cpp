#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <omp.h>
#include <numeric>

using namespace cv;
using namespace std;
using namespace std::chrono;

class CascadeStopSignDetector {
private:
    CascadeClassifier classifier;
public:
    bool initialize(const string& modelFile) {
        return classifier.load(modelFile);
    }

    vector<Rect> detect(const Mat& inputFrame) {
        vector<Rect> candidates;
        Mat gray;
        cvtColor(inputFrame, gray, COLOR_BGR2GRAY);
        Rect roi(0, 0, inputFrame.cols, inputFrame.rows / 2);
        Mat cropped = gray(roi);
        classifier.detectMultiScale(cropped, candidates, 1.1, 3);
        for (auto& box : candidates) {
            box.y += roi.y;
        }
        return candidates;
    }
};

queue<Mat> frameQueue, outputQueue;
mutex mtx, out_mtx;
condition_variable cv_in, cv_out;
bool finished = false;

Vec2f prev_left(0, 0), prev_right(0, 0);
vector<Vec2f> left_history, right_history;
const int history_size = 5;

Mat regionOfInterest(Mat image) {
    int height = image.rows;
    int width = image.cols;
    Point pts[1][4] = {
        { Point(width * 0.0, height), Point(width * 1.0, height),
          Point(width * 0.7, height * 0.6), Point(width * 0.3, height * 0.6) }
    };
    Mat mask = Mat::zeros(image.size(), image.type());
    const Point* ppt[1] = { pts[0] };
    int npt[] = { 4 };
    fillPoly(mask, ppt, npt, 1, Scalar(255));
    Mat masked;
    bitwise_and(image, mask, masked);
    return masked;
}


vector<int> makeCoordinates(Mat image, Vec2f line_params) {
    float slope = line_params[0];
    float intercept = line_params[1];
    int y1 = image.rows;
    int y2 = int(y1 * 0.6);
    int x1 = int((y1 - intercept) / slope);
    int x2 = int((y2 - intercept) / slope);
    return { x1, y1, x2, y2 };
}

vector<Vec4i> averageLines(Mat image, vector<Vec4i>& lines) {
    vector<Vec2f> left_fit, right_fit;
    for (Vec4i l : lines) {
        int x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
        float dx = x2 - x1;
        float dy = y2 - y1;
        if (abs(dx) < 1e-6) continue;
        float slope = dy / dx;
        if (abs(slope) < 0.5 || abs(slope) > 2.0) continue;
        float intercept = y1 - slope * x1;
        if (slope < 0) left_fit.push_back(Vec2f(slope, intercept));
        else right_fit.push_back(Vec2f(slope, intercept));
    }
    vector<Vec4i> output;
    if (!left_fit.empty()) {
        Vec2f avg(0, 0);
        for (auto& f : left_fit) avg += f;
        avg /= float(left_fit.size());
        left_history.push_back(avg);
        if (left_history.size() > history_size) left_history.erase(left_history.begin());
        Vec2f smooth_left(0, 0);
        for (auto& l : left_history) smooth_left += l;
        smooth_left /= float(left_history.size());
        prev_left = smooth_left;
        vector<int> coords = makeCoordinates(image, prev_left);
        output.push_back(Vec4i(coords[0], coords[1], coords[2], coords[3]));
    }
    if (!right_fit.empty()) {
        Vec2f avg(0, 0);
        for (auto& f : right_fit) avg += f;
        avg /= float(right_fit.size());
        right_history.push_back(avg);
        if (right_history.size() > history_size) right_history.erase(right_history.begin());
        Vec2f smooth_right(0, 0);
        for (auto& r : right_history) smooth_right += r;
        smooth_right /= float(right_history.size());
        prev_right = smooth_right;
        vector<int> coords = makeCoordinates(image, prev_right);
        output.push_back(Vec4i(coords[0], coords[1], coords[2], coords[3]));
    }
    return output;
}

Mat displayLines(Mat image, vector<Vec4i> lines) {
    Mat line_image = Mat::zeros(image.size(), image.type());
    for (Vec4i l : lines) {
        line(line_image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 10);
    }
    return line_image;
}

void annotateDetections(Mat& frame, const vector<Rect>& boxes, Scalar color = Scalar(0, 255, 0)) {
    for (const Rect& rect : boxes) {
        rectangle(frame, rect, color, 2);
    }
}

void reader(VideoCapture& cap, int& totalFramesRead) {
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        resize(frame, frame, Size(640, 360));
        {
            unique_lock<mutex> lock(mtx);
            frameQueue.push(frame.clone());
        }
        totalFramesRead++;
        cv_in.notify_one();
    }
    cout << "Total Frames Read: " << totalFramesRead << endl;
    finished = true;
    cv_in.notify_all();
    cv_out.notify_all();
}

void processor(CascadeStopSignDetector& detector, int& totalFramesProcessed) {
    while (true) {
        Mat frame;
        {
            unique_lock<mutex> lock(mtx);
            cv_in.wait(lock, [] { return !frameQueue.empty() || finished; });
            if (frameQueue.empty() && finished) break;
            frame = frameQueue.front();
            frameQueue.pop();
        }

        Mat gray, blur, canny_img;
        #pragma omp parallel sections
        {
            #pragma omp section
            { cvtColor(frame, gray, COLOR_BGR2GRAY); }

            #pragma omp section
            { GaussianBlur(frame, blur, Size(5, 5), 0); }
        }

        Canny(blur, canny_img, 50, 150);
        Mat cropped = regionOfInterest(canny_img);

        vector<Vec4i> lines;
        HoughLinesP(cropped, lines, 2, CV_PI / 180, 50, 50, 20);
        vector<Vec4i> averaged_lines = averageLines(frame, lines);
        Mat line_image = displayLines(frame, averaged_lines);
        Mat combo;
        addWeighted(frame, 0.8, line_image, 1.0, 1, combo);

        vector<Rect> detections = detector.detect(frame);
        annotateDetections(combo, detections);

        {
            unique_lock<mutex> lock(out_mtx);
            outputQueue.push(combo);
        }
        totalFramesProcessed++;
        cv_out.notify_one();
    }
    cv_out.notify_all();
}

void writer(VideoWriter& out, int& totalFramesWritten) {
    vector<double> fpsBuffer;
    const int fpsWindow = 10;
    auto last_time = high_resolution_clock::now();
    while (true) {
        Mat frame;
        {
            unique_lock<mutex> lock(out_mtx);
            cv_out.wait(lock, [] { return !outputQueue.empty() || (finished && frameQueue.empty()); });
            if (outputQueue.empty() && finished && frameQueue.empty()) break;
            if (!outputQueue.empty()) {
                frame = outputQueue.front();
                outputQueue.pop();
            } else {
                continue;
            }
        }

        out.write(frame);
        totalFramesWritten++;
        imshow("Combined Detection", frame);

        auto now = high_resolution_clock::now();
        double fps = 1.0 / duration<double>(now - last_time).count();
        last_time = now;

        fpsBuffer.push_back(fps);
        if (fpsBuffer.size() > fpsWindow) fpsBuffer.erase(fpsBuffer.begin());
        double avgFPS = accumulate(fpsBuffer.begin(), fpsBuffer.end(), 0.0) / fpsBuffer.size();

        cout << "Smoothed FPS: " << avgFPS << endl;

        this_thread::sleep_for(chrono::milliseconds(5));

        if (waitKey(1) == 27) break;
    }
}

int main() {
    const string inputVideo = "test00.mp4";
    const string outputVideo = "op_test00.mp4";
    const string cascadeFile = "stop_sign_classifier_2.xml";

    VideoCapture cap(inputVideo);
    if (!cap.isOpened()) {
        cout << "Error opening video" << endl;
        return -1;
    }

    int width = 640, height = 360;
    double fps = cap.get(CAP_PROP_FPS);
    double frameCount = cap.get(CAP_PROP_FRAME_COUNT);
    double duration = frameCount / fps;

    cout << "Video FPS: " << fps << endl;
    cout << "Total Frames: " << frameCount << endl;
    cout << "Video Duration (sec): " << duration << endl;

    VideoWriter out(outputVideo, VideoWriter::fourcc('m','p','4','v'), fps, Size(width, height));

    CascadeStopSignDetector detector;
    if (!detector.initialize(cascadeFile)) {
        cerr << "Failed to load Haar cascade classifier." << endl;
        return -1;
    }

    int totalFramesRead = 0;
    int totalFramesProcessed = 0;
    int totalFramesWritten = 0;

    thread t1(reader, ref(cap), ref(totalFramesRead));
    thread t2(processor, ref(detector), ref(totalFramesProcessed));
    thread t3(writer, ref(out), ref(totalFramesWritten));

    t1.join();
    t2.join();
    t3.join();

    cout << "All frames processed. Video output complete." << endl;
    cout << "Total Frames Written: " << totalFramesWritten << endl;

    cap.release();
    out.release();
    destroyAllWindows();
    return 0;
}

