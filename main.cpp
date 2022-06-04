#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <float.h>
#include <algorithm>
#include <set>
#include <chrono>
using namespace std;
using namespace std::chrono;
using ll = long long;

long double distance(vector<long double> &a, vector<long double> &b, auto &feats) {
    double sqsum = 0.0;
    auto sqr = [&] (const double x) { return x * x; };

    // Only features that are relevant should
    // count towards the distance function
    for (auto i:feats) {
        sqsum += sqr(a[i] - b[i]);
    }

    return sqrt(sqsum);
}

double nnClassify(vector<vector<long double>> &data, auto &feats) {
    double res = 0;

    // For all of the points in the data
    for (int i = 0; i < data.size(); i++) {
        int cat = 1;
        long double best = DBL_MAX;

        vector<long double> point = data[i];

        // Compare the distance of the current point with the
        // rest of the points in the training data
        for (int j = 0; j < data.size(); j++) {
            if (i == j) continue;
            long double curr = distance(point, data[j], feats);
            if (curr < best) {
                cat = data[j][0];
                best = curr;
            }
        }

        if (cat == point[0]) res += 1;
    }

    return res / double(data.size());
}

void featureSelectionForward(vector<vector<long double>> data) {
    int samples = data.size();
    int features = data[0].size();

    vector<int> optimalFeatures;
    double optimalAccuracy = 0.0;

    vector<int> feats;
    for (int i = 0; i < features; i++) {
        int tokeep = -1;
        double best = 0.0;
        if (i == 0) {
            best = nnClassify(data, feats);
        }
        else {
            for (int j = 1; j < features; j++) {
                // Skip it if it is already in the features
                auto it = find(feats.begin(), feats.end(), j);
                if (it != feats.end()) continue;

                // Otherwise try it out!
                feats.push_back(j);
                // cout << "       Using feature set { ";
                // for (auto x:feats) cout << x << " ";
                // cout << "}, ";

                // Calculate the accuracy
                double res = nnClassify(data, feats);

                if (best < res) {
                    best = res;
                    tokeep = j;
                }
                // cout << "accuracy is " << res << endl;
                feats.pop_back();
            }
        }
        if (tokeep != -1) feats.push_back(tokeep);

        //cout << "Feature set { ";
        for (auto x:feats) cout << x << " ";
        //cout << "} was the best. Accuracy is " << best << endl << endl;
        cout << ", " << best << endl;
        if (best > optimalAccuracy) {
            optimalAccuracy = best;
            optimalFeatures = feats;
        }
    }

    // cout << "Done!" << endl;
    // cout << "The best feature subset was { ";
    // for (auto x:optimalFeatures) cout << x << " ";
    // cout << "}, that has an accuracy of " << optimalAccuracy << endl;
}

void featureSelectionBackward(vector<vector<long double>> data) {
    int samples = data.size();
    int features = data[0].size();

    set<int> optimalFeatures;
    double optimalAccuracy = 0.0;

    set<int> feats;
    for (int i = 1; i < features; i++) feats.insert(i);

    for (int i = 0; i < features; i++) {
        int toremove = -1;
        double best = 0.0;
        if (feats.empty()) {
            best = nnClassify(data, feats);
        }
        else {
            for (int j = 1; j < features; j++) {
                // Skip it if it is already removed
                if (feats.find(j) == feats.end()) continue;

                // Otherwise try it out!
                feats.erase(j);
                // cout << "       Using feature set { ";
                // for (auto x:feats) cout << x << " ";
                // cout << "}, ";

                // Calculate the accuracy
                double res = nnClassify(data, feats);

                if (best < res) {
                    best = res;
                    toremove = j;
                }
                // cout << "accuracy is " << res << endl;
                feats.insert(j);
            }
        }
        if (toremove != -1) feats.erase(toremove);

        //cout << "Feature set { ";
        for (auto x:feats) cout << x << " ";
        // cout << "} was the best. Accuracy is ";
        cout << ", " << best << endl;
        if (best > optimalAccuracy) {
            optimalAccuracy = best;
            optimalFeatures = feats;
        }
    }

    // cout << "Done!" << endl;
    // cout << "The best feature subset was { ";
    // for (auto x:optimalFeatures) cout << x << " ";
    // cout << "}, that has an accuracy of " << optimalAccuracy << endl;
}

int main() {
    cout << "Input 1) Small dataset or 2) Large dataset" << endl;
    int choice = 2;
    // cin >> choice;
    cout << choice << endl;
    int n, m;
    if (choice == 1) {
        cout << "Using small dataset!" << endl;
        freopen("smallTestData.txt", "r", stdin);
        n = 297;
        m = 11;
    }
    else if (choice == 2) {
        cout << "Using large dataset!" << endl;
        freopen("largeTestData.txt", "r", stdin);
        n = 1000;
        m = 41;
    }

    vector<vector<long double>> data(n, vector<long double> (m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> data[i][j];
        }
    }

    cout << "Starting forward selection" << endl;
    auto start = high_resolution_clock::now();
    featureSelectionForward(data);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds> (stop - start);
    cout << "Time: " << duration.count() << " milliseconds" << endl;
    cout << endl;

    cout << "Starting backward selection" << endl;
    start = high_resolution_clock::now();
    featureSelectionBackward(data);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds> (stop - start);
    cout << "Time: " << duration.count() << " milliseconds" << endl;
    cout << endl;
}
