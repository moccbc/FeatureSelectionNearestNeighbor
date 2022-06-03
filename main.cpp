#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <float.h>
#include <algorithm>
using namespace std;
using ll = long long;

long double distance(vector<long double> &a, vector<long double> &b, vector<int> &feats) {
    double sqsum = 0.0;
    auto sqr = [&] (const double x) { return x * x; };

    for (auto i:feats) {
        sqsum += sqr(a[i] - b[i]);
    }

    return sqrt(sqsum);
}

double nnClassify(vector<vector<long double>> &train, vector<vector<long double>> &test, vector<int> &feats) {
    double res = 0;

    for (auto point:test) {
        int cat = 1;
        long double best = DBL_MAX;

        int n = train.size();
        for (int i = 0; i < n; i++) {
            long double curr = distance(train[i], point, feats);
            if (curr < best) {
                cat = train[i][0];
                best = curr;
            }
        }

        if (cat == point[0]) res += 1;
    }

    return res / double(test.size());
}

void featureSelection(vector<vector<long double>> data) {
    int samples = data.size();
    int features = data[0].size();

    vector<vector<long double>> train(data.begin(), data.begin() + 200);
    vector<vector<long double>> test(data.begin()+200, data.end());

    int sum = 0;
    for (int i = 0; i < test.size(); i++) {
        if (test[i][0] == 1) sum++;
    }

    vector<int> optimalFeatures;
    double optimalAccuracy = 0.0;
    vector<int> feats;

    for (int i = 0; i < features; i++) {
        int tokeep = -1;
        double best = 0.0;
        if (i == 0) {
            best = nnClassify(train, test, feats);
        }
        else {
            for (int j = 1; j < features; j++) {
                // Skip it if it is already in the features
                auto it = find(feats.begin(), feats.end(), j);
                if (it != feats.end()) continue;

                // Otherwise try it out!
                feats.push_back(j);
                cout << "       Using feature set { ";
                for (auto x:feats) cout << x << " ";
                cout << "}, ";
                double res = nnClassify(train, test, feats);
                if (best < res) {
                    best = res;
                    tokeep = j;
                }
                cout << "accuracy is " << res << endl;
                feats.pop_back();
            }
        }
        if (tokeep != -1) feats.push_back(tokeep);

        cout << "Feature set { ";
        for (auto x:feats) cout << x << " ";
        cout << "} was the best. Accuracy is " << best << endl << endl;
        if (best > optimalAccuracy) {
            optimalAccuracy = best;
            optimalFeatures = feats;
        }
    }

    cout << "Done!" << endl;
    cout << "The best feature subset was { ";
    for (auto x:optimalFeatures) cout << x << " ";
    cout << "}, that has an accuracy of " << optimalAccuracy << endl;
}

int main() {
    freopen("smallTestData.txt", "r", stdin);

    int n = 297, m = 11;
    vector<vector<long double>> data(n, vector<long double> (m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> data[i][j];
        }
    }

    featureSelection(data);
}
