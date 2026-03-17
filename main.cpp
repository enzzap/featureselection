#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace std;

// i will try my best to give a good description of each
class FeatureSelection {
public:
    vector<vector<double>> data; // Stores the feature values for each row in the dataset
    vector<int> classLabel; // Stores the class label for each row
    vector<int> bestSet; // Stores the best feature subset found during the search
    double bestAccuracy = 0.0; // Stores the accuracy of that best subset

    void readData(const string& fileName) {
        ifstream inFS(fileName);
        string line;

        
        // gives error if no file
        if (!inFS.is_open()) {
            cout << "Unable to open " << fileName << ".\n";
            return;
        }

        // reading input from the file line by line
        while (getline(inFS, line)) {
            if (line.empty()) continue;

            stringstream ss(line);
            double value;
            vector<double> row;

            ss >> value;
            classLabel.push_back((int)value); // first number becomes the classlabel

            while (ss >> value) {
                row.push_back(value); // rest of the numbers become feature values
            }

            data.push_back(row);
        }

        inFS.close();
    }

    //////////////////////////////////////////////////////////
    // HELPER FUNCTIONS
    //////////////////////////////////////////////////////////

    // checks if feature is already in the set
    bool contains(const vector<int>& set, int feature) {
        for (int x : set) {
            if (x == feature) return true;
        }
        return false;
    }

    // prints the set in a neat format
    void printSet(const vector<int>& set) {
        cout << "{";
        for (int i = 0; i < (int)set.size(); i++) {
            if (i > 0) cout << ",";
            cout << set[i];
        }
        cout << "}";
    }
    
    //////////////////////////////////////////////////////////
    // ACCURACY FUNCTION
    //////////////////////////////////////////////////////////
    double accuracy(const vector<int>& featureSet) {
        int correct = 0;

        for (int i = 0; i < (int)data.size(); i++) {
            double bestDistance = 1e18; // c++ doesnt have infinity like the matlab code given so i put it to a very high value
            int nearestLabel = -1;

            for (int k = 0; k < (int)data.size(); k++) {
                if (i == k) continue; // leave one out (skips itself)

                double distance = 0.0;

                for (int j = 0; j < (int)featureSet.size(); j++) {
                    int feature = featureSet[j] - 1; // this just makes the indices from 1-.... to 0-.... so it aligns with vector index numbers as it starts from 0
                    double diff = data[i][feature] - data[k][feature];
                    distance += diff * diff; // euclidean distance calculation (i don't square root to save calculation time as the comparison still holds true if it was square rooted or not, so it was redundant to root it)
                }

                if (distance < bestDistance) { // distance comparison
                    bestDistance = distance;
                    nearestLabel = classLabel[k];
                }
            }

            if (nearestLabel == classLabel[i]) { // checks if prediction is correct
                correct++;
            }
        }

        return (double)correct / data.size();
    }

    //////////////////////////////////////////////////////////
    // FORWARD SELECTION
    //////////////////////////////////////////////////////////
    void forwardSelection() {
        vector<int> currentSet; // no features selected yet, same as "{}"
        int totalFeatures = data[0].size();

        for (int level = 1; level <= totalFeatures; level++) {
            int bestFeature = -1;
            double bestSoFar = 0.0;
            vector<int> bestCandidateSet;

            for (int feature = 1; feature <= totalFeatures; feature++) { // adds a feature on each loop
                if (contains(currentSet, feature)) continue; // skips already chosen features

                vector<int> tempSet = currentSet;
                tempSet.push_back(feature);

                double acc = accuracy(tempSet); // runs the accuracy function on that subset

                cout << "\tUsing feature(s) ";
                printSet(tempSet);
                cout << " accuracy is " << fixed << setprecision(1)
                     << acc * 100 << "%\n";

                if (acc > bestSoFar) { // chooses the feature with highest accuracy
                    bestSoFar = acc;
                    bestFeature = feature;
                    bestCandidateSet = tempSet;
                }
            }

            currentSet = bestCandidateSet; // adds that selected feature

            cout << "Feature set ";
            printSet(currentSet);
            cout << " was best, accuracy is " << fixed << setprecision(1)
                 << bestSoFar * 100 << "%\n\n";

            if (bestSoFar > bestAccuracy) { // another check to make sure that the overall highest is kept track of
                bestAccuracy = bestSoFar;
                bestSet = currentSet;
            }
        }
    }

    //////////////////////////////////////////////////////////
    // BACKWARD ELIMINATION
    //////////////////////////////////////////////////////////
    void backwardElimination() {
        vector<int> currentSet;
        int totalFeatures = data[0].size();

        for (int i = 1; i <= totalFeatures; i++) { // starts with all features already selected
            currentSet.push_back(i);
        }

        // start with all features as the first known best candidate
        bestSet = currentSet;
        bestAccuracy = accuracy(currentSet);

        // keep removing until only one feature is left
        while (currentSet.size() > 1) {
            int bestFeatureToRemoveIndex = -1;
            double bestSoFar = 0.0;
            vector<int> bestCandidateSet; // gets the baseline accuracy using every feature

            for (int i = 0; i < (int)currentSet.size(); i++) {
                vector<int> tempSet = currentSet;
                tempSet.erase(tempSet.begin() + i);

                double acc = accuracy(tempSet); // runs the accuracy function on that smaller subset

                cout << "\tUsing feature(s) ";
                printSet(tempSet);
                cout << " accuracy is " << fixed << setprecision(1)
                     << acc * 100 << "%\n";

                if (acc > bestSoFar) { // keeps the removal that gives the highest accuracy
                    bestSoFar = acc;
                    bestFeatureToRemoveIndex = i;
                    bestCandidateSet = tempSet;
                }
            }

            currentSet = bestCandidateSet; // removes whichever feature gave the best result

            cout << "Feature set ";
            printSet(currentSet);
            cout << " was best, accuracy is " << fixed << setprecision(1)
                 << bestSoFar * 100 << "%\n\n";

            if (bestSoFar > bestAccuracy) { // another check to make sure that the overall highest is kept track of
                bestAccuracy = bestSoFar;
                bestSet = currentSet;
            }
        }
    }
};

//////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////

int main() {
    // I formatted this almost 1:1 to the examples given in the project pdf
    string inputFile;
    int choice;

    cout << "Welcome to Chahith Srikanth's Feature Selection Algorithm.\n";
    cout << "Type in the name of the file to test: ";
    cin >> inputFile;

    FeatureSelection fs;
    fs.readData(inputFile);

    if (fs.data.empty()) { // stops the program if the file failed to load
        cout << "No data loaded.\n";
        return 1;
    }

    cout << "\nType the number of the algorithm you want to run.\n\n";
    cout << "\t1) Forward Selection\n";
    cout << "\t2) Backward Elimination\n\n";
    cin >> choice;

    cout << "\n";
    cout << "This dataset has " << fs.data[0].size()
         << " features (not including the class attribute), with "
         << fs.data.size() << " instances.\n\n";

    vector<int> allFeatures;
    for (int i = 1; i <= (int)fs.data[0].size(); i++) { // builds the full feature set
        allFeatures.push_back(i);
    }

    double allAcc = fs.accuracy(allFeatures); // gets the accuracy using all features first

    cout << "Running nearest neighbor with all " << fs.data[0].size()
         << " features, using \"leaving-one-out\" evaluation, I get an accuracy of "
         << fixed << setprecision(1) << allAcc * 100 << "%\n\n";

    cout << "Beginning search.\n\n";
    // I added this timer for my own reference and for the report statistic to know how long the code took to run
    auto start = chrono::high_resolution_clock::now();

    if (choice == 1) {
        fs.forwardSelection();
    } else if (choice == 2) {
        fs.backwardElimination();
    } else {
        cout << "Invalid choice\n";
        return 1;
    }

    cout << "Finished search!! The best feature subset is ";
    fs.printSet(fs.bestSet);
    cout << ", which has an accuracy of "
         << fixed << setprecision(1) << fs.bestAccuracy * 100 << "%\n";
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "\nTime taken: " << elapsed.count() << " seconds\n";

    return 0;
}