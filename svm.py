"""
    SVMアルゴリズムで手書き文字の判定を学習し、また結果を評価します.
"""
import os
import joblib
from sklearn import svm, metrics

if __name__ == "__main__":

    if not os.path.exists("result"):
        os.mkdir("result")

    """
        **** ここを実装します（基礎課題） ****
        `csv`フォルダからデータを読み込み、SVMアルゴリズムを用いた学習を行ってください。
        そして学習結果を`result`フォルダに`svm.pkl`という名前で保存してください。

        実装ステップ：
            ・トレーニングデータを読み込む
            ・SVGアルゴリズムによる学習を行う
            ・テストデータを読み込む
            ・精度とメトリクスによる性能評価を行う
            ・学習結果を`result/svm.pkl`ファイルとして保存する

        参考になる情報
            講義スライドや答えを適宜確認しながら実装してみてください。
            サンプルを見ながら手を動かしながら学ぶという感じがお勧めです。

        ここが一番大変なところです。
        ぜひぜひ頑張ってください！！
    """
    # Load training data.
    with open("./csv/train-images.csv") as f:
        training_images = f.read().split("\n")[:60000]
    with open("./csv/train-labels.csv") as f:
        training_labels = f.read().split("\n")[:60000]

    # Convert data.
    training_images = [[int(i)/256 for i in image.split(",")] for image in training_images]
    training_labels = [int(l) for l in training_labels]

    # Use SVM.
    clf = svm.SVC()
    clf.fit(training_images, training_labels)

    # Load test data.
    with open("./csv/t10k-images.csv") as f:
        test_images = f.read().split("\n")[:500]
    with open("./csv/t10k-labels.csv") as f:
        test_labels = f.read().split("\n")[:500]

    # Convert data.
    test_images = [[int(i)/256 for i in image.split(",")] for image in test_images]
    test_labels = [int(l) for l in test_labels]

    # Evaluate the model.
    test_predict = clf.predict(test_images)

    # Show results.
    ac_score = metrics.accuracy_score(test_labels, test_predict)
    print("Accuracy:", ac_score)
    cl_report = metrics.classification_report(test_labels, test_predict)
    print(cl_report)

    # Save the training result
    joblib.dump(clf, "./result/svm.pkl")