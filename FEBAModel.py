import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Concatenate, LayerNormalization
from transformers import TFBertModel, BertTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import itertools
import shap
from mlxtend.plotting import plot_decision_regions


class BoostingClassifier:

    def __init__(self, config):
        self.num_iterations = config.get('num_iterations', 10)
        self.epsilon = config.get('epsilon', 0.1)
        self.lstm_units = config.get('lstm_units', 32)
        self.bert_model_name = config.get(
            'bert_model_name', 'bert-base-uncased')
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.error_rate = config.get('error_rate', 0.1)
        self.classifier_weight = config.get('classifier_weight', 0.5)
        self.bert_activation = config.get('bert_activation', 'relu')
        self.lstm_activation = config.get('lstm_activation', 'relu')
        self.bert_loss = config.get('bert_loss', 'binary_crossentropy')
        self.lstm_loss = config.get('lstm_loss', 'binary_crossentropy')
        self.optimizer = config.get('optimizer', 'adam')

        # Stratified KFold parameters
        self.n_splits = config.get('n_splits', 5)
        self.shuffle = config.get('shuffle', True)
        self.random_state = config.get('random_state', 30)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.weak_classifiers = []
        self.classifier_weights = []
        self.explainer = None

    def fit(self, X_train, y_train, sample_weights=None):
        tf.get_logger().setLevel('ERROR')
        num_samples, num_features = X_train.shape
        num_classes = len(np.unique(y_train))

        if sample_weights is None:
            sample_weights = np.ones(num_samples) / num_samples

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        fold_accuracies = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            for _ in range(self.num_iterations):
                weak_classifiers = []

                for class_label in range(num_classes):
                    binary_labels = np.where(
                        y_fold_train == class_label, 0.9, 0)
                    combined_model = self._train_weak_classifier(
                        X_fold_train, binary_labels, sample_weights, num_features)
                    weak_classifiers.append(combined_model)

                    if self.epsilon > 0:
                        X_adv = self._fgsm_attack(
                            X_fold_train, binary_labels, combined_model)
                        binary_labels = np.where(
                            y_fold_train == class_label, 0.9, 0)
                        combined_model = self._train_weak_classifier(
                            X_adv, binary_labels, sample_weights, num_features)
                        weak_classifiers.append(combined_model)

                self.weak_classifiers.append(weak_classifiers)
                combined_pred = self._get_combined_predictions(X_fold_val)

                sample_weights = sample_weights.astype(np.float32)
                error_rate = 1.0 - accuracy_score(y_fold_val, np.argmax(
                    combined_pred, axis=1), sample_weight=sample_weights[val_idx])
                error_rate = max(error_rate, 1e-10)
                classifier_weight = 0.5 * \
                    np.log((1.0 - error_rate) / error_rate)
                self.classifier_weights.append(classifier_weight)

                sample_weights *= np.exp(classifier_weight *
                                         (y_fold_val != np.argmax(combined_pred, axis=1)))
                sample_weights /= np.sum(sample_weights)

            # Evaluating on validation fold
            y_pred_val = self.predict(X_fold_val)
            fold_accuracy = accuracy_score(y_fold_val, y_pred_val)
            fold_accuracies.append(fold_accuracy)

        # Plotting Model Performance across Folds
        self.plot_model_performance_across_folds(fold_accuracies)

    def _train_weak_classifier(self, X_train, binary_labels, sample_weights, num_features):
        bert_inputs = self._prepare_bert_input(X_train)
        bilstm_model = self._build_bilstm_model(num_features)

        combined_model = self._build_combined_model(bert_inputs, bilstm_model)
        optimizer = self._get_optimizer()
        combined_model.compile(loss=self.bert_loss,
                               optimizer=optimizer, metrics=['accuracy'])

        for _ in range(self.num_epochs):
            combined_model.fit([bert_inputs, np.expand_dims(X_train, axis=2)], binary_labels,
                               sample_weight=sample_weights, epochs=1, batch_size=self.batch_size, verbose=0)
            if self.epsilon > 0:
                X_adv = self._fgsm_attack(
                    X_train, binary_labels, combined_model)
                combined_model.fit([bert_inputs, np.expand_dims(X_adv, axis=2)], binary_labels,
                                   sample_weight=sample_weights, epochs=1, batch_size=self.batch_size, verbose=0)

        return combined_model

    def _build_bilstm_model(self, num_features):
        bilstm_input = Input(shape=(num_features, 1))
        bilstm_output = Bidirectional(
            LSTM(self.lstm_units, activation=self.lstm_activation))(bilstm_input)
        bilstm_output = LayerNormalization()(bilstm_output)
        return Model(inputs=bilstm_input, outputs=bilstm_output)

    def _build_combined_model(self, bert_inputs, bilstm_model):
        bert_model = TFBertModel.from_pretrained(self.bert_model_name)
        bert_output = bert_model(bert_inputs)[1]
        bert_output = LayerNormalization()(bert_output)
        combined = Concatenate()([bert_output, bilstm_model.output])
        output = Dense(1, activation=self.bert_activation)(combined)
        return Model(inputs=[bert_model.input, bilstm_model.input], outputs=output)

    def _prepare_bert_input(self, texts):
        tokenized = self.tokenizer(
            texts.tolist(), padding=True, truncation=True, return_tensors='tf')
        return tokenized

    def _get_combined_predictions(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.weak_classifiers[0])
        combined_pred = np.zeros((num_samples, num_classes))

        for i, classifiers in enumerate(self.weak_classifiers):
            pred = np.zeros((num_samples, num_classes))

            for class_label, classifier in enumerate(classifiers):
                bert_inputs = self._prepare_bert_input(X)
                bilstm_input = np.expand_dims(X, axis=2)
                pred[:, class_label] += classifier.predict(
                    [bert_inputs, bilstm_input]).flatten()

            combined_pred += self.classifier_weights[i] * pred

        return combined_pred

    def predict(self, X):
        combined_pred = self._get_combined_predictions(X)
        return np.argmax(combined_pred, axis=1)

    def explain(self, X):
        if not self.explainer:
            explainer_model = Model(
                inputs=self.weak_classifiers[0][0].input, outputs=self.weak_classifiers[0][0].output)
            self.explainer = shap.Explainer(explainer_model, X)

        shap_values = self.explainer(X)
        return shap_values

    def plot_summary(self, X):
        shap_values = self.explain(X)
        shap.summary_plot(shap_values, X)

    def plot_dependence(self, X, feature):
        shap_values = self.explain(X)
        shap.dependence_plot(feature, shap_values, X)

    def local_explanations(self, X, index):
        if not self.explainer:
            explainer_model = Model(
                inputs=self.weak_classifiers[0][0].input, outputs=self.weak_classifiers[0][0].output)
            self.explainer = shap.Explainer(explainer_model, X)

        shap_values = self.explainer(X[index:index+1])
        return shap_values

    def global_feature_importance(self, X):
        if not self.explainer:
            explainer_model = Model(
                inputs=self.weak_classifiers[0][0].input, outputs=self.weak_classifiers[0][0].output)
            self.explainer = shap.Explainer(explainer_model, X)

        shap_values = self.explainer(X)
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        return feature_importance

    def compute_metrics(self, X_test, y_test):
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        return metrics

    def plot_model_performance_across_folds(self, fold_accuracies):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(fold_accuracies) + 1), fold_accuracies,
                marker='o', linestyle='-', color='b')
        plt.title('Model Performance across Folds')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, X_test, y_test):
        y_scores = self._get_combined_predictions(X_test)
        # Assuming binary classification
        fpr, tpr, _ = roc_curve(y_test, y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curve(self, X_test, y_test):
        y_scores = self._get_combined_predictions(X_test)
        precision, recall, _ = precision_recall_curve(
            y_test, y_scores[:, 1])  # Assuming binary classification

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue',
                lw=2, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(self, X_train, y_train):
        train_accuracies = []
        val_accuracies = []

        for _ in range(self.num_epochs):
            self.fit(X_train, y_train)
            y_pred_train = self.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_accuracies.append(train_accuracy)

            y_pred_val = self.predict(X_train)
            val_accuracy = accuracy_score(y_train, y_pred_val)
            val_accuracies.append(val_accuracy)

        plt.figure(figsize=(8, 6))
        epochs = range(1, self.num_epochs + 1)
        plt.plot(epochs, train_accuracies, marker='o',
                 linestyle='-', color='b', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, marker='o', linestyle='-',
                 color='r', label='Validation Accuracy')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm, classes, normalize=False, cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix'

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    def plot_decision_boundary(self, X, y):
        combined_pred = self._get_combined_predictions(X)
        plt.figure(figsize=(10, 8))
        plot_decision_regions(X, np.argmax(
            combined_pred, axis=1), clf=self, markers='o', colors='blue,red')
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_model_confidence_distribution(self, X):
        combined_pred = self._get_combined_predictions(X)
        plt.figure(figsize=(8, 6))
        plt.hist(np.max(combined_pred, axis=1),
                 bins=20, edgecolor='black', alpha=0.7)
        plt.title('Model Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _fgsm_attack(self, X, y, model):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X)
            bert_inputs = self._prepare_bert_input(X)
            bilstm_input = np.expand_dims(X, axis=2)
            prediction = model([bert_inputs, bilstm_input], training=True)
            loss = loss_object(y, prediction)

        gradient = tape.gradient(loss, X)
        perturbation = self.epsilon * tf.sign(gradient)
        adv_X = X + perturbation
        adv_X = tf.clip_by_value(adv_X, clip_value_min=0.0, clip_value_max=1.0)
        return adv_X

    def _get_optimizer(self):
        if self.optimizer == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        else:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. Supported optimizers are 'adam' and 'sgd'.")

    @staticmethod
    def main():
        config = {
            'num_iterations': 5,
            'epsilon': 0.1,
            'lstm_units': 64,
            'bert_en_uncased_L-12_H-768_A-12_4': 'bert_en_uncased_preprocess_3',
            'num_epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001,
            'error_rate': 0.1,
            'classifier_weight': 0.5,
            'bert_activation': 'relu',
            'lstm_activation': 'relu',
            'bert_loss': 'binary_crossentropy',
            'lstm_loss': 'binary_crossentropy',
            'optimizer': 'adam',
            'n_splits': 5,
            'shuffle': True,
            'random_state': 30
        }

        feat_df = pd.read_csv('./80clp_feat_df (1).csv')
        feat_df.drop(feat_df.columns[0], axis=1, inplace=True)
        X, y = feat_df.iloc[:, :-1].values, feat_df.iloc[:, -1].values.astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=40)
        
        classifier = BoostingClassifier(config)
        classifier.fit(X_train, y_train)

        metrics = classifier.compute_metrics(X_test, y_test)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        classifier.plot_model_performance_across_folds()
        classifier.plot_roc_curve(X_test, y_test)
        classifier.plot_precision_recall_curve(X_test, y_test)
        classifier.plot_learning_curves(X_train, y_train)
        classifier.plot_confusion_matrix(
            metrics['confusion_matrix'], classes=np.unique(y_train))
        classifier.plot_decision_boundary(X_test, y_test)
        classifier.plot_model_confidence_distribution(X_test)


if __name__ == "__main__":
    BoostingClassifier.main()
