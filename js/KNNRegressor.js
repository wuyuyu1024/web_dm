// import * as tf from '@tensorflow/tfjs';

class KNNRegressor {
    constructor(k=4) {
        this.k = k;
        this.points = null;
        this.labels = null;
    }

    // Fit the model with data (points and vector labels)
    fit(points, labels) {
        // console.log(points)
        // console.log(labels)
        this.points = tf.tensor2d(points);
        this.labels = tf.tensor2d(labels); // Change to tensor2d for vector labels
    }

    // Predict the labels for multiple points
    predict(queryPoints) {
        // const queryPoints = tf.tensor2d(points, [points.length, 2]);
        // console.log('queryPoints')
        // console.log(queryPoints)
        let predictions_list = [];

        const epsilon = 1e-8; // Threshold for near-zero distances


        for (let i = 0; i < queryPoints.shape[0]; i++) {
            const queryPoint = queryPoints.slice([i, 0], [1, -1]).reshape([-1]);

            // Calculate distances
            const diff = this.points.sub(queryPoint);
            const distances = diff.pow(2).sum(1).sqrt();

            // Check for zero or near-zero distances
            const isNearZero = distances.lessEqual(epsilon);
            if (isNearZero.any().dataSync()[0]) {
                // Handle the near-zero case
                const nearZeroIndex = isNearZero.argMax().dataSync()[0];
                predictions_list.push(this.labels.slice([nearZeroIndex, 0], [1, -1]).dataSync());
                continue;
            }
            // Find the indices of the k nearest neighbors
            const sortedIndices = this.bubble_sort(distances.dataSync());
            // const sortedIndices = customTensorSort(distances, this.k);
            // console.log(sortedIndices)

            // Retrieve the labels and distances of the k nearest neighbors
            const nearestDistances = distances.gather(sortedIndices);
            const nearestLabels = this.labels.gather(sortedIndices);

            // Calculate weighted average for each label dimension
            const weights = nearestDistances.pow(-1).expandDims(1);
            const weightedLabels = nearestLabels.mul(weights);
            const summedWeights = weights.sum(0);
            const weightedAverage = weightedLabels.sum(0).div(summedWeights);

            predictions_list.push(weightedAverage);
        }
        const predictions = tf.stack(predictions_list)
        // free memory
        // queryPoints.dispose();
        predictions_list.forEach(tensor => tensor.dispose());
        return predictions;
    }

    bubble_sort(distances){
        let indices_sorted = []
        length = distances.length
        for (let i = 0; i < this.k; i++){
            let min = distances[0]
            let min_index = 0
            for (let j = 0; j < length; j++){
                if (distances[j] < min){
                    min = distances[j]
                    min_index = j
                }
            }
            indices_sorted.push(min_index)
            distances[min_index] = 1e10
        }
        // free memory
        distances = null
        return indices_sorted
    }
}



// class KNNRegressor {
//     constructor(k = 4) {
//         this.k = k;
//         this.points = null;
//         this.labels = null;
//     }

//     // Fit the model with data (points and vector labels)
//     fit(points, labels) {
//         this.points = tf.tensor2d(points);
//         this.labels = tf.tensor2d(labels);
//     }

//     // Custom function to find the indices of the k smallest elements in a tensor
//     async customTensorSort(distances, k) {
//         let kIndices = [];

//         for (let i = 0; i < k; i++) {
//             let minIndex = await distances.argMin().data();
//             kIndices.push(minIndex[0]);

//             // Set the found minimum distance to a large number so it's not found again
//             distances = tf.tidy(() => {
//                 return tf.tensor1d(distances.arraySync().map((x, idx) => {
//                     return idx === minIndex[0] ? Number.MAX_SAFE_INTEGER : x;
//                 }));
//             });
//         }

//         return kIndices;
//     }

//     // Predict the labels for multiple points
//     async predict(points) {
//         const queryPoints = tf.tensor2d(points, [points.length, 2]);
//         let predictions = [];

//         const epsilon = 1e-8; // Threshold for near-zero distances

//         for (let i = 0; i < queryPoints.shape[0]; i++) {
//             const queryPoint = queryPoints.slice([i, 0], [1, -1]).reshape([-1]);

//             // Calculate distances
//             const diff = this.points.sub(queryPoint);
//             const distances = diff.pow(2).sum(1).sqrt();

//             // Check for zero or near-zero distances
//             const isNearZero = distances.lessEqual(epsilon);
//             if (await isNearZero.any().data()) {
//                 const nearZeroIndex = await isNearZero.argMax().data();
//                 predictions.push(this.labels.slice([nearZeroIndex, 0], [1, -1]).arraySync());
//                 continue;
//             }

//             // Find the indices of the k nearest neighbors
//             const sortedIndices = await this.customTensorSort(distances, this.k);

//             // Retrieve the labels and distances of the k nearest neighbors
//             const nearestDistances = distances.gather(sortedIndices);
//             const nearestLabels = this.labels.gather(sortedIndices);

//             // Calculate weighted average for each label dimension
//             const weights = nearestDistances.pow(-1).expandDims(1);
//             const weightedLabels = nearestLabels.mul(weights);
//             const summedWeights = weights.sum(0);
//             const weightedAverage = weightedLabels.sum(0).div(summedWeights);

//             predictions.push(weightedAverage.arraySync());
//         }

//         queryPoints.dispose();
//         return predictions;
//     }
// }




async function customTensorSort(distances, k) {
    let kIndices = [];

    for (let i = 0; i < k; i++) {
        let minIndex = await distances.argMin().data();
        kIndices.push(minIndex[0]);

        // Set the found minimum distance to a large number so it's not found again
        distances = tf.tidy(() => {
            return tf.tensor1d(distances.arraySync().map((x, idx) => {
                return idx === minIndex[0] ? Number.MAX_SAFE_INTEGER : x;
            }));
        });
    }

    return kIndices;
}