/**
 * PC-ALM (Predictive Coding with Augmented Lagrangian Multipliers)
 * Pure JavaScript High-Performance Neural Engine for In-Browser Inference & Continual Learning
 * 
 * Based on the mathematical formulation:
 * L(h, theta, lambda) = 0.5 * ||y - W_L h_{L-1} - b_L||^2 + sum(lambda_i^T (h_i - mu_i)) + 0.5 * sum(||h_i - mu_i||^2)
 */

(function (global) {
    'use strict';

    class PcalmLayer {
        constructor(inDim, outDim) {
            this.inDim = inDim;
            this.outDim = outDim;

            const numWeights = inDim * outDim;
            this.W = new Float32Array(numWeights);
            this.v_W = new Float32Array(numWeights);
            this.b = new Float32Array(outDim);
            this.v_b = new Float32Array(outDim);

            this.h = new Float32Array(outDim);
            this.z = new Float32Array(outDim);
            this.mu = new Float32Array(outDim);
            this.lambda = new Float32Array(outDim);
            this.r_tilde = new Float32Array(outDim);
            this.grad_h = new Float32Array(outDim);
        }

        initHe() {
            const std = Math.sqrt(2.0 / this.inDim);
            for (let i = 0; i < this.W.length; i++) {
                // Box-Muller transform for normal distribution
                const u1 = Math.random() + 1e-10;
                const u2 = Math.random() + 1e-10;
                this.W[i] = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * std;
            }
            this.v_W.fill(0);
            this.b.fill(0);
            this.v_b.fill(0);
        }
    }

    class PcalmModel {
        constructor(dims = [784, 128, 64, 10], config = {}) {
            this.dims = dims;
            this.numLayers = dims.length - 1;

            this.config = {
                inference_steps: config.inference_steps !== undefined ? config.inference_steps : 10,
                eta_h: config.eta_h !== undefined ? config.eta_h : 0.08,
                alpha: config.alpha !== undefined ? config.alpha : 0.04,
                eta_theta: config.eta_theta !== undefined ? config.eta_theta : 0.005,
                momentum: config.momentum !== undefined ? config.momentum : 0.85,
                weight_decay: config.weight_decay !== undefined ? config.weight_decay : 1e-4
            };

            this.layers = [];
            for (let l = 0; l < this.numLayers; l++) {
                const layer = new PcalmLayer(dims[l], dims[l + 1]);
                layer.initHe();
                this.layers.push(layer);
            }

            this.input_h0 = new Float32Array(dims[0]);
            this.probs = new Float32Array(dims[this.numLayers]);
        }

        // Softmax
        softmax(input, output) {
            const size = input.length;
            let maxVal = input[0];
            for (let i = 1; i < size; i++) {
                if (input[i] > maxVal) maxVal = input[i];
            }
            let sum = 0.0;
            for (let i = 0; i < size; i++) {
                output[i] = Math.exp(input[i] - maxVal);
                sum += output[i];
            }
            if (sum > 0) {
                for (let i = 0; i < size; i++) {
                    output[i] /= sum;
                }
            }
        }

        // Forward Pass (Inference initialization)
        forward(input_x, out_probs = null) {
            this.input_h0.set(input_x);

            let prev_h = this.input_h0;
            for (let l = 0; l < this.numLayers; l++) {
                const layer = this.layers[l];
                const isOutput = (l === this.numLayers - 1);
                const inDim = layer.inDim;
                const outDim = layer.outDim;

                for (let i = 0; i < outDim; i++) {
                    let sum = layer.b[i];
                    const rowOffset = i * inDim;
                    for (let j = 0; j < inDim; j++) {
                        sum += layer.W[rowOffset + j] * prev_h[j];
                    }
                    layer.z[i] = sum;

                    if (isOutput) {
                        layer.mu[i] = sum;
                    } else {
                        layer.mu[i] = (sum > 0) ? sum : 0.1 * sum; // LeakyReLU
                    }
                    layer.h[i] = layer.mu[i];
                    layer.lambda[i] = 0.0;
                    layer.r_tilde[i] = 0.0;
                }
                prev_h = layer.h;
            }

            const lastLayer = this.layers[this.numLayers - 1];
            const targetProbs = out_probs || this.probs;
            this.softmax(lastLayer.z, targetProbs);

            let bestClass = 0;
            let bestVal = targetProbs[0];
            for (let i = 1; i < lastLayer.outDim; i++) {
                if (targetProbs[i] > bestVal) {
                    bestVal = targetProbs[i];
                    bestClass = i;
                }
            }

            return bestClass;
        }

        // Single sample PC-ALM training iteration
        trainSample(input_x, target_label) {
            const pred = this.forward(input_x, this.probs);
            const lastLayer = this.layers[this.numLayers - 1];
            const numClasses = lastLayer.outDim;

            let targetProb = (target_label >= 0 && target_label < numClasses) ? this.probs[target_label] : 1e-7;
            if (targetProb < 1e-7) targetProb = 1e-7;
            const sampleLoss = -Math.log(targetProb);

            const T = this.config.inference_steps;
            const eta_h = this.config.eta_h;
            const alpha = this.config.alpha;

            // 1. PC-ALM Inference Phase (T steps of Primal-Dual dynamics)
            for (let t = 0; t < T; t++) {
                this.softmax(lastLayer.z, this.probs);
                for (let i = 0; i < numClasses; i++) {
                    const y_i = (i === target_label) ? 1.0 : 0.0;
                    lastLayer.r_tilde[i] = this.probs[i] - y_i;
                }

                // Dual step: Lagrange multiplier ascent for hidden layers
                for (let l = 0; l < this.numLayers - 1; l++) {
                    const layer = this.layers[l];
                    for (let i = 0; i < layer.outDim; i++) {
                        const err = layer.h[i] - layer.mu[i];
                        layer.lambda[i] += alpha * err;
                        layer.r_tilde[i] = err + layer.lambda[i];
                    }
                }

                // Primal step: update hidden activations h_l
                for (let l = 0; l < this.numLayers - 1; l++) {
                    const layer = this.layers[l];
                    const nextLayer = this.layers[l + 1];
                    const nextIsOutput = (l + 1 === this.numLayers - 1);

                    for (let i = 0; i < layer.outDim; i++) {
                        let backErr = 0.0;
                        for (let k = 0; k < nextLayer.outDim; k++) {
                            let delta_k = nextLayer.r_tilde[k];
                            if (!nextIsOutput) {
                                delta_k *= (nextLayer.z[k] > 0) ? 1.0 : 0.1; // d_LeakyReLU
                            }
                            backErr += nextLayer.W[k * nextLayer.inDim + i] * delta_k;
                        }

                        const grad = layer.r_tilde[i] - backErr;
                        layer.grad_h[i] = grad;
                        layer.h[i] -= eta_h * grad;
                    }

                    // Recompute predicted activation mu_l = sigma(W_l * h_{l-1} + b_l)
                    const prev_h = (l === 0) ? this.input_h0 : this.layers[l - 1].h;
                    for (let i = 0; i < layer.outDim; i++) {
                        let sum = layer.b[i];
                        const rowOffset = i * layer.inDim;
                        for (let j = 0; j < layer.inDim; j++) {
                            sum += layer.W[rowOffset + j] * prev_h[j];
                        }
                        layer.z[i] = sum;
                        layer.mu[i] = (sum > 0) ? sum : 0.1 * sum;
                    }
                }

                // Update output layer pre-activation z_L based on updated h_{L-2}
                const prev_h_last = this.layers[this.numLayers - 2].h;
                for (let i = 0; i < lastLayer.outDim; i++) {
                    let sum = lastLayer.b[i];
                    const rowOffset = i * lastLayer.inDim;
                    for (let j = 0; j < lastLayer.inDim; j++) {
                        sum += lastLayer.W[rowOffset + j] * prev_h_last[j];
                    }
                    lastLayer.z[i] = sum;
                    lastLayer.mu[i] = sum;
                }
            }

            // 2. Learning Phase: Parameter updates (W and b with momentum & decay)
            const eta_theta = this.config.eta_theta;
            const momentum = this.config.momentum;
            const weight_decay = this.config.weight_decay;

            for (let l = 0; l < this.numLayers; l++) {
                const layer = this.layers[l];
                const isOutput = (l === this.numLayers - 1);
                const prev_h = (l === 0) ? this.input_h0 : this.layers[l - 1].h;

                // Layer-wise learning rate: preserve low-level features (Layer 0) while adapting higher classification layers
                const layerLr = eta_theta * (l === 0 ? 0.4 : (l === 1 ? 0.75 : 1.0));

                for (let i = 0; i < layer.outDim; i++) {
                    let delta_i = layer.r_tilde[i];
                    if (!isOutput) {
                        delta_i *= (layer.z[i] > 0) ? 1.0 : 0.1;
                    }

                    // Bias update
                    layer.v_b[i] = momentum * layer.v_b[i] + delta_i;
                    layer.b[i] -= layerLr * layer.v_b[i];

                    // Weights update
                    const rowOffset = i * layer.inDim;
                    for (let j = 0; j < layer.inDim; j++) {
                        const idx = rowOffset + j;
                        const grad_w = delta_i * prev_h[j] + weight_decay * layer.W[idx];
                        layer.v_W[idx] = momentum * layer.v_W[idx] + grad_w;
                        layer.W[idx] -= layerLr * layer.v_W[idx];
                    }
                }
            }

            // Compute diagnostic metrics
            let maxErr = 0.0;
            let maxLam = 0.0;
            for (let l = 0; l < this.numLayers - 1; l++) {
                const layer = this.layers[l];
                for (let i = 0; i < layer.outDim; i++) {
                    const err = Math.abs(layer.h[i] - layer.mu[i]);
                    if (err > maxErr) maxErr = err;
                    const lam = Math.abs(layer.lambda[i]);
                    if (lam > maxLam) maxLam = lam;
                }
            }

            return {
                loss: sampleLoss,
                predictedClass: pred,
                maxError: maxErr,
                maxLambda: maxLam
            };
        }

        // Spatial shift utility (shifts 28x28 image by dx, dy)
        shiftImage(pixels, dx, dy) {
            const shifted = new Float32Array(784);
            for (let y = 0; y < 28; y++) {
                const ny = y + dy;
                if (ny < 0 || ny >= 28) continue;
                const row = y * 28;
                const nrow = ny * 28;
                for (let x = 0; x < 28; x++) {
                    const nx = x + dx;
                    if (nx >= 0 && nx < 28) {
                        shifted[nrow + nx] = pixels[row + x];
                    }
                }
            }
            return shifted;
        }

        // Retrieve balanced rehearsal anchors of other digits to prevent catastrophic forgetting
        getRehearsalAnchors(excludeLabel, count = 3) {
            const anchors = [];
            if (typeof window !== 'undefined' && Array.isArray(window.MNIST_SAMPLES) && window.MNIST_SAMPLES.length > 0) {
                const candidates = window.MNIST_SAMPLES.filter(s => s.label !== excludeLabel);
                const usedLabels = new Set();
                for (let i = 0; i < 50 && anchors.length < count; i++) {
                    const randSample = candidates[Math.floor(Math.random() * candidates.length)];
                    if (!usedLabels.has(randSample.label)) {
                        usedLabels.add(randSample.label);
                        anchors.push(randSample);
                    }
                }
            }
            return anchors;
        }

        // Real-time Online Adaptation with Spatial Jitter Augmentation & Experience Replay
        adaptSingle(input_x, target_label, adapt_steps = 8, lr = 0.016) {
            const preProbs = new Float32Array(10);
            const oldPred = this.forward(input_x, preProbs);
            const oldTargetProb = preProbs[target_label];

            const origLr = this.config.eta_theta;
            this.config.eta_theta = (lr > 0) ? lr : 0.016;

            // 1. Get rehearsal anchors (other digits) to preserve existing decision boundaries
            const anchors = this.getRehearsalAnchors(target_label, 3);

            // 2. Schedule interleaved sequence: user drawing + jitter + rehearsal anchors
            const trainingSequence = [
                { pixels: input_x, label: target_label }, // Exact user sample
                ...(anchors[0] ? [{ pixels: anchors[0].pixels, label: anchors[0].label }] : []), // Rehearsal anchor
                { pixels: this.shiftImage(input_x, 1, 0), label: target_label }, // Jitter right
                ...(anchors[1] ? [{ pixels: anchors[1].pixels, label: anchors[1].label }] : []), // Rehearsal anchor
                { pixels: this.shiftImage(input_x, -1, 0), label: target_label }, // Jitter left
                ...(anchors[2] ? [{ pixels: anchors[2].pixels, label: anchors[2].label }] : []), // Rehearsal anchor
                { pixels: this.shiftImage(input_x, 0, 1), label: target_label }, // Jitter down
                { pixels: input_x, label: target_label } // Final exact anchor
            ];

            let lastStats = null;
            for (let i = 0; i < trainingSequence.length; i++) {
                const item = trainingSequence[i];
                const stats = this.trainSample(item.pixels, item.label);
                if (item.label === target_label) {
                    lastStats = stats;
                }
            }

            this.config.eta_theta = origLr;

            const postProbs = new Float32Array(10);
            const newPred = this.forward(input_x, postProbs);
            const newTargetProb = postProbs[target_label];

            const layerActs = this.getLayerActivationNorms();

            return {
                target: target_label,
                old_prediction: oldPred,
                old_target_prob: oldTargetProb,
                new_prediction: newPred,
                new_target_prob: newTargetProb,
                prob_diff: (newTargetProb - oldTargetProb),
                loss: -Math.log(Math.max(newTargetProb, 1e-7)),
                max_error: lastStats ? lastStats.maxError : 0,
                max_lambda: lastStats ? lastStats.maxLambda : 0,
                probabilities: Array.from(postProbs),
                layer_activations: layerActs
            };
        }

        // Get layer activation norms for diagnostic gauges
        getLayerActivationNorms() {
            const acts = [];
            for (let l = 0; l < this.numLayers; l++) {
                const layer = this.layers[l];
                let sumSq = 0.0;
                for (let k = 0; k < layer.outDim; k++) {
                    sumSq += layer.h[k] * layer.h[k];
                }
                acts.push(Math.sqrt(sumSq / layer.outDim));
            }
            return acts;
        }

        // Return raw activations for all layers (for real-time heatmap visualization)
        getLayerActivations() {
            return this.layers.map(layer => Float32Array.from(layer.h));
        }

        // Return weight matrix and metadata for a specified layer
        getLayerWeights(layerIdx) {
            if (layerIdx < 0 || layerIdx >= this.numLayers) return null;
            const layer = this.layers[layerIdx];
            return {
                inDim: layer.inDim,
                outDim: layer.outDim,
                W: Float32Array.from(layer.W),
                b: Float32Array.from(layer.b)
            };
        }

        // Return the 28x28 (784 floats) receptive field of a specific neuron in Layer 0
        getNeuronReceptiveField(neuronIdx) {
            if (this.numLayers <= 0) return null;
            const layer0 = this.layers[0];
            if (neuronIdx < 0 || neuronIdx >= layer0.outDim) return null;
            const offset = neuronIdx * layer0.inDim;
            return layer0.W.slice(offset, offset + layer0.inDim);
        }

        // Serialize current parameters to plain JS object (for IndexedDB)
        toObject() {
            const layersData = [];
            for (let l = 0; l < this.numLayers; l++) {
                const layer = this.layers[l];
                layersData.push({
                    inDim: layer.inDim,
                    outDim: layer.outDim,
                    W: Array.from(layer.W),
                    b: Array.from(layer.b)
                });
            }
            return {
                dims: this.dims,
                config: { ...this.config },
                layers: layersData,
                timestamp: Date.now()
            };
        }

        // Restore parameters from JS object
        fromObject(obj) {
            if (!obj || !obj.layers || obj.layers.length !== this.numLayers) {
                throw new Error('Invalid model data format');
            }
            if (obj.config) {
                Object.assign(this.config, obj.config);
            }
            for (let l = 0; l < this.numLayers; l++) {
                const layer = this.layers[l];
                const saved = obj.layers[l];
                layer.W.set(saved.W);
                layer.b.set(saved.b);
                layer.v_W.fill(0);
                layer.v_b.fill(0);
            }
        }

        // Load model from ArrayBuffer (.bin binary file format)
        loadFromBinaryBuffer(buffer) {
            const dataView = new DataView(buffer);
            let offset = 0;

            // 1. Magic
            const magic = String.fromCharCode(
                dataView.getUint8(offset++),
                dataView.getUint8(offset++),
                dataView.getUint8(offset++),
                dataView.getUint8(offset++)
            );
            if (magic !== 'PCAL') {
                throw new Error('Invalid model file: magic is not PCAL');
            }

            // 2. Version
            const version = dataView.getUint32(offset, true);
            offset += 4;

            // 3. Number of layers
            const fileNumLayers = dataView.getInt32(offset, true);
            offset += 4;
            if (fileNumLayers !== this.numLayers) {
                throw new Error(`Layer count mismatch: expected ${this.numLayers}, file has ${fileNumLayers}`);
            }

            // 4. Layer dimensions
            const fileDims = [];
            for (let i = 0; i <= fileNumLayers; i++) {
                fileDims.push(dataView.getInt32(offset, true));
                offset += 4;
            }

            // 5. Config
            this.config.inference_steps = dataView.getInt32(offset, true); offset += 4;
            this.config.eta_h = dataView.getFloat32(offset, true); offset += 4;
            this.config.alpha = dataView.getFloat32(offset, true); offset += 4;
            this.config.eta_theta = dataView.getFloat32(offset, true); offset += 4;
            this.config.momentum = dataView.getFloat32(offset, true); offset += 4;
            this.config.weight_decay = dataView.getFloat32(offset, true); offset += 4;

            // 6. Layer Weights and Biases
            for (let l = 0; l < this.numLayers; l++) {
                const layer = this.layers[l];
                const numWeights = layer.inDim * layer.outDim;
                for (let i = 0; i < numWeights; i++) {
                    layer.W[i] = dataView.getFloat32(offset, true);
                    offset += 4;
                }
                for (let i = 0; i < layer.outDim; i++) {
                    layer.b[i] = dataView.getFloat32(offset, true);
                    offset += 4;
                }
                layer.v_W.fill(0);
                layer.v_b.fill(0);
            }

            return true;
        }

        // Export current weights to ArrayBuffer (.bin format for download)
        saveToBinaryBuffer() {
            let totalFloats = 0;
            for (let l = 0; l < this.numLayers; l++) {
                totalFloats += this.layers[l].inDim * this.layers[l].outDim + this.layers[l].outDim;
            }
            const headerSize = 4 + 4 + 4 + (this.numLayers + 1) * 4 + 24;
            const totalBytes = headerSize + totalFloats * 4;

            const buffer = new ArrayBuffer(totalBytes);
            const dataView = new DataView(buffer);
            let offset = 0;

            // Magic 'PCAL'
            dataView.setUint8(offset++, 'P'.charCodeAt(0));
            dataView.setUint8(offset++, 'C'.charCodeAt(0));
            dataView.setUint8(offset++, 'A'.charCodeAt(0));
            dataView.setUint8(offset++, 'L'.charCodeAt(0));

            // Version 1
            dataView.setUint32(offset, 1, true); offset += 4;

            // Num layers
            dataView.setInt32(offset, this.numLayers, true); offset += 4;

            // Layer dims
            for (let i = 0; i <= this.numLayers; i++) {
                dataView.setInt32(offset, this.dims[i], true); offset += 4;
            }

            // Config
            dataView.setInt32(offset, this.config.inference_steps, true); offset += 4;
            dataView.setFloat32(offset, this.config.eta_h, true); offset += 4;
            dataView.setFloat32(offset, this.config.alpha, true); offset += 4;
            dataView.setFloat32(offset, this.config.eta_theta, true); offset += 4;
            dataView.setFloat32(offset, this.config.momentum, true); offset += 4;
            dataView.setFloat32(offset, this.config.weight_decay, true); offset += 4;

            // Weights and Biases
            for (let l = 0; l < this.numLayers; l++) {
                const layer = this.layers[l];
                const numWeights = layer.inDim * layer.outDim;
                for (let i = 0; i < numWeights; i++) {
                    dataView.setFloat32(offset, layer.W[i], true); offset += 4;
                }
                for (let i = 0; i < layer.outDim; i++) {
                    dataView.setFloat32(offset, layer.b[i], true); offset += 4;
                }
            }

            return buffer;
        }
    }

    // Export to global scope
    global.PcalmModel = PcalmModel;

})(typeof window !== 'undefined' ? window : this);
