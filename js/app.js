/**
 * PC-ALM Neural Studio - Application Controller
 * Pure Client-Side Implementation with Browser IndexedDB Integration
 */

(function () {
    'use strict';

    // Core Model and Database Instances
    let model = null;
    let isModelReady = false;

    // DOM Elements - Canvas
    const drawCanvas = document.getElementById('drawCanvas');
    const drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
    const miniCanvas = document.getElementById('miniCanvas');
    const miniCtx = miniCanvas.getContext('2d', { willReadFrequently: true });

    // DOM Elements - Controls
    const penSizeInput = document.getElementById('penSize');
    const penSizeVal = document.getElementById('penSizeVal');
    const clearBtn = document.getElementById('clearBtn');
    const randomMnistBtn = document.getElementById('randomMnistBtn');

    // DOM Elements - Prediction
    const statusBadge = document.getElementById('statusBadge');
    const statusText = document.getElementById('statusText');
    const predictedDigit = document.getElementById('predictedDigit');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const predictionDesc = document.getElementById('predictionDesc');
    const probBarsContainer = document.getElementById('probBars');
    const topProbVal = document.getElementById('topProbVal');
    const inferenceTime = document.getElementById('inferenceTime');

    // DOM Elements - Adaptation
    const digitButtons = document.getElementById('digitButtons');
    const targetDigitLabel = document.getElementById('targetDigitLabel');
    const adaptBtn = document.getElementById('adaptBtn');
    const adaptationFeedback = document.getElementById('adaptationFeedback');
    const fbOldPred = document.getElementById('fbOldPred');
    const fbNewPred = document.getElementById('fbNewPred');
    const fbProbDiff = document.getElementById('fbProbDiff');
    const fbLossVal = document.getElementById('fbLossVal');

    // DOM Elements - Diagnostics
    const gaugeH1 = document.getElementById('gaugeH1');
    const gaugeH1Val = document.getElementById('gaugeH1Val');
    const gaugeH2 = document.getElementById('gaugeH2');
    const gaugeH2Val = document.getElementById('gaugeH2Val');
    const gaugeH3 = document.getElementById('gaugeH3');
    const gaugeH3Val = document.getElementById('gaugeH3Val');
    const maxLambdaVal = document.getElementById('maxLambdaVal');
    const maxErrorVal = document.getElementById('maxErrorVal');
    const saveModelBtn = document.getElementById('saveModelBtn');
    const resetWeightsBtn = document.getElementById('resetWeightsBtn');
    const downloadBinBtn = document.getElementById('downloadBinBtn');

    // DOM Elements - IndexedDB Modal & Actions
    const openDbModalBtn = document.getElementById('openDbModalBtn');
    const dbCountBadge = document.getElementById('dbCountBadge');
    const dbModal = document.getElementById('dbModal');
    const closeDbModalBtn = document.getElementById('closeDbModalBtn');
    const dbTotalCount = document.getElementById('dbTotalCount');
    const dbWeightsStatus = document.getElementById('dbWeightsStatus');
    const dbLastUpdate = document.getElementById('dbLastUpdate');
    const dbHistoryList = document.getElementById('dbHistoryList');
    const exportDbBtn = document.getElementById('exportDbBtn');
    const importDbInput = document.getElementById('importDbInput');
    const clearDbBtn = document.getElementById('clearDbBtn');

    // DOM Elements - Heatmap Modal
    const openHeatmapBtn = document.getElementById('openHeatmapBtn');
    const closeHeatmapModalBtn = document.getElementById('closeHeatmapModalBtn');
    const heatmapModal = document.getElementById('heatmapModal');
    const heatCanvasL1 = document.getElementById('heatCanvasL1');
    const heatCanvasL2 = document.getElementById('heatCanvasL2');
    const heatCanvasL3 = document.getElementById('heatCanvasL3');
    const l1Meta = document.getElementById('l1Meta');
    const l2Meta = document.getElementById('l2Meta');
    const l3Meta = document.getElementById('l3Meta');
    const heatCanvasWeights = document.getElementById('heatCanvasWeights');
    const weightLayerSelect = document.getElementById('weightLayerSelect');
    const weightMatrixTitle = document.getElementById('weightMatrixTitle');
    const weightMeta = document.getElementById('weightMeta');
    const filterNeuronSlider = document.getElementById('filterNeuronSlider');
    const filterNeuronIndex = document.getElementById('filterNeuronIndex');
    const filterCanvas28 = document.getElementById('filterCanvas28');
    const filterWeightRange = document.getElementById('filterWeightRange');
    const heatmapTooltip = document.getElementById('heatmapTooltip');

    let currentHeatmapTab = 'activations';

    // Application State
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let hasDrawn = false;
    let selectedTargetDigit = null;
    let lastPixels = new Float32Array(784);
    let lastPredictTime = 0;
    let predictThrottleTimer = null;

    // Helper: Base64 to ArrayBuffer
    function base64ToArrayBuffer(base64) {
        const binaryString = window.atob(base64);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }

    // 1. Initialize Neural Network & Load Weights
    async function initModel() {
        model = new window.PcalmModel([784, 128, 64, 10]);

        statusText.textContent = '가중치 로딩 중...';

        try {
            // Check IndexedDB for custom user-trained weights first
            const dbLoaded = await window.appDB.loadModelWeights(model);
            if (dbLoaded && dbLoaded.success) {
                console.log('✓ Restored adapted weights from browser IndexedDB (updated: ' + dbLoaded.updatedAt + ')');
                statusText.textContent = '브라우저 DB 가중치 활성';
                isModelReady = true;
                await refreshDbStats();
                return;
            }

            // Otherwise, load default weights from binary file or embedded base64
            let arrayBuffer = null;
            try {
                // Attempt fetch (works on GitHub Pages / HTTP server)
                const resp = await fetch('data/pcalm_mnist.bin');
                if (resp.ok) {
                    arrayBuffer = await resp.arrayBuffer();
                }
            } catch (fetchErr) {
                console.log('Fetch pcalm_mnist.bin skipped or CORS restricted (file://), using embedded fallback.');
            }

            // Fallback to embedded base64 weights (works offline / file:// protocol)
            if (!arrayBuffer && window.DEFAULT_MODEL_BASE64) {
                arrayBuffer = base64ToArrayBuffer(window.DEFAULT_MODEL_BASE64);
            }

            if (arrayBuffer) {
                model.loadFromBinaryBuffer(arrayBuffer);
                console.log('✓ Default PC-ALM model loaded successfully!');
                statusText.textContent = '기본 모델 (96.5% 정확도)';
            } else {
                console.warn('Using procedurally initialized weights');
                statusText.textContent = 'He 초기화 모델';
            }

            isModelReady = true;
            await refreshDbStats();
        } catch (err) {
            console.error('Failed to initialize model:', err);
            statusText.textContent = '초기화 오류';
        }
    }

    // 2. Initialize Canvas
    function initCanvas() {
        drawCtx.fillStyle = '#000000';
        drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        drawCtx.strokeStyle = '#FFFFFF';
        drawCtx.lineWidth = parseInt(penSizeInput.value, 10);

        miniCtx.fillStyle = '#000000';
        miniCtx.fillRect(0, 0, miniCanvas.width, miniCanvas.height);
    }

    // 3. Initialize Softmax Probability Bars
    function initProbBars() {
        probBarsContainer.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'prob-bar-row';
            row.id = `probRow_${i}`;
            row.innerHTML = `
                <span class="prob-digit">${i}</span>
                <div class="prob-track">
                    <div class="prob-fill" id="probFill_${i}"></div>
                </div>
                <span class="prob-val" id="probVal_${i}">0.0%</span>
            `;
            probBarsContainer.appendChild(row);
        }
    }

    // Coordinate conversion
    function getCanvasCoords(e) {
        const rect = drawCanvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const scaleX = drawCanvas.width / rect.width;
        const scaleY = drawCanvas.height / rect.height;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function setLiveStatus(streaming) {
        if (!statusText) return;
        if (streaming) {
            statusText.textContent = '실시간 인식 중';
        } else if (statusText.textContent === '실시간 인식 중') {
            statusText.textContent = '준비됨';
        }
    }

    // 4. Drawing Event Handlers
    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        hasDrawn = true;
        setLiveStatus(true);

        const coords = getCanvasCoords(e);
        lastX = coords.x;
        lastY = coords.y;

        drawCtx.beginPath();
        drawCtx.arc(lastX, lastY, drawCtx.lineWidth / 2, 0, Math.PI * 2);
        drawCtx.fillStyle = '#FFFFFF';
        drawCtx.fill();

        lastPredictTime = performance.now();
        schedulePrediction(30);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCanvasCoords(e);

        drawCtx.beginPath();
        drawCtx.moveTo(lastX, lastY);
        drawCtx.lineTo(coords.x, coords.y);
        drawCtx.stroke();

        lastX = coords.x;
        lastY = coords.y;

        // Continuous real-time live prediction (~30ms throttle)
        const now = performance.now();
        if (now - lastPredictTime >= 30) {
            lastPredictTime = now;
            runPrediction();
        } else if (!predictThrottleTimer) {
            predictThrottleTimer = setTimeout(() => {
                predictThrottleTimer = null;
                runPrediction();
            }, 30);
        }
    }

    function stopDrawing(e) {
        if (!isDrawing) return;
        isDrawing = false;
        if (predictThrottleTimer) {
            clearTimeout(predictThrottleTimer);
            predictThrottleTimer = null;
        }
        runPrediction();
        setLiveStatus(false);
    }

    function schedulePrediction(delayMs) {
        if (predictThrottleTimer) clearTimeout(predictThrottleTimer);
        predictThrottleTimer = setTimeout(() => {
            predictThrottleTimer = null;
            if (hasDrawn) runPrediction();
        }, delayMs);
    }

    // 5. Clear Canvas
    function clearCanvas() {
        drawCtx.fillStyle = '#000000';
        drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
        miniCtx.fillStyle = '#000000';
        miniCtx.fillRect(0, 0, miniCanvas.width, miniCanvas.height);
        hasDrawn = false;
        lastPixels.fill(0);

        setLiveStatus(false);
        predictedDigit.textContent = '-';
        predictedDigit.style.transform = 'scale(1)';
        confidenceBadge.textContent = '신뢰도: 0.0%';
        predictionDesc.textContent = '캔버스에 숫자를 그리면 실시간으로 인식합니다.';
        topProbVal.textContent = '0.0%';
        adaptationFeedback.classList.add('hidden');

        for (let i = 0; i < 10; i++) {
            document.getElementById(`probFill_${i}`).style.width = '0%';
            document.getElementById(`probVal_${i}`).textContent = '0.0%';
            document.getElementById(`probRow_${i}`).classList.remove('top-rank');
        }
        updateGauges([0, 0, 0], 0, 0);
        updateAdaptButton();

        if (heatmapModal && !heatmapModal.classList.contains('hidden') && currentHeatmapTab === 'activations') {
            renderActivationHeatmaps();
        }
    }

    // 6. MNIST Center-of-Mass (CoM) Normalization
    function extractNormalized28x28() {
        const rawImg = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
        const w = drawCanvas.width;
        const h = drawCanvas.height;

        // 1. Find bounding box of drawing
        let minX = w, minY = h, maxX = 0, maxY = 0;
        let totalMass = 0;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const brightness = rawImg.data[idx]; // R channel
                if (brightness > 20) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    totalMass += brightness;
                }
            }
        }

        if (totalMass === 0 || minX > maxX || minY > maxY) {
            miniCtx.fillStyle = '#000000';
            miniCtx.fillRect(0, 0, 28, 28);
            lastPixels.fill(0);
            return lastPixels;
        }

        // Bounding box dimensions
        const boxW = (maxX - minX + 1);
        const boxH = (maxY - minY + 1);

        // Render centered inside 20x20 box on offscreen canvas
        const offCanvas = document.createElement('canvas');
        offCanvas.width = 28;
        offCanvas.height = 28;
        const offCtx = offCanvas.getContext('2d');
        offCtx.fillStyle = '#000000';
        offCtx.fillRect(0, 0, 28, 28);

        // Scale to fit 20x20 box preserving aspect ratio
        const scale = 20.0 / Math.max(boxW, boxH);
        const scaledW = boxW * scale;
        const scaledH = boxH * scale;

        // Position initial bounding box center at (14, 14)
        const dx = 14.0 - scaledW / 2.0;
        const dy = 14.0 - scaledH / 2.0;

        offCtx.drawImage(drawCanvas, minX, minY, boxW, boxH, dx, dy, scaledW, scaledH);

        // 2. Adjust translation by Center of Mass (CoM) to match MNIST distribution
        const boxImg = offCtx.getImageData(0, 0, 28, 28);
        let cx = 0, cy = 0, mass = 0;
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const v = boxImg.data[(y * 28 + x) * 4];
                cx += x * v;
                cy += y * v;
                mass += v;
            }
        }

        miniCtx.fillStyle = '#000000';
        miniCtx.fillRect(0, 0, 28, 28);

        if (mass > 0) {
            cx /= mass;
            cy /= mass;
            const shiftX = Math.round(13.5 - cx);
            const shiftY = Math.round(13.5 - cy);
            miniCtx.drawImage(offCanvas, shiftX, shiftY);
        } else {
            miniCtx.drawImage(offCanvas, 0, 0);
        }

        // 3. Extract normalized float values [0.0, 1.0] (clamping background noise)
        const finalImg = miniCtx.getImageData(0, 0, 28, 28);
        for (let i = 0; i < 784; i++) {
            const val = finalImg.data[i * 4];
            lastPixels[i] = val > 18 ? val / 255.0 : 0.0;
        }

        return lastPixels;
    }

    // 7. Run In-Browser Prediction
    function runPrediction() {
        if (!hasDrawn || !isModelReady) return;

        const pixels = extractNormalized28x28();

        let strokeSum = 0;
        for (let i = 0; i < 784; i++) strokeSum += pixels[i];
        if (strokeSum < 0.2) return;

        const startTime = performance.now();
        const probs = new Float32Array(10);
        const pred = model.forward(pixels, probs);
        const elapsed = (performance.now() - startTime).toFixed(2);

        inferenceTime.textContent = `${elapsed} ms`;
        renderPrediction(pred, probs);
        updateAdaptButton();
    }

    // 8. Render Prediction Results
    function renderPrediction(pred, probs) {
        const conf = (probs[pred] * 100).toFixed(1);

        if (predictedDigit.textContent !== String(pred)) {
            predictedDigit.textContent = pred;
            predictedDigit.style.transform = 'scale(1.18)';
            setTimeout(() => { predictedDigit.style.transform = 'scale(1)'; }, 120);
        }

        confidenceBadge.textContent = `신뢰도: ${conf}%`;
        predictionDesc.textContent = `숫자 '${pred}'로 가장 높은 확률(${conf}%)로 판정되었습니다.`;
        topProbVal.textContent = `${conf}%`;

        for (let i = 0; i < 10; i++) {
            const p = probs[i];
            const pct = (p * 100).toFixed(1);
            const fill = document.getElementById(`probFill_${i}`);
            const val = document.getElementById(`probVal_${i}`);
            const row = document.getElementById(`probRow_${i}`);

            fill.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
            val.textContent = `${pct}%`;

            if (i === pred) {
                row.classList.add('top-rank');
            } else {
                row.classList.remove('top-rank');
            }
        }

        // Layer Activation Gauges
        const layerActs = model.getLayerActivationNorms();
        updateGauges(layerActs, 0, 0);

        if (heatmapModal && !heatmapModal.classList.contains('hidden') && currentHeatmapTab === 'activations') {
            renderActivationHeatmaps();
        }
    }

    // Update Diagnostics Gauges
    function updateGauges(layerActs, maxErr, maxLam) {
        if (layerActs && layerActs.length >= 3) {
            gaugeH1.style.width = `${Math.min(100, layerActs[0] * 35)}%`;
            gaugeH1Val.textContent = layerActs[0].toFixed(2);

            gaugeH2.style.width = `${Math.min(100, layerActs[1] * 40)}%`;
            gaugeH2Val.textContent = layerActs[1].toFixed(2);

            gaugeH3.style.width = `${Math.min(100, layerActs[2] * 25)}%`;
            gaugeH3Val.textContent = layerActs[2].toFixed(2);
        }
        if (maxLam !== undefined) maxLambdaVal.textContent = maxLam.toFixed(4);
        if (maxErr !== undefined) maxErrorVal.textContent = maxErr.toFixed(4);
    }

    // 9. Real-Time Online Adaptation & IndexedDB Storage
    async function triggerOnlineAdaptation() {
        if (selectedTargetDigit === null || !hasDrawn || !isModelReady) return;

        adaptBtn.disabled = true;
        adaptBtn.innerHTML = '<span class="pulse-dot"></span> <span>학습 및 DB 저장 중...</span>';

        try {
            // 1. Perform in-browser PC-ALM adaptation
            const res = model.adaptSingle(lastPixels, selectedTargetDigit, 8, 0.03);

            // 2. Create thumbnail data URL from miniCanvas
            const thumbUrl = miniCanvas.toDataURL('image/png');

            // 3. Save training record to IndexedDB
            await window.appDB.addTrainingRecord({
                timestamp: Date.now(),
                target: res.target,
                old_prediction: res.old_prediction,
                old_target_prob: res.old_target_prob,
                new_prediction: res.new_prediction,
                new_target_prob: res.new_target_prob,
                prob_diff: res.prob_diff,
                loss: res.loss,
                thumbnail: thumbUrl,
                pixels: lastPixels
            });

            // 4. Save updated model weights to IndexedDB
            await window.appDB.saveModelWeights(model);

            // 5. Update UI feedback
            adaptationFeedback.classList.remove('hidden');
            fbOldPred.textContent = res.old_prediction;
            fbNewPred.textContent = res.new_prediction;

            const oldPct = (res.old_target_prob * 100).toFixed(1);
            const newPct = (res.new_target_prob * 100).toFixed(1);
            const diffPct = (res.prob_diff * 100).toFixed(1);
            fbProbDiff.textContent = (res.prob_diff >= 0 ? '+' : '') + `${diffPct}%`;
            fbLossVal.textContent = res.loss.toFixed(3);

            // Re-render prediction
            renderPrediction(res.new_prediction, res.probabilities);

            maxErrorVal.textContent = res.max_error.toFixed(4);
            maxLambdaVal.textContent = res.max_lambda.toFixed(4);

            predictionDesc.textContent = `정답 '${selectedTargetDigit}'(으)로 학습 완료`;
            statusText.textContent = 'DB 가중치 갱신됨';

            // Refresh DB stats badge
            await refreshDbStats();
        } catch (err) {
            console.error('Adaptation error:', err);
            alert('런타임 학습 중 오류가 발생했습니다: ' + err.message);
        } finally {
            adaptBtn.disabled = false;
            updateAdaptButton();
        }
    }

    // Target digit selection
    function selectTargetDigit(digit) {
        selectedTargetDigit = digit;
        targetDigitLabel.textContent = digit;

        const btns = digitButtons.querySelectorAll('.btn-digit, .pill-digit, .digit-btn');
        btns.forEach(btn => {
            if (parseInt(btn.getAttribute('data-digit'), 10) === digit) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        updateAdaptButton();
    }

    function updateAdaptButton() {
        if (selectedTargetDigit !== null && hasDrawn) {
            adaptBtn.disabled = false;
            adaptBtn.innerHTML = `<span>정답 [<strong>${selectedTargetDigit}</strong>]로 즉시 학습</span>`;
        } else {
            adaptBtn.disabled = true;
            adaptBtn.innerHTML = '<span>정답 [?]로 즉시 학습</span>';
        }
    }

    // 10. Load Random MNIST Sample
    function loadRandomMnistSample() {
        if (!window.MNIST_SAMPLES || window.MNIST_SAMPLES.length === 0) {
            alert('내장된 MNIST 샘플을 찾을 수 없습니다.');
            return;
        }

        const idx = Math.floor(Math.random() * window.MNIST_SAMPLES.length);
        const sample = window.MNIST_SAMPLES[idx];

        // Draw onto miniCanvas
        const imgData = miniCtx.createImageData(28, 28);
        for (let i = 0; i < 784; i++) {
            const val = Math.round(sample.pixels[i] * 255);
            imgData.data[i * 4 + 0] = val;
            imgData.data[i * 4 + 1] = val;
            imgData.data[i * 4 + 2] = val;
            imgData.data[i * 4 + 3] = 255;
            lastPixels[i] = sample.pixels[i];
        }
        miniCtx.putImageData(imgData, 0, 0);

        // Scale up onto main 280x280 canvas
        drawCtx.imageSmoothingEnabled = false;
        drawCtx.drawImage(miniCanvas, 0, 0, drawCanvas.width, drawCanvas.height);
        hasDrawn = true;

        // Auto select true MNIST ground truth label
        selectTargetDigit(sample.label);

        // Run prediction
        setLiveStatus(true);
        runPrediction();
        setTimeout(() => setLiveStatus(false), 200);
    }

    // 11. Save Model to IndexedDB
    async function saveModel() {
        if (!isModelReady) return;
        try {
            saveModelBtn.disabled = true;
            saveModelBtn.textContent = '브라우저 DB에 저장 중...';

            await window.appDB.saveModelWeights(model);
            await refreshDbStats();
            alert('✓ 현재 학습된 모델 가중치가 브라우저 IndexedDB에 성공적으로 저장되었습니다!\n페이지를 새로고침하거나 브라우저를 다시 열어도 유지됩니다.');
        } catch (err) {
            alert('가중치 저장 실패: ' + err.message);
        } finally {
            saveModelBtn.disabled = false;
            saveModelBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 01-2-2V5a2 2 0 012-2h11l5 5v11a2 2 0 01-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg>
                현재 가중치를 브라우저 DB에 저장
            `;
        }
    }

    // 12. Reset Model Weights to Default
    async function resetModelWeights() {
        if (!confirm('모델 가중치를 기본 학습 상태로 복원하시겠습니까?\n브라우저 DB에 저장된 사용자 맞춤 가중치는 초기화됩니다.')) {
            return;
        }

        try {
            await window.appDB.clearModelWeights();

            // Re-load default weights
            let arrayBuffer = null;
            if (window.DEFAULT_MODEL_BASE64) {
                arrayBuffer = base64ToArrayBuffer(window.DEFAULT_MODEL_BASE64);
            }
            if (arrayBuffer) {
                model.loadFromBinaryBuffer(arrayBuffer);
            } else {
                model = new window.PcalmModel([784, 128, 64, 10]);
            }

            statusText.textContent = '기본 모델 (초기화 완료)';
            await refreshDbStats();
            if (hasDrawn) runPrediction();
            alert('✓ 모델 가중치가 기본 상태로 복원되었습니다.');
        } catch (err) {
            alert('모델 초기화 실패: ' + err.message);
        }
    }

    // 13. Download Model Weights as .bin file
    function downloadModelBinary() {
        if (!isModelReady) return;
        try {
            const buffer = model.saveToBinaryBuffer();
            const blob = new Blob([buffer], { type: 'application/octet-stream' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'pcalm_mnist_adapted.bin';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            alert('바이너리 다운로드 실패: ' + err.message);
        }
    }

    // 14. IndexedDB Management & Stats
    async function refreshDbStats() {
        try {
            const stats = await window.appDB.getStats();
            dbCountBadge.textContent = `${stats.totalSamples}개`;
            dbTotalCount.textContent = `${stats.totalSamples}개`;

            if (stats.hasCustomWeights) {
                dbWeightsStatus.textContent = '맞춤 가중치 저장됨';
                dbLastUpdate.textContent = stats.lastUpdate ? new Date(stats.lastUpdate).toLocaleString() : '';
            } else {
                dbWeightsStatus.textContent = '기본 가중치';
                dbLastUpdate.textContent = '저장된 맞춤 가중치 없음';
            }
        } catch (err) {
            console.error('Failed to get DB stats:', err);
        }
    }

    // Render training history list in modal
    async function renderDbHistoryList() {
        try {
            const records = await window.appDB.getAllTrainingRecords();
            dbHistoryList.innerHTML = '';

            if (records.length === 0) {
                dbHistoryList.innerHTML = '<div class="db-empty-state">아직 저장된 실시간 학습 데이터가 없습니다.<br/>손글씨를 그리고 "즉시 런타임 학습"을 진행해보세요.</div>';
                return;
            }

            records.forEach(rec => {
                const item = document.createElement('div');
                item.className = 'db-history-item';
                const diff = (rec.prob_diff * 100).toFixed(1);
                item.innerHTML = `
                    <div class="item-left">
                        <img src="${rec.thumbnail}" class="item-thumb" alt="숫자 ${rec.target}">
                        <div class="item-info">
                            <span class="item-target">정답 [${rec.target}] (이전 예측: ${rec.old_prediction})</span>
                            <span class="item-time">${rec.isoDate}</span>
                        </div>
                    </div>
                    <div class="item-right">
                        <span class="item-diff">+${diff}%</span>
                        <button class="item-del-btn" data-id="${rec.id}" title="기록 삭제">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path></svg>
                        </button>
                    </div>
                `;
                dbHistoryList.appendChild(item);
            });

            // Attach delete handlers
            dbHistoryList.querySelectorAll('.item-del-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    const id = parseInt(e.currentTarget.getAttribute('data-id'), 10);
                    await window.appDB.deleteTrainingRecord(id);
                    await refreshDbStats();
                    renderDbHistoryList();
                });
            });
        } catch (err) {
            console.error('Failed to render history list:', err);
        }
    }

    // Export DB to JSON
    async function exportDatabase() {
        try {
            const jsonStr = await window.appDB.exportToJson();
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `NumWriting_backup_${new Date().toISOString().slice(0,10)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            alert('내보내기 실패: ' + err.message);
        }
    }

    // Import DB from JSON
    async function importDatabase(e) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = async (evt) => {
            try {
                await window.appDB.importFromJson(evt.target.result, model);
                await refreshDbStats();
                renderDbHistoryList();
                alert('✓ 학습 데이터 및 가중치를 성공적으로 가져왔습니다!');
            } catch (err) {
                alert('가져오기 실패: ' + err.message);
            }
        };
        reader.readAsText(file);
        e.target.value = ''; // reset file input
    }

    // Clear all history records
    async function clearAllHistory() {
        if (!confirm('저장된 실시간 학습 기록을 모두 삭제하시겠습니까?')) return;
        try {
            await window.appDB.clearAllTrainingHistory();
            await refreshDbStats();
            renderDbHistoryList();
        } catch (err) {
            alert('기록 삭제 실패: ' + err.message);
        }
    }

    // 15. Setup Event Listeners
    function setupEventListeners() {
        // Mouse events
        drawCanvas.addEventListener('mousedown', startDrawing);
        window.addEventListener('mousemove', draw);
        window.addEventListener('mouseup', stopDrawing);

        // Touch events
        drawCanvas.addEventListener('touchstart', startDrawing, { passive: false });
        window.addEventListener('touchmove', draw, { passive: false });
        window.addEventListener('touchend', stopDrawing);

        // Brush slider
        penSizeInput.addEventListener('input', (e) => {
            const size = e.target.value;
            drawCtx.lineWidth = parseInt(size, 10);
            penSizeVal.textContent = `${size}px`;
        });

        // Clear canvas
        clearBtn.addEventListener('click', clearCanvas);

        // Random MNIST button
        randomMnistBtn.addEventListener('click', loadRandomMnistSample);

        // Target digit selector buttons
        digitButtons.addEventListener('click', (e) => {
            const btn = e.target.closest('.btn-digit') || e.target.closest('.pill-digit') || e.target.closest('.digit-btn');
            if (btn) {
                const digit = parseInt(btn.getAttribute('data-digit'), 10);
                selectTargetDigit(digit);
            }
        });

        // Online Adaptation button
        adaptBtn.addEventListener('click', triggerOnlineAdaptation);

        // Save & Reset & Download buttons
        saveModelBtn.addEventListener('click', saveModel);
        resetWeightsBtn.addEventListener('click', resetModelWeights);
        downloadBinBtn.addEventListener('click', downloadModelBinary);

        // IndexedDB Modal
        openDbModalBtn.addEventListener('click', () => {
            dbModal.classList.remove('hidden');
            renderDbHistoryList();
        });

        closeDbModalBtn.addEventListener('click', () => {
            dbModal.classList.add('hidden');
        });

        dbModal.addEventListener('click', (e) => {
            if (e.target === dbModal) dbModal.classList.add('hidden');
        });

        exportDbBtn.addEventListener('click', exportDatabase);
        importDbInput.addEventListener('change', importDatabase);
        clearDbBtn.addEventListener('click', clearAllHistory);

        setupHeatmapInteractions();
    }

    // =========================================================================
    // 16. HEATMAP VISUALIZATION ENGINE
    // =========================================================================

    // Colormap for neuron activations: 0.0 (dark) -> 1.0 (hot red/white)
    function getActivationColor(v) {
        const val = Math.max(0, Math.min(1, v));
        let r, g, b;
        if (val < 0.25) {
            const t = val / 0.25;
            r = Math.round(9 + t * 21);
            g = Math.round(13 + t * 45);
            b = Math.round(22 + t * 116);
        } else if (val < 0.5) {
            const t = (val - 0.25) / 0.25;
            r = Math.round(30 - t * 24);
            g = Math.round(58 + t * 124);
            b = Math.round(138 + t * 74);
        } else if (val < 0.75) {
            const t = (val - 0.5) / 0.25;
            r = Math.round(6 + t * 239);
            g = Math.round(182 - t * 24);
            b = Math.round(212 - t * 201);
        } else {
            const t = (val - 0.75) / 0.25;
            r = Math.round(245 - t * 6);
            g = Math.round(158 - t * 90);
            b = Math.round(11 + t * 57);
        }
        return `rgb(${r},${g},${b})`;
    }

    // Diverging colormap for weights: Negative (Red) < 0 (Neutral Dark) < Positive (Green)
    function getWeightColor(w, maxAbs) {
        const norm = maxAbs > 1e-6 ? Math.max(-1, Math.min(1, w / maxAbs)) : 0;
        let r, g, b;
        if (norm < 0) {
            const t = -norm;
            r = Math.round(24 + t * 215);
            g = Math.round(24 - t * 12);
            b = Math.round(27 - t * 15);
        } else {
            const t = norm;
            r = Math.round(24 - t * 8);
            g = Math.round(24 + t * 161);
            b = Math.round(27 + t * 102);
        }
        return `rgb(${r},${g},${b})`;
    }

    // Render Tab 1: Real-time Layer Activations
    function renderActivationHeatmaps() {
        if (!model) return;
        const acts = model.getLayerActivations();
        if (!acts || acts.length < 3) return;

        // Layer 1: 128 Neurons (16 cols x 8 rows)
        if (heatCanvasL1) {
            const ctx1 = heatCanvasL1.getContext('2d');
            const cols = 16, rows = 8;
            const cellW = heatCanvasL1.width / cols;
            const cellH = heatCanvasL1.height / rows;
            const l1Acts = acts[0];

            let sum1 = 0, max1 = 0;
            for (let i = 0; i < 128; i++) {
                const v = l1Acts[i];
                sum1 += v;
                if (v > max1) max1 = v;
            }
            const normMax1 = Math.max(max1, 1.0);

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const idx = r * cols + c;
                    const val = l1Acts[idx] || 0;
                    ctx1.fillStyle = getActivationColor(val / normMax1);
                    ctx1.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
                }
            }
            if (l1Meta) l1Meta.textContent = `평균: ${(sum1 / 128).toFixed(2)} | 최대: ${max1.toFixed(2)}`;
        }

        // Layer 2: 64 Neurons (8 cols x 8 rows)
        if (heatCanvasL2) {
            const ctx2 = heatCanvasL2.getContext('2d');
            const cols = 8, rows = 8;
            const cellW = heatCanvasL2.width / cols;
            const cellH = heatCanvasL2.height / rows;
            const l2Acts = acts[1];

            let sum2 = 0, max2 = 0;
            for (let i = 0; i < 64; i++) {
                const v = l2Acts[i];
                sum2 += v;
                if (v > max2) max2 = v;
            }
            const normMax2 = Math.max(max2, 1.0);

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const idx = r * cols + c;
                    const val = l2Acts[idx] || 0;
                    ctx2.fillStyle = getActivationColor(val / normMax2);
                    ctx2.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
                }
            }
            if (l2Meta) l2Meta.textContent = `평균: ${(sum2 / 64).toFixed(2)} | 최대: ${max2.toFixed(2)}`;
        }

        // Layer 3: Output 10 Neurons (10 cols x 1 row)
        if (heatCanvasL3) {
            const ctx3 = heatCanvasL3.getContext('2d');
            const cols = 10;
            const cellW = heatCanvasL3.width / cols;
            const cellH = heatCanvasL3.height;
            const l3Acts = acts[2];

            let max3 = -Infinity, maxDigit = 0;
            for (let i = 0; i < 10; i++) {
                if (l3Acts[i] > max3) {
                    max3 = l3Acts[i];
                    maxDigit = i;
                }
            }
            const min3 = Math.min(...l3Acts);
            const range3 = Math.max(max3 - min3, 1e-4);

            for (let c = 0; c < cols; c++) {
                const val = l3Acts[c];
                const normVal = (val - min3) / range3;
                ctx3.fillStyle = getActivationColor(normVal);
                ctx3.fillRect(c * cellW, 0, cellW - 1, cellH);

                // Draw digit number label
                ctx3.fillStyle = normVal > 0.6 ? '#000000' : '#FFFFFF';
                ctx3.font = 'bold 12px "JetBrains Mono", monospace';
                ctx3.textAlign = 'center';
                ctx3.textBaseline = 'middle';
                ctx3.fillText(String(c), c * cellW + cellW / 2, cellH / 2);
            }
            if (l3Meta) l3Meta.textContent = `최고 예측: 숫자 '${maxDigit}' (${max3.toFixed(2)})`;
        }
    }

    // Render Tab 2: Weight Matrix
    function renderWeightHeatmap() {
        if (!model || !heatCanvasWeights) return;
        const layerIdx = parseInt(weightLayerSelect.value, 10);
        const data = model.getLayerWeights(layerIdx);
        if (!data) return;

        const ctx = heatCanvasWeights.getContext('2d');
        const W = data.W;
        const rows = data.outDim;
        const cols = data.inDim;

        let maxAbs = 0;
        for (let i = 0; i < W.length; i++) {
            const abs = Math.abs(W[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        ctx.clearRect(0, 0, heatCanvasWeights.width, heatCanvasWeights.height);

        // Adjust canvas pixel density
        heatCanvasWeights.width = Math.max(320, cols);
        heatCanvasWeights.height = Math.max(120, rows * 4);

        const cellW = heatCanvasWeights.width / cols;
        const cellH = heatCanvasWeights.height / rows;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const weight = W[r * cols + c];
                ctx.fillStyle = getWeightColor(weight, maxAbs);
                ctx.fillRect(c * cellW, r * cellH, Math.max(1, cellW), Math.max(1, cellH));
            }
        }

        if (weightMatrixTitle) {
            weightMatrixTitle.textContent = `Layer ${layerIdx} 가중치 (${rows}행 × ${cols}열)`;
        }
        if (weightMeta) {
            weightMeta.textContent = `절대치 최대: ±${maxAbs.toFixed(3)}`;
        }
    }

    // Render Tab 3: Receptive Field
    function renderFilterReceptiveField() {
        if (!model || !filterCanvas28) return;
        const neuronIdx = parseInt(filterNeuronSlider.value, 10);
        const rf = model.getNeuronReceptiveField(neuronIdx);
        if (!rf) return;

        const ctx = filterCanvas28.getContext('2d');
        let maxAbs = 0;
        for (let i = 0; i < 784; i++) {
            const abs = Math.abs(rf[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        filterCanvas28.width = 196;
        filterCanvas28.height = 196;
        const scale = 196 / 28; // 7px per cell

        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const w = rf[y * 28 + x];
                ctx.fillStyle = getWeightColor(w, maxAbs);
                ctx.fillRect(x * scale, y * scale, scale, scale);
            }
        }

        if (filterNeuronIndex) filterNeuronIndex.textContent = `#${neuronIdx}`;
        if (filterWeightRange) filterWeightRange.textContent = `±${maxAbs.toFixed(4)}`;
    }

    function setupHeatmapInteractions() {
        // Tab switching
        const tabBtns = document.querySelectorAll('.heatmap-tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentHeatmapTab = btn.getAttribute('data-tab');

                document.querySelectorAll('.heatmap-tab-pane').forEach(p => p.classList.remove('active'));
                if (currentHeatmapTab === 'activations') {
                    document.getElementById('tabActivations').classList.add('active');
                    renderActivationHeatmaps();
                } else if (currentHeatmapTab === 'weights') {
                    document.getElementById('tabWeights').classList.add('active');
                    renderWeightHeatmap();
                } else if (currentHeatmapTab === 'filters') {
                    document.getElementById('tabFilters').classList.add('active');
                    renderFilterReceptiveField();
                }
            });
        });

        // Layer selection in weights tab
        if (weightLayerSelect) {
            weightLayerSelect.addEventListener('change', renderWeightHeatmap);
        }

        // Neuron slider in filters tab
        if (filterNeuronSlider) {
            filterNeuronSlider.addEventListener('input', renderFilterReceptiveField);
        }

        // Modal Open / Close
        if (openHeatmapBtn) {
            openHeatmapBtn.addEventListener('click', () => {
                heatmapModal.classList.remove('hidden');
                if (currentHeatmapTab === 'activations') renderActivationHeatmaps();
                else if (currentHeatmapTab === 'weights') renderWeightHeatmap();
                else if (currentHeatmapTab === 'filters') renderFilterReceptiveField();
            });
        }

        if (closeHeatmapModalBtn) {
            closeHeatmapModalBtn.addEventListener('click', () => {
                heatmapModal.classList.add('hidden');
                if (heatmapTooltip) heatmapTooltip.classList.add('hidden');
            });
        }

        if (heatmapModal) {
            heatmapModal.addEventListener('click', (e) => {
                if (e.target === heatmapModal) {
                    heatmapModal.classList.add('hidden');
                    if (heatmapTooltip) heatmapTooltip.classList.add('hidden');
                }
            });
        }

        // Tooltip tracking on Layer 1 canvas
        function handleCanvasHover(canvas, cols, rows, labelPrefix, getValFn) {
            if (!canvas) return;
            canvas.addEventListener('mousemove', (e) => {
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const c = Math.floor((x / rect.width) * cols);
                const r = Math.floor((y / rect.height) * rows);
                if (c >= 0 && c < cols && r >= 0 && r < rows) {
                    const idx = r * cols + c;
                    const val = getValFn(idx);
                    if (val !== undefined && heatmapTooltip) {
                        heatmapTooltip.textContent = `${labelPrefix} #${idx}: ${val.toFixed(3)}`;
                        heatmapTooltip.style.left = `${e.clientX}px`;
                        heatmapTooltip.style.top = `${e.clientY - 10}px`;
                        heatmapTooltip.classList.remove('hidden');
                    }
                }
            });

            canvas.addEventListener('mouseleave', () => {
                if (heatmapTooltip) heatmapTooltip.classList.add('hidden');
            });
        }

        handleCanvasHover(heatCanvasL1, 16, 8, 'L1 뉴런', (idx) => {
            const acts = model ? model.getLayerActivations() : null;
            return acts && acts[0] ? acts[0][idx] : 0;
        });

        handleCanvasHover(heatCanvasL2, 8, 8, 'L2 뉴런', (idx) => {
            const acts = model ? model.getLayerActivations() : null;
            return acts && acts[1] ? acts[1][idx] : 0;
        });

        handleCanvasHover(heatCanvasL3, 10, 1, '숫자 클래스', (idx) => {
            const acts = model ? model.getLayerActivations() : null;
            return acts && acts[2] ? acts[2][idx] : 0;
        });
    }

    // Main Init
    window.addEventListener('DOMContentLoaded', async () => {
        initCanvas();
        initProbBars();
        setupEventListeners();
        selectTargetDigit(0);
        await initModel();
    });
})();
