/**
 * NumWriting Browser Database (IndexedDB) Module
 * Manages persistent storage for:
 * 1. Real-time adapted model weights
 * 2. User training handwriting samples and history (with thumbnail images)
 */

(function (global) {
    'use strict';

    const DB_NAME = 'NumWritingDB';
    const DB_VERSION = 1;
    const STORE_WEIGHTS = 'model_weights';
    const STORE_HISTORY = 'training_history';

    class NumWritingDB {
        constructor() {
            this.db = null;
            this.initPromise = null;
        }

        // Initialize IndexedDB
        async init() {
            if (this.initPromise) return this.initPromise;

            this.initPromise = new Promise((resolve, reject) => {
                if (!('indexedDB' in window)) {
                    console.warn('IndexedDB not supported in this environment');
                    resolve(null);
                    return;
                }

                const request = indexedDB.open(DB_NAME, DB_VERSION);

                request.onupgradeneeded = (event) => {
                    const db = event.target.result;

                    // 1. Model Weights store
                    if (!db.objectStoreNames.contains(STORE_WEIGHTS)) {
                        db.createObjectStore(STORE_WEIGHTS, { keyPath: 'id' });
                    }

                    // 2. Real-time Training History store
                    if (!db.objectStoreNames.contains(STORE_HISTORY)) {
                        const historyStore = db.createObjectStore(STORE_HISTORY, {
                            keyPath: 'id',
                            autoIncrement: true
                        });
                        historyStore.createIndex('timestamp', 'timestamp', { unique: false });
                        historyStore.createIndex('target', 'target', { unique: false });
                    }
                };

                request.onsuccess = (event) => {
                    this.db = event.target.result;
                    resolve(this.db);
                };

                request.onerror = (event) => {
                    console.error('IndexedDB open error:', event.target.error);
                    reject(event.target.error);
                };
            });

            return this.initPromise;
        }

        // Save model weights to IndexedDB
        async saveModelWeights(model) {
            await this.init();
            if (!this.db) return false;

            const modelData = model.toObject();
            return new Promise((resolve, reject) => {
                const tx = this.db.transaction([STORE_WEIGHTS], 'readwrite');
                const store = tx.objectStore(STORE_WEIGHTS);

                const item = {
                    id: 'current_weights',
                    updatedAt: new Date().toISOString(),
                    modelData: modelData
                };

                const req = store.put(item);
                req.onsuccess = () => resolve(true);
                req.onerror = (e) => reject(e.target.error);
            });
        }

        // Load model weights from IndexedDB
        async loadModelWeights(model) {
            await this.init();
            if (!this.db) return false;

            return new Promise((resolve, reject) => {
                const tx = this.db.transaction([STORE_WEIGHTS], 'readonly');
                const store = tx.objectStore(STORE_WEIGHTS);
                const req = store.get('current_weights');

                req.onsuccess = () => {
                    if (req.result && req.result.modelData) {
                        try {
                            model.fromObject(req.result.modelData);
                            resolve({
                                success: true,
                                updatedAt: req.result.updatedAt
                            });
                        } catch (err) {
                            console.error('Failed to parse saved weights:', err);
                            resolve({ success: false });
                        }
                    } else {
                        resolve({ success: false });
                    }
                };

                req.onerror = (e) => reject(e.target.error);
            });
        }

        // Clear stored model weights
        async clearModelWeights() {
            await this.init();
            if (!this.db) return false;

            return new Promise((resolve, reject) => {
                const tx = this.db.transaction([STORE_WEIGHTS], 'readwrite');
                const store = tx.objectStore(STORE_WEIGHTS);
                const req = store.delete('current_weights');
                req.onsuccess = () => resolve(true);
                req.onerror = (e) => reject(e.target.error);
            });
        }

        // Add a training sample record to history
        async addTrainingRecord(record) {
            await this.init();
            if (!this.db) return null;

            const entry = {
                timestamp: record.timestamp || Date.now(),
                isoDate: new Date().toLocaleString(),
                target: record.target,
                old_prediction: record.old_prediction,
                old_target_prob: record.old_target_prob,
                new_prediction: record.new_prediction,
                new_target_prob: record.new_target_prob,
                prob_diff: record.prob_diff,
                loss: record.loss,
                thumbnail: record.thumbnail || '',
                pixels: record.pixels ? Array.from(record.pixels) : []
            };

            return new Promise((resolve, reject) => {
                const tx = this.db.transaction([STORE_HISTORY], 'readwrite');
                const store = tx.objectStore(STORE_HISTORY);
                const req = store.add(entry);

                req.onsuccess = () => {
                    entry.id = req.result;
                    resolve(entry);
                };
                req.onerror = (e) => reject(e.target.error);
            });
        }

        // Get all training records
        async getAllTrainingRecords() {
            await this.init();
            if (!this.db) return [];

            return new Promise((resolve, reject) => {
                const tx = this.db.transaction([STORE_HISTORY], 'readonly');
                const store = tx.objectStore(STORE_HISTORY);
                const req = store.getAll();

                req.onsuccess = () => {
                    const list = req.result || [];
                    // Sort descending by timestamp (newest first)
                    list.sort((a, b) => b.timestamp - a.timestamp);
                    resolve(list);
                };
                req.onerror = (e) => reject(e.target.error);
            });
        }

        // Delete a specific training record by ID
        async deleteTrainingRecord(id) {
            await this.init();
            if (!this.db) return false;

            return new Promise((resolve, reject) => {
                const tx = this.db.transaction([STORE_HISTORY], 'readwrite');
                const store = tx.objectStore(STORE_HISTORY);
                const req = store.delete(id);
                req.onsuccess = () => resolve(true);
                req.onerror = (e) => reject(e.target.error);
            });
        }

        // Clear all training history records
        async clearAllTrainingHistory() {
            await this.init();
            if (!this.db) return false;

            return new Promise((resolve, reject) => {
                const tx = this.db.transaction([STORE_HISTORY], 'readwrite');
                const store = tx.objectStore(STORE_HISTORY);
                const req = store.clear();
                req.onsuccess = () => resolve(true);
                req.onerror = (e) => reject(e.target.error);
            });
        }

        // Get summary statistics of the database
        async getStats() {
            await this.init();
            if (!this.db) return { totalSamples: 0, hasCustomWeights: false, lastUpdate: null };

            const records = await this.getAllTrainingRecords();

            let hasCustomWeights = false;
            let lastUpdate = null;

            await new Promise((resolve) => {
                const tx = this.db.transaction([STORE_WEIGHTS], 'readonly');
                const req = tx.objectStore(STORE_WEIGHTS).get('current_weights');
                req.onsuccess = () => {
                    if (req.result) {
                        hasCustomWeights = true;
                        lastUpdate = req.result.updatedAt;
                    }
                    resolve();
                };
                req.onerror = () => resolve();
            });

            return {
                totalSamples: records.length,
                hasCustomWeights: hasCustomWeights,
                lastUpdate: lastUpdate,
                recentRecords: records.slice(0, 10)
            };
        }

        // Export entire training database to JSON format for backup
        async exportToJson() {
            const records = await this.getAllTrainingRecords();
            let weightsData = null;

            await new Promise((resolve) => {
                const tx = this.db.transaction([STORE_WEIGHTS], 'readonly');
                const req = tx.objectStore(STORE_WEIGHTS).get('current_weights');
                req.onsuccess = () => {
                    weightsData = req.result ? req.result.modelData : null;
                    resolve();
                };
                req.onerror = () => resolve();
            });

            return JSON.stringify({
                version: 1,
                exportedAt: new Date().toISOString(),
                totalSamples: records.length,
                trainingHistory: records,
                modelWeights: weightsData
            }, null, 2);
        }

        // Import training database from JSON
        async importFromJson(jsonStr, model = null) {
            const data = JSON.parse(jsonStr);
            if (!data.trainingHistory) throw new Error('Invalid JSON backup file');

            await this.init();

            // Import training records
            const tx = this.db.transaction([STORE_HISTORY], 'readwrite');
            const store = tx.objectStore(STORE_HISTORY);
            for (const item of data.trainingHistory) {
                const clone = { ...item };
                delete clone.id; // allow new ID assignment
                store.add(clone);
            }

            await new Promise((resolve, reject) => {
                tx.oncomplete = () => resolve();
                tx.onerror = (e) => reject(e.target.error);
            });

            // If model weights are present and model instance provided, restore weights
            if (data.modelWeights && model) {
                model.fromObject(data.modelWeights);
                await this.saveModelWeights(model);
            }

            return true;
        }
    }

    // Export singleton instance to window
    global.appDB = new NumWritingDB();

})(typeof window !== 'undefined' ? window : this);
