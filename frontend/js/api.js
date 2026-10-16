/**
 * Cognisec AI - API Client Module
 * Provides typed, error-handled HTTP requests to the backend server.
 */

const API = {
  baseUrl: '',

  /**
   * Helper for standard JSON GET requests
   */
  async get(endpoint) {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`);
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      console.error(`API GET ${endpoint} failed:`, err);
      throw err;
    }
  },

  /**
   * Helper for POST requests with FormData or JSON
   */
  async post(endpoint, data, isFormData = false) {
    try {
      const options = {
        method: 'POST',
      };

      if (isFormData) {
        options.body = data;
      } else {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(data);
      }

      const response = await fetch(`${this.baseUrl}${endpoint}`, options);
      const json = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(json.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return json;
    } catch (err) {
      console.error(`API POST ${endpoint} failed:`, err);
      throw err;
    }
  },

  // Specific API calls
  async getHealth() {
    return this.get('/api/health');
  },

  async getMetadata() {
    return this.get('/api/metadata');
  },

  async getExperiments() {
    return this.get('/api/experiments');
  },

  async getComparison() {
    return this.get('/api/results/comparison');
  },

  async getSampleCsv() {
    return this.get('/api/sample-csv');
  },

  async previewCsv(data) {
    const isForm = data instanceof FormData;
    return this.post('/api/analyze/csv-preview', data, isForm);
  },

  async runXGBoost(data) {
    const isForm = data instanceof FormData;
    return this.post('/api/analyze/xgboost', data, isForm);
  },

  async runVAE(data) {
    const isForm = data instanceof FormData;
    return this.post('/api/analyze/vae', data, isForm);
  },

  async runDual(data) {
    const isForm = data instanceof FormData;
    return this.post('/api/analyze/dual', data, isForm);
  },

  // Live monitoring endpoints
  async getLiveInterfaces() {
    return this.get('/api/live/interfaces');
  },

  async startLiveCapture(data) {
    return this.post('/api/live/start', data);
  },

  async stopLiveCapture() {
    return this.post('/api/live/stop', {});
  },

  async getLiveStatus() {
    return this.get('/api/live/status');
  },

  async getLiveFlows(limit = 50, offset = 0) {
    return this.get(`/api/live/flows?limit=${limit}&offset=${offset}`);
  },

  async clearLiveBuffer() {
    return this.post('/api/live/clear', {});
  },

  async getLiveAnalytics() {
    return this.get('/api/analytics/live');
  },

  async getLiveHistory() {
    return this.get('/api/analytics/live/history');
  },

  async getAnalyticsSummary() {
    return this.get('/api/analytics/summary');
  },

  async getResearchFigures() {
    return this.get('/api/results/research-figures');
  },

  async getCsvAnalytics(data) {
    const isForm = data instanceof FormData;
    return this.post('/api/analyze/csv-analytics', data, isForm);
  },

  async getUploadAnalytics(analysisId) {
    return this.get(`/api/analytics/upload/${analysisId}`);
  }
};



window.API = API;
