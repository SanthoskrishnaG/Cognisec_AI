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

  async previewCsv(formData) {
    return this.post('/api/analyze/csv-preview', formData, true);
  },

  async runXGBoost(formData) {
    return this.post('/api/analyze/xgboost', formData, true);
  },

  async runVAE(formData) {
    return this.post('/api/analyze/vae', formData, true);
  },

  async runDual(formData) {
    return this.post('/api/analyze/dual', formData, true);
  }
};

window.API = API;
