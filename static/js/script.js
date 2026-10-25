/**
 * Plant Disease Detection System — Frontend JavaScript
 * =====================================================
 * Handles file upload, drag-and-drop, API communication,
 * and dynamic result rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------
  // DOM References
  // -------------------------------------------------------
  const navbar = document.getElementById('navbar');
  const navToggle = document.getElementById('nav-toggle');
  const navMenu = document.getElementById('nav-menu');
  const navLinks = document.querySelectorAll('.nav-link');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const dropZonePrompt = document.getElementById('drop-zone-prompt');
  const previewArea = document.getElementById('preview-area');
  const previewImage = document.getElementById('preview-image');
  const removeImageBtn = document.getElementById('remove-image-btn');
  const fileMeta = document.getElementById('file-meta');
  const fileName = document.getElementById('file-name');
  const fileSize = document.getElementById('file-size');
  const analyzeBtn = document.getElementById('analyze-btn');
  const loadingOverlay = document.getElementById('loading-overlay');
  const resultsSection = document.getElementById('results-section');
  const resultImage = document.getElementById('result-image');
  const resultPlantName = document.getElementById('result-plant-name');
  const resultStatusBadge = document.getElementById('result-status-badge');
  const resultDiseaseName = document.getElementById('result-disease-name');
  const resultConfidencePercent = document.getElementById('result-confidence-percent');
  const resultConfidenceBar = document.getElementById('result-confidence-bar');
  const topPredictionsList = document.getElementById('top-predictions-list');
  const diseaseDescription = document.getElementById('disease-description');
  const diseasePreventionList = document.getElementById('disease-prevention-list');
  const tryAnotherBtn = document.getElementById('try-another-btn');
  const toastContainer = document.getElementById('toast-container');

  // -------------------------------------------------------
  // State
  // -------------------------------------------------------
  let selectedFile = null;

  // -------------------------------------------------------
  // Constants
  // -------------------------------------------------------
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB
  const ALLOWED_TYPES = ['image/jpeg', 'image/png'];

  // -------------------------------------------------------
  // 1. Navigation
  // -------------------------------------------------------

  // Mobile toggle
  if (navToggle) {
    navToggle.addEventListener('click', () => {
      navMenu.classList.toggle('open');
    });
  }

  // Smooth scroll + active state
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      navMenu.classList.remove('open');

      const targetId = link.getAttribute('href');
      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        targetEl.scrollIntoView({ behavior: 'smooth' });
      }

      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });

  // Update active nav on scroll
  const sections = document.querySelectorAll('section[id], header[id]');
  window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
      const top = section.offsetTop - 100;
      if (window.scrollY >= top) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  });

  // -------------------------------------------------------
  // 2. Drag & Drop
  // -------------------------------------------------------

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  // -------------------------------------------------------
  // 3. File Input
  // -------------------------------------------------------

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
    }
  });

  // -------------------------------------------------------
  // 4. File Validation
  // -------------------------------------------------------

  function validateFile(file) {
    if (!file) {
      showToast('No file selected. Please choose an image.', 'error');
      return false;
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
      showToast('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'error');
      return false;
    }

    if (file.size > MAX_FILE_SIZE) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      showToast(`File is too large (${sizeMB} MB). Maximum allowed size is 10 MB.`, 'error');
      return false;
    }

    return true;
  }

  // -------------------------------------------------------
  // 5. File Handling & Preview
  // -------------------------------------------------------

  function handleFile(file) {
    if (!validateFile(file)) {
      return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      dropZonePrompt.classList.add('hidden');
      previewArea.classList.remove('hidden');
      fileName.textContent = file.name;
      fileSize.textContent = formatFileSize(file.size);
      analyzeBtn.disabled = false;

      // Hide results if showing from previous run
      resultsSection.classList.add('hidden');
    };
    reader.readAsDataURL(file);
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  // Remove image button
  removeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  // Sample chips quick-select handler
  const sampleChips = document.querySelectorAll('.sample-chip');
  sampleChips.forEach(chip => {
    chip.addEventListener('click', async () => {
      const sampleFile = chip.getAttribute('data-sample');
      try {
        const response = await fetch(`/static/samples/${sampleFile}`);
        if (!response.ok) throw new Error('Could not load sample image');
        const blob = await response.blob();
        const file = new File([blob], sampleFile, { type: 'image/jpeg' });
        handleFile(file);
        showToast('Sample leaf loaded! Click "Analyze Leaf" to diagnose.', 'success');
      } catch (err) {
        showToast('Error loading sample image. Please try uploading a file.', 'error');
      }
    });
  });

  // -------------------------------------------------------
  // 6. Analyze Button → API Call
  // -------------------------------------------------------

  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
      showToast('Please select an image first.', 'error');
      return;
    }

    // Show loading
    analyzeBtn.disabled = true;
    loadingOverlay.classList.remove('hidden');

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const savedKey = localStorage.getItem('plant_ai_gemini_key') || '';
      const headers = {};
      if (savedKey) {
        headers['X-Gemini-Key'] = savedKey;
      }

      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
        headers: headers,
      });

      const data = await response.json();

      if (data.success) {
        displayResults(data);
      } else {
        showToast(data.error || 'Prediction failed. Please try again.', 'error');
      }

    } catch (err) {
      console.error('Prediction error:', err);
      showToast('Network error. Please check your connection and try again.', 'error');
    } finally {
      loadingOverlay.classList.add('hidden');
      analyzeBtn.disabled = false;
    }
  });

  // -------------------------------------------------------
  // 7. Display Results
  // -------------------------------------------------------

  function displayResults(data) {
    const { predictions, disease_info, image_url, warning } = data;
    const top = predictions[0];

    // Set image
    resultImage.src = image_url;
    resultImage.alt = `Analyzed leaf: ${top.disease_name}`;

    const unsupportedAlert = document.getElementById('unsupported-alert');
    const unsupportedAlertTitle = document.getElementById('unsupported-alert-title');
    const unsupportedAlertDesc = document.getElementById('unsupported-alert-desc');
    const isReliable = (data.is_reliable !== false) && (top.confidence >= 50.0);

    if (!isReliable) {
      // Image is an unsupported plant (like Paddy) or confidence is too low
      if (unsupportedAlert) {
        unsupportedAlert.classList.remove('hidden');
        if (data.warning_type === 'unsupported_plant' || top.confidence < 50.0) {
          unsupportedAlertTitle.textContent = '⚠️ Inconclusive Diagnosis — Unsupported Crop Detected';
          unsupportedAlertDesc.innerHTML = data.warning || 
            `The AI confidence is low (<strong>${top.confidence.toFixed(1)}%</strong>). This image does not reliably match our 14 supported crops. If this is <strong>Paddy/Rice</strong>, Wheat, or another unlisted plant, please note that diagnoses for unsupported crops cannot be guaranteed.`;
        } else {
          unsupportedAlertTitle.textContent = '⚠️ Ambiguous / Low Confidence Diagnosis';
          unsupportedAlertDesc.textContent = data.warning || 'The AI is uncertain about this prediction. Please try a clearer, closer leaf photo.';
        }
      }

      // Update badge to reflect uncertainty
      resultStatusBadge.textContent = 'Inconclusive';
      resultStatusBadge.className = 'status-badge unsupported';

      // Update plant & disease display with disclaimer
      resultPlantName.textContent = `Plant: Unknown / Unsupported (Best match: ${top.plant_name})`;
      resultDiseaseName.textContent = `${top.disease_name} (Uncertain)`;
    } else {
      // Diagnosis is reliable
      if (unsupportedAlert) {
        unsupportedAlert.classList.add('hidden');
      }

      resultPlantName.textContent = `Plant: ${top.plant_name}`;
      resultStatusBadge.textContent = top.is_healthy ? 'Healthy' : 'Diseased';
      resultStatusBadge.className = `status-badge ${top.is_healthy ? 'healthy' : 'diseased'}`;
      resultDiseaseName.textContent = top.disease_name;
    }

    // Confidence bar (animate)
    const conf = top.confidence;
    resultConfidencePercent.textContent = `${conf.toFixed(1)}%`;
    resultConfidenceBar.style.width = '0%';
    resultConfidenceBar.className = 'confidence-fill';

    if (conf < 40) {
      resultConfidenceBar.classList.add('low');
    } else if (conf < 70) {
      resultConfidenceBar.classList.add('medium');
    }

    // Trigger the animation after a small delay
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        resultConfidenceBar.style.width = `${conf}%`;
      });
    });

    // Top predictions list
    topPredictionsList.innerHTML = '';
    predictions.forEach((pred, index) => {
      const item = document.createElement('div');
      item.className = 'prediction-item';
      item.innerHTML = `
        <span class="prediction-rank">${index + 1}</span>
        <span class="prediction-name" title="${pred.plant_name} — ${pred.disease_name}">
          ${pred.plant_name} — ${pred.disease_name}
        </span>
        <div class="prediction-bar-track">
          <div class="prediction-bar-fill" style="width: 0%;"></div>
        </div>
        <span class="prediction-conf">${pred.confidence.toFixed(1)}%</span>
      `;
      topPredictionsList.appendChild(item);

      // Animate prediction bars
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const barFill = item.querySelector('.prediction-bar-fill');
          barFill.style.width = `${pred.confidence}%`;
        });
      });
    });

    // Disease info
    if (disease_info) {
      diseaseDescription.textContent = disease_info.description || 'No additional information available.';

      diseasePreventionList.innerHTML = '';
      if (disease_info.prevention && disease_info.prevention.length > 0) {
        disease_info.prevention.forEach(tip => {
          const li = document.createElement('li');
          li.textContent = tip;
          diseasePreventionList.appendChild(li);
        });
      }
    }

    // Show warning toast if confidence is low
    if (warning) {
      showToast(warning, 'warning');
    }

    // Show results section
    resultsSection.classList.remove('hidden');

    // Smooth scroll to results
    setTimeout(() => {
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  }

  // -------------------------------------------------------
  // 8. Try Another Image
  // -------------------------------------------------------

  tryAnotherBtn.addEventListener('click', () => {
    resetUpload();
    resultsSection.classList.add('hidden');

    // Scroll to upload section
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
      uploadSection.scrollIntoView({ behavior: 'smooth' });
    }
  });

  function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImage.src = '';
    previewArea.classList.add('hidden');
    dropZonePrompt.classList.remove('hidden');
    analyzeBtn.disabled = true;
  }

  // -------------------------------------------------------
  // 9. Toast Notifications
  // -------------------------------------------------------

  function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'error' ? '❌' : type === 'warning' ? '⚠️' : '✅';
    toast.innerHTML = `<span>${icon}</span><span>${message}</span>`;

    toastContainer.appendChild(toast);

    // Auto-remove after animation finishes
    setTimeout(() => {
      if (toast.parentNode) {
        toast.remove();
      }
    }, 4000);
  }

  // Modal handlers
  const apiKeyModal = document.getElementById('api-key-modal');
  const closeModalBtn = document.getElementById('close-modal-btn');
  const saveKeyBtn = document.getElementById('save-key-btn');
  const clearKeyBtn = document.getElementById('clear-key-btn');
  const geminiKeyInput = document.getElementById('gemini-key-input');

  if (geminiKeyInput) {
    geminiKeyInput.value = localStorage.getItem('plant_ai_gemini_key') || '';
  }

  const openModalBtn = document.getElementById('open-api-modal-btn');
  if (openModalBtn && apiKeyModal) {
    openModalBtn.addEventListener('click', () => apiKeyModal.classList.remove('hidden'));
  }

  if (closeModalBtn) {
    closeModalBtn.addEventListener('click', () => apiKeyModal.classList.add('hidden'));
  }

  if (saveKeyBtn) {
    saveKeyBtn.addEventListener('click', () => {
      const key = geminiKeyInput.value.trim();
      if (key) {
        localStorage.setItem('plant_ai_gemini_key', key);
        showToast('Gemini Vision AI enabled successfully!', 'success');
      } else {
        localStorage.removeItem('plant_ai_gemini_key');
        showToast('Using high-speed built-in AI engine.', 'success');
      }
      apiKeyModal.classList.add('hidden');
    });
  }

  if (clearKeyBtn) {
    clearKeyBtn.addEventListener('click', () => {
      geminiKeyInput.value = '';
      localStorage.removeItem('plant_ai_gemini_key');
      showToast('Key cleared. Using built-in AI.', 'success');
      apiKeyModal.classList.add('hidden');
    });
  }
});