'use strict';
const imageForm = document.querySelector('#image-form');
const imageFile = document.querySelector('#image-file');
const imageTask = document.querySelector('#image-task');
const imageCheckpoint = document.querySelector('#image-checkpoint');
const imageButton = document.querySelector('#image-predict');
const imageClear = document.querySelector('#image-clear');
const imageError = document.querySelector('#image-error');
const imageResults = document.querySelector('#image-results');
const imageProgress = document.querySelector('#image-progress');
const imagePreview = document.querySelector('#upload-preview');
const imageSource = document.querySelector('#image-source');
let imageMetadata = null;
let imageBusy = false;
let previewSequence = 0;
let selectedImage = null;
let imageMaxBytes = Number(imageForm.dataset.maxImageBytes) || 8 * 1024 * 1024;

function imageSizeError() { return `Choose an image no larger than ${imageMaxBytes / 1024 / 1024} MB.`; }

function imageNode(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function readable(label) { return label.replaceAll('_', ' '); }
function imageFail(message) { imageError.textContent = message; imageError.hidden = false; }
function resetImageResults() {
  imageResults.replaceChildren(); imageError.hidden = true;
  document.querySelector('#processed-image').hidden = true;
  document.querySelector('#image-timing').textContent = '';
}
function updateImageAvailability() {
  if (!imageMetadata) return;
  const kind = imageCheckpoint.value;
  document.querySelector('#checkpoint-help').textContent = kind === 'final'
    ? 'Both models at step 10,000: the final states of the training runs.'
    : imageMetadata.models.map(m => `${m.name}: step ${m.checkpoints[kind].step.toLocaleString()}`).join('; ')
      + '. Selected by highest mean validation accuracy, with lower loss breaking ties.';
  let count = 0;
  for (const input of imageForm.querySelectorAll('input[name="image-method"]')) {
    const entry = imageMetadata.models.find(m => m.id === input.value);
    const available = Boolean(entry && entry.checkpoints[kind].available);
    input.disabled = imageBusy || !available;
    if (available) count++;
    else input.checked = false;
  }
  document.querySelector('#image-model-status').textContent = imageMetadata.message || `${count} of 2 image models available ${imageMetadata.location === 'remote' ? 'on the model server' : 'locally'}`;
}
function updateTaskClasses() {
  const task = imageMetadata?.tasks.find(t => t.id === Number(imageTask.value));
  document.querySelector('#task-classes').textContent = imageTask.value !== '' && task
    ? `Possible classes: ${task.class_names.map(readable).join(', ')}.`
    : 'Select a group to see its five possible classes.';
}
imageTask.addEventListener('change', () => { resetImageResults(); updateTaskClasses(); });
imageCheckpoint.addEventListener('change', () => { resetImageResults(); updateImageAvailability(); });
function selectImage(file, source = 'file') {
  imageButton.disabled = imageBusy;
  for (const example of imageForm.querySelectorAll('[data-image-example]')) example.setAttribute('aria-pressed', 'false');
  resetImageResults();
  selectedImage = null;
  imageSource.textContent = 'No image selected.';
  imagePreview.hidden = true;
  imagePreview.removeAttribute('src');
  document.querySelector('#preview-placeholder').hidden = false;
  const sequence = ++previewSequence;
  if (!file) return;
  if (file.size > imageMaxBytes) { imageFail(imageSizeError()); return; }
  if (file.type && !['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    imageFail('Use a PNG, JPEG or WebP image.'); return;
  }
  selectedImage = file;
  imageSource.textContent = `${source === 'clipboard' ? 'Pasted image' : source === 'example' ? 'Example image' : 'Selected image'} ready (${Math.max(1, Math.ceil(file.size / 1024))} KB).`;
  imageProgress.textContent = imageTask.value === ''
    ? 'Image ready. Choose its known superclass.'
    : 'Image ready. Select Predict to compare the models.';
  const reader = new FileReader();
  reader.onload = () => {
    if (sequence !== previewSequence) return;
    imagePreview.src = reader.result; imagePreview.hidden = false;
    document.querySelector('#preview-placeholder').hidden = true;
  };
  reader.onerror = () => {
    if (sequence !== previewSequence) return;
    selectedImage = null; imageSource.textContent = 'No image selected.';
    imageFail('This image could not be read.');
  };
  reader.readAsDataURL(file);
}
imageFile.addEventListener('change', () => selectImage(imageFile.files[0]));

for (const example of imageForm.querySelectorAll('[data-image-example]')) {
  example.addEventListener('click', async () => {
    if (imageBusy) return;
    selectImage(null);
    imageFile.value = '';
    const sequence = previewSequence;
    imageButton.disabled = true;
    imageSource.textContent = 'Loading example image…';
    try {
      const response = await fetch(example.dataset.imageExample);
      if (!response.ok) throw new Error('Example unavailable');
      const blob = await response.blob();
      // A newer example, upload, paste or Clear wins over this request.
      if (sequence !== previewSequence) return;
      const file = new File([blob], example.dataset.imageExample.split('/').pop(), {type: blob.type});
      imageTask.value = example.dataset.task;
      updateTaskClasses();
      selectImage(file, 'example');
      if (selectedImage) {
        example.setAttribute('aria-pressed', 'true');
        imageSource.textContent = `${example.dataset.label} example ready. Superclass selected automatically.`;
      }
    } catch (_) {
      if (sequence !== previewSequence) return;
      imageButton.disabled = false;
      imageSource.textContent = 'No image selected.';
      imageFail('The example image could not be loaded. Try another example, upload a file, or paste an image.');
    }
  });
}
document.addEventListener('paste', event => {
  // Clipboard image bytes arrive with the user-triggered paste event. No
  // clipboard permission prompt, background read, or image-URL fetch is needed.
  const clipboard = event.clipboardData;
  if (!clipboard) return;
  const files = [...(clipboard.items || [])]
    .filter(item => item.kind === 'file' && item.type.startsWith('image/'))
    .map(item => item.getAsFile()).filter(Boolean);
  const file = files[0] || [...(clipboard.files || [])].find(item => item.type.startsWith('image/'));
  if (!file) return; // Leave ordinary text pastes alone.
  event.preventDefault();
  if (imageBusy) return;
  imageFile.value = '';
  selectImage(file, 'clipboard');
  if (selectedImage) {
    // Reflect the pasted image in the native picker where supported; selectedImage
    // remains the source of truth so paste also works without this assignment.
    try {
      const transfer = new DataTransfer(); transfer.items.add(file);
      imageFile.files = transfer.files;
    } catch (_) { /* The preview and prediction still use the pasted File. */ }
  }
});
imageClear.addEventListener('click', () => {
  if (imageBusy) return;
  imageForm.reset(); previewSequence++; selectedImage = null;
  imageButton.disabled = false;
  for (const example of imageForm.querySelectorAll('[data-image-example]')) example.setAttribute('aria-pressed', 'false');
  imageSource.textContent = 'No image selected.';
  imagePreview.removeAttribute('src'); imagePreview.hidden = true;
  document.querySelector('#preview-placeholder').hidden = false;
  resetImageResults(); updateImageAvailability(); updateTaskClasses();
  imageProgress.textContent = 'Upload an image and choose its superclass.';
});
imageForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (imageBusy) return;
  resetImageResults();
  const file = selectedImage;
  const methods = [...imageForm.querySelectorAll('input[name="image-method"]:checked')].map(x => x.value);
  if (!file) { imageFail('Choose an image first.'); return; }
  if (file.size > imageMaxBytes) { imageFail(imageSizeError()); return; }
  if (imageTask.value === '') { imageFail('Choose the known superclass.'); return; }
  if (!methods.length) { imageFail('Select at least one available method.'); return; }
  const params = new URLSearchParams({task: imageTask.value, checkpoint: imageCheckpoint.value});
  for (const method of methods) params.append('methods', method);
  imageBusy = true;
  for (const input of imageForm.querySelectorAll('input, select, button')) input.disabled = true;
  imageButton.textContent = 'Predicting…';
  document.querySelector('#image-results-section').setAttribute('aria-busy', 'true');
  imageProgress.textContent = 'Running the selected models. The first request also loads their weights.';
  try {
    const response = await fetch(`/api/predict-image?${params}`, {
      method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The image or selected options could not be processed.');
    for (const result of data.results) {
      const card = imageNode('article', undefined, 'image-result-card');
      card.append(imageNode('h3', result.name),
        imageNode('p', `${result.checkpoint === 'final' ? 'Final' : 'Best validation'} · Step ${result.step.toLocaleString()} · Seed ${result.seed}`, 'helper'),
        imageNode('p', `Predicted: ${readable(result.predicted_class)}`, 'image-predicted-class'));
      const table = imageNode('table');
      const caption = imageNode('caption', `${result.name}: probabilities within ${readable(data.task.name)}`, 'sr-only');
      table.append(caption);
      const head = imageNode('thead'); const heading = imageNode('tr');
      for (const title of ['Class', 'Probability']) { const th = imageNode('th', title); th.scope = 'col'; heading.append(th); }
      head.append(heading); table.append(head);
      const body = imageNode('tbody');
      for (const item of result.probabilities) {
        const row = imageNode('tr'); const label = imageNode('th', readable(item.label)); label.scope = 'row';
        if (item.label === result.predicted_class) row.className = 'image-predicted-row';
        row.append(label, imageNode('td', `${(100 * item.probability).toFixed(2)}%`)); body.append(row);
      }
      table.append(body); card.append(table);
      card.append(imageNode('p', `${result.inference_ms.toFixed(1)} ms inference${result.first_load ? ` + ${(result.loading_ms / 1000).toFixed(2)} s first load` : ''}`, 'helper'));
      imageResults.append(card);
    }
    const preview = document.querySelector('#model-preview'); preview.src = data.image.preview;
    document.querySelector('#processed-image-note').textContent = `${data.image.width} × ${data.image.height} original → 32 × 32 RGB input. Shown enlarged; the model uses the same processed pixels.`;
    document.querySelector('#processed-image').hidden = false;
    imageProgress.textContent = `Predictions within ${readable(data.task.name)}. The superclass was supplied, not predicted.`;
    document.querySelector('#image-timing').textContent = `${(data.total_ms / 1000).toFixed(2)} s total · ${data.device.toUpperCase()}`;
  } catch (err) {
    imageProgress.textContent = 'No predictions to display.';
    imageFail(err instanceof TypeError ? 'Cannot reach the prediction server. Make sure it is running.' : err.message);
  } finally {
    imageBusy = false;
    for (const input of imageForm.querySelectorAll('input, select, button')) input.disabled = false;
    updateImageAvailability(); imageButton.textContent = 'Predict';
    document.querySelector('#image-results-section').setAttribute('aria-busy', 'false');
  }
});
fetch('/api/image-models').then(async response => {
  if (!response.ok) throw new Error('Cannot read image models');
  return response.json();
}).then(data => { imageMetadata = data; imageMaxBytes = data.max_image_bytes || imageMaxBytes; updateImageAvailability(); updateTaskClasses(); })
  .catch(() => { document.querySelector('#image-model-status').textContent = 'Image model status unavailable. Check the server.'; });
