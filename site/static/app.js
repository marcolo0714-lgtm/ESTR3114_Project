'use strict';
const form = document.querySelector('#prediction-form');
const sentence = document.querySelector('#sentence');
const button = document.querySelector('#predict-button');
const clear = document.querySelector('#clear-button');
const error = document.querySelector('#error');
const progress = document.querySelector('#progress');
const results = document.querySelector('#results-body');
const wrapper = document.querySelector('#results-wrapper');
const section = document.querySelector('#results-section');
const timing = document.querySelector('#result-timing');
const truncation = document.querySelector('#truncation');
const processed = document.querySelector('#processed-input');
let busy = false;
const unavailable = new Set();
const desktopPredictions = window.matchMedia('(min-width: 761px)');
const measuringCanvas = document.createElement('canvas').getContext('2d');

function fitPredictionGroup(group) {
  if (!desktopPredictions.matches || !group.clientWidth) {
    group.classList.remove('wrap-predictions');
    return;
  }
  // Measure the labels at their natural width, rather than splitting a token
  // into tiny fragments to force all five predictions onto a single row.
  const items = [...group.children];
  const widths = items.map(item => Math.max(...[...item.children].map(label => {
    const style = getComputedStyle(label);
    measuringCanvas.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
    return measuringCanvas.measureText(label.textContent).width;
  })));
  const gap = parseFloat(getComputedStyle(group).columnGap) || 0;
  const fullRowWidth = Math.max(...widths) * items.length + gap * (items.length - 1);
  group.classList.toggle('wrap-predictions', fullRowWidth > group.clientWidth);
}
const predictionLayout = new ResizeObserver(entries => {
  // Reflow outside the observer delivery cycle when the number of rows changes.
  for (const entry of entries) requestAnimationFrame(() => fitPredictionGroup(entry.target));
});
desktopPredictions.addEventListener('change', () => {
  for (const group of results.querySelectorAll('.token-list, .sentiment-list')) fitPredictionGroup(group);
});

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function percentage(value) { return `${(value * 100).toFixed(2)}%`; }
function showError(message) { error.textContent = message; error.hidden = false; }
function resetResults() {
  predictionLayout.disconnect();
  results.replaceChildren(); wrapper.hidden = true; error.hidden = true;
  truncation.hidden = true; processed.hidden = true; timing.textContent = '';
}
function renderResult(result) {
  const row = element('tr');
  const name = element('th', result.name); name.scope = 'row'; row.append(name);
  const tokenCell = element('td');
  tokenCell.dataset.label = 'Top five next tokens';
  const list = element('ol', undefined, 'token-list');
  for (const token of result.tokens) {
    const item = element('li');
    item.append(element('code', token.is_eos ? 'EOS' : JSON.stringify(token.text)),
                element('span', percentage(token.probability)));
    list.append(item);
  }
  tokenCell.append(list); row.append(tokenCell);
  const sentimentCell = element('td');
  sentimentCell.dataset.label = 'Sentiment';
  if (result.sentiment) {
    const sentiments = element('ul', undefined, 'sentiment-list');
    for (const [label, probability] of Object.entries(result.sentiment.probabilities)) {
      const line = element('li', undefined, 'probability-row');
      if (label === result.sentiment.label) {
        line.classList.add('predicted-sentiment');
        line.setAttribute('aria-label', `${label}, predicted sentiment`);
      }
      line.append(element('span', label[0].toUpperCase() + label.slice(1)), element('span', percentage(probability)));
      sentiments.append(line);
    }
    sentimentCell.append(sentiments);
  } else {
    sentimentCell.append(element('p', 'Untrained head', 'untrained'),
                         element('p', 'No sentiment supervision.', 'helper'));
  }
  row.append(sentimentCell);
  const runtime = element('td');
  runtime.dataset.label = 'Inference';
  runtime.append(element('div', `${result.inference_ms.toFixed(0)} ms`, 'inference-time'));
  if (result.first_load) runtime.append(element('div', `+ ${(result.loading_ms/1000).toFixed(1)} s first load`, 'helper'));
  row.append(runtime); results.append(row);
  for (const group of row.querySelectorAll('.token-list, .sentiment-list')) predictionLayout.observe(group);
}

for (const example of document.querySelectorAll('[data-example]')) {
  example.addEventListener('click', () => { if (!busy) { sentence.value = example.dataset.example; sentence.focus(); } });
}
clear.addEventListener('click', () => {
  if (busy) return;
  sentence.value = ''; resetResults(); progress.textContent = 'Enter a sentence to compare the models.'; sentence.focus();
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy) return;
  resetResults();
  const methods = [...document.querySelectorAll('input[name="method"]:checked')].map(input => input.value);
  if (!sentence.value.trim()) { showError('Enter a sentence first.'); return; }
  if (!methods.length) { showError('Select at least one method.'); return; }
  const text = sentence.value;
  busy = true; button.disabled = true; clear.disabled = true; sentence.disabled = true;
  for (const input of document.querySelectorAll('input[name="method"]')) input.disabled = true;
  button.textContent = 'Predicting…'; section.setAttribute('aria-busy', 'true');
  progress.textContent = 'Running the selected models. The first request also loads their weights; please wait.';
  try {
    const response = await fetch('/api/predict-text', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, methods}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The input could not be processed. Use at most 12,000 characters and choose a valid method.');
    for (const result of data.results) renderResult(result);
    const input = data.results[0];
    progress.textContent = `${input.used_text_tokens} text tokens + EOS · Predictions for your submitted sentence.`;
    timing.textContent = `${(data.total_ms / 1000).toFixed(2)} s total · ${data.device.toUpperCase()}`;
    document.querySelector('#processed-text').textContent = input.processed_text;
    processed.hidden = false;
    if (input.truncated) {
      truncation.textContent = `Your input contains ${input.original_tokens} tokens. Only its first 95 text tokens were used, followed by EOS.`;
      truncation.hidden = false;
    }
    wrapper.hidden = false;
  } catch (err) {
    progress.textContent = 'No predictions to display.';
    showError(err instanceof TypeError ? 'Cannot reach the prediction server. Make sure it is running, then try again.' : err.message);
  } finally {
    busy = false; button.disabled = false; clear.disabled = false; sentence.disabled = false;
    for (const input of document.querySelectorAll('input[name="method"]')) input.disabled = unavailable.has(input.value);
    button.textContent = 'Predict'; section.setAttribute('aria-busy', 'false');
  }
});

fetch('/api/models').then(async response => {
  if (!response.ok) throw new Error('Unavailable');
  return response.json();
}).then(data => {
  const available = data.models.filter(model => model.available).length;
  document.querySelector('#model-status').textContent = data.message || `${available} of ${data.models.length} models available ${data.location === 'remote' ? 'on the model server' : 'locally'}`;
  for (const model of data.models) {
    if (!model.available) {
      unavailable.add(model.id);
      const input = document.querySelector(`input[value="${model.id}"]`);
      input.checked = false; input.disabled = true;
    }
  }
}).catch(() => {
  document.querySelector('#model-status').textContent = 'Model status unavailable. Check that the server is running.';
});
