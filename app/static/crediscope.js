const form = document.getElementById('applicant-form');
const button = document.getElementById('predict-btn');
const buttonLabel = document.getElementById('button-label');
const score = document.getElementById('score');
const scoreBar = document.getElementById('score-bar');
const status = document.getElementById('risk-status');
const factorsContainer = document.getElementById('factors');
const notice = document.getElementById('notice');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  button.disabled = true;
  buttonLabel.textContent = 'Assessing applicant…';
  status.className = 'risk-status';
  status.textContent = 'Running model and explanations…';
  factorsContainer.replaceChildren();
  const busy = document.createElement('p');
  busy.className = 'busy-text';
  busy.textContent = 'Calculating feature contributions…';
  factorsContainer.appendChild(busy);

  const payload = Object.fromEntries(new FormData(form).entries());
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'The assessment could not be completed.');

    const pct = Math.max(0, Math.min(100, result.probability * 100));
    score.textContent = pct.toFixed(1) + '%';
    scoreBar.style.width = pct.toFixed(2) + '%';
    scoreBar.className = 'bar-fill' + (result.higher_risk ? ' higher' : '');
    status.textContent = result.risk;
    status.className = 'risk-status ' + (result.higher_risk ? 'positive' : 'negative');
    factorsContainer.replaceChildren();
    result.factors.forEach((factor) => {
      const row = document.createElement('div');
      row.className = 'factor';
      const name = document.createElement('div');
      name.className = 'factor-name';
      // UK-facing wording only: preserve the original model feature and category values.
    name.textContent = factor.feature
      .replace(/^Existing credit lines$/i, 'Existing Credit')
      .replace(/^Checking Account Status:/i, 'Current account position:')
      .replace(/^Has Checking Account\b/i, 'Has a current account')
      .replace(/0 To 500 Gbp/gi, 'In credit (up to £500)')
      .replace(/500 Gbp Or More/gi, 'In credit (£500 or more)')
      .replace(/No Account/gi, 'No current account');
      const direction = document.createElement('span');
      direction.className = 'factor-direction ' + factor.direction;
      direction.textContent = factor.direction === 'higher' ? '↑ Higher risk' : '↓ Lower risk';
      row.append(name, direction);
      factorsContainer.appendChild(row);
    });
    notice.textContent = result.notice;
  } catch (error) {
    status.textContent = 'Assessment unavailable';
    factorsContainer.replaceChildren();
    const message = document.createElement('div');
    message.className = 'error';
    message.textContent = error.message;
    factorsContainer.appendChild(message);
    notice.textContent = 'No applicant notice was generated.';
    score.textContent = '—';
    scoreBar.style.width = '0%';
  } finally {
    button.disabled = false;
    buttonLabel.textContent = 'Generate risk assessment';
  }
});
