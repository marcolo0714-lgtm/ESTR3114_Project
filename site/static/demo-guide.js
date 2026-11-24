'use strict';

const demoDialog = document.querySelector('#live-demo-dialog');
if (demoDialog) {
  // A changed tunnel URL gets a fresh announcement, even in an existing session.
  const demoKey = `estr3114-demo-seen:${demoDialog.dataset.demoUrl || 'pending'}`;
  let alreadySeen = false;
  try { alreadySeen = sessionStorage.getItem(demoKey) === 'yes'; } catch (_) { /* Storage may be blocked. */ }
  if (!alreadySeen) demoDialog.showModal();
  function rememberDemoVisit() {
    try { sessionStorage.setItem(demoKey, 'yes'); } catch (_) { /* Navigation still works without storage. */ }
  }
  demoDialog.addEventListener('close', rememberDemoVisit);
  for (const link of demoDialog.querySelectorAll('a')) link.addEventListener('click', rememberDemoVisit);
}
