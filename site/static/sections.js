'use strict';

// Both reports keep their desktop sidebar. On smaller screens, the menu is
// closed initially and closes after navigation, Escape, or an outside click.
const sectionsToggle = document.querySelector('#report-sections-toggle');
const sectionsSidebar = document.querySelector('#report-sidebar');
const sectionsMobile = window.matchMedia('(max-width: 1023px)');

if (sectionsToggle && sectionsSidebar) {
  function setSectionsOpen(open, restoreFocus = false) {
    const expanded = sectionsMobile.matches && open;
    document.body.classList.toggle('sections-open', expanded);
    sectionsToggle.setAttribute('aria-expanded', String(expanded));
    sectionsToggle.textContent = expanded ? 'Hide sections' : 'Show sections';
    if (restoreFocus && sectionsMobile.matches) sectionsToggle.focus({preventScroll: true});
  }
  sectionsToggle.addEventListener('click', () => {
    setSectionsOpen(sectionsToggle.getAttribute('aria-expanded') !== 'true');
  });
  sectionsSidebar.addEventListener('click', event => {
    if (sectionsMobile.matches && event.target.closest('a[href^="#"]')) setSectionsOpen(false, true);
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && sectionsToggle.getAttribute('aria-expanded') === 'true') {
      setSectionsOpen(false, true);
    }
  });
  document.addEventListener('click', event => {
    if (sectionsMobile.matches && !sectionsSidebar.contains(event.target) && !sectionsToggle.contains(event.target)) {
      setSectionsOpen(false);
    }
  });
  sectionsMobile.addEventListener('change', () => {
    const focusedInside = sectionsSidebar.contains(document.activeElement);
    setSectionsOpen(false, focusedInside);
  });
  setSectionsOpen(false);
}
