function navbar($elements) {
  $elements.forEach($navbar => {
    // TODO: Use data-toggle approach.
    const $toggle = $navbar.querySelector('.navbar__toggle');
    const $sidebar = $navbar.querySelector('.navbar-sidebar');
    const $sidebarClose = $navbar.querySelector('.navbar-sidebar__close');
    const $backdrop = $navbar.querySelector('.navbar-sidebar__backdrop');

    if ($toggle == null || $sidebarClose == null) {
      return;
    }

    $toggle.addEventListener('click', e => {
      $navbar.classList.add('navbar-sidebar--show');
    });

    [$backdrop, $sidebarClose].forEach($el =>
      $el.addEventListener('click', e => {
        $navbar.classList.remove('navbar-sidebar--show');
      }),
    );
  });
}
