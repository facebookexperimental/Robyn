function menu($elements) {
  $elements.forEach($menu => {
    $menu.addEventListener('click', event => {
      let $listItem = event.target;
      while ($listItem) {
        if ($listItem.classList.contains('menu')) {
          return;
        }

        if ($listItem.classList.contains('menu__list-item')) {
          break;
        }
        $listItem = $listItem.parentNode;
      }

      // Not clicking on a list item.
      if (!$listItem) {
        return;
      }

      event.preventDefault();
      if ($listItem.classList.contains('menu__list-item')) {
        $listItem.classList.toggle('menu__list-item--collapsed');
      }

      // Don't add any active class if non-leaf item selected.
      if ($listItem.querySelector('.menu__list')) {
        return;
      }

      $menu
        .querySelectorAll('.menu__link')
        .forEach($elItem => $elItem.classList.remove('menu__link--active'));

      // Traverse parents and add active class.
      while ($listItem) {
        if ($listItem.classList.contains('menu')) {
          return;
        }

        if ($listItem.classList.contains('menu__list-item')) {
          const $link = $listItem.querySelector('.menu__link');
          if ($link) {
            $link.classList.add('menu__link--active');
          }
        }

        $listItem = $listItem.parentNode;
      }
    });

    $button = $menu.querySelector('.menu__button');
    if ($button) {
      $button.addEventListener('click', e => {
        $menu.classList.toggle('menu--show');
      });
    }
  });
}
