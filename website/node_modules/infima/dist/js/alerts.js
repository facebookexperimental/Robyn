function alert($elements) {
  $elements.forEach($alert => {
    $alert.addEventListener('click', (e) => {
      if (e.target && e.target.classList.contains('close')) {
        $alert.remove();
      }
    });
  });
}
