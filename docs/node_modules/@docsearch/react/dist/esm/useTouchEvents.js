import React from 'react';
export function useTouchEvents(_ref) {
  var getEnvironmentProps = _ref.getEnvironmentProps,
      panelElement = _ref.panelElement,
      searchBoxElement = _ref.searchBoxElement,
      inputElement = _ref.inputElement;
  React.useEffect(function () {
    if (!(panelElement && searchBoxElement && inputElement)) {
      return undefined;
    }

    var _getEnvironmentProps = getEnvironmentProps({
      panelElement: panelElement,
      searchBoxElement: searchBoxElement,
      inputElement: inputElement
    }),
        onTouchStart = _getEnvironmentProps.onTouchStart,
        onTouchMove = _getEnvironmentProps.onTouchMove;

    window.addEventListener('touchstart', onTouchStart);
    window.addEventListener('touchmove', onTouchMove);
    return function () {
      window.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchmove', onTouchMove);
    };
  }, [getEnvironmentProps, panelElement, searchBoxElement, inputElement]);
}