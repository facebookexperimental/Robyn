import { getActiveItem } from './utils';
export function getCompletion(_ref) {
  var state = _ref.state;

  if (state.isOpen === false || state.activeItemId === null) {
    return null;
  }

  var _ref2 = getActiveItem(state),
      itemInputValue = _ref2.itemInputValue;

  return itemInputValue || null;
}