function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { _defineProperty(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }

function _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }

function _objectWithoutProperties(source, excluded) { if (source == null) return {}; var target = _objectWithoutPropertiesLoose(source, excluded); var key, i; if (Object.getOwnPropertySymbols) { var sourceSymbolKeys = Object.getOwnPropertySymbols(source); for (i = 0; i < sourceSymbolKeys.length; i++) { key = sourceSymbolKeys[i]; if (excluded.indexOf(key) >= 0) continue; if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue; target[key] = source[key]; } } return target; }

function _objectWithoutPropertiesLoose(source, excluded) { if (source == null) return {}; var target = {}; var sourceKeys = Object.keys(source); var key, i; for (i = 0; i < sourceKeys.length; i++) { key = sourceKeys[i]; if (excluded.indexOf(key) >= 0) continue; target[key] = source[key]; } return target; }

import { onInput } from './onInput';
import { getActiveItem } from './utils';
export function onKeyDown(_ref) {
  var event = _ref.event,
      props = _ref.props,
      refresh = _ref.refresh,
      store = _ref.store,
      setters = _objectWithoutProperties(_ref, ["event", "props", "refresh", "store"]);

  if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
    // Default browser behavior changes the caret placement on ArrowUp and
    // Arrow down.
    event.preventDefault();
    store.dispatch(event.key, null);
    var nodeItem = props.environment.document.getElementById("".concat(props.id, "-item-").concat(store.getState().activeItemId));

    if (nodeItem) {
      if (nodeItem.scrollIntoViewIfNeeded) {
        nodeItem.scrollIntoViewIfNeeded(false);
      } else {
        nodeItem.scrollIntoView(false);
      }
    }

    var highlightedItem = getActiveItem(store.getState());

    if (store.getState().activeItemId !== null && highlightedItem) {
      var item = highlightedItem.item,
          itemInputValue = highlightedItem.itemInputValue,
          itemUrl = highlightedItem.itemUrl,
          source = highlightedItem.source;
      source.onActive(_objectSpread({
        event: event,
        item: item,
        itemInputValue: itemInputValue,
        itemUrl: itemUrl,
        refresh: refresh,
        source: source,
        state: store.getState()
      }, setters));
    }
  } else if (event.key === 'Escape') {
    // This prevents the default browser behavior on `input[type="search"]`
    // from removing the query right away because we first want to close the
    // panel.
    event.preventDefault();
    store.dispatch(event.key, null);
  } else if (event.key === 'Enter') {
    // No active item, so we let the browser handle the native `onSubmit` form
    // event.
    if (store.getState().activeItemId === null || store.getState().collections.every(function (collection) {
      return collection.items.length === 0;
    })) {
      return;
    } // This prevents the `onSubmit` event to be sent because an item is
    // highlighted.


    event.preventDefault();

    var _ref2 = getActiveItem(store.getState()),
        _item = _ref2.item,
        _itemInputValue = _ref2.itemInputValue,
        _itemUrl = _ref2.itemUrl,
        _source = _ref2.source;

    if (event.metaKey || event.ctrlKey) {
      if (_itemUrl !== undefined) {
        _source.onSelect(_objectSpread({
          event: event,
          item: _item,
          itemInputValue: _itemInputValue,
          itemUrl: _itemUrl,
          refresh: refresh,
          source: _source,
          state: store.getState()
        }, setters));

        props.navigator.navigateNewTab({
          itemUrl: _itemUrl,
          item: _item,
          state: store.getState()
        });
      }
    } else if (event.shiftKey) {
      if (_itemUrl !== undefined) {
        _source.onSelect(_objectSpread({
          event: event,
          item: _item,
          itemInputValue: _itemInputValue,
          itemUrl: _itemUrl,
          refresh: refresh,
          source: _source,
          state: store.getState()
        }, setters));

        props.navigator.navigateNewWindow({
          itemUrl: _itemUrl,
          item: _item,
          state: store.getState()
        });
      }
    } else if (event.altKey) {// Keep native browser behavior
    } else {
      if (_itemUrl !== undefined) {
        _source.onSelect(_objectSpread({
          event: event,
          item: _item,
          itemInputValue: _itemInputValue,
          itemUrl: _itemUrl,
          refresh: refresh,
          source: _source,
          state: store.getState()
        }, setters));

        props.navigator.navigate({
          itemUrl: _itemUrl,
          item: _item,
          state: store.getState()
        });
        return;
      }

      onInput(_objectSpread({
        event: event,
        nextState: {
          isOpen: false
        },
        props: props,
        query: _itemInputValue,
        refresh: refresh,
        store: store
      }, setters)).then(function () {
        _source.onSelect(_objectSpread({
          event: event,
          item: _item,
          itemInputValue: _itemInputValue,
          itemUrl: _itemUrl,
          refresh: refresh,
          source: _source,
          state: store.getState()
        }, setters));
      });
    }
  }
}