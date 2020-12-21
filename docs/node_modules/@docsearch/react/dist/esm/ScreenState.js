function _extends() { _extends = Object.assign || function (target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i]; for (var key in source) { if (Object.prototype.hasOwnProperty.call(source, key)) { target[key] = source[key]; } } } return target; }; return _extends.apply(this, arguments); }

import React from 'react';
import { ErrorScreen } from './ErrorScreen';
import { NoResultsScreen } from './NoResultsScreen';
import { ResultsScreen } from './ResultsScreen';
import { StartScreen } from './StartScreen';
export var ScreenState = React.memo(function (props) {
  if (props.state.status === 'error') {
    return /*#__PURE__*/React.createElement(ErrorScreen, null);
  }

  var hasCollections = props.state.collections.some(function (collection) {
    return collection.items.length > 0;
  });

  if (!props.state.query) {
    return /*#__PURE__*/React.createElement(StartScreen, _extends({}, props, {
      hasCollections: hasCollections
    }));
  }

  if (hasCollections === false) {
    return /*#__PURE__*/React.createElement(NoResultsScreen, props);
  }

  return /*#__PURE__*/React.createElement(ResultsScreen, props);
}, function areEqual(_prevProps, nextProps) {
  // We don't update the screen when Autocomplete is loading or stalled to
  // avoid UI flashes:
  //  - Empty screen → Results screen
  //  - NoResults screen → NoResults screen with another query
  return nextProps.state.status === 'loading' || nextProps.state.status === 'stalled';
});