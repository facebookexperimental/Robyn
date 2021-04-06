var _Symbol = require("@babel/runtime-corejs3/core-js/symbol");

var _isIterable = require("@babel/runtime-corejs3/core-js/is-iterable");

var _getIterator = require("@babel/runtime-corejs3/core-js/get-iterator");

function _iterableToArrayLimitLoose(arr, i) {
  if (typeof _Symbol === "undefined" || !_isIterable(Object(arr))) return;
  var _arr = [];

  for (var _iterator = _getIterator(arr), _step; !(_step = _iterator.next()).done;) {
    _arr.push(_step.value);

    if (i && _arr.length === i) break;
  }

  return _arr;
}

module.exports = _iterableToArrayLimitLoose;
module.exports["default"] = module.exports, module.exports.__esModule = true;