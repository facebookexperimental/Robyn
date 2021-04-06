import _Symbol from "@babel/runtime-corejs3/core-js/symbol";
import _isIterable from "@babel/runtime-corejs3/core-js/is-iterable";
import _getIterator from "@babel/runtime-corejs3/core-js/get-iterator";
export default function _iterableToArrayLimitLoose(arr, i) {
  if (typeof _Symbol === "undefined" || !_isIterable(Object(arr))) return;
  var _arr = [];

  for (var _iterator = _getIterator(arr), _step; !(_step = _iterator.next()).done;) {
    _arr.push(_step.value);

    if (i && _arr.length === i) break;
  }

  return _arr;
}