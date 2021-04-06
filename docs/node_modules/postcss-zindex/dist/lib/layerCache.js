'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _has = require('has');

var _has2 = _interopRequireDefault(_has);

var _uniqs = require('uniqs');

var _uniqs2 = _interopRequireDefault(_uniqs);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function LayerCache(opts) {
    this._values = [];
    this._startIndex = opts.startIndex || 1;
}

function ascending(a, b) {
    return a - b;
}

function reduceValues(list, value, index) {
    list[value] = index + this._startIndex;
    return list;
}

LayerCache.prototype._findValue = function (value) {
    if ((0, _has2.default)(this._values, value)) {
        return this._values[value];
    }
    return false;
};

LayerCache.prototype.optimizeValues = function () {
    this._values = (0, _uniqs2.default)(this._values).sort(ascending).reduce(reduceValues.bind(this), {});
};

LayerCache.prototype.addValue = function (value) {
    let parsedValue = parseInt(value, 10);
    // pass only valid values
    if (!parsedValue || parsedValue < 0) {
        return;
    }
    this._values.push(parsedValue);
};

LayerCache.prototype.getValue = function (value) {
    let parsedValue = parseInt(value, 10);
    return this._findValue(parsedValue) || value;
};

exports.default = LayerCache;
module.exports = exports['default'];