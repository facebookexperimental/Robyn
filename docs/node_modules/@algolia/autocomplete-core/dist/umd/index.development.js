/*! @algolia/autocomplete-core 1.0.0-alpha.44 | MIT License | © Algolia, Inc. and contributors | https://github.com/algolia/autocomplete */
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global['@algolia/autocomplete-core'] = {}));
}(this, (function (exports) { 'use strict';

  function _typeof(obj) {
    "@babel/helpers - typeof";

    if (typeof Symbol === "function" && typeof Symbol.iterator === "symbol") {
      _typeof = function (obj) {
        return typeof obj;
      };
    } else {
      _typeof = function (obj) {
        return obj && typeof Symbol === "function" && obj.constructor === Symbol && obj !== Symbol.prototype ? "symbol" : typeof obj;
      };
    }

    return _typeof(obj);
  }

  function _defineProperty(obj, key, value) {
    if (key in obj) {
      Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
      });
    } else {
      obj[key] = value;
    }

    return obj;
  }

  function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);

    if (Object.getOwnPropertySymbols) {
      var symbols = Object.getOwnPropertySymbols(object);
      if (enumerableOnly) symbols = symbols.filter(function (sym) {
        return Object.getOwnPropertyDescriptor(object, sym).enumerable;
      });
      keys.push.apply(keys, symbols);
    }

    return keys;
  }

  function _objectSpread2(target) {
    for (var i = 1; i < arguments.length; i++) {
      var source = arguments[i] != null ? arguments[i] : {};

      if (i % 2) {
        ownKeys(Object(source), true).forEach(function (key) {
          _defineProperty(target, key, source[key]);
        });
      } else if (Object.getOwnPropertyDescriptors) {
        Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
      } else {
        ownKeys(Object(source)).forEach(function (key) {
          Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
      }
    }

    return target;
  }

  function _objectWithoutPropertiesLoose(source, excluded) {
    if (source == null) return {};
    var target = {};
    var sourceKeys = Object.keys(source);
    var key, i;

    for (i = 0; i < sourceKeys.length; i++) {
      key = sourceKeys[i];
      if (excluded.indexOf(key) >= 0) continue;
      target[key] = source[key];
    }

    return target;
  }

  function _objectWithoutProperties(source, excluded) {
    if (source == null) return {};

    var target = _objectWithoutPropertiesLoose(source, excluded);

    var key, i;

    if (Object.getOwnPropertySymbols) {
      var sourceSymbolKeys = Object.getOwnPropertySymbols(source);

      for (i = 0; i < sourceSymbolKeys.length; i++) {
        key = sourceSymbolKeys[i];
        if (excluded.indexOf(key) >= 0) continue;
        if (!Object.prototype.propertyIsEnumerable.call(source, key)) continue;
        target[key] = source[key];
      }
    }

    return target;
  }

  function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
  }

  function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
  }

  function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
  }

  function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
  }

  function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;

    for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i];

    return arr2;
  }

  function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }

  function getItemsCount(state) {
    if (state.collections.length === 0) {
      return 0;
    }

    return state.collections.reduce(function (sum, collection) {
      return sum + collection.items.length;
    }, 0);
  }

  /**
   * Throws an error if the condition is not met in development mode.
   * This is used to make development a better experience to provide guidance as
   * to where the error comes from.
   */
  function invariant(condition, message) {

    if (!condition) {
      throw new Error("[Autocomplete] ".concat(message));
    }
  }

  var warnCache = {
    current: {}
  };
  /**
   * Logs a warning if the condition is not met.
   * This is used to log issues in development environment only.
   */

  function warn(condition, message) {

    if (condition) {
      return;
    }

    var sanitizedMessage = message.trim();
    var hasAlreadyPrinted = warnCache.current[sanitizedMessage];

    if (!hasAlreadyPrinted) {
      warnCache.current[sanitizedMessage] = true; // eslint-disable-next-line no-console

      console.warn("[Autocomplete] ".concat(sanitizedMessage));
    }
  }

  function checkOptions(options) {
    "development" !== 'production' ? warn(!options.debug, 'The `debug` option is meant for development debugging and should not be used in production.') : void 0;
  }

  function createStore(reducer, props, onStoreStateChange) {
    var state = props.initialState;
    return {
      getState: function getState() {
        return state;
      },
      dispatch: function dispatch(action, payload) {
        var prevState = _objectSpread2({}, state);

        state = reducer(state, {
          type: action,
          props: props,
          payload: payload
        });
        onStoreStateChange({
          state: state,
          prevState: prevState
        });
      }
    };
  }

  function flatten(values) {
    return values.reduce(function (a, b) {
      return a.concat(b);
    }, []);
  }

  var autocompleteId = 0;
  function generateAutocompleteId() {
    return "autocomplete-".concat(autocompleteId++);
  }

  /**
   * Returns the next active item ID from the current state.
   *
   * We allow circular keyboard navigation from the base index.
   * The base index can either be `null` (nothing is highlighted) or `0`
   * (the first item is highlighted).
   * The base index is allowed to get assigned `null` only if
   * `props.defaultActiveItemId` is `null`. This pattern allows to "stop"
   * by the actual query before navigating to other suggestions as seen on
   * Google or Amazon.
   *
   * @param moveAmount The offset to increment (or decrement) the last index
   * @param baseIndex The current index to compute the next index from
   * @param itemCount The number of items
   * @param defaultActiveItemId The default active index to fallback to
   */
  function getNextActiveItemId(moveAmount, baseIndex, itemCount, defaultActiveItemId) {
    if (moveAmount < 0 && (baseIndex === null || defaultActiveItemId !== null && baseIndex === 0)) {
      return itemCount + moveAmount;
    }

    var numericIndex = (baseIndex === null ? -1 : baseIndex) + moveAmount;

    if (numericIndex <= -1 || numericIndex >= itemCount) {
      return defaultActiveItemId === null ? null : 0;
    }

    return numericIndex;
  }

  var noop = function noop() {};

  function getNormalizedSources(getSources, params) {
    return Promise.resolve(getSources(params)).then(function (sources) {
      invariant(Array.isArray(sources), "The `getSources` function must return an array of sources but returned type ".concat(JSON.stringify(_typeof(sources)), ":\n\n").concat(JSON.stringify(sources, null, 2)));
      return Promise.all(sources // We allow `undefined` and `false` sources to allow users to use
      // `Boolean(query) && source` (=> `false`).
      // We need to remove these values at this point.
      .filter(function (maybeSource) {
        return Boolean(maybeSource);
      }).map(function (source) {
        invariant(typeof source.sourceId === 'string', 'A source must provide a `sourceId` string.');

        var normalizedSource = _objectSpread2({
          getItemInputValue: function getItemInputValue(_ref) {
            var state = _ref.state;
            return state.query;
          },
          getItemUrl: function getItemUrl() {
            return undefined;
          },
          onSelect: function onSelect(_ref2) {
            var setIsOpen = _ref2.setIsOpen;
            setIsOpen(false);
          },
          onActive: noop
        }, source);

        return Promise.resolve(normalizedSource);
      }));
    });
  }

  // We don't have access to the autocomplete source when we call `onKeyDown`
  // or `onClick` because those are native browser events.
  // However, we can get the source from the suggestion index.
  function getCollectionFromActiveItemId(state) {
    // Given 3 sources with respectively 1, 2 and 3 suggestions: [1, 2, 3]
    // We want to get the accumulated counts:
    // [1, 1 + 2, 1 + 2 + 3] = [1, 3, 3 + 3] = [1, 3, 6]
    var accumulatedCollectionsCount = state.collections.map(function (collections) {
      return collections.items.length;
    }).reduce(function (acc, collectionsCount, index) {
      var previousValue = acc[index - 1] || 0;
      var nextValue = previousValue + collectionsCount;
      acc.push(nextValue);
      return acc;
    }, []); // Based on the accumulated counts, we can infer the index of the suggestion.

    var collectionIndex = accumulatedCollectionsCount.reduce(function (acc, current) {
      if (current <= state.activeItemId) {
        return acc + 1;
      }

      return acc;
    }, 0);
    return state.collections[collectionIndex];
  }
  /**
   * Gets the highlighted index relative to a suggestion object (not the absolute
   * highlighted index).
   *
   * Example:
   *  [['a', 'b'], ['c', 'd', 'e'], ['f']]
   *                      ↑
   *         (absolute: 3, relative: 1)
   */


  function getRelativeActiveItemId(_ref) {
    var state = _ref.state,
        collection = _ref.collection;
    var isOffsetFound = false;
    var counter = 0;
    var previousItemsOffset = 0;

    while (isOffsetFound === false) {
      var currentCollection = state.collections[counter];

      if (currentCollection === collection) {
        isOffsetFound = true;
        break;
      }

      previousItemsOffset += currentCollection.items.length;
      counter++;
    }

    return state.activeItemId - previousItemsOffset;
  }

  function getActiveItem(state) {
    var collection = getCollectionFromActiveItemId(state);

    if (!collection) {
      return null;
    }

    var item = collection.items[getRelativeActiveItemId({
      state: state,
      collection: collection
    })];
    var source = collection.source;
    var itemInputValue = source.getItemInputValue({
      item: item,
      state: state
    });
    var itemUrl = source.getItemUrl({
      item: item,
      state: state
    });
    return {
      item: item,
      itemInputValue: itemInputValue,
      itemUrl: itemUrl,
      source: source
    };
  }

  function isOrContainsNode(parent, child) {
    return parent === child || parent.contains(child);
  }

  function getAutocompleteSetters(_ref) {
    var store = _ref.store;

    var setActiveItemId = function setActiveItemId(value) {
      store.dispatch('setActiveItemId', value);
    };

    var setQuery = function setQuery(value) {
      store.dispatch('setQuery', value);
    };

    var setCollections = function setCollections(rawValue) {
      var baseItemId = 0;
      var value = rawValue.map(function (collection) {
        return _objectSpread2(_objectSpread2({}, collection), {}, {
          // We flatten the stored items to support calling `getAlgoliaHits`
          // from the source itself.
          items: flatten(collection.items).map(function (item) {
            return _objectSpread2(_objectSpread2({}, item), {}, {
              __autocomplete_id: baseItemId++
            });
          })
        });
      });
      store.dispatch('setCollections', value);
    };

    var setIsOpen = function setIsOpen(value) {
      store.dispatch('setIsOpen', value);
    };

    var setStatus = function setStatus(value) {
      store.dispatch('setStatus', value);
    };

    var setContext = function setContext(value) {
      store.dispatch('setContext', value);
    };

    return {
      setActiveItemId: setActiveItemId,
      setQuery: setQuery,
      setCollections: setCollections,
      setIsOpen: setIsOpen,
      setStatus: setStatus,
      setContext: setContext
    };
  }

  function getDefaultProps(props, pluginSubscribers) {
    var _props$id;

    var environment = typeof window !== 'undefined' ? window : {};
    var plugins = props.plugins || [];
    return _objectSpread2(_objectSpread2({
      debug: false,
      openOnFocus: false,
      placeholder: '',
      autoFocus: false,
      defaultActiveItemId: null,
      stallThreshold: 300,
      environment: environment,
      shouldPanelOpen: function shouldPanelOpen(_ref) {
        var state = _ref.state;
        return getItemsCount(state) > 0;
      }
    }, props), {}, {
      // Since `generateAutocompleteId` triggers a side effect (it increments
      // and internal counter), we don't want to execute it if unnecessary.
      id: (_props$id = props.id) !== null && _props$id !== void 0 ? _props$id : generateAutocompleteId(),
      plugins: plugins,
      // The following props need to be deeply defaulted.
      initialState: _objectSpread2({
        activeItemId: null,
        query: '',
        completion: null,
        collections: [],
        isOpen: false,
        status: 'idle',
        context: {}
      }, props.initialState),
      onStateChange: function onStateChange(params) {
        var _props$onStateChange;

        (_props$onStateChange = props.onStateChange) === null || _props$onStateChange === void 0 ? void 0 : _props$onStateChange.call(props, params);
        plugins.forEach(function (x) {
          var _x$onStateChange;

          return (_x$onStateChange = x.onStateChange) === null || _x$onStateChange === void 0 ? void 0 : _x$onStateChange.call(x, params);
        });
      },
      onSubmit: function onSubmit(params) {
        var _props$onSubmit;

        (_props$onSubmit = props.onSubmit) === null || _props$onSubmit === void 0 ? void 0 : _props$onSubmit.call(props, params);
        plugins.forEach(function (x) {
          var _x$onSubmit;

          return (_x$onSubmit = x.onSubmit) === null || _x$onSubmit === void 0 ? void 0 : _x$onSubmit.call(x, params);
        });
      },
      onReset: function onReset(params) {
        var _props$onReset;

        (_props$onReset = props.onReset) === null || _props$onReset === void 0 ? void 0 : _props$onReset.call(props, params);
        plugins.forEach(function (x) {
          var _x$onReset;

          return (_x$onReset = x.onReset) === null || _x$onReset === void 0 ? void 0 : _x$onReset.call(x, params);
        });
      },
      getSources: function getSources(params) {
        return Promise.all([].concat(_toConsumableArray(plugins.map(function (plugin) {
          return plugin.getSources;
        })), [props.getSources]).filter(Boolean).map(function (getSources) {
          return getNormalizedSources(getSources, params);
        })).then(function (nested) {
          return flatten(nested);
        }).then(function (sources) {
          return sources.map(function (source) {
            return _objectSpread2(_objectSpread2({}, source), {}, {
              onSelect: function onSelect(params) {
                source.onSelect(params);
                pluginSubscribers.forEach(function (x) {
                  var _x$onSelect;

                  return (_x$onSelect = x.onSelect) === null || _x$onSelect === void 0 ? void 0 : _x$onSelect.call(x, params);
                });
              },
              onActive: function onActive(params) {
                source.onActive(params);
                pluginSubscribers.forEach(function (x) {
                  var _x$onActive;

                  return (_x$onActive = x.onActive) === null || _x$onActive === void 0 ? void 0 : _x$onActive.call(x, params);
                });
              }
            });
          });
        });
      },
      navigator: _objectSpread2({
        navigate: function navigate(_ref2) {
          var itemUrl = _ref2.itemUrl;
          environment.location.assign(itemUrl);
        },
        navigateNewTab: function navigateNewTab(_ref3) {
          var itemUrl = _ref3.itemUrl;
          var windowReference = environment.open(itemUrl, '_blank', 'noopener');
          windowReference === null || windowReference === void 0 ? void 0 : windowReference.focus();
        },
        navigateNewWindow: function navigateNewWindow(_ref4) {
          var itemUrl = _ref4.itemUrl;
          environment.open(itemUrl, '_blank', 'noopener');
        }
      }, props.navigator)
    });
  }

  var lastStalledId = null;
  function onInput(_ref) {
    var event = _ref.event,
        _ref$nextState = _ref.nextState,
        nextState = _ref$nextState === void 0 ? {} : _ref$nextState,
        props = _ref.props,
        query = _ref.query,
        refresh = _ref.refresh,
        store = _ref.store,
        setters = _objectWithoutProperties(_ref, ["event", "nextState", "props", "query", "refresh", "store"]);

    if (lastStalledId) {
      props.environment.clearTimeout(lastStalledId);
    }

    var setCollections = setters.setCollections,
        setIsOpen = setters.setIsOpen,
        setQuery = setters.setQuery,
        setActiveItemId = setters.setActiveItemId,
        setStatus = setters.setStatus;
    setQuery(query);
    setActiveItemId(props.defaultActiveItemId);

    if (!query && props.openOnFocus === false) {
      var _nextState$isOpen;

      setStatus('idle');
      setCollections(store.getState().collections.map(function (collection) {
        return _objectSpread2(_objectSpread2({}, collection), {}, {
          items: []
        });
      }));
      setIsOpen((_nextState$isOpen = nextState.isOpen) !== null && _nextState$isOpen !== void 0 ? _nextState$isOpen : props.shouldPanelOpen({
        state: store.getState()
      }));
      return Promise.resolve();
    }

    setStatus('loading');
    lastStalledId = props.environment.setTimeout(function () {
      setStatus('stalled');
    }, props.stallThreshold);
    return props.getSources(_objectSpread2({
      query: query,
      refresh: refresh,
      state: store.getState()
    }, setters)).then(function (sources) {
      setStatus('loading'); // @TODO: convert `Promise.all` to fetching strategy.

      return Promise.all(sources.map(function (source) {
        return Promise.resolve(source.getItems(_objectSpread2({
          query: query,
          refresh: refresh,
          state: store.getState()
        }, setters))).then(function (items) {
          invariant(Array.isArray(items), "The `getItems` function must return an array of items but returned type ".concat(JSON.stringify(_typeof(items)), ":\n\n").concat(JSON.stringify(items, null, 2)));
          return {
            source: source,
            items: items
          };
        });
      })).then(function (collections) {
        var _nextState$isOpen2;

        setStatus('idle');
        setCollections(collections);
        var isPanelOpen = props.shouldPanelOpen({
          state: store.getState()
        });
        setIsOpen((_nextState$isOpen2 = nextState.isOpen) !== null && _nextState$isOpen2 !== void 0 ? _nextState$isOpen2 : props.openOnFocus && !query && isPanelOpen || isPanelOpen);
        var highlightedItem = getActiveItem(store.getState());

        if (store.getState().activeItemId !== null && highlightedItem) {
          var item = highlightedItem.item,
              itemInputValue = highlightedItem.itemInputValue,
              itemUrl = highlightedItem.itemUrl,
              source = highlightedItem.source;
          source.onActive(_objectSpread2({
            event: event,
            item: item,
            itemInputValue: itemInputValue,
            itemUrl: itemUrl,
            refresh: refresh,
            source: source,
            state: store.getState()
          }, setters));
        }
      }).finally(function () {
        if (lastStalledId) {
          props.environment.clearTimeout(lastStalledId);
        }
      });
    });
  }

  function onKeyDown(_ref) {
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
        source.onActive(_objectSpread2({
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
          _source.onSelect(_objectSpread2({
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
          _source.onSelect(_objectSpread2({
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
      } else if (event.altKey) ; else {
        if (_itemUrl !== undefined) {
          _source.onSelect(_objectSpread2({
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

        onInput(_objectSpread2({
          event: event,
          nextState: {
            isOpen: false
          },
          props: props,
          query: _itemInputValue,
          refresh: refresh,
          store: store
        }, setters)).then(function () {
          _source.onSelect(_objectSpread2({
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

  function getPropGetters(_ref) {
    var props = _ref.props,
        refresh = _ref.refresh,
        store = _ref.store,
        setters = _objectWithoutProperties(_ref, ["props", "refresh", "store"]);

    var getEnvironmentProps = function getEnvironmentProps(providedProps) {
      var inputElement = providedProps.inputElement,
          formElement = providedProps.formElement,
          panelElement = providedProps.panelElement,
          rest = _objectWithoutProperties(providedProps, ["inputElement", "formElement", "panelElement"]);

      return _objectSpread2({
        // On touch devices, we do not rely on the native `blur` event of the
        // input to close the panel, but rather on a custom `touchstart` event
        // outside of the autocomplete elements.
        // This ensures a working experience on mobile because we blur the input
        // on touch devices when the user starts scrolling (`touchmove`).
        onTouchStart: function onTouchStart(event) {
          if (store.getState().isOpen === false || event.target === inputElement) {
            return;
          } // @TODO: support cases where there are multiple Autocomplete instances.
          // Right now, a second instance makes this computation return false.


          var isTargetWithinAutocomplete = [formElement, panelElement].some(function (contextNode) {
            return isOrContainsNode(contextNode, event.target) || isOrContainsNode(contextNode, props.environment.document.activeElement);
          });

          if (isTargetWithinAutocomplete === false) {
            store.dispatch('blur', null);
          }
        },
        // When scrolling on touch devices (mobiles, tablets, etc.), we want to
        // mimic the native platform behavior where the input is blurred to
        // hide the virtual keyboard. This gives more vertical space to
        // discover all the suggestions showing up in the panel.
        onTouchMove: function onTouchMove(event) {
          if (store.getState().isOpen === false || inputElement !== props.environment.document.activeElement || event.target === inputElement) {
            return;
          }

          inputElement.blur();
        }
      }, rest);
    };

    var getRootProps = function getRootProps(rest) {
      return _objectSpread2({
        role: 'combobox',
        'aria-expanded': store.getState().isOpen,
        'aria-haspopup': 'listbox',
        'aria-owns': store.getState().isOpen ? "".concat(props.id, "-list") : undefined,
        'aria-labelledby': "".concat(props.id, "-label")
      }, rest);
    };

    var getFormProps = function getFormProps(providedProps) {
      var inputElement = providedProps.inputElement,
          rest = _objectWithoutProperties(providedProps, ["inputElement"]);

      return _objectSpread2({
        action: '',
        noValidate: true,
        role: 'search',
        onSubmit: function onSubmit(event) {
          var _providedProps$inputE;

          event.preventDefault();
          props.onSubmit(_objectSpread2({
            event: event,
            refresh: refresh,
            state: store.getState()
          }, setters));
          store.dispatch('submit', null);
          (_providedProps$inputE = providedProps.inputElement) === null || _providedProps$inputE === void 0 ? void 0 : _providedProps$inputE.blur();
        },
        onReset: function onReset(event) {
          var _providedProps$inputE2;

          event.preventDefault();
          props.onReset(_objectSpread2({
            event: event,
            refresh: refresh,
            state: store.getState()
          }, setters));
          store.dispatch('reset', null);
          (_providedProps$inputE2 = providedProps.inputElement) === null || _providedProps$inputE2 === void 0 ? void 0 : _providedProps$inputE2.focus();
        }
      }, rest);
    };

    var getInputProps = function getInputProps(providedProps) {
      function onFocus(event) {
        // We want to trigger a query when `openOnFocus` is true
        // because the panel should open with the current query.
        if (props.openOnFocus || Boolean(store.getState().query)) {
          onInput(_objectSpread2({
            event: event,
            props: props,
            query: store.getState().completion || store.getState().query,
            refresh: refresh,
            store: store
          }, setters));
        }

        store.dispatch('focus', null);
      }

      var isTouchDevice = ('ontouchstart' in props.environment);

      var _ref2 = providedProps || {},
          inputElement = _ref2.inputElement,
          _ref2$maxLength = _ref2.maxLength,
          maxLength = _ref2$maxLength === void 0 ? 512 : _ref2$maxLength,
          rest = _objectWithoutProperties(_ref2, ["inputElement", "maxLength"]);

      var activeItem = getActiveItem(store.getState());
      return _objectSpread2({
        'aria-autocomplete': 'both',
        'aria-activedescendant': store.getState().isOpen && store.getState().activeItemId !== null ? "".concat(props.id, "-item-").concat(store.getState().activeItemId) : undefined,
        'aria-controls': store.getState().isOpen ? "".concat(props.id, "-list") : undefined,
        'aria-labelledby': "".concat(props.id, "-label"),
        value: store.getState().completion || store.getState().query,
        id: "".concat(props.id, "-input"),
        autoComplete: 'off',
        autoCorrect: 'off',
        autoCapitalize: 'off',
        enterKeyHint: activeItem !== null && activeItem !== void 0 && activeItem.itemUrl ? 'go' : 'search',
        spellCheck: 'false',
        autoFocus: props.autoFocus,
        placeholder: props.placeholder,
        maxLength: maxLength,
        type: 'search',
        onChange: function onChange(event) {
          onInput(_objectSpread2({
            event: event,
            props: props,
            query: event.currentTarget.value.slice(0, maxLength),
            refresh: refresh,
            store: store
          }, setters));
        },
        onKeyDown: function onKeyDown$1(event) {
          onKeyDown(_objectSpread2({
            event: event,
            props: props,
            refresh: refresh,
            store: store
          }, setters));
        },
        onFocus: onFocus,
        onBlur: function onBlur() {
          // We do rely on the `blur` event on touch devices.
          // See explanation in `onTouchStart`.
          if (!isTouchDevice) {
            store.dispatch('blur', null);
          }
        },
        onClick: function onClick(event) {
          // When the panel is closed and you click on the input while
          // the input is focused, the `onFocus` event is not triggered
          // (default browser behavior).
          // In an autocomplete context, it makes sense to open the panel in this
          // case.
          // We mimic this event by catching the `onClick` event which
          // triggers the `onFocus` for the panel to open.
          if (providedProps.inputElement === props.environment.document.activeElement && !store.getState().isOpen) {
            onFocus(event);
          }
        }
      }, rest);
    };

    var getLabelProps = function getLabelProps(rest) {
      return _objectSpread2({
        htmlFor: "".concat(props.id, "-input"),
        id: "".concat(props.id, "-label")
      }, rest);
    };

    var getListProps = function getListProps(rest) {
      return _objectSpread2({
        role: 'listbox',
        'aria-labelledby': "".concat(props.id, "-label"),
        id: "".concat(props.id, "-list")
      }, rest);
    };

    var getPanelProps = function getPanelProps(rest) {
      return _objectSpread2({
        onMouseDown: function onMouseDown(event) {
          // Prevents the `activeElement` from being changed to the panel so
          // that the blur event is not triggered, otherwise it closes the
          // panel.
          event.preventDefault();
        },
        onMouseLeave: function onMouseLeave() {
          store.dispatch('mouseleave', null);
        }
      }, rest);
    };

    var getItemProps = function getItemProps(providedProps) {
      var item = providedProps.item,
          source = providedProps.source,
          rest = _objectWithoutProperties(providedProps, ["item", "source"]);

      return _objectSpread2({
        id: "".concat(props.id, "-item-").concat(item.__autocomplete_id),
        role: 'option',
        'aria-selected': store.getState().activeItemId === item.__autocomplete_id,
        onMouseMove: function onMouseMove(event) {
          if (item.__autocomplete_id === store.getState().activeItemId) {
            return;
          }

          store.dispatch('mousemove', item.__autocomplete_id);
          var activeItem = getActiveItem(store.getState());

          if (store.getState().activeItemId !== null && activeItem) {
            var _item = activeItem.item,
                itemInputValue = activeItem.itemInputValue,
                itemUrl = activeItem.itemUrl,
                _source = activeItem.source;

            _source.onActive(_objectSpread2({
              event: event,
              item: _item,
              itemInputValue: itemInputValue,
              itemUrl: itemUrl,
              refresh: refresh,
              source: _source,
              state: store.getState()
            }, setters));
          }
        },
        onMouseDown: function onMouseDown(event) {
          // Prevents the `activeElement` from being changed to the item so it
          // can remain with the current `activeElement`.
          event.preventDefault();
        },
        onClick: function onClick(event) {
          var itemInputValue = source.getItemInputValue({
            item: item,
            state: store.getState()
          });
          var itemUrl = source.getItemUrl({
            item: item,
            state: store.getState()
          }); // If `getItemUrl` is provided, it means that the suggestion
          // is a link, not plain text that aims at updating the query.
          // We can therefore skip the state change because it will update
          // the `activeItemId`, resulting in a UI flash, especially
          // noticeable on mobile.

          var runPreCommand = itemUrl ? Promise.resolve() : onInput(_objectSpread2({
            event: event,
            nextState: {
              isOpen: false
            },
            props: props,
            query: itemInputValue,
            refresh: refresh,
            store: store
          }, setters));
          runPreCommand.then(function () {
            source.onSelect(_objectSpread2({
              event: event,
              item: item,
              itemInputValue: itemInputValue,
              itemUrl: itemUrl,
              refresh: refresh,
              source: source,
              state: store.getState()
            }, setters));
          });
        }
      }, rest);
    };

    return {
      getEnvironmentProps: getEnvironmentProps,
      getRootProps: getRootProps,
      getFormProps: getFormProps,
      getLabelProps: getLabelProps,
      getInputProps: getInputProps,
      getPanelProps: getPanelProps,
      getListProps: getListProps,
      getItemProps: getItemProps
    };
  }

  function getCompletion(_ref) {
    var state = _ref.state;

    if (state.isOpen === false || state.activeItemId === null) {
      return null;
    }

    var _ref2 = getActiveItem(state),
        itemInputValue = _ref2.itemInputValue;

    return itemInputValue || null;
  }

  var stateReducer = function stateReducer(state, action) {
    switch (action.type) {
      case 'setActiveItemId':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: action.payload
          });
        }

      case 'setQuery':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            query: action.payload,
            completion: null
          });
        }

      case 'setCollections':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            collections: action.payload
          });
        }

      case 'setIsOpen':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            isOpen: action.payload
          });
        }

      case 'setStatus':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            status: action.payload
          });
        }

      case 'setContext':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            context: _objectSpread2(_objectSpread2({}, state.context), action.payload)
          });
        }

      case 'ArrowDown':
        {
          var nextState = _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: getNextActiveItemId(1, state.activeItemId, getItemsCount(state), action.props.defaultActiveItemId)
          });

          return _objectSpread2(_objectSpread2({}, nextState), {}, {
            completion: getCompletion({
              state: nextState
            })
          });
        }

      case 'ArrowUp':
        {
          var _nextState = _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: getNextActiveItemId(-1, state.activeItemId, getItemsCount(state), action.props.defaultActiveItemId)
          });

          return _objectSpread2(_objectSpread2({}, _nextState), {}, {
            completion: getCompletion({
              state: _nextState
            })
          });
        }

      case 'Escape':
        {
          if (state.isOpen) {
            return _objectSpread2(_objectSpread2({}, state), {}, {
              isOpen: false,
              completion: null
            });
          }

          return _objectSpread2(_objectSpread2({}, state), {}, {
            query: '',
            status: 'idle',
            collections: []
          });
        }

      case 'submit':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: null,
            isOpen: false,
            status: 'idle'
          });
        }

      case 'reset':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: // Since we open the panel on reset when openOnFocus=true
            // we need to restore the highlighted index to the defaultActiveItemId. (DocSearch use-case)
            // Since we close the panel when openOnFocus=false
            // we lose track of the highlighted index. (Query-suggestions use-case)
            action.props.openOnFocus === true ? action.props.defaultActiveItemId : null,
            status: 'idle',
            query: ''
          });
        }

      case 'focus':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: action.props.defaultActiveItemId,
            isOpen: (action.props.openOnFocus || Boolean(state.query)) && action.props.shouldPanelOpen({
              state: state
            })
          });
        }

      case 'blur':
        {
          if (action.props.debug) {
            return state;
          }

          return _objectSpread2(_objectSpread2({}, state), {}, {
            isOpen: false,
            activeItemId: null
          });
        }

      case 'mousemove':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: action.payload
          });
        }

      case 'mouseleave':
        {
          return _objectSpread2(_objectSpread2({}, state), {}, {
            activeItemId: action.props.defaultActiveItemId
          });
        }

      default:
        invariant(false, "The reducer action ".concat(JSON.stringify(action.type), " is not supported."));
        return state;
    }
  };

  function createAutocomplete(options) {
    checkOptions(options);
    var subscribers = [];
    var props = getDefaultProps(options, subscribers);
    var store = createStore(stateReducer, props, onStoreStateChange);
    var setters = getAutocompleteSetters({
      store: store
    });
    var propGetters = getPropGetters(_objectSpread2({
      props: props,
      refresh: refresh,
      store: store
    }, setters));

    function onStoreStateChange(_ref) {
      var prevState = _ref.prevState,
          state = _ref.state;
      props.onStateChange(_objectSpread2({
        prevState: prevState,
        state: state,
        refresh: refresh
      }, setters));
    }

    function refresh() {
      return onInput(_objectSpread2({
        event: new Event('input'),
        nextState: {
          isOpen: store.getState().isOpen
        },
        props: props,
        query: store.getState().query,
        refresh: refresh,
        store: store
      }, setters));
    }

    props.plugins.forEach(function (plugin) {
      var _plugin$subscribe;

      return (_plugin$subscribe = plugin.subscribe) === null || _plugin$subscribe === void 0 ? void 0 : _plugin$subscribe.call(plugin, _objectSpread2(_objectSpread2({}, setters), {}, {
        refresh: refresh,
        onSelect: function onSelect(fn) {
          subscribers.push({
            onSelect: fn
          });
        },
        onActive: function onActive(fn) {
          subscribers.push({
            onActive: fn
          });
        }
      }));
    });
    return _objectSpread2(_objectSpread2({
      refresh: refresh
    }, propGetters), setters);
  }

  var version = '1.0.0-alpha.44';

  exports.createAutocomplete = createAutocomplete;
  exports.getDefaultProps = getDefaultProps;
  exports.version = version;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=index.development.js.map
