const postcss = require('postcss');
const parser = require('postcss-selector-parser');
const {name} = require('../package.json');

/**
 * Ensure that attributes with different quotes match.
 * @param {Object} selector - postcss selector node
 */
function normalizeAttributes(selector) {
  selector.walkAttributes((node) => {
    if (node.value) {
      // remove quotes
      node.value = node.value.replace(/'|\\'|"|\\"/g, '');
    }
  });
}

/**
 * Sort class and id groups alphabetically
 * @param {Object} selector - postcss selector node
 */
function sortGroups(selector) {
  selector.each((subSelector) => {
    subSelector.nodes.sort((a, b) => {
      // different types cannot be sorted
      if (a.type !== b.type) {
        return 0;
      }

      // sort alphabetically
      return a.value < b.value ? -1 : 1;
    });
  });

  selector.sort((a, b) => (a.nodes.join('') < b.nodes.join('') ? -1 : 1));
}

/**
 * Remove duplicated properties
 * @param {Object} selector - postcss selector node
 * @param {Boolean} exact
 */
function removeDupProperties(selector, exact) {
  // Remove duplicated properties from bottom to top ()
  for (let actIndex = selector.nodes.length - 1; actIndex >= 1; actIndex--) {
    for (let befIndex = actIndex - 1; befIndex >= 0; befIndex--) {
      if (selector.nodes[actIndex].prop === selector.nodes[befIndex].prop) {
        if (
          !exact ||
          (exact &&
            selector.nodes[actIndex].value === selector.nodes[befIndex].value)
        ) {
          selector.nodes[befIndex].remove();
          actIndex--;
        }
      }
    }
  }
}

const uniformStyle = parser((selector) => {
  normalizeAttributes(selector);
  sortGroups(selector);
});

const defaultOptions = {
  removeDuplicatedProperties: false,
};

module.exports = postcss.plugin(name, (options) => {
  options = Object.assign({}, defaultOptions, options);
  return (css) => {
    // Create a map to store maps
    const mapTable = new Map();
    // root map to store root selectors
    mapTable.set('root', new Map());

    css.walkRules((rule) => {
      let map;
      // Check selector parent for any at rule
      if (rule.parent.type === 'atrule') {
        // Use name and query params as the key
        const query =
          rule.parent.name.toLowerCase() +
          rule.parent.params.replace(/\s+/g, '');

        // See if this query key is already in the map table
        map = mapTable.has(query) ? // If it is use it
          mapTable.get(query) : // if not set it and get it
          mapTable.set(query, new Map()).get(query);
      } else {
        // Otherwise we are dealing with a selector in the root
        map = mapTable.get('root');
      }

      const selector = uniformStyle.processSync(rule.selector, {
        lossless: false,
      });

      if (map.has(selector)) {
        // store original rule as destination
        const destination = map.get(selector);
        // move declarations to original rule
        while (rule.nodes.length > 0) {
          destination.append(rule.nodes[0]);
        }
        // remove duplicated rule
        rule.remove();

        if (options.removeDuplicatedProperties ||
          options.removeDuplicatedValues) {
          removeDupProperties(
              destination,
              options.removeDuplicatedValues,
          );
        }
      } else {
        if (options.removeDuplicatedProperties ||
          options.removeDuplicatedValues) {
          removeDupProperties(
              rule,
              options.removeDuplicatedValues,
          );
        }
        // add new selector to symbol table
        map.set(selector, rule);
      }
    });
  };
});
