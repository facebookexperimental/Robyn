# Postcss combine duplicated selectors

<!-- current project status -->

[![npm](https://img.shields.io/npm/v/postcss-combine-duplicated-selectors.svg)](https://www.npmjs.com/package/postcss-combine-duplicated-selectors)
[![build status](https://github.com/ChristianMurphy/postcss-combine-duplicated-selectors/workflows/CI/badge.svg)](https://github.com/ChristianMurphy/postcss-combine-duplicated-selectors/actions)
[![dependency status](https://david-dm.org/ChristianMurphy/postcss-combine-duplicated-selectors.svg)](https://david-dm.org/ChristianMurphy/postcss-combine-duplicated-selectors)
[![devDependency status](https://david-dm.org/ChristianMurphy/postcss-combine-duplicated-selectors/dev-status.svg)](https://david-dm.org/ChristianMurphy/postcss-combine-duplicated-selectors?type=dev)

<!-- standards and technologies used in project -->

[![Built with PostCSS](https://img.shields.io/badge/built_with-postcss-blue.svg)](http://postcss.org/)
[![Semver](https://img.shields.io/badge/SemVer-2.0.0-green.svg)](http://semver.org/spec/v2.0.0.html)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-green.svg)](https://conventionalcommits.org)
[![Google code style](https://img.shields.io/badge/code_style-Google-brightgreen.svg?style=flat)](https://google.github.io/styleguide/jsguide.html)

Automatically detects and combines duplicated css selectors so you don't have to
:smile:

## Usage

### Requirements

In order to use this you will need to have [postcss](https://github.com/postcss/postcss) installed. Depending on whether or not you want to use the CLI you need to install [postcss-cli](https://github.com/postcss/postcss-cli).

### Using PostCSS JS API

```js
'use strict';

const fs = require('fs');
const postcss = require('postcss');
const css = fs.readFileSync('src/app.css');

postcss([require('postcss-combine-duplicated-selectors')])
    .process(css, {from: 'src/app.css', to: 'app.css'})
    .then((result) => {
      fs.writeFileSync('app.css', result.css);
      if (result.map) fs.writeFileSync('app.css.map', result.map);
    });
```

### Using PostCSS CLI

```sh
postcss style.css --use postcss-combine-duplicated-selectors --output newcss.css
```

## Example

Input

```css
.module {
  color: green;
}
.another-module {
  color: blue;
}
.module {
  background: red;
}
.another-module {
  background: yellow;
}
```

Output

```css
.module {
  color: green;
  background: red;
}
.another-module {
  color: blue;
  background: yellow;
}
```

### Duplicated Properties

Duplicated properties can optionally be combined.

Set the `removeDuplicatedProperties` option to `true` to enable.

```js
const postcss = require('postcss');
const combineSelectors = require('postcss-combine-duplicated-selectors');

postcss([combineSelectors({removeDuplicatedProperties: true})]);
```

When enabled the following css

```css
.a {
  height: 10px;
  background: orange;
  background: rgba(255, 165, 0, 0.5);
}
```

will combine into

```css
.a {
  height: 10px;
  background: rgba(255, 165, 0, 0.5);
}
```

In order to limit this to only combining properties when the values are equal, set the `removeDuplicatedValues` option to `true` instead. This could clean up duplicated properties, but allow for conscious duplicates such as fallbacks for custom properties.

```js
const postcss = require('postcss');
const combineSelectors = require('postcss-combine-duplicated-selectors');

postcss([combineSelectors({removeDuplicatedValues: true})]);
```

This will transform the following css

```css
.a {
  height: 10px;
}

.a {
  width: 20px;
  background: var(--custom-color);
  background: rgba(255, 165, 0, 0.5);
}
```

into

```css
.a {
  height: 10px;
  width: 20px;
  background: var(--custom-color);
  background: rgba(255, 165, 0, 0.5);
}
```

### Media Queries

If you have code with media queries, pass code through [_postcss-combine-media-query_](https://github.com/SassNinja/postcss-combine-media-query) or [_css-mquery-packer_](https://github.com/n19htz/css-mquery-packer) before _postcss-combine-duplicated-selectors_ to ensure optimal results.
