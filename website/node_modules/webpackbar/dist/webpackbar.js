'use strict';

function _interopDefault (ex) { return (ex && (typeof ex === 'object') && 'default' in ex) ? ex['default'] : ex; }

var webpack = require('webpack');
var env = _interopDefault(require('std-env'));
var prettyTime = _interopDefault(require('pretty-time'));
var path = require('path');
var path__default = _interopDefault(path);
var chalk = _interopDefault(require('chalk'));
var Consola = _interopDefault(require('consola'));
var textTable = _interopDefault(require('text-table'));
var figures = require('figures');
var ansiEscapes = _interopDefault(require('ansi-escapes'));
var wrapAnsi = _interopDefault(require('wrap-ansi'));

function first(arr) {
  return arr[0];
}
function last(arr) {
  return arr.length ? arr[arr.length - 1] : null;
}
function startCase(str) {
  return str[0].toUpperCase() + str.substr(1);
}
function firstMatch(regex, str) {
  const m = regex.exec(str);
  return m ? m[0] : null;
}
function hasValue(s) {
  return s && s.length;
}
function removeAfter(delimiter, str) {
  return first(str.split(delimiter)) || '';
}
function removeBefore(delimiter, str) {
  return last(str.split(delimiter)) || '';
}
function range(len) {
  const arr = [];

  for (let i = 0; i < len; i++) {
    arr.push(i);
  }

  return arr;
}
function shortenPath(path$1 = '') {
  const cwd = process.cwd() + path.sep;
  return String(path$1).replace(cwd, '');
}
function objectValues(obj) {
  return Object.keys(obj).map(key => obj[key]);
}

const nodeModules = `${path__default.delimiter}node_modules${path__default.delimiter}`;
const BAR_LENGTH = 25;
const BLOCK_CHAR = '█';
const BLOCK_CHAR2 = '█';
const NEXT = ' ' + chalk.blue(figures.pointerSmall) + ' ';
const BULLET = figures.bullet;
const TICK = figures.tick;
const CROSS = figures.cross;
const CIRCLE_OPEN = figures.radioOff;

const consola = Consola.withTag('webpackbar');
const colorize = color => {
  if (color[0] === '#') {
    return chalk.hex(color);
  }

  return chalk[color] || chalk.keyword(color);
};
const renderBar = (progress, color) => {
  const w = progress * (BAR_LENGTH / 100);
  const bg = chalk.white(BLOCK_CHAR);
  const fg = colorize(color)(BLOCK_CHAR2);
  return range(BAR_LENGTH).map(i => i < w ? fg : bg).join('');
};
function createTable(data) {
  return textTable(data, {
    align: data[0].map(() => 'l')
  }).replace(/\n/g, '\n\n');
}
function ellipsisLeft(str, n) {
  if (str.length <= n - 3) {
    return str;
  }

  return `...${str.substr(str.length - n - 1)}`;
}

const parseRequest = requestStr => {
  const parts = (requestStr || '').split('!');
  const file = path__default.relative(process.cwd(), removeAfter('?', removeBefore(nodeModules, parts.pop())));
  const loaders = parts.map(part => firstMatch(/[a-z0-9-@]+-loader/, part)).filter(hasValue);
  return {
    file: hasValue(file) ? file : null,
    loaders
  };
};
const formatRequest = request => {
  const loaders = request.loaders.join(NEXT);

  if (!loaders.length) {
    return request.file || '';
  }

  return `${loaders}${NEXT}${request.file}`;
}; // Hook helper for webpack 3 + 4 support

function hook(compiler, hookName, fn) {
  if (compiler.hooks) {
    compiler.hooks[hookName].tap('WebpackBar:' + hookName, fn);
  } else {
    compiler.plugin(hookName, fn);
  }
}

const originalWrite = Symbol('webpackbarWrite');
class LogUpdate {
  constructor() {
    this.prevLineCount = 0;
    this.listening = false;
    this.extraLines = '';
    this._onData = this._onData.bind(this);
    this._streams = [process.stdout, process.stderr];
  }

  render(lines) {
    this.listen();
    const wrappedLines = wrapAnsi(lines, this.columns, {
      trim: false,
      hard: true,
      wordWrap: false
    });
    const data = ansiEscapes.eraseLines(this.prevLineCount) + wrappedLines + '\n' + this.extraLines;
    this.write(data);
    this.prevLineCount = data.split('\n').length;
  }

  get columns() {
    return (process.stderr.columns || 80) - 2;
  }

  write(data) {
    const stream = process.stderr;

    if (stream.write[originalWrite]) {
      stream.write[originalWrite].call(stream, data, 'utf-8');
    } else {
      stream.write(data, 'utf-8');
    }
  }

  clear() {
    this.done();
    this.write(ansiEscapes.eraseLines(this.prevLineCount));
  }

  done() {
    this.stopListen();
    this.prevLineCount = 0;
    this.extraLines = '';
  }

  _onData(data) {
    const str = String(data);
    const lines = str.split('\n').length - 1;

    if (lines > 0) {
      this.prevLineCount += lines;
      this.extraLines += data;
    }
  }

  listen() {
    // Prevent listening more than once
    if (this.listening) {
      return;
    } // Spy on all streams


    for (const stream of this._streams) {
      // Prevent overriding more than once
      if (stream.write[originalWrite]) {
        continue;
      } // Create a wrapper fn


      const write = (data, ...args) => {
        if (!stream.write[originalWrite]) {
          return stream.write(data, ...args);
        }

        this._onData(data);

        return stream.write[originalWrite].call(stream, data, ...args);
      }; // Backup original write fn


      write[originalWrite] = stream.write; // Override write fn

      stream.write = write;
    }

    this.listening = true;
  }

  stopListen() {
    // Restore original write fns
    for (const stream of this._streams) {
      if (stream.write[originalWrite]) {
        stream.write = stream.write[originalWrite];
      }
    }

    this.listening = false;
  }

}

/* eslint-disable no-console */
const logUpdate = new LogUpdate();
let lastRender = Date.now();
class FancyReporter {
  allDone() {
    logUpdate.done();
  }

  done(context) {
    this._renderStates(context.statesArray);

    if (context.hasErrors) {
      logUpdate.done();
    }
  }

  progress(context) {
    if (Date.now() - lastRender > 50) {
      this._renderStates(context.statesArray);
    }
  }

  _renderStates(statesArray) {
    lastRender = Date.now();
    const renderedStates = statesArray.map(c => this._renderState(c)).join('\n\n');
    logUpdate.render('\n' + renderedStates + '\n');
  }

  _renderState(state) {
    const color = colorize(state.color);
    let line1;
    let line2;

    if (state.progress >= 0 && state.progress < 100) {
      // Running
      line1 = [color(BULLET), color(state.name), renderBar(state.progress, state.color), state.message, `(${state.progress || 0}%)`, chalk.grey(state.details[0] || ''), chalk.grey(state.details[1] || '')].join(' ');
      line2 = state.request ? ' ' + chalk.grey(ellipsisLeft(formatRequest(state.request), logUpdate.columns)) : '';
    } else {
      let icon = ' ';

      if (state.hasErrors) {
        icon = CROSS;
      } else if (state.progress === 100) {
        icon = TICK;
      } else if (state.progress === -1) {
        icon = CIRCLE_OPEN;
      }

      line1 = color(`${icon} ${state.name}`);
      line2 = chalk.grey('  ' + state.message);
    }

    return line1 + '\n' + line2;
  }

}

class SimpleReporter {
  start(context) {
    consola.info(`Compiling ${context.state.name}`);
  }

  change(context, {
    shortPath
  }) {
    consola.debug(`${shortPath} changed.`, `Rebuilding ${context.state.name}`);
  }

  done(context) {
    const {
      hasError,
      message,
      name
    } = context.state;
    consola[hasError ? 'error' : 'success'](`${name}: ${message}`);
  }

}

const DB = {
  loader: {
    get: loader => startCase(loader)
  },
  ext: {
    get: ext => `${ext} files`,
    vue: 'Vue Single File components',
    js: 'JavaScript files',
    sass: 'SASS files',
    scss: 'SASS files',
    unknown: 'Unknown files'
  }
};
function getDescription(category, keyword) {
  if (!DB[category]) {
    return startCase(keyword);
  }

  if (DB[category][keyword]) {
    return DB[category][keyword];
  }

  if (DB[category].get) {
    return DB[category].get(keyword);
  }

  return '-';
}

function formatStats(allStats) {
  const lines = [];
  Object.keys(allStats).forEach(category => {
    const stats = allStats[category];
    lines.push(`> Stats by ${chalk.bold(startCase(category))}`);
    let totalRequests = 0;
    const totalTime = [0, 0];
    const data = [[startCase(category), 'Requests', 'Time', 'Time/Request', 'Description']];
    Object.keys(stats).forEach(item => {
      const stat = stats[item];
      totalRequests += stat.count || 0;
      const description = getDescription(category, item);
      totalTime[0] += stat.time[0];
      totalTime[1] += stat.time[1];
      const avgTime = [stat.time[0] / stat.count, stat.time[1] / stat.count];
      data.push([item, stat.count || '-', prettyTime(stat.time), prettyTime(avgTime), description]);
    });
    data.push(['Total', totalRequests, prettyTime(totalTime), '', '']);
    lines.push(createTable(data));
  });
  return `${lines.join('\n\n')}\n`;
}

class Profiler {
  constructor() {
    this.requests = [];
  }

  onRequest(request) {
    if (!request) {
      return;
    } // Measure time for last request


    if (this.requests.length) {
      const lastReq = this.requests[this.requests.length - 1];

      if (lastReq.start) {
        lastReq.time = process.hrtime(lastReq.start);
        delete lastReq.start;
      }
    } // Ignore requests without any file or loaders


    if (!request.file || !request.loaders.length) {
      return;
    }

    this.requests.push({
      request,
      start: process.hrtime()
    });
  }

  getStats() {
    const loaderStats = {};
    const extStats = {};

    const getStat = (stats, name) => {
      if (!stats[name]) {
        // eslint-disable-next-line no-param-reassign
        stats[name] = {
          count: 0,
          time: [0, 0]
        };
      }

      return stats[name];
    };

    const addToStat = (stats, name, count, time) => {
      const stat = getStat(stats, name);
      stat.count += count;
      stat.time[0] += time[0];
      stat.time[1] += time[1];
    };

    this.requests.forEach(({
      request,
      time = [0, 0]
    }) => {
      request.loaders.forEach(loader => {
        addToStat(loaderStats, loader, 1, time);
      });
      const ext = request.file && path__default.extname(request.file).substr(1);
      addToStat(extStats, ext && ext.length ? ext : 'unknown', 1, time);
    });
    return {
      ext: extStats,
      loader: loaderStats
    };
  }

  getFormattedStats() {
    return formatStats(this.getStats());
  }

}

class ProfileReporter {
  progress(context) {
    if (!context.state.profiler) {
      context.state.profiler = new Profiler();
    }

    context.state.profiler.onRequest(context.state.request);
  }

  done(context) {
    if (context.state.profiler) {
      context.state.profile = context.state.profiler.getFormattedStats();
      delete context.state.profiler;
    }
  }

  allDone(context) {
    let str = '';

    for (const state of context.statesArray) {
      const color = colorize(state.color);

      if (state.profile) {
        str += color(`\nProfile results for ${chalk.bold(state.name)}\n`) + `\n${state.profile}\n`;
        delete state.profile;
      }
    }

    process.stderr.write(str);
  }

}

class StatsReporter {
  constructor(options) {
    this.options = Object.assign({
      chunks: false,
      children: false,
      modules: false,
      colors: true,
      warnings: true,
      errors: true
    }, options);
  }

  done(context, {
    stats
  }) {
    const str = stats.toString(this.options);

    if (context.hasErrors) {
      process.stderr.write('\n' + str + '\n');
    } else {
      context.state.statsString = str;
    }
  }

  allDone(context) {
    let str = '';

    for (const state of context.statesArray) {
      if (state.statsString) {
        str += '\n' + state.statsString + '\n';
        delete state.statsString;
      }
    }

    process.stderr.write(str);
  }

}



var reporters = /*#__PURE__*/Object.freeze({
  fancy: FancyReporter,
  basic: SimpleReporter,
  profile: ProfileReporter,
  stats: StatsReporter
});

const DEFAULTS = {
  name: 'webpack',
  color: 'green',
  reporters: env.minimalCLI ? ['basic'] : ['fancy'],
  reporter: null // Default state object

};
const DEFAULT_STATE = {
  start: null,
  progress: -1,
  done: false,
  message: '',
  details: [],
  request: null,
  hasErrors: false // Mapping from name => State

};
const globalStates = {};
class WebpackBarPlugin extends webpack.ProgressPlugin {
  constructor(options) {
    super();
    this.options = Object.assign({}, DEFAULTS, options); // Assign a better handler to base ProgressPlugin

    this.handler = (percent, message, ...details) => {
      this.updateProgress(percent, message, details);
    }; // Reporters


    this.reporters = Array.from(this.options.reporters || []);

    if (this.options.reporter) {
      this.reporters.push(this.options.reporter);
    } // Resolve reporters


    this.reporters = this.reporters.filter(Boolean).map(_reporter => {
      if (this.options[_reporter] === false) {
        return false;
      }

      let reporter = _reporter;
      let reporterOptions = this.options[reporter] || {};

      if (Array.isArray(_reporter)) {
        reporter = _reporter[0];

        if (_reporter[1] === false) {
          return false;
        }

        if (_reporter[1]) {
          reporterOptions = _reporter[1];
        }
      }

      if (typeof reporter === 'string') {
        if (reporters[reporter]) {
          // eslint-disable-line import/namespace
          reporter = reporters[reporter]; // eslint-disable-line import/namespace
        } else {
          reporter = require(reporter);
        }
      }

      if (typeof reporter === 'function') {
        if (typeof reporter.constructor === 'function') {
          const Reporter = reporter;
          reporter = new Reporter(reporterOptions);
        } else {
          reporter = reporter(reporterOptions);
        }
      }

      return reporter;
    }).filter(Boolean);
  }

  callReporters(fn, payload = {}) {
    for (const reporter of this.reporters) {
      if (typeof reporter[fn] === 'function') {
        try {
          reporter[fn](this, payload);
        } catch (e) {
          process.stdout.write(e.stack + '\n');
        }
      }
    }
  }

  get hasRunning() {
    return objectValues(this.states).some(state => !state.done);
  }

  get hasErrors() {
    return objectValues(this.states).some(state => state.hasErrors);
  }

  get statesArray() {
    return objectValues(this.states).sort((s1, s2) => s1.name.localeCompare(s2.name));
  }

  get states() {
    return globalStates;
  }

  get state() {
    return globalStates[this.options.name];
  }

  _ensureState() {
    // Keep our state in shared object
    if (!this.states[this.options.name]) {
      this.states[this.options.name] = { ...DEFAULT_STATE,
        color: this.options.color,
        name: startCase(this.options.name)
      };
    }
  }

  apply(compiler) {
    // Prevent adding multi instances to the same compiler
    if (compiler.webpackbar) {
      return;
    }

    compiler.webpackbar = this; // Apply base hooks

    super.apply(compiler); // Register our state after all plugins initialized

    hook(compiler, 'afterPlugins', () => {
      this._ensureState();
    }); // Hook into the compiler before a new compilation is created.

    hook(compiler, 'compile', () => {
      this._ensureState();

      Object.assign(this.state, { ...DEFAULT_STATE,
        start: process.hrtime()
      });
      this.callReporters('start');
    }); // Watch compilation has been invalidated.

    hook(compiler, 'invalid', (fileName, changeTime) => {
      this._ensureState();

      this.callReporters('change', {
        path: fileName,
        shortPath: shortenPath(fileName),
        time: changeTime
      });
    }); // Compilation has completed

    hook(compiler, 'done', stats => {
      this._ensureState(); // Prevent calling done twice


      if (this.state.done) {
        return;
      }

      const hasErrors = stats.hasErrors();
      const status = hasErrors ? 'with some errors' : 'successfully';
      const time = this.state.start ? ' in ' + prettyTime(process.hrtime(this.state.start), 2) : '';
      Object.assign(this.state, { ...DEFAULT_STATE,
        progress: 100,
        done: true,
        message: `Compiled ${status}${time}`,
        hasErrors
      });
      this.callReporters('progress');
      this.callReporters('done', {
        stats
      });

      if (!this.hasRunning) {
        this.callReporters('beforeAllDone');
        this.callReporters('allDone');
        this.callReporters('afterAllDone');
      }
    });
  }

  updateProgress(percent = 0, message = '', details = []) {
    const progress = Math.floor(percent * 100);
    Object.assign(this.state, {
      progress,
      message: message || '',
      details,
      request: parseRequest(details[2])
    });
    this.callReporters('progress');
  }

}

module.exports = WebpackBarPlugin;
