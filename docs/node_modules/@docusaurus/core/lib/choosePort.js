"use strict";
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * This feature was heavily inspired by create-react-app and
 * uses many of the same utility functions to implement it.
 */
const child_process_1 = require("child_process");
const detect_port_1 = __importDefault(require("detect-port"));
const is_root_1 = __importDefault(require("is-root"));
const chalk_1 = __importDefault(require("chalk"));
const inquirer_1 = __importDefault(require("inquirer"));
const isInteractive = process.stdout.isTTY;
const execOptions = {
    encoding: 'utf8',
    stdio: [
        'pipe',
        'pipe',
        'ignore',
    ],
};
// Clears console
function clearConsole() {
    process.stdout.write(process.platform === 'win32' ? '\x1B[2J\x1B[0f' : '\x1B[2J\x1B[3J\x1B[H');
}
// Gets process id of what is on port
function getProcessIdOnPort(port) {
    return child_process_1.execSync(`lsof -i:${port} -P -t -sTCP:LISTEN`, execOptions)
        .toString()
        .split('\n')[0]
        .trim();
}
// Gets process command
function getProcessCommand(processId) {
    let command = child_process_1.execSync(`ps -o command -p ${processId} | sed -n 2p`, execOptions);
    command = command.toString().replace(/\n$/, '');
    return command;
}
// Gets directory of a process from its process id
function getDirectoryOfProcessById(processId) {
    return child_process_1.execSync(`lsof -p ${processId} | awk '$4=="cwd" {for (i=9; i<=NF; i++) printf "%s ", $i}'`, execOptions)
        .toString()
        .trim();
}
// Gets process on port
function getProcessForPort(port) {
    try {
        const processId = getProcessIdOnPort(port);
        const directory = getDirectoryOfProcessById(processId);
        const command = getProcessCommand(processId);
        return (chalk_1.default.cyan(command) +
            chalk_1.default.grey(` (pid ${processId})\n`) +
            chalk_1.default.blue('  in ') +
            chalk_1.default.cyan(directory));
    }
    catch (e) {
        return null;
    }
}
/**
 * Detects if program is running on port and prompts user
 * to choose another if port is already being used
 */
async function choosePort(host, defaultPort) {
    // @ts-expect-error: bad lib typedef?
    return detect_port_1.default(defaultPort, host).then((port) => new Promise((resolve) => {
        if (port === defaultPort) {
            return resolve(port);
        }
        const message = process.platform !== 'win32' && defaultPort < 1024 && !is_root_1.default()
            ? `Admin permissions are required to run a server on a port below 1024.`
            : `Something is already running on port ${defaultPort}.`;
        if (isInteractive) {
            clearConsole();
            const existingProcess = getProcessForPort(defaultPort);
            const question = {
                type: 'confirm',
                name: 'shouldChangePort',
                message: `${chalk_1.default.yellow(`${message}${existingProcess ? ` Probably:\n  ${existingProcess}` : ''}`)}\n\nWould you like to run the app on another port instead?`,
                default: true,
            };
            inquirer_1.default.prompt(question).then((answer) => {
                if (answer.shouldChangePort) {
                    resolve(port);
                }
                else {
                    resolve(null);
                }
            });
        }
        else {
            console.log(chalk_1.default.red(message));
            resolve(null);
        }
        return null;
    }), (err) => {
        throw new Error(`${chalk_1.default.red(`Could not find an open port at ${chalk_1.default.bold(host)}.`)}\n${`Network error message: ${err.message}` || err}\n`);
    });
}
exports.default = choosePort;
