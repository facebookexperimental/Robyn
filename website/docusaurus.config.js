/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

module.exports = {
  title: 'Robyn',
  tagline:
    'Robyn is an automated Marketing Mix Modeling (MMM) code. It aims to reduce human bias by means of ridge regression and evolutionary algorithms, enables actionable decision making providing a budget allocator and diminishing returns curves and allows ground-truth calibration to account for causation',
  url: 'https://facebookexperimental.github.io',
  baseUrl: '/Robyn/',
  onBrokenLinks: 'throw',
  favicon: 'img/robyn_logo.png',
  organizationName: 'facebookexperimental', // Usually your GitHub org/user name.
  projectName: 'Robyn', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'Robyn',
      logo: {
        alt: 'Robyn Logo',
        src: 'img/robyn_logo.png',
      },
      items: [
        {
          to: 'docs/',
          label: 'Docs',
          position: 'left',
        },
        {
          to: 'docs/about',
          label: 'About',
          position: 'left',
        },
        // Please keep GitHub link to the right for consistency.
        {
          href: 'https://github.com/facebookexperimental/Robyn',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/facebookexperimental/Robyn',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Discord chat',
              href: 'https://discord.gg/BYhqMedCcN',
            },
          ],
        },
        {
          title: 'Legal',
          // Please do not remove the privacy and terms, it's a legal requirement.
          items: [
            {
              label: 'Privacy',
              href: 'https://opensource.facebook.com/legal/privacy/',
            },
            {
              label: 'Terms',
              href: 'https://opensource.facebook.com/legal/terms/',
            },
          ],
        },
      ],
      logo: {
        alt: 'Facebook Open Source Logo',
        src: 'img/oss_logo.png',
        href: 'https://opensource.facebook.com',
      },
      // Please do not remove the credits, help to publicize Docusaurus :)
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc. Built with Docusaurus.`,
    },
    googleAnalytics: {
      trackingID: 'G-TMZK4YFDGL',
      anonymizeIP: true,
    },
    gtag: {
      trackingID: 'G-TMZK4YFDGL',
      anonymizeIP: true,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/facebookexperimental/Robyn',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
