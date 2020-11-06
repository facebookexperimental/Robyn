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
  tagline: 'Robyn is a Marketing Mix Modeling (MMM) code. It can be used to build end-to-end time series regression models and as an econometrics code library. Automated, built for very large data sets, and is suitable for digital and complex consumer behaviour',
  url: 'https://facebookexperimental.github.io/Robyn/',
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
          to: 'docs/doc11',
          activeBasePath: 'docs',
          label: 'About Robyn',
          position: 'right',
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
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          // It is recommended to set document id as docs home page (`docs/` path).
          homePageId: 'doc2',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebookexperimental/Robyn',
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/master/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
