/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

module.exports = {
<<<<<<< HEAD
  title: 'Robyn',
  tagline: 'Robyn is a Marketing Mix Modeling (MMM) code. It can be used to build end-to-end time series regression models and as an econometrics code library. Automated, built for very large data sets, and is suitable for digital and complex consumer behaviour',
  url: 'https://facebookexperimental.github.io/Robyn/',
  baseUrl: '/Robyn/',
  onBrokenLinks: 'throw',
  favicon: 'img/robyn_logo.png',
  organizationName: 'facebook', // Usually your GitHub org/user name.
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
=======
  title: 'MMM for all',
  tagline: 'Democratizing marketing mix models',
  url: 'https://your-docusaurus-test-site.com',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  favicon: 'img/favicon.ico',
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'MMM for all', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'MMM for all',
      logo: {
        alt: 'My Facebook Project Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: 'docs/',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        {to: 'blog', label: 'Blog', position: 'left'},
        // Please keep GitHub link to the right for consistency.
        {
          href: 'https://github.com/facebook/docusaurus',
>>>>>>> 88a95a753303f58e6c9f4b7b4e8c8c7a4488fdec
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
<<<<<<< HEAD
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/facebookexperimental/Robyn',
=======
          title: 'Learn',
          items: [
            {
              label: 'Style Guide',
              to: 'docs/',
            },
            {
              label: 'Second Doc',
              to: 'docs/doc2',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/docusaurus',
            },
            {
              label: 'Twitter',
              href: 'https://twitter.com/docusaurus',
            },
            {
              label: 'Discord',
              href: 'https://discordapp.com/invite/docusaurus',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: 'blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/facebook/docusaurus',
>>>>>>> 88a95a753303f58e6c9f4b7b4e8c8c7a4488fdec
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
<<<<<<< HEAD
            'https://github.com/facebookexperimental/Robyn',
=======
            'https://github.com/facebook/docusaurus/edit/master/website/',
>>>>>>> 88a95a753303f58e6c9f4b7b4e8c8c7a4488fdec
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
