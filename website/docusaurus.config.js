/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

module.exports = {
  title: 'Robyn',
  tagline: 'Robyn aims to reduce human bias in the modeling process, esp. by automating modelers decisions like adstocking, saturation, trend & seasonality as well as model validation. Moreover, the budget allocator & calibration enable actionability and causality of the results.',
  url: 'https://facebookexperimental.github.io',
  baseUrl: '/Robyn/',
  onBrokenLinks: 'throw',
  favicon: 'img/robyn_logo.png',
  organizationName: 'facebookexperimental', // Usually your GitHub org/user name.
  projectName: 'Robyn', // Usually your repo name.
  themeConfig: {
    announcementBar: {
      id: 'support_ukraine',
      content:
        'Support Ukraine 🇺🇦 <a target="_blank" rel="noopener noreferrer" href="https://opensource.facebook.com/support-ukraine"> Help Provide Humanitarian Aid to Ukraine</a>.',
      backgroundColor: '#20232a',
      textColor: '#fff',
      isCloseable: false,
    },
    navbar: {
      title: 'Robyn',
      logo: {
        alt: 'Robyn Logo',
        src: 'img/robyn_logo.png',
      },
      items: [
        {
          to: 'docs/quick-start/',
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
            {
              label: 'Case study: Resident',
              href: 'https://www.facebook.com/business/success/resident',
            },
            {
              label: 'Case study: Central Group',
              href: 'https://www.facebook.com/business/success/central-retail-corporation',
            },
            {
              label: 'Case study: Bark',
              href: 'https://www.facebook.com/business/measurement/case-studies/bark',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Robyn MMM Users Facebook Group',
              href: 'https://www.facebook.com/groups/954715125296621',
            },
            {
              label: 'Raise an issue on Github',
              href: 'https://github.com/facebookexperimental/Robyn/issues',
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
      copyright: `Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/facebookexperimental/Robyn/edit/main/website',
        },

        googleAnalytics: {
          trackingID: 'G-TMZK4YFDGL',
          anonymizeIP: true,
        },
        gtag: {
          trackingID: 'G-TMZK4YFDGL',
          anonymizeIP: true,
        },

        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
