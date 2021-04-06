export default {
  "title": "Robyn",
  "tagline": "Robyn is an automated Marketing Mix Modeling (MMM) code. It aims to reduce human bias by means of ridge regression and evolutionary algorithms, enables actionable decision making providing a budget allocator and diminishing returns curves and allows ground-truth calibration to account for causation",
  "url": "https://facebookexperimental.github.io/Robyn/",
  "baseUrl": "/Robyn/",
  "onBrokenLinks": "throw",
  "favicon": "img/robyn_logo.png",
  "organizationName": "facebookexperimental",
  "projectName": "Robyn",
  "themeConfig": {
    "navbar": {
      "title": "Robyn",
      "logo": {
        "alt": "Robyn Logo",
        "src": "img/robyn_logo.png"
      },
      "items": [
        {
          "to": "docs/",
          "label": "Docs",
          "position": "left"
        },
        {
          "to": "docs/about",
          "label": "About",
          "position": "left"
        },
        {
          "href": "https://github.com/facebookexperimental/Robyn",
          "label": "GitHub",
          "position": "right"
        }
      ],
      "hideOnScroll": false
    },
    "footer": {
      "style": "dark",
      "links": [
        {
          "title": "More",
          "items": [
            {
              "label": "GitHub",
              "href": "https://github.com/facebookexperimental/Robyn"
            }
          ]
        },
        {
          "title": "Community",
          "items": [
            {
              "label": "Discord chat",
              "href": "https://discord.gg/BYhqMedCcN"
            }
          ]
        },
        {
          "title": "Legal",
          "items": [
            {
              "label": "Privacy",
              "href": "https://opensource.facebook.com/legal/privacy/"
            },
            {
              "label": "Terms",
              "href": "https://opensource.facebook.com/legal/terms/"
            }
          ]
        }
      ],
      "logo": {
        "alt": "Facebook Open Source Logo",
        "src": "img/oss_logo.png",
        "href": "https://opensource.facebook.com"
      },
      "copyright": "Copyright © 2021 Facebook, Inc. Built with Docusaurus."
    },
    "colorMode": {
      "defaultMode": "light",
      "disableSwitch": false,
      "respectPrefersColorScheme": false,
      "switchConfig": {
        "darkIcon": "🌜",
        "darkIconStyle": {},
        "lightIcon": "🌞",
        "lightIconStyle": {}
      }
    },
    "docs": {
      "versionPersistence": "localStorage"
    },
    "metadatas": [],
    "prism": {
      "additionalLanguages": []
    },
    "hideableSidebar": false
  },
  "presets": [
    [
      "@docusaurus/preset-classic",
      {
        "docs": {
          "sidebarPath": "/Users/leonelsentana/Robyn/docs/sidebars.js",
          "editUrl": "https://github.com/facebookexperimental/Robyn"
        },
        "theme": {
          "customCss": "/Users/leonelsentana/Robyn/docs/src/css/custom.css"
        }
      }
    ]
  ],
  "baseUrlIssueBanner": true,
  "i18n": {
    "defaultLocale": "en",
    "locales": [
      "en"
    ],
    "localeConfigs": {}
  },
  "onBrokenMarkdownLinks": "warn",
  "onDuplicateRoutes": "warn",
  "customFields": {},
  "plugins": [],
  "themes": [],
  "titleDelimiter": "|",
  "noIndex": false
};