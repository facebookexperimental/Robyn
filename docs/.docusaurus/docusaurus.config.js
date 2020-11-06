export default {
  "title": "Robyn",
  "tagline": "Robyn is a Marketing Mix Modeling (MMM) code. It can be used to build end-to-end time series regression models and as an econometrics code library. Automated, built for very large data sets, and is suitable for digital and complex consumer behaviour",
  "url": "https://facebookexperimental.github.io/Robyn/",
  "baseUrl": "/Robyn/",
  "onBrokenLinks": "throw",
  "favicon": "img/robyn_logo.png",
  "organizationName": "facebook",
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
          "to": "docs/doc11",
          "activeBasePath": "docs",
          "label": "About Robyn",
          "position": "right"
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
      "copyright": "Copyright © 2020 Facebook, Inc. Built with Docusaurus."
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
    }
  },
  "presets": [
    [
      "@docusaurus/preset-classic",
      {
        "docs": {
          "homePageId": "doc2",
          "sidebarPath": "/Users/leonelsentana/Robyn/docs/sidebars.js",
          "editUrl": "https://github.com/facebookexperimental/Robyn"
        },
        "blog": {
          "showReadingTime": true,
          "editUrl": "https://github.com/facebook/docusaurus/edit/master/website/blog/"
        },
        "theme": {
          "customCss": "/Users/leonelsentana/Robyn/docs/src/css/custom.css"
        }
      }
    ]
  ],
  "onDuplicateRoutes": "warn",
  "customFields": {},
  "plugins": [],
  "themes": []
};