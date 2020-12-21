"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = _interopRequireDefault(require("react"));

var _clsx = _interopRequireDefault(require("clsx"));

var _Link = _interopRequireDefault(require("@docusaurus/Link"));

var _themeCommon = require("@docusaurus/theme-common");

var _useBaseUrl = _interopRequireDefault(require("@docusaurus/useBaseUrl"));

var _stylesModule = _interopRequireDefault(require("./styles.module.css"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
function FooterLink({
  to,
  href,
  label,
  prependBaseUrlToHref,
  ...props
}) {
  const toUrl = (0, _useBaseUrl.default)(to);
  const normalizedHref = (0, _useBaseUrl.default)(href, {
    forcePrependBaseUrl: true
  });
  return <_Link.default className="footer__link-item" {...href ? {
    target: '_blank',
    rel: 'noopener noreferrer',
    href: prependBaseUrlToHref ? normalizedHref : href
  } : {
    to: toUrl
  }} {...props}>
      {label}
    </_Link.default>;
}

const FooterLogo = ({
  url,
  alt
}) => <img className="footer__logo" alt={alt} src={url} />;

function Footer() {
  const {
    footer
  } = (0, _themeCommon.useThemeConfig)();
  const {
    copyright,
    links = [],
    logo = {}
  } = footer || {};
  const logoUrl = (0, _useBaseUrl.default)(logo.src);

  if (!footer) {
    return null;
  }

  return <footer className={(0, _clsx.default)('footer', {
    'footer--dark': footer.style === 'dark'
  })}>
      <div className="container">
        {links && links.length > 0 && <div className="row footer__links">
            {links.map((linkItem, i) => <div key={i} className="col footer__col">
                {linkItem.title != null ? <h4 className="footer__title">{linkItem.title}</h4> : null}
                {linkItem.items != null && Array.isArray(linkItem.items) && linkItem.items.length > 0 ? <ul className="footer__items">
                    {linkItem.items.map((item, key) => item.html ? <li key={key} className="footer__item" // Developer provided the HTML, so assume it's safe.
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{
              __html: item.html
            }} /> : <li key={item.href || item.to} className="footer__item">
                          <FooterLink {...item} />
                        </li>)}
                  </ul> : null}
              </div>)}
          </div>}
        {(logo || copyright) && <div className="footer__bottom text--center">
            {logo && logo.src && <div className="margin-bottom--sm">
                {logo.href ? <a href={logo.href} target="_blank" rel="noopener noreferrer" className={_stylesModule.default.footerLogoLink}>
                    <FooterLogo alt={logo.alt} url={logoUrl} />
                  </a> : <FooterLogo alt={logo.alt} url={logoUrl} />}
              </div>}
            {copyright ? <div className="footer__copyright" // Developer provided the HTML, so assume it's safe.
        // eslint-disable-next-line react/no-danger
        dangerouslySetInnerHTML={{
          __html: copyright
        }} /> : null}
          </div>}
      </div>
    </footer>;
}

var _default = Footer;
exports.default = _default;