/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
<<<<<<< HEAD
=======

>>>>>>> 88a95a753303f58e6c9f4b7b4e8c8c7a4488fdec
import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const features = [
  {
<<<<<<< HEAD
    title: <>Advanced</>,
    imageUrl: 'img/take_control.svg',
    description: (
      <>
      <ul>
      <li> Automated code, making it significantly faster to run </li>
      <li> Can be calibrated and validated using real world experiments</li>
      <li> Fully customisable adstock to suit your business</li>
      <li> Automated seasonality and richer external variables using Facebook code ‘Prophet’, increasing interpretability and model fit</li>
      <li> Uses Ridge Regression to solve for multicollinearity</li>
      <li> Built to manage large data sets and numbers of variables making it ideal for digital marketing and complex consumer behaviour</li>
    </ul>
      </>
    ),
  },
  {
    title: <>Controllable and Scalable</>,
    imageUrl: 'img/calibrate.svg',
    description: (
      <>
    <ul>
      <li> Standardised and stable code to limit analyst bias and subjectivity, making models scaleable and transferable</li>
      <li> Fully customisable to accommodate multiple unique variables that matter your business</li>
      <li> Increase the number of models and frequency as faster to run and automated</li>
      <li> Model all of the outputs that matter to your business</li>
    </ul>
=======
    title: <>Easy to Use</>,
    imageUrl: 'img/undraw_docusaurus_mountain.svg',
    description: (
      <>
        Docusaurus was designed from the ground up to be easily installed and
        used to get your website up and running quickly.
>>>>>>> 88a95a753303f58e6c9f4b7b4e8c8c7a4488fdec
      </>
    ),
  },
  {
<<<<<<< HEAD
    title: <>Actionable</>,
    imageUrl: 'img/focus_on_what matters.svg',
    description: (
      <>
      <ul>
      <li> No need to wait until your campaigns have finished. Faster modeling allows for inflight campaign optimization.</li>
      <li> Continuous modeling helps you to understand the performance of your marketing in almost real time.</li>
      <li> Includes integrated marketing budget optimizer with the ability to apply custom restraints</li>
      <li> Not dependent on Facebook advertising but can connect to measurement products in Facebook Ads Manager for granular insights</li>
    </ul>
=======
    title: <>Focus on What Matters</>,
    imageUrl: 'img/undraw_docusaurus_tree.svg',
    description: (
      <>
        Docusaurus lets you focus on your docs, and we&apos;ll do the chores. Go
        ahead and move your docs into the <code>docs</code> directory.
>>>>>>> 88a95a753303f58e6c9f4b7b4e8c8c7a4488fdec
      </>
    ),
  },
  {
<<<<<<< HEAD
    description: (
      <>
      </>
    ),
  },
  {
    title: <>Privacy by Design</>,
    description: (
      <>
      <ul>
      <li> Privacy safe with no requirement for PII or Individual log level data</li>
      <li> Not dependent on Cookies or Pixel data</li>
      </ul>
      </>
    ),
  }
=======
    title: <>Powered by React</>,
    imageUrl: 'img/undraw_docusaurus_react.svg',
    description: (
      <>
        Extend or customize your website layout by reusing React. Docusaurus can
        be extended while reusing the same header and footer.
      </>
    ),
  },
>>>>>>> 88a95a753303f58e6c9f4b7b4e8c8c7a4488fdec
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--4', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/')}>
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map(({title, imageUrl, description}) => (
                  <Feature
                    key={title}
                    title={title}
                    imageUrl={imageUrl}
                    description={description}
                  />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
