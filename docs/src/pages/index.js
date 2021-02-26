/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const features = [
  {
    title: <>Advanced</>,
    imageUrl: 'img/take_control.svg',
    description: (
      <>
        <ul>
          <li> Automated code, making it significantly faster to run </li>
          <li> Can be calibrated and validated using real world experiments</li>
          <li> Fully customisable adstock to suit your business</li>
          <li>
            {' '}
            Automated seasonality and richer external variables using Facebook
            code ‘Prophet’, increasing interpretability and model fit
          </li>
          <li> Uses Ridge Regression to solve for multicollinearity</li>
          <li>
            {' '}
            Built to manage large data sets and numbers of variables making it
            ideal for digital marketing and complex consumer behaviour
          </li>
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
          <li>
            {' '}
            Standardised and stable code to limit analyst bias and subjectivity,
            making models scaleable and transferable
          </li>
          <li>
            {' '}
            Fully customisable to accommodate multiple unique variables that
            matter your business
          </li>
          <li>
            {' '}
            Increase the number of models and frequency as faster to run and
            automated
          </li>
          <li> Model all of the outputs that matter to your business</li>
        </ul>
      </>
    ),
  },
  {
    title: <>Actionable</>,
    imageUrl: 'img/focus_on_what matters.svg',
    description: (
      <>
        <ul>
          <li>
            {' '}
            No need to wait until your campaigns have finished. Faster modeling
            allows for inflight campaign optimization.
          </li>
          <li>
            {' '}
            Continuous modeling helps you to understand the performance of your
            marketing in almost real time.
          </li>
          <li>
            {' '}
            Includes integrated marketing budget optimizer with the ability to
            apply custom restraints
          </li>
          <li>
            {' '}
            Not dependent on Facebook advertising but can connect to measurement
            products in Facebook Ads Manager for granular insights
          </li>
        </ul>
      </>
    ),
  },
  {
    description: <></>,
  },
  {
    title: <>Privacy by Design</>,
    description: (
      <>
        <ul>
          <li>
            {' '}
            Privacy safe with no requirement for PII or Individual log level
            data
          </li>
          <li> Not dependent on Cookies or Pixel data</li>
        </ul>
      </>
    ),
  },
];

function Feature({ imageUrl, title, description }) {
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
  const { siteConfig = {} } = context;
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />"
    >
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
              to={useBaseUrl('docs/')}
            >
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>
        <div className="container padding-top--lg">
          <div className="row">
            <div className="col col--6 col--offset-3">
              <iframe
                width="560"
                height="315"
                src="https://www.youtube.com/embed/8SyKRpsXn44"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen
              />
            </div>
          </div>
        </div>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map(({ title, imageUrl, description }) => (
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
