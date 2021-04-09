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
    title: <>Reduces human bias</>,
    imageUrl: 'img/take_control.svg',
    description: (
      <>
        <ul>
          <li> Automated hyperparameter optimization with evolutionary algorithms from Facebook's AI library <a href="https://facebookresearch.github.io/nevergrad">Nevergrad</a></li>
          <li> Ridge regression with cross-validation to regularize multi-collinearity and prevent overfitting</li>
          <li> Facebook's <a href="https://facebook.github.io/prophet/">Prophet</a> library to automatically decompose the trend, seasonality and holidays patterns</li>
        </ul>
      </>
    ),
  },
  {
    title: <>Aligns with the ground-truth</>,
    imageUrl: 'img/calibrate.svg',
    description: (
      <>
        <ul>
          <li>
            {' '}
            It calibrates models based on ground-truth methodologies (Geo-based, Facebook lift, MTA, etc.)
          </li>
          <li>
            {' '}
            Facebook <a href="https://facebookresearch.github.io/nevergrad">Nevergrad</a>'s multi-objective optimization minimizing the error between MMM prediction and ground-truth
          </li>
        </ul>
      </>
    ),
  },
  {
    title: <>Enables actionable decision making</>,
    imageUrl: 'img/focus_on_what_matters.svg',
    description: (
      <>
        <ul>
          <li>
            {' '}
            Budget allocator using a gradient-based constrained non-linear solver to maximize the outcome by reallocating budgets
          </li>
          <li>
            {' '}
            Enables frequent modeling outcomes due to stronger automation
          </li>
          <li>
            {' '}
            Allows intuitive model comparisons via automatically generated model one-pagers
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
