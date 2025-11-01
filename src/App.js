import './App.css';

function App() {
  return (
    <div className="App">
      <div className="container">
        <header className="paper-header">
          <h1 className="paper-title">Dataset-Adaptive Dimensionality Reduction</h1>
          <div className="authors">
            <span className="author"><strong>Hyeon Jeon¹</strong></span>
            <span className="author"><strong>Jeongin Park¹</strong></span>
            <span className="author"><strong>Soohyun Lee¹</strong></span>
            <span className="author"><strong>Dae Hyun Kim²</strong></span>
            <span className="author"><strong>Sungbok Shin³</strong></span>
            <span className="author"><strong>Jinwook Seo¹</strong></span>
          </div>
          <div className="affiliations">
            <p>¹Seoul National University, ²Yonsei University, ³Inria-Saclay and Université Paris-Saclay</p>
          </div>
          <div className="links">
            <a href="./pdf/jeon25tvcg.pdf" className="paper-link">Paper PDF</a>
            <a href="https://github.com/hj-n/dataset-adaptive-dr" className="paper-link" target="_blank" rel="noopener noreferrer">GitHub Repository</a>
            <a href="./pdf/jeon25tvcg_appendix.pdf" className="paper-link">Appendix</a>
          </div>
        </header>

        <main className="paper-content">
          <section className="section">
            <p>
              Dimensionality reduction (DR) is essential for visualizing high-dimensional data, but finding optimal DR projections typically requires extensive trial and error. Practitioners must test multiple techniques with various hyperparameter settings, often running unnecessary computations without clear stopping criteria. This process becomes particularly inefficient because different datasets have vastly different structural properties - some can be accurately projected into 2D while others inherently resist low-dimensional representation.
            </p>
            <p>
              We introduce a <strong>dataset-adaptive approach</strong> that uses <strong>structural complexity metrics</strong> to quantify how difficult a dataset is to project into lower dimensions. By measuring this complexity upfront, we can predict which DR techniques will succeed and when optimization has reached its limits. Our method achieves 5-13x speedup while maintaining comparable accuracy to exhaustive optimization.
            </p>
          </section>

          <section className="section">
            <h2>Understanding Structural Complexity</h2>
            
            <h3>What Makes a Dataset Complex?</h3>
            <p>
              When we project high-dimensional data into 2D for visualization, we inevitably lose information. The question is: how much information must we lose? This depends on the dataset's structural complexity. Consider a dataset of images containing only horizontal and vertical lines versus one containing natural photographs. The former has simple, low-dimensional structure that can be well-preserved in 2D, while the latter contains rich, complex patterns that cannot be fully captured in any 2D representation.
            </p>
            <p>
              Structural complexity emerges from several factors. In high-dimensional spaces, points tend to become equidistant from each other, a phenomenon known as the "curse of dimensionality." This makes it difficult to preserve meaningful distance relationships in lower dimensions. Additionally, local neighborhood structures can be inconsistent - points that seem related from one perspective may appear unrelated from another. These challenges mean that some datasets will never achieve high-accuracy 2D representations, regardless of which DR technique or parameters we use.
            </p>
            <p>
              Understanding a dataset's structural complexity before optimization allows us to set realistic expectations and avoid wasting computation on impossible goals. If we know a dataset is highly complex, we can accept a certain level of distortion as inevitable and stop optimization once we reach that threshold.
            </p>

            <h3>Our Complexity Metrics</h3>
            <p>We developed two complementary metrics that capture different aspects of structural complexity:</p>
            
            <p>
              <strong>Pairwise Distance Shift (PDS)</strong> measures global structural complexity by quantifying how pairwise distances between data points behave in high-dimensional space. In high dimensions, distances tend to concentrate around their mean value with small variance - a phenomenon that makes it difficult to distinguish between near and far neighbors. PDS captures this effect through a scale-invariant ratio of distance statistics. Lower PDS values indicate higher complexity, suggesting the dataset will be challenging to project accurately while preserving global structure.
            </p>
            
            <p>
              <strong>Mutual Neighbor Consistency (MNC)</strong> focuses on local structural complexity by examining the consistency between different neighborhood definitions. It compares k-nearest neighbor relationships with shared nearest neighbor relationships. In complex high-dimensional datasets, these two notions of neighborhood often disagree - points that are k-nearest neighbors may share few common neighbors. MNC quantifies this inconsistency, with lower values indicating more complex local structures that will be difficult to preserve during dimensionality reduction.
            </p>
            
            <p>
              These metrics work synergistically. PDS excels at predicting performance for DR techniques that prioritize global structure (like PCA), while MNC better predicts performance for local techniques (like t-SNE and UMAP). By combining both metrics, we achieve robust complexity assessment across diverse datasets and DR techniques.
            </p>
          </section>

          <section className="section">
            <h2>The Dataset-Adaptive Workflow</h2>
            <p>
              The traditional approach tests multiple DR techniques with numerous hyperparameter configurations, running each for a fixed number of iterations. This wastes time on unsuitable techniques and either stops too early or runs too long. Our dataset-adaptive workflow transforms this through intelligent prediction and early stopping, as illustrated in Figure 1.
            </p>

            <div className="figure-container">
              <img src="./imgs/workflow.png" alt="Comparison between conventional and dataset-adaptive workflows" className="figure" />
              <p className="figure-caption">
                <strong>Figure 1</strong>: Comparison between the traditional approach (top) and our dataset-adaptive workflow (bottom), showing how our method reduces computational overhead through intelligent technique selection and early termination.
              </p>
            </div>

            <p>
              As shown in Figure 1, our approach first computes PDS and MNC scores for the dataset. These feed into pre-trained regression models that predict maximum achievable accuracy for each DR technique. The workflow then focuses only on promising techniques and terminates optimization early once the predicted maximum is approached. For simple datasets, it might test several options but terminate each quickly. For complex datasets where only one or two techniques are suitable, it focuses efforts accordingly. This dynamic allocation ensures efficiency without sacrificing quality.
            </p>
          </section>

          <section className="section">
            <h2>Evaluation: Effectiveness of the Dataset-Adaptive Workflow</h2>
            <p>
              We evaluated our dataset-adaptive workflow on 96 real-world datasets from diverse domains including computer vision, bioinformatics, and social networks. The results, summarized in Figure 2, demonstrate substantial efficiency improvements without compromising projection quality.
            </p>

            <div className="figure-container">
              <img src="./imgs/accuracy_time.png" alt="Accuracy vs Time comparison showing performance gains" className="figure" />
              <p className="figure-caption">
                <strong>Figure 2</strong>: Performance comparison between traditional exhaustive optimization and our dataset-adaptive approach, showing accuracy versus computation time across multiple datasets and DR techniques.
              </p>
            </div>
            
            <p>
              <strong>Performance Gains</strong>: As evident in Figure 2, the dataset-adaptive workflow achieves median speedups of <strong>5.3× for single-technique optimization</strong> and <strong>13× when selecting among multiple techniques</strong>. These improvements come from two sources: eliminating time spent on unsuitable techniques and early termination when near-optimal parameters are found.
            </p>
            
            <p>
              <strong>Accuracy Preservation</strong>: Despite the dramatic speedup shown in Figure 2, the workflow maintains high-quality results. Statistical analysis shows no significant difference in projection accuracy between our adaptive approach and exhaustive optimization (p &gt; 0.05 for both Trustworthiness & Continuity and Label-Trustworthiness & Label-Continuity metrics). The final projections achieve accuracy within 2% of conventional optimization.
            </p>
            
            <p>
              <strong>Real-World Impact</strong>: In practice, these improvements translate to substantial time savings, as illustrated in Figure 3. For example, optimizing UMAP on the Fashion-MNIST dataset reduced from 837 seconds to 52 seconds while maintaining accuracy of 0.97. On the Optical Recognition dataset, optimization time dropped from 415 seconds to 14 seconds. The approach shows particular strength on complex datasets where traditional optimization would waste significant time testing unsuitable techniques - our workflow quickly identifies that UMAP or t-SNE will outperform PCA on locally-structured data, avoiding hours of fruitless PCA optimization.
            </p>

            <div className="figure-container">
              <img src="./imgs/projections.png" alt="Real-world projection examples demonstrating the effectiveness" className="figure" />
              <p className="figure-caption">
                <strong>Figure 3</strong>: Example projections demonstrating how our dataset-adaptive approach achieves comparable visual quality to exhaustive optimization while requiring significantly less computation time across different datasets.
              </p>
            </div>
          </section>

          <section className="section">
            <h2>Citation</h2>
            <div className="citation-container">
              <p>If you use this work in your research, please cite:</p>
              <div className="citation-box">
                <code>
                  TBA (To Be Announced)
                </code>
              </div>
              <p className="citation-note">
                <em>The paper is currently under review. Citation information will be updated upon acceptance.</em>
              </p>
            </div>
          </section>

          <section className="section">
            <h2>Contact</h2>
            <ul className="contact-list">
              <li>Hyeon Jeon: hj@hcil.snu.ac.kr</li>
            </ul>
          </section>

          <section className="section faq-section">
            <h2>Frequently Asked Questions</h2>
            
            <h3>How do I choose between PDS and MNC?</h3>
            <p>
              While we recommend using both metrics together (PDS+MNC) for best results, understanding their individual strengths helps in special cases. PDS excels at capturing global patterns and works well when you're interested in overall data structure or when using globally-oriented techniques like PCA or classical MDS. MNC shines for local structure analysis and pairs naturally with local techniques like t-SNE and UMAP. For most applications, the combined PDS+MNC score provides robust predictions across all DR techniques.
            </p>

            <h3>What's the computational overhead of computing complexity metrics?</h3>
            <p>
              The complexity metrics are designed to be negligible compared to DR optimization time. For a dataset with 10,000 points and 100 dimensions, computing both PDS and MNC typically takes less than a second on a modern CPU. The metrics scale roughly as O(n²) with the number of points, but our implementation uses efficient parallelization and optional sampling strategies for very large datasets. In practice, the time spent computing metrics is recovered many times over through optimized DR computation.
            </p>

            <h3>Can I use this with my custom DR technique?</h3>
            <p>
              Yes! The framework is extensible to any DR technique. You'll need to provide a wrapper that follows our simple interface and collect training data to calibrate the prediction models. We provide utilities to help with this process, and the community has already contributed wrappers for several additional techniques beyond our original set.
            </p>

            <h3>How accurate are the performance predictions?</h3>
            <p>
              Our regression models achieve R² values above 0.8 for most technique-metric combinations, meaning they explain over 80% of the variance in DR performance. The predictions are most accurate for well-studied techniques (t-SNE, UMAP, PCA) and become more reliable as more training data is collected. Even when predictions aren't perfect, they're accurate enough to guide optimization decisions effectively.
            </p>
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
