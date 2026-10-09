import {
  type ChangeEvent,
  type DragEvent,
  type SubmitEvent,
  useEffect,
  useState,
} from 'react';

import './App.css';

import {
  getInspectionStatus,
  inspectProduct,
  type InspectionResult,
  type InspectionStatus,
} from './services/inspection-api';


type ViewMode =
  | 'original'
  | 'heatmap'
  | 'overlay';


function App() {
  const [status, setStatus] =
    useState<InspectionStatus | null>(null);

  const [selectedFile, setSelectedFile] =
    useState<File | null>(null);

  const [inspectionResult, setInspectionResult] =
    useState<InspectionResult | null>(null);

  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  const [isInspecting, setIsInspecting] =
    useState(false);

  const [isDragging, setIsDragging] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const [activeView, setActiveView] =
    useState<ViewMode>('original');


  useEffect(() => {
    async function loadStatus() {
      try {
        const result = await getInspectionStatus();

        setStatus(result);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : 'Unknown error',
        );
      }
    }

    void loadStatus();
  }, []);


  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);


  function selectFile(
    file: File | null,
  ) {
    if (
      file
      && !file.type.startsWith('image/')
    ) {
      setError(
        'The selected file must be an image.',
      );

      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(file);
    setInspectionResult(null);
    setError(null);
    setActiveView('original');

    if (file) {
      setPreviewUrl(
        URL.createObjectURL(file),
      );
    } else {
      setPreviewUrl(null);
    }
  }


  function handleFileChange(
    event: ChangeEvent<HTMLInputElement>,
  ) {
    const file =
      event.target.files?.[0] ?? null;

    selectFile(file);
  }


  function handleDrop(
    event: DragEvent<HTMLLabelElement>,
  ) {
    event.preventDefault();

    setIsDragging(false);

    const file =
      event.dataTransfer.files?.[0] ?? null;

    selectFile(file);
  }


  async function handleSubmit(
    event: SubmitEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    if (!selectedFile) {
      setError('Select an image first.');
      return;
    }

    try {
      setIsInspecting(true);
      setError(null);

      const result = await inspectProduct(
        selectedFile,
      );

      setInspectionResult(result);

      // Após a inspeção, mostra automaticamente
      // a visualização mais útil para o operador.
      setActiveView('overlay');
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Unknown error',
      );
    } finally {
      setIsInspecting(false);
    }
  }


  function getActiveImage(): string | null {
    if (!inspectionResult) {
      return previewUrl;
    }

    if (activeView === 'heatmap') {
      return (
        `data:image/png;base64,` +
        inspectionResult.heatmapBase64
      );
    }

    if (activeView === 'overlay') {
      return (
        `data:image/png;base64,` +
        inspectionResult.overlayBase64
      );
    }

    return previewUrl;
  }


  const activeImage = getActiveImage();

  const decisionClass =
    inspectionResult?.decision === 'APPROVED'
      ? 'decision-approved'
      : 'decision-rejected';


  return (
    <div className="app">
      <div className="app-container">

        {/* Top bar */}
        <header className="topbar">
          <div className="brand">
            <div className="brand-icon">
              V
            </div>

            <div>
              <strong className="brand-name">
                VisionInspect
              </strong>

              <span className="brand-subtitle">
                AI Quality Control
              </span>
            </div>
          </div>


          <div className="topbar-status">
            <span className="status-dot" />

            <span>
              {status
                ? status.message
                : 'Connecting...'}
            </span>

            <span className="gpu-badge">
              GPU
            </span>
          </div>
        </header>


        {/* Workspace */}
        <div className="workspace">

          {/* Sidebar */}
          <aside className="sidebar">

            <div className="sidebar-heading">
              <span className="sidebar-eyebrow">
                Inspection
              </span>

              <h1>
                Product Analysis
              </h1>

              <p>
                Upload a product image and run
                anomaly detection using the trained
                reference model.
              </p>
            </div>


            <form
              className="inspection-form"
              onSubmit={handleSubmit}
            >
              <label
                className={
                  isDragging
                    ? 'upload-zone dragging'
                    : 'upload-zone'
                }
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => {
                  setIsDragging(false);
                }}
                onDrop={handleDrop}
              >
                <span className="upload-icon">
                  +
                </span>

                <span className="upload-title">
                  Drop image here or click to browse
                </span>

                <span className="upload-hint">
                  PNG, JPG or JPEG
                </span>

                <input
                  className="upload-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </label>


              {selectedFile && (
                <div className="selected-file">
                  <span className="selected-file-indicator" />

                  <div className="selected-file-info">
                    <strong>
                      {selectedFile.name}
                    </strong>

                    <span>
                      {(
                        selectedFile.size
                        / 1024
                      ).toFixed(1)} KB
                    </span>
                  </div>
                </div>
              )}


              <button
                className="inspect-button"
                type="submit"
                disabled={
                  !selectedFile
                  || isInspecting
                }
              >
                {isInspecting
                  ? 'Analyzing...'
                  : 'Run Inspection'}
              </button>
            </form>


            {error && (
              <div className="error-message">
                {error}
              </div>
            )}


            <div className="sidebar-result">
              <div className="sidebar-section-title">
                Result
              </div>


              {!inspectionResult && (
                <div className="waiting-result">
                  Run an inspection to view
                  the analysis.
                </div>
              )}


              {inspectionResult && (
                <>
                  <div className="decision-row">
                    <span
                      className={
                        `decision-badge ${decisionClass}`
                      }
                    >
                      {inspectionResult.decision}
                    </span>
                  </div>


                  <div className="metric-list">
                    <div className="metric-row">
                      <span>
                        Anomaly score
                      </span>

                      <strong>
                        {inspectionResult.score.toFixed(4)}
                      </strong>
                    </div>


                    <div className="metric-row">
                      <span>
                        Threshold
                      </span>

                      <strong>
                        {inspectionResult.threshold.toFixed(4)}
                      </strong>
                    </div>
                  </div>
                </>
              )}
            </div>

          </aside>


          {/* Inspection viewport */}
          <section className="inspection-viewport">

            <div className="viewport-header">
              <div>
                <span className="viewport-eyebrow">
                  Visual Analysis
                </span>

                <h2>
                  Inspection View
                </h2>
              </div>


              {inspectionResult && (
                <div className="view-tabs">

                  <button
                    type="button"
                    className={
                      activeView === 'original'
                        ? 'view-tab active'
                        : 'view-tab'
                    }
                    onClick={() => {
                      setActiveView('original');
                    }}
                  >
                    Original
                  </button>


                  <button
                    type="button"
                    className={
                      activeView === 'heatmap'
                        ? 'view-tab active'
                        : 'view-tab'
                    }
                    onClick={() => {
                      setActiveView('heatmap');
                    }}
                  >
                    Heatmap
                  </button>


                  <button
                    type="button"
                    className={
                      activeView === 'overlay'
                        ? 'view-tab active'
                        : 'view-tab'
                    }
                    onClick={() => {
                      setActiveView('overlay');
                    }}
                  >
                    Overlay
                  </button>

                </div>
              )}
            </div>


            <div className="viewport-body">

              {!activeImage && (
                <div className="empty-viewport">
                  <div className="empty-viewport-icon">
                    ◇
                  </div>

                  <h3>
                    No image selected
                  </h3>

                  <p>
                    Select or drop a product image
                    from the inspection panel.
                  </p>
                </div>
              )}


              {activeImage && (
                <div className="inspection-stage">

                  <img
                    className="inspection-image"
                    src={activeImage}
                    alt={`Inspection ${activeView}`}
                  />


                  {inspectionResult
                    && inspectionResult.decision === 'REJECTED'
                    && activeView === 'overlay'
                    && inspectionResult.boundingBoxes.length > 0
                    && (
                      <svg
                        className="bounding-box-layer"
                        viewBox={
                          `0 0 ${inspectionResult.imageWidth} ${inspectionResult.imageHeight}`
                        }
                        preserveAspectRatio="xMidYMid meet"
                        aria-hidden="true"
                      >
                        {inspectionResult.boundingBoxes.map(
                          (box, index) => (
                            <rect
                              key={
                                `${box.x}-${box.y}-${index}`
                              }
                              className="defect-box"
                              x={box.x}
                              y={box.y}
                              width={box.width}
                              height={box.height}
                              vectorEffect="non-scaling-stroke"
                            />
                          ),
                        )}
                      </svg>
                    )}

                </div>
              )}


              {isInspecting && (
                <div className="inspection-loading">
                  <div className="loading-spinner" />

                  <span>
                    Running GPU inference...
                  </span>
                </div>
              )}

            </div>


            <footer className="viewport-footer">
              <span>
                {inspectionResult
                  ? (
                    <>
                      View:{' '}
                      <strong>
                        {activeView}
                      </strong>
                    </>
                  )
                  : 'Waiting for inspection'}
              </span>

              <span>
                Detection model ready
              </span>
            </footer>

          </section>

        </div>
      </div>
    </div>
  );
}


export default App;