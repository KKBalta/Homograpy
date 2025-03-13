\documentclass{article}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{xcolor}

\title{Homography Project}
\author{}
\date{}

\begin{document}

\maketitle

\section{Introduction}
This project implements various homography-based computer vision techniques, including feature detection, matching, homography computation, and applications such as image warping and augmented reality.

\section{Project Structure}
\begin{verbatim}
Homograpy/
├── data/            # Input data (images and videos)
├── figs/            # Image figures for documentation
├── python/          # Source code
│   ├── ar.py             # Augmented reality implementation
│   ├── briefRotTest.py   # Tests feature matching under rotation
│   ├── HarryPotterize.py # Replace book cover using homography
│   ├── helper.py         # Utility functions for feature detection
│   ├── loadSaveVid.py    # Video I/O functions
│   ├── matchPics.py      # Image feature matching functions
│   ├── panaroma.py       # Panorama stitching (stub)
│   ├── planarH.py        # Homography calculation implementations
│   ├── q3_4.py           # Feature matching example script
│   └── test_computeH.py  # Test for homography computation
├── requirements.txt      # Required Python packages
└── results/         # Output directory for results
\end{verbatim}

\section{Installation}
To set up the environment, install the required dependencies:
\begin{lstlisting}
pip install -r requirements.txt
\end{lstlisting}

\section{Features and Implementations}

\subsection{Feature Detection and Matching}
The project utilizes FAST corner detection and BRIEF descriptors for feature extraction and matching:
\begin{itemize}
    \item \texttt{corner\_detection} detects corners using the FAST algorithm.
    \item \texttt{computeBrief} computes BRIEF descriptors for features.
    \item \texttt{matchPics} matches features between images.
\end{itemize}

\subsection{Homography Computation}
The project implements multiple homography computation methods with increasing robustness:
\begin{itemize}
    \item \texttt{computeH}: Direct Linear Transform (DLT) for computing homography.
    \item \texttt{computeH\_norm}: Normalized DLT for improved numerical stability.
    \item \texttt{computeH\_ransac}: RANSAC-based homography computation to handle outliers.
\end{itemize}

\subsection{Image Warping and Compositing}
\texttt{compositeH} applies a homography to overlay one image onto another.

\section{Applications}

\subsection{Feature Matching Visualization}
To run the feature matching demonstration:
\begin{lstlisting}
cd python
python q3_4.py
\end{lstlisting}
This matches features between book cover and desk images and visualizes the matches.

\subsection{Harry Potter Book Cover Replacement}
Replace a book cover in an image with another using homography:
\begin{lstlisting}
cd python
python HarryPotterize.py
\end{lstlisting}

\subsection{Feature Matching under Rotation}
Test the robustness of feature matching under rotation:
\begin{lstlisting}
cd python
python briefRotTest.py
\end{lstlisting}
This rotates an image at different angles and plots the number of matches to assess descriptor robustness.

\subsection{Augmented Reality}
The \texttt{ar.py} script overlays video content onto a book cover in a video sequence, implementing an augmented reality effect.
\begin{lstlisting}
cd python
python ar.py
\end{lstlisting}

\section{Testing}
Test the homography computation with:
\begin{lstlisting}
cd python
python test_computeH.py
\end{lstlisting}
This verifies the correctness of the homography computation by mapping known points.

\section{Dependencies}
This project relies on several Python libraries:
\begin{itemize}
    \item OpenCV (\texttt{cv2})
    \item NumPy
    \item SciPy
    \item scikit-image
    \item Matplotlib
\end{itemize}
All dependencies are listed in \texttt{requirements.txt}.

\section{Future Work}
\begin{itemize}
    \item Complete the panorama stitching functionality in \texttt{panaroma.py}.
    \item Optimize the augmented reality implementation for real-time performance.
\end{itemize}

\section{License}
This project is open-source and free to use for educational and research purposes.

\end{document}
