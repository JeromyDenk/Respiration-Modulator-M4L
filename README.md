{
    "profile_name": "Test Profile with EVM Disabled",
    "description": "Configuration for testing the pipeline with default settings, Nth frame pose detection, and EVM refinement disabled initially.",
  
    "pipeline_manager": {
      "POSE_DETECTION_FRAME_INTERVAL": 30,
      "PIPELINE_RECALIBRATION_INTERVAL_SEC": 300
    },
  
    "pose_detector": {
      "POSE_MODEL_COMPLEXITY": 0,
      "POSE_STATIC_IMAGE_MODE": false,
      "POSE_MIN_DETECTION_CONFIDENCE": 0.5,
      "POSE_MIN_TRACKING_CONFIDENCE": 0.5
    },
  
    "coarse_roi_calculator": {
      "ROI_STRATEGY": "single_chest_abdomen",
      "ROI_PADDING_FACTOR": 1.05,
      "POSE_MIN_LANDMARK_VISIBILITY": 0.6,
      "ROI_SHOULDER_ONLY_ASPECT_RATIO": 1.8
      // ROI_LANDMARKS_SHOULDERS and ROI_LANDMARKS_HIPS usually use MediaPipe defaults, no need to override unless necessary
    },
  
    "evm_processor": {
      "EVM_ENABLED": false, // Set to true to enable EVM refinement
      "EVM_BUFFER_SECONDS": 2.0, // How much history for EVM analysis (used by PipelineManager)
      "EVM_REFINED_ROI_SIZE_FACTOR": 0.3, // Size of refined ROI relative to coarse ROI min dimension
      "EVM_MIN_BUFFER_FRAMES": 15, // Min frames needed in buffer for EVM analysis (can override calculated default)
      "EVM_ROI_SCORING_METHOD": "pixel_variance", // Currently only 'pixel_variance' supported
      // Parameters for full EVM (if implemented later)
      "EVM_PYRAMID_LEVELS": 4,
      "EVM_TEMPORAL_FILTER_LOW_HZ": 0.1, // Match signal processor
      "EVM_TEMPORAL_FILTER_HIGH_HZ": 2.0, // Match signal processor
      "EVM_ALPHA": 50, // Amplification factor
      "EVM_ROI_SELECTION_LEVEL": 3 // Which pyramid level to analyze
    },
  
    "feature_tracker": {
      "OPTICAL_FLOW_PARAMS": {
        "feature_params": {
          "maxCorners": 100,
          "qualityLevel": 0.3,
          "minDistance": 7,
          "blockSize": 7
        },
        "lk_params": {
          "winSize": [15, 15], // Use list format for JSON compatibility
          "maxLevel": 2,
          // Criteria: (type, max_iter, epsilon) - Use integer for type flags
          // cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT = 1 | 2 = 3
          "criteria": [3, 10, 0.03]
        }
      },
      "FEATURE_REDETECT_THRESHOLD": 30 // Redetect if fewer than 30 features remain
    },
  
    "signal_generator": {
      "SIGNAL_MIN_FEATURES_FOR_PCA": 3,
      "SIGNAL_PCA_METHOD": "numpy" // Currently only 'numpy' supported
    },
  
    "signal_processor": {
      "SIGNAL_BUFFER_SECONDS": 10.0, // Analysis window
      "SIGNAL_FUSION_STRATEGY": "first", // 'first' or 'average'
      "SIGNAL_FILTER_METHOD": "lfilter", // 'lfilter' (causal) or 'filtfilt' (zero-phase)
      "SIGNAL_FILTER_TYPE": "butterworth",
      "SIGNAL_FILTER_ORDER": 2,
      "SIGNAL_FILTER_LOW_HZ": 0.1, // ~6 BPM
      "SIGNAL_FILTER_HIGH_HZ": 2.0, // ~120 BPM
      "PEAK_DETECT_MIN_HEIGHT": 0.0, // Set relative to filtered signal amplitude after testing
      "PEAK_DETECT_MIN_DISTANCE_SEC": 0.5, // Allows up to 120 BPM
      "PEAK_DETECT_PROMINENCE": null, // Set to a float value after visual tuning (e.g., 0.1, 0.2)
      "BPM_AVERAGING_SECONDS": 5.0, // Smoothing window for final BPM
      "PHASE_SLOPE_WINDOW_MS": 100 // Window for inhale/exhale detection
    },
  
    "osc_sender": {
      "OSC_IP_ADDRESS": "127.0.0.1",
      "OSC_PORT": 8888,
      "OSC_BPM_ADDRESS": "/respiration/bpm",
      "OSC_PHASE_ADDRESS": "/respiration/phase", // 1=inhale, -1=exhale, 0=unknown
      "OSC_VALIDITY_ADDRESS": "/respiration/valid" // 1=valid, 0=invalid
      // Add addresses for raw/filtered signal if needed
    },
  
    "ui": {
        "MATPLOTLIB_UI_ENABLED": true, // Set to false if running headless
        "PLOT_UPDATE_INTERVAL_MS": 100 // How often to redraw plots
    }
  
  }
  