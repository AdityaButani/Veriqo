// Modern Chart.js configuration and utilities for AI Platform
// Enhanced with animations, responsive design, and interactive features

// Chart.js global defaults
Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
Chart.defaults.color = '#6b7280';
Chart.defaults.plugins.legend.position = 'bottom';
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.padding = 20;

// Dark mode support
const isDarkMode = () => document.documentElement.classList.contains('dark');

// Color schemes for different themes
const colorSchemes = {
    light: {
        primary: '#0ea5e9',
        accent: '#d946ef',
        success: '#22c55e',
        warning: '#f59e0b',
        danger: '#ef4444',
        neutral: '#6b7280',
        background: '#ffffff',
        surface: '#f9fafb',
        border: '#e5e7eb'
    },
    dark: {
        primary: '#38bdf8',
        accent: '#e879f9',
        success: '#4ade80',
        warning: '#fbbf24',
        danger: '#f87171',
        neutral: '#9ca3af',
        background: '#1f2937',
        surface: '#374151',
        border: '#4b5563'
    }
};

// Get current color scheme
const getColorScheme = () => isDarkMode() ? colorSchemes.dark : colorSchemes.light;

// Enhanced chart configurations
const chartConfigs = {
    // Doughnut chart for classification results
    classification: {
        type: 'doughnut',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: getColorScheme().background,
                    titleColor: getColorScheme().neutral,
                    bodyColor: getColorScheme().neutral,
                    borderColor: getColorScheme().border,
                    borderWidth: 1,
                    cornerRadius: 8,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                animateScale: true,
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    },

    // Bar chart for confidence distribution
    confidence: {
        type: 'bar',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: getColorScheme().background,
                    titleColor: getColorScheme().neutral,
                    bodyColor: getColorScheme().neutral,
                    borderColor: getColorScheme().border,
                    borderWidth: 1,
                    cornerRadius: 8
                }
            },
            scales: {
                x: {
                    grid: {
                        color: getColorScheme().border,
                        drawBorder: false
                    },
                    ticks: {
                        color: getColorScheme().neutral,
                        font: {
                            size: 12
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: getColorScheme().border,
                        drawBorder: false
                    },
                    ticks: {
                        color: getColorScheme().neutral,
                        font: {
                            size: 12
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    },

    // Line chart for timeline analysis
    timeline: {
        type: 'line',
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: {
                            size: 12,
                            weight: '500'
                        }
                    }
                },
                tooltip: {
                    backgroundColor: getColorScheme().background,
                    titleColor: getColorScheme().neutral,
                    bodyColor: getColorScheme().neutral,
                    borderColor: getColorScheme().border,
                    borderWidth: 1,
                    cornerRadius: 8,
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: getColorScheme().border,
                        drawBorder: false
                    },
                    ticks: {
                        color: getColorScheme().neutral,
                        font: {
                            size: 12
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: getColorScheme().border,
                        drawBorder: false
                    },
                    ticks: {
                        color: getColorScheme().neutral,
                        font: {
                            size: 12
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    }
};

// Chart utility functions
const ChartUtils = {
    // Create classification chart
    createClassificationChart: (canvasId, data) => {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const colors = [
            getColorScheme().success,
            getColorScheme().danger,
            getColorScheme().warning,
            getColorScheme().primary
        ];

        return new Chart(ctx, {
            type: chartConfigs.classification.type,
            data: {
                labels: data.labels,
                datasets: [{
                    data: data.values,
                    backgroundColor: colors,
                    borderColor: getColorScheme().background,
                    borderWidth: 3,
                    hoverBorderWidth: 4,
                    hoverOffset: 10
                }]
            },
            options: chartConfigs.classification.options
        });
    },

    // Create confidence distribution chart
    createConfidenceChart: (canvasId, data) => {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        return new Chart(ctx, {
            type: chartConfigs.confidence.type,
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Number of Reviews',
                    data: data.values,
                    backgroundColor: getColorScheme().primary,
                    borderColor: getColorScheme().primary,
                    borderWidth: 1,
                    borderRadius: 6,
                    hoverBackgroundColor: getColorScheme().accent,
                    hoverBorderColor: getColorScheme().accent
                }]
            },
            options: chartConfigs.confidence.options
        });
    },

    // Create timeline chart
    createTimelineChart: (canvasId, data) => {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        return new Chart(ctx, {
            type: chartConfigs.timeline.type,
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Genuine Reviews',
                        data: data.genuine,
                        borderColor: getColorScheme().success,
                        backgroundColor: getColorScheme().success + '20',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: getColorScheme().success,
                        pointBorderColor: getColorScheme().background,
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Suspicious Reviews',
                        data: data.suspicious,
                        borderColor: getColorScheme().danger,
                        backgroundColor: getColorScheme().danger + '20',
                        borderWidth: 3,
                        fill: true,
                        tension: 0.4,
                        pointBackgroundColor: getColorScheme().danger,
                        pointBorderColor: getColorScheme().background,
                        pointBorderWidth: 2,
                        pointRadius: 6,
                        pointHoverRadius: 8
                    }
                ]
            },
            options: chartConfigs.timeline.options
        });
    },

    // Update chart colors for theme changes
    updateChartColors: (chart) => {
        if (!chart) return;
        
        const colors = getColorScheme();
        
        // Update chart colors based on type
        if (chart.config.type === 'doughnut') {
            chart.data.datasets[0].backgroundColor = [
                colors.success,
                colors.danger,
                colors.warning,
                colors.primary
            ];
            chart.data.datasets[0].borderColor = colors.background;
        } else if (chart.config.type === 'bar') {
            chart.data.datasets[0].backgroundColor = colors.primary;
            chart.data.datasets[0].borderColor = colors.primary;
            chart.data.datasets[0].hoverBackgroundColor = colors.accent;
            chart.data.datasets[0].hoverBorderColor = colors.accent;
        } else if (chart.config.type === 'line') {
            chart.data.datasets[0].borderColor = colors.success;
            chart.data.datasets[0].backgroundColor = colors.success + '20';
            chart.data.datasets[0].pointBackgroundColor = colors.success;
            
            if (chart.data.datasets[1]) {
                chart.data.datasets[1].borderColor = colors.danger;
                chart.data.datasets[1].backgroundColor = colors.danger + '20';
                chart.data.datasets[1].pointBackgroundColor = colors.danger;
            }
        }

        // Update tooltip colors
        chart.options.plugins.tooltip.backgroundColor = colors.background;
        chart.options.plugins.tooltip.titleColor = colors.neutral;
        chart.options.plugins.tooltip.bodyColor = colors.neutral;
        chart.options.plugins.tooltip.borderColor = colors.border;

        // Update scale colors
        chart.options.scales.x.grid.color = colors.border;
        chart.options.scales.x.ticks.color = colors.neutral;
        chart.options.scales.y.grid.color = colors.border;
        chart.options.scales.y.ticks.color = colors.neutral;

        chart.update();
    },

    // Animate chart entrance
    animateChart: (chart, delay = 0) => {
        if (!chart) return;
        
        setTimeout(() => {
            const canvas = chart.canvas;
            canvas.style.opacity = '0';
            canvas.style.transform = 'translateY(20px)';
            
            requestAnimationFrame(() => {
                canvas.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
                canvas.style.opacity = '1';
                canvas.style.transform = 'translateY(0)';
            });
        }, delay);
    },

    // Create loading state for charts
    createLoadingState: (canvasId) => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 30;

        let angle = 0;
        
        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.save();
            ctx.translate(centerX, centerY);
            ctx.rotate(angle);
            
            ctx.strokeStyle = getColorScheme().primary;
            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.arc(0, 0, radius, 0, Math.PI * 1.5);
            ctx.stroke();
            
            ctx.restore();
            
            angle += 0.1;
            requestAnimationFrame(animate);
        };
        
        animate();
    }
};

// Theme change handler
const handleThemeChange = () => {
    // Update all existing charts
    Object.values(Chart.instances).forEach(chart => {
        ChartUtils.updateChartColors(chart);
    });
};

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Listen for theme changes
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                handleThemeChange();
            }
        });
    });

    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['class']
    });

    // Initialize any charts that are already in the DOM
    const chartCanvases = document.querySelectorAll('canvas[data-chart]');
    chartCanvases.forEach(canvas => {
        const chartType = canvas.dataset.chart;
        const chartData = JSON.parse(canvas.dataset.chartData || '{}');
        
        let chart = null;
        switch (chartType) {
            case 'classification':
                chart = ChartUtils.createClassificationChart(canvas.id, chartData);
                break;
            case 'confidence':
                chart = ChartUtils.createConfidenceChart(canvas.id, chartData);
                break;
            case 'timeline':
                chart = ChartUtils.createTimelineChart(canvas.id, chartData);
                break;
        }
        
        if (chart) {
            ChartUtils.animateChart(chart, 200);
        }
    });
});

// Export utilities for use in other scripts
window.ChartUtils = ChartUtils; 