"use client";

import { ReactNode } from "react";

export function Card({
  title,
  children,
  className = "",
  glow = false,
  compact = false,
}: {
  title?: string;
  children: ReactNode;
  className?: string;
  glow?: boolean;
  compact?: boolean;
}) {
  return (
    <div
      className={`
        bg-surface-raised border border-white/[0.06] rounded-2xl
        ${compact ? "p-4" : "p-5"}
        shadow-card transition-all duration-300
        hover:border-white/[0.1] hover:shadow-card-hover
        ${glow ? "animate-pulse-glow" : ""}
        ${className}
      `}
    >
      {title && (
        <h3 className="text-xs font-semibold text-gray-400 mb-3.5 uppercase tracking-[0.08em] flex items-center gap-2">
          <span className="w-1 h-3.5 bg-brand-500 rounded-full" />
          {title}
        </h3>
      )}
      {children}
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  color,
  icon,
  size = "default",
}: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  icon?: ReactNode;
  size?: "sm" | "default" | "lg";
}) {
  const valueSize = {
    sm: "text-sm",
    default: "text-lg",
    lg: "text-2xl",
  }[size];

  return (
    <div className="group">
      <div className="metric-label mb-1 flex items-center gap-1.5">
        {icon && <span className="text-gray-600">{icon}</span>}
        {label}
      </div>
      <div className={`${valueSize} font-bold font-mono tracking-tight tabular-nums ${color || "text-white"}`}>
        {value}
      </div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

export function Badge({
  text,
  variant = "neutral",
  dot = false,
  size = "default",
}: {
  text: string;
  variant?: "positive" | "negative" | "neutral" | "warning" | "info";
  dot?: boolean;
  size?: "sm" | "default";
}) {
  const colors = {
    positive: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    negative: "bg-red-500/10 text-red-400 border-red-500/20",
    neutral: "bg-gray-500/10 text-gray-400 border-gray-500/20",
    warning: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    info: "bg-sky-500/10 text-sky-400 border-sky-500/20",
  };

  const dotColors = {
    positive: "bg-emerald-400",
    negative: "bg-red-400",
    neutral: "bg-gray-400",
    warning: "bg-amber-400",
    info: "bg-sky-400",
  };

  const sizeClasses = size === "sm" ? "px-1.5 py-px text-[10px]" : "px-2.5 py-0.5 text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 ${sizeClasses} rounded-full border font-medium ${colors[variant]}`}
    >
      {dot && <span className={`w-1.5 h-1.5 rounded-full ${dotColors[variant]}`} />}
      {text}
    </span>
  );
}

export function Skeleton({
  className = "",
  lines = 1,
}: {
  className?: string;
  lines?: number;
}) {
  if (lines > 1) {
    return (
      <div className="space-y-2">
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={`skeleton h-4 ${i === lines - 1 ? "w-3/4" : "w-full"} ${className}`}
          />
        ))}
      </div>
    );
  }
  return <div className={`skeleton h-4 w-full ${className}`} />;
}

export function EmptyState({
  message,
  icon,
}: {
  message: string;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-500 animate-fade-in">
      {icon && <div className="text-3xl mb-3 text-gray-600">{icon}</div>}
      <p className="text-sm">{message}</p>
    </div>
  );
}
