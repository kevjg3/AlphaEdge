"use client";

interface Props {
  matrix: Record<string, Record<string, number>>;
  labels: Record<string, string>;
}

function getColor(val: number): string {
  // -1 = deep red, 0 = neutral gray, +1 = deep blue/green
  if (val >= 0.8) return "bg-emerald-500/80 text-white";
  if (val >= 0.5) return "bg-emerald-600/50 text-emerald-100";
  if (val >= 0.2) return "bg-emerald-800/30 text-emerald-200";
  if (val >= -0.2) return "bg-gray-800 text-gray-300";
  if (val >= -0.5) return "bg-red-800/30 text-red-200";
  if (val >= -0.8) return "bg-red-600/50 text-red-100";
  return "bg-red-500/80 text-white";
}

export default function CorrelationHeatmap({ matrix, labels }: Props) {
  if (!matrix || Object.keys(matrix).length === 0) return null;

  const keys = Object.keys(matrix);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="p-1.5 text-left text-gray-500 font-normal" />
            {keys.map((k) => (
              <th key={k} className="p-1.5 text-center text-gray-400 font-medium min-w-[52px]">
                {k === "ticker" ? labels.ticker || k : k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {keys.map((row) => (
            <tr key={row}>
              <td className="p-1.5 text-gray-400 font-medium whitespace-nowrap">
                {row === "ticker" ? labels.ticker || row : labels[row] || row}
              </td>
              {keys.map((col) => {
                const val = matrix[row]?.[col] ?? 0;
                return (
                  <td
                    key={col}
                    className={`p-1.5 text-center font-mono rounded-sm ${getColor(val)} ${row === col ? "ring-1 ring-gray-600" : ""}`}
                  >
                    {val.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
