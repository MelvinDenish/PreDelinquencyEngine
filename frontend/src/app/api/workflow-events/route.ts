/**
 * GET /api/workflow-events?since=<ISO-timestamp>&limit=<n>
 * Returns the latest n8n pipeline events for the frontend live panel.
 * The frontend polls this every 3 s to show the live workflow feed.
 */
import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const limit = Math.min(parseInt(searchParams.get("limit") ?? "20", 10), 50);
  const since = searchParams.get("since");

  let events = [];
  try {
    const dataPath = path.resolve(process.cwd(), "pdi_events.json");
    if (fs.existsSync(dataPath)) {
      events = JSON.parse(fs.readFileSync(dataPath, "utf-8"));
    }
  } catch (e) {
    console.error("Failed to read events file", e);
  }

  if (since) {
    const sinceMs = new Date(since).getTime();
    events = events.filter((e: any) => new Date(e.event_time).getTime() > sinceMs);
  }

  return NextResponse.json({
    events: events.slice(0, limit),
    total: events.length,
    fetched_at: new Date().toISOString(),
  });
}
