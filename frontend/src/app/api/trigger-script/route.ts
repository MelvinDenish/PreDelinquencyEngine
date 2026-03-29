import { NextRequest, NextResponse } from "next/server";
import { exec } from "child_process";
import path from "path";

export async function POST(req: NextRequest): Promise<NextResponse> {
  return new Promise<NextResponse>((resolve) => {
    // Determine the root of the project. Assuming frontend is at root/frontend
    const scriptPath = path.resolve(process.cwd(), "..", "safety_workflow.py");
    const frontendUrl = `${req.nextUrl.origin}/api/workflow-event`;

    exec(`python "${scriptPath}" --once`, {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', FRONTEND_URL: frontendUrl }
    }, (error, stdout, stderr) => {
      if (error) {
        console.error("Error executing Python script:", error);
        console.error("Stderr:", stderr);
        resolve(NextResponse.json({ status: "error", message: error.message }, { status: 500 }));
        return;
      }
      
      resolve(NextResponse.json({ status: "ok", output: stdout }));
    });
  });
}
