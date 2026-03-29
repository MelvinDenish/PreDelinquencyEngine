const fs = require("fs");
const d = JSON.parse(fs.readFileSync("/tmp/pdi_workflow.json", "utf8"));
d.id = "1";
d.active = true;
fs.writeFileSync("/tmp/pdi_fixed.json", JSON.stringify(d));
console.log("Fixed workflow written with id=1, active=true");
