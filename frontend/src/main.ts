// Entry point. Bootstraps the controller and guards startup so an error
// surfaces in the console instead of a silently blank page.

import { FinThesisApp } from "./app";

try {
  FinThesisApp.mount("#app");
} catch (err) {
  console.error("Failed to start the FinThesis UI", err);
}
