// Friendly labels for the scrub categories the backend reports (from harness models).

const CATEGORY_LABELS: Record<string, string> = {
  name: "Name",
  email: "Email",
  phone: "Phone",
  address: "Address",
  location: "Location",
  employer: "Employer",
  account_id: "Account / ID",
  secret: "Secret",
  exact_amount: "Amount",
  exact_date: "Date",
  medical_specifics: "Medical",
  legal_specifics: "Legal",
  other_identifier: "Identifier",
  age: "Age",
  dob: "Date of birth",
  handle: "Handle",
}

export function categoryLabel(category: string): string {
  return CATEGORY_LABELS[category] ?? category
}
