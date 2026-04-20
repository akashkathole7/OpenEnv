# Scope Card — reconcile_gst2b_env

Scope fence for v1. Anything outside this list is a future-version concern;
the environment must not model, validate, or reward it.

## IN scope

- B2B domestic supply of **goods** (not services) under GST 2.0
- GST 2.0 slab structure: 0 / 5 / 18 / 40 primary; 3 / 12 / 28 / 0.25 residual
- GSTR-2B matching against the buyer's purchase register
- Rule 36(4) per-supplier ITC cap compliance
- Circular-trading ring detection across supplier GSTINs

## OUT of scope

- Reverse Charge Mechanism (RCM)
- Input Service Distributor (ISD)
- Supply to / from SEZ units
- Import of services or goods
- Composition dealers (GSTR-4 regime)
- E-invoicing / IRN generation
- E-way bill generation or validation

> HSN → slab mappings are synthetic; not a tax-advice tool.
