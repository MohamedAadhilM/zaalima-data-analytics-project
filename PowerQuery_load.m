// =====================================================================================
// POWER QUERY (M) — loads all 20 tables from the CSV folder.
// How to use:
//  1. Home > Transform data > New Source > Blank Query, name it  fnLoad,  paste the FIRST block, Done.
//  2. New Parameter (Manage Parameters) named  DataFolder  (Text), value e.g.  C:\causal-pricing-engine\powerbi\data\
//     (must end with a backslash).
//  3. For each table below create a Blank Query with that exact name and paste its one-line body:
//        let Source = fnLoad("DimMarket") in Source
//  4. Close & Apply.
// =====================================================================================

// ---------- QUERY NAME:  fnLoad  ----------------------------------------------------
(tableName as text) as table =>
let
    Raw     = Csv.Document(File.Contents(DataFolder & tableName & ".csv"),
                           [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headed  = Table.PromoteHeaders(Raw, [PromoteAllScalars = true]),
    Typed   = List.Accumulate(
        Table.ColumnNames(Headed), Headed,
        (state, col) =>
            let
                vals   = List.Select(Table.Column(state, col), each _ <> null and _ <> ""),
                isBool = List.Count(vals) > 0 and List.AllTrue(List.Transform(vals, each _ = "True" or _ = "False")),
                isDate = List.Count(vals) > 0 and List.AllTrue(List.Transform(vals,
                            each (try Date.FromText(_, [Format = "yyyy-MM-dd", Culture = "en-US"]) otherwise null) <> null)),
                isNum  = List.Count(vals) > 0 and List.AllTrue(List.Transform(vals,
                            each (try Number.FromText(_, "en-US") otherwise null) <> null))
            in
                if isBool then Table.TransformColumns(state, {{col, each if _ = null or _ = "" then null else Logical.FromText(_), type logical}})
                else if isDate then Table.TransformColumnTypes(state, {{col, type date}}, "en-US")
                else if isNum  then Table.TransformColumnTypes(state, {{col, type number}}, "en-US")
                else Table.TransformColumnTypes(state, {{col, type text}})
    )
in
    Typed

// ---------- ONE QUERY PER TABLE (query name = table name) ---------------------------
// DimDate            let Source = fnLoad("DimDate") in Source
// DimMarket          let Source = fnLoad("DimMarket") in Source
// FactMarketWeek     let Source = fnLoad("FactMarketWeek") in Source
// FactCounterfactual let Source = fnLoad("FactCounterfactual") in Source
// FactPlacebo        let Source = fnLoad("FactPlacebo") in Source
// SegmentEffect      let Source = fnLoad("SegmentEffect") in Source
// CateMarket         let Source = fnLoad("CateMarket") in Source
// Estimators         let Source = fnLoad("Estimators") in Source
// Summary            let Source = fnLoad("Summary") in Source
// MspeFit            let Source = fnLoad("MspeFit") in Source
// PlaceboInTime      let Source = fnLoad("PlaceboInTime") in Source
// LeaveOneOut        let Source = fnLoad("LeaveOneOut") in Source
// Refuters           let Source = fnLoad("Refuters") in Source
// CvFolds            let Source = fnLoad("CvFolds") in Source
// RollingCv          let Source = fnLoad("RollingCv") in Source
// RobustnessValue    let Source = fnLoad("RobustnessValue") in Source
// OvbGrid            let Source = fnLoad("OvbGrid") in Source
// WhatIfGrid         let Source = fnLoad("WhatIfGrid") in Source
// ReviewChecklist    let Source = fnLoad("ReviewChecklist") in Source
// DataQuality        let Source = fnLoad("DataQuality") in Source
