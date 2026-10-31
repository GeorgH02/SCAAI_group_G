
"""Convert internal evidence into a restricted public API response."""

SAFE_LISTING_FIELDS = {
    "listing_id",
    "host_id",
    "neighbourhood",
    "room_type",
    "accommodates",
    "bedrooms",
    "price",
    "host_is_superhost",
    "snapshot_date",
    "distance_m",
    "lnight_road_band",
    "lnight_rail_band",
}

SAFE_GROUP_FIELDS = {
    "host_is_superhost",
    "listing_count",
    "priced_listing_count",
    "missing_price_count",
    "average_base_price",
}

SAFE_AVAILABILITY_FIELDS = {
    "listing_id",
    "status",
    "failures",
    "unknowns",
}


def public_evidence(results):
    output = []

    for result in results:
        if not isinstance(result, dict):
            continue

        item = {"tool": result.get("tool")}

        if "rows" in result:
            item["rows"] = [
                {key: value for key, value in row.items()
                 if key in SAFE_LISTING_FIELDS}
                for row in result["rows"]
                if isinstance(row, dict)
            ]
            item["total_matches"] = result.get("total_matches")
            item["has_more"] = result.get("has_more")

        if "groups" in result:
            item["groups"] = [
                {
                    **{key: group.get(key) for key in SAFE_GROUP_FIELDS},
                    "listing_ids": group.get("listing_ids", [])[:20],
                }
                for group in result["groups"]
                if isinstance(group, dict)
            ]

        if "results" in result:
            item["availability"] = [
                {key: stay.get(key) for key in SAFE_AVAILABILITY_FIELDS}
                for stay in result["results"]
                if isinstance(stay, dict)
            ]

        output.append(item)

    return output
