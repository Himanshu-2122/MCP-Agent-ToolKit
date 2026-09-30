import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("real-tools")


@mcp.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city name. Temperature in Celsius, wind in km/h."""
    # Step A: sheher ka naam -> latitude/longitude
    geo = httpx.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    if not geo.get("results"):
        return {"error": f"City '{city}' not found"}
    place = geo["results"][0]

    # Step B: lat/long -> current weather
    weather = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        },
        timeout=10,
    ).json()
    cur = weather["current"]

    return {
        "city": place["name"],
        "country": place.get("country"),
        "temperature_c": cur["temperature_2m"],
        "humidity_percent": cur["relative_humidity_2m"],
        "wind_kmh": cur["wind_speed_10m"],
    }


@mcp.tool()
def get_github_repos(username: str, limit: int = 5) -> list | dict:
    """List a GitHub user's most recently updated public repositories."""
    r = httpx.get(
        f"https://api.github.com/users/{username}/repos",
        params={"sort": "updated", "per_page": limit},
        timeout=10,
    )
    if r.status_code != 200:
        return {"error": f"GitHub returned status {r.status_code}"}

    return [
        {
            "name": repo["name"],
            "description": repo["description"],
            "stars": repo["stargazers_count"],
            "language": repo["language"],
            "url": repo["html_url"],
        }
        for repo in r.json()
    ]


@mcp.tool()
def get_hackernews_top(limit: int = 5) -> list:
    """Get the current top stories from Hacker News (tech and AI news)."""
    ids = httpx.get(
        "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
    ).json()[:limit]

    stories = []
    for story_id in ids:
        s = httpx.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", timeout=10
        ).json()
        stories.append(
            {"title": s.get("title"), "url": s.get("url"), "score": s.get("score")}
        )
    return stories


if __name__ == "__main__":
    mcp.run()