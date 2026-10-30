from qloo_client import (
    QlooAPIError,
    get_movie_recommendations,
)


def main():
    try:
        response = get_movie_recommendations(
            taste_tags=[
                "urn:tag:theme:qloo:ambition",
                "urn:tag:character:qloo:determined",
            ],
            required_tags=[
                "urn:tag:subgenre:qloo:sports_drama",
            ],
            limit=10,
        )

        movies = response.get("results", {}).get("entities", [])

        print("API request successful!")
        print("Number of movies returned:", len(movies))
        print()

        for index, movie in enumerate(movies, start=1):
            print(f"\nMovie {index}")
            print("Name:", movie.get("name"))
            print("Entity ID:", movie.get("entity_id"))
            print("Affinity:", movie.get("query", {}).get("affinity"))
            print("Properties:", movie.get("properties"))
            print("Tags:", movie.get("tags"))
            print("Available fields:", list(movie.keys()))
    except (QlooAPIError, ValueError) as exc:
        print("Request failed:", exc)


if __name__ == "__main__":
    main()