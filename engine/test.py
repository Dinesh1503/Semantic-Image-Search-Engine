from inference import Inference
from database.db import VectorDatabase
from paths import images_dir
from storage import load_records, save_records


def build_records(inf: Inference):
    """Caption, vectorise and persist every image in `images_dir()` to data.json."""
    indexes = inf.get_index(str(images_dir()))
    captions = inf.get_image_captions(indexes)
    vectors = inf.vectorise_captions(captions)
    captured_at = [inf.get_image_datetime(i["image_path"]) for i in indexes]

    save_records(
        [
            (index["image_filename"], index["image_path"], caption, vector, meta)
            for index, caption, vector, meta in zip(indexes, captions, vectors, captured_at)
        ]
    )


def populate_db(db: VectorDatabase):
    file_data = load_records()

    print(len(file_data))

    db.connect()
    db.check_data()

    db.delete_all_data()
    db.check_data()

    db.add_data(file_data)
    db.check_data()


def test():
    inf = Inference()

    mock_results = [
        ('golden_retriever_1.jpg', '/images/dogs/golden_retriever_1.jpg', 0.8842),  # Should be GOOD
        ('yellow_lab.jpg', '/images/dogs/yellow_lab.jpg', 0.5210),                # Should be GOOD
        ('brown_cat.jpg', '/images/cats/brown_cat.jpg', 0.3155),                 # Should be BAD
        ('park_bench.jpg', '/images/scenery/park_bench.jpg', 0.1201),            # Should be IRRELEVANT
        ('night_sky.jpg', '/images/space/night_sky.jpg', -0.0542)                # Should be IRRELEVANT
    ]
    classified_results = inf.classify_results(mock_results)

    print(classified_results)


if __name__ == "__main__":

    populate_db(VectorDatabase.from_env())
