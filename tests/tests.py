import pytest
from worker.models import Models


@pytest.mark.celery(result_backend='redis://')
def test_create_task(celery_app, celery_worker):
    @celery_app.task
    def classes():
        models_dict = Models()
        return {"classes": list(models_dict.classes.keys())}
    assert classes.delay().get(timeout=10) == b'{"classes":["Linear regression","Gradient Boosting regression","Logistic regression","Gradient Boosting classifier"]}\n'

