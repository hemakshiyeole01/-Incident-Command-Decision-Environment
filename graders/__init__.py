from graders.grader1 import grade as grade_task1
from graders.grader2 import grade as grade_task2
from graders.grader3 import grade as grade_task3

GRADERS = {
    "task1_resource": grade_task1,
    "task2_multiagency": grade_task2,
    "task3_cascade": grade_task3,
}

__all__ = ["GRADERS", "grade_task1", "grade_task2", "grade_task3"]
