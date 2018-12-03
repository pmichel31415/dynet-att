#!/usr/bin/env python3

from .tasks import get_task


def main():
    # Get task
    task = get_task()
    # Run
    task.main()


if __name__ == "__main__":
    main()
