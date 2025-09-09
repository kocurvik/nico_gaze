import vectors

class Error:
    def __init__(self, view_point, intersection_point, dVector_correct, dVector_computed):
        self.distance_error = vectors.distance_3D(view_point, intersection_point)
        self.angle_error = vectors.vectors_angle(dVector_correct, dVector_computed)

    def __str__(self):
        return f"Distance Error: {self.distance_error}, Angle Error: {self.angle_error}"


