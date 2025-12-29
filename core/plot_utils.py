#!/usr/bin/python3
"""
Data visualization utility functions.

Author(s):
    Michael Yao @michael-s-yao

Licensed under the MIT License. Copyright University of Pennsylvania 2025.
"""
import matplotlib.lines as mlines
import matplotlib.path as mpath
import numpy as np
from matplotlib.patches import Circle, Polygon, RegularPolygon
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from typing import Any, Dict, Final, List, Sequence, Tuple, Union


BROAD_SUBJECTS: Final[Dict[str, str]] = {
    "Medicine": "medicine.parquet",
    "Rheum": "rheumatology.parquet",
    "Pulm": "pulmonology.parquet",
    "PC": "primarycare.parquet",
    "Onc": "oncology.parquet",
    "Nephrol": "nephrology.parquet",
    "ID": "infectiousdisease.parquet",
    "Geriatrics": "geriatrics.parquet",
    "GI": "gastroenterology.parquet",
    "Endo": "endocrinology.parquet",
    "CC": "criticalcare.parquet",
    "Cards": "cardiology.parquet",
    "Allergy": "allergy.parquet"
}


def radar_factory(num_vars: int, frame: str = "polygon"):
    """
    Create a radar chart with `num_vars` axes.
    Input:
        num_vars: number of variables for radar chart.
        frame: the shape of the frame surrounding the axes. One of [`circle`,
            `polygon`].
    Returns:
    """
    theta = np.linspace(0.0, 2.0 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path) -> mpath.Path:
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return mpath.Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name: Final[str] = "radar"

        PolarTransform = RadarTransform

        def __init__(self, *args: Tuple[Any], **kwargs: Any):
            super(RadarAxes, self).__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(
            self,
            *args: Tuple[Any],
            closed: bool = True,
            **kwargs: Dict[str, Any]
        ) -> List[Polygon]:
            return super(RadarAxes, self).fill(*args, closed=closed, **kwargs)

        def plot(
            self,
            *args: Any,
            scalex: bool = True,
            scaley: bool = True,
            data: Any = None,
            **kwargs: Any
        ) -> List[mlines.Line2D]:
            ells = super(RadarAxes, self).plot(
                *args, scalex=scalex, scaley=scaley, data=data, **kwargs
            )
            for line in ells:
                self._close_line(line)
            return ells

        def _close_line(self, line) -> None:
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels: Sequence[str]) -> None:
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self) -> Union[Circle, RegularPolygon]:
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon(
                    (0.5, 0.5), num_vars, radius=0.5, edgecolor="k"
                )
            raise ValueError(f"Unknown value for `frame`: {frame}")

        def _gen_axes_spines(self) -> Dict[str, Any]:
            if frame == "circle":
                return (
                    super(RadarAxes, self)._gen_axes_spines()  # type: ignore
                )
            elif frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=mpath.Path.unit_regular_polygon(num_vars)
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            raise ValueError(f"Unknown value for `frame`: {frame}")

    register_projection(RadarAxes)
    return theta
