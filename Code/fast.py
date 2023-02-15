import numpy as np
from HARK.interpolation import HARKinterpolator1D, HARKinterpolator2D
from interpolation.splines import CGrid, eval_linear


class LinearInterpFast(HARKinterpolator1D):
    distance_criteria = ["x_list", "y_list"]

    def __init__(
        self,
        x_list,
        y_list,
    ):
        self.x_list = CGrid(x_list)
        self.y_list = y_list

    def __call__(self, x):
        return eval_linear(self.x_list, self.y_list, x.reshape(-1, 1)).reshape(x.shape)


class BilinearInterpFast(HARKinterpolator2D):
    distance_criteria = ["x_list", "y_list", "f_values"]

    def __init__(self, f_values, x_list, y_list):
        self.f_values = f_values
        self.x_list = x_list
        self.y_list = y_list

        self.grid = CGrid(x_list, y_list)

    def __call__(self, x, y):
        flat_x = x.flatten()
        flat_y = y.flatten()
        joint = np.stack((flat_x, flat_y), axis=1)

        out = np.empty_like(flat_x)
        eval_linear(self.grid, self.f_values, joint, out)

        return out.reshape(x.shape)


class Curvilinear2DInterp(HARKinterpolator2D):
    """
    A 2D interpolation method for curvilinear or "warped grid" interpolation, as
    in White (2015).  Used for models with two endogenous states that are solved
    with the endogenous grid method.

    Parameters
    ----------
    f_values: numpy.array
        A 2D array of function values such that f_values[i,j] =
        f(x_values[i,j],y_values[i,j]).
    x_values: numpy.array
        A 2D array of x values of the same size as f_values.
    y_values: numpy.array
        A 2D array of y values of the same size as f_values.
    """

    distance_criteria = ["f_values", "x_values", "y_values"]

    def __init__(self, f_values, x_values, y_values):
        self.f_values = f_values
        self.x_values = x_values
        self.y_values = y_values
        my_shape = f_values.shape
        self.x_n = my_shape[0]
        self.y_n = my_shape[1]
        self.update_polarity()

    def update_polarity(self):
        """
        Fills in the polarity attribute of the interpolation, determining whether
        the "plus" (True) or "minus" (False) solution of the system of equations
        should be used for each sector.  Needs to be called in __init__.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Grab a point known to be inside each sector: the midway point between
        # the lower left and upper right vertex of each sector
        x_temp = 0.5 * (
            self.x_values[0 : (self.x_n - 1), 0 : (self.y_n - 1)]
            + self.x_values[1 : self.x_n, 1 : self.y_n]
        )
        y_temp = 0.5 * (
            self.y_values[0 : (self.x_n - 1), 0 : (self.y_n - 1)]
            + self.y_values[1 : self.x_n, 1 : self.y_n]
        )
        size = (self.x_n - 1) * (self.y_n - 1)
        x_temp = np.reshape(x_temp, size)
        y_temp = np.reshape(y_temp, size)
        y_pos = np.tile(np.arange(0, self.y_n - 1), self.x_n - 1)
        x_pos = np.reshape(
            np.tile(np.arange(0, self.x_n - 1), (self.y_n - 1, 1)).transpose(), size
        )

        # Set the polarity of all sectors to "plus", then test each sector
        self.polarity = np.ones((self.x_n - 1, self.y_n - 1), dtype=bool)
        alpha, beta = self.find_coords(x_temp, y_temp, x_pos, y_pos)
        polarity = np.logical_and(
            np.logical_and(alpha > 0, alpha < 1), np.logical_and(beta > 0, beta < 1)
        )

        # Update polarity: if (alpha,beta) not in the unit square, then that
        # sector must use the "minus" solution instead
        self.polarity = np.reshape(polarity, (self.x_n - 1, self.y_n - 1))

    def find_sector(self, x, y):
        """
        Finds the quadrilateral "sector" for each (x,y) point in the input.
        Only called as a subroutine of _evaluate().

        Parameters
        ----------
        x : np.array
            Values whose sector should be found.
        y : np.array
            Values whose sector should be found.  Should be same size as x.

        Returns
        -------
        x_pos : np.array
            Sector x-coordinates for each point of the input, of the same size.
        y_pos : np.array
            Sector y-coordinates for each point of the input, of the same size.
        """
        # Initialize the sector guess
        m = x.size
        x_pos_guess = (np.ones(m) * self.x_n / 2).astype(int)
        y_pos_guess = (np.ones(m) * self.y_n / 2).astype(int)

        # Define a function that checks whether a set of points violates a linear
        # boundary defined by (x_bound_1,y_bound_1) and (x_bound_2,y_bound_2),
        # where the latter is *COUNTER CLOCKWISE* from the former.  Returns
        # 1 if the point is outside the boundary and 0 otherwise.
        violation_check = (
            lambda x_check, y_check, x_bound_1, y_bound_1, x_bound_2, y_bound_2: (
                (y_bound_2 - y_bound_1) * x_check - (x_bound_2 - x_bound_1) * y_check
                > x_bound_1 * y_bound_2 - y_bound_1 * x_bound_2
            )
            + 0
        )

        # Identify the correct sector for each point to be evaluated
        these = np.ones(m, dtype=bool)
        max_loops = self.x_n + self.y_n
        loops = 0
        while np.any(these) and loops < max_loops:
            # Get coordinates for the four vertices: (xA,yA),...,(xD,yD)
            x_temp = x[these]
            y_temp = y[these]
            xA = self.x_values[x_pos_guess[these], y_pos_guess[these]]
            xB = self.x_values[x_pos_guess[these] + 1, y_pos_guess[these]]
            xC = self.x_values[x_pos_guess[these], y_pos_guess[these] + 1]
            xD = self.x_values[x_pos_guess[these] + 1, y_pos_guess[these] + 1]
            yA = self.y_values[x_pos_guess[these], y_pos_guess[these]]
            yB = self.y_values[x_pos_guess[these] + 1, y_pos_guess[these]]
            yC = self.y_values[x_pos_guess[these], y_pos_guess[these] + 1]
            yD = self.y_values[x_pos_guess[these] + 1, y_pos_guess[these] + 1]

            # Check the "bounding box" for the sector: is this guess plausible?
            move_down = (y_temp < np.minimum(yA, yB)) + 0
            move_right = (x_temp > np.maximum(xB, xD)) + 0
            move_up = (y_temp > np.maximum(yC, yD)) + 0
            move_left = (x_temp < np.minimum(xA, xC)) + 0

            # Check which boundaries are violated (and thus where to look next)
            c = (move_down + move_right + move_up + move_left) == 0
            move_down[c] = violation_check(
                x_temp[c], y_temp[c], xA[c], yA[c], xB[c], yB[c]
            )
            move_right[c] = violation_check(
                x_temp[c], y_temp[c], xB[c], yB[c], xD[c], yD[c]
            )
            move_up[c] = violation_check(
                x_temp[c], y_temp[c], xD[c], yD[c], xC[c], yC[c]
            )
            move_left[c] = violation_check(
                x_temp[c], y_temp[c], xC[c], yC[c], xA[c], yA[c]
            )

            # Update the sector guess based on the violations
            x_pos_next = x_pos_guess[these] - move_left + move_right
            x_pos_next[x_pos_next < 0] = 0
            x_pos_next[x_pos_next > (self.x_n - 2)] = self.x_n - 2
            y_pos_next = y_pos_guess[these] - move_down + move_up
            y_pos_next[y_pos_next < 0] = 0
            y_pos_next[y_pos_next > (self.y_n - 2)] = self.y_n - 2

            # Check which sectors have not changed, and mark them as complete
            no_move = np.array(
                np.logical_and(
                    x_pos_guess[these] == x_pos_next, y_pos_guess[these] == y_pos_next
                )
            )
            x_pos_guess[these] = x_pos_next
            y_pos_guess[these] = y_pos_next
            temp = these.nonzero()
            these[temp[0][no_move]] = False

            # Move to the next iteration of the search
            loops += 1

        # Return the output
        x_pos = x_pos_guess
        y_pos = y_pos_guess
        return x_pos, y_pos

    def find_coords(self, x, y, x_pos, y_pos):
        """
        Calculates the relative coordinates (alpha,beta) for each point (x,y),
        given the sectors (x_pos,y_pos) in which they reside.  Only called as
        a subroutine of __call__().

        Parameters
        ----------
        x : np.array
            Values whose sector should be found.
        y : np.array
            Values whose sector should be found.  Should be same size as x.
        x_pos : np.array
            Sector x-coordinates for each point in (x,y), of the same size.
        y_pos : np.array
            Sector y-coordinates for each point in (x,y), of the same size.

        Returns
        -------
        alpha : np.array
            Relative "horizontal" position of the input in their respective sectors.
        beta : np.array
            Relative "vertical" position of the input in their respective sectors.
        """
        # Calculate relative coordinates in the sector for each point
        xA = self.x_values[x_pos, y_pos]
        xB = self.x_values[x_pos + 1, y_pos]
        xC = self.x_values[x_pos, y_pos + 1]
        xD = self.x_values[x_pos + 1, y_pos + 1]
        yA = self.y_values[x_pos, y_pos]
        yB = self.y_values[x_pos + 1, y_pos]
        yC = self.y_values[x_pos, y_pos + 1]
        yD = self.y_values[x_pos + 1, y_pos + 1]
        polarity = 2.0 * self.polarity[x_pos, y_pos] - 1.0
        a = xA
        b = xB - xA
        c = xC - xA
        d = xA - xB - xC + xD
        e = yA
        f = yB - yA
        g = yC - yA
        h = yA - yB - yC + yD
        denom = d * g - h * c
        mu = (h * b - d * f) / denom
        tau = (h * (a - x) - d * (e - y)) / denom
        zeta = a - x + c * tau
        eta = b + c * mu + d * tau
        theta = d * mu
        alpha = (-eta + polarity * np.sqrt(eta**2.0 - 4.0 * zeta * theta)) / (
            2.0 * theta
        )
        beta = mu * alpha + tau

        # Alternate method if there are sectors that are "too regular"
        z = np.logical_or(
            np.isnan(alpha), np.isnan(beta)
        )  # These points weren't able to identify coordinates
        if np.any(z):
            these = np.isclose(
                f / b, (yD - yC) / (xD - xC)
            )  # iso-beta lines have equal slope
            if np.any(these):
                kappa = f[these] / b[these]
                int_bot = yA[these] - kappa * xA[these]
                int_top = yC[these] - kappa * xC[these]
                int_these = y[these] - kappa * x[these]
                beta_temp = (int_these - int_bot) / (int_top - int_bot)
                x_left = beta_temp * xC[these] + (1.0 - beta_temp) * xA[these]
                x_right = beta_temp * xD[these] + (1.0 - beta_temp) * xB[these]
                alpha_temp = (x[these] - x_left) / (x_right - x_left)
                beta[these] = beta_temp
                alpha[these] = alpha_temp

            # print(np.sum(np.isclose(g/c,(yD-yB)/(xD-xB))))

        return alpha, beta

    def _evaluate(self, x, y):
        """
        Returns the level of the interpolated function at each value in x,y.
        Only called internally by HARKinterpolator2D.__call__ (etc).
        """
        x_pos, y_pos = self.find_sector(x, y)
        alpha, beta = self.find_coords(x, y, x_pos, y_pos)

        # Calculate the function at each point using bilinear interpolation
        f = (
            (1 - alpha) * (1 - beta) * self.f_values[x_pos, y_pos]
            + (1 - alpha) * beta * self.f_values[x_pos, y_pos + 1]
            + alpha * (1 - beta) * self.f_values[x_pos + 1, y_pos]
            + alpha * beta * self.f_values[x_pos + 1, y_pos + 1]
        )
        return f

    def _derX(self, x, y):
        """
        Returns the derivative with respect to x of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeX.
        """
        x_pos, y_pos = self.find_sector(x, y)
        alpha, beta = self.find_coords(x, y, x_pos, y_pos)

        # Get four corners data for each point
        xA = self.x_values[x_pos, y_pos]
        xB = self.x_values[x_pos + 1, y_pos]
        xC = self.x_values[x_pos, y_pos + 1]
        xD = self.x_values[x_pos + 1, y_pos + 1]
        yA = self.y_values[x_pos, y_pos]
        yB = self.y_values[x_pos + 1, y_pos]
        yC = self.y_values[x_pos, y_pos + 1]
        yD = self.y_values[x_pos + 1, y_pos + 1]
        fA = self.f_values[x_pos, y_pos]
        fB = self.f_values[x_pos + 1, y_pos]
        fC = self.f_values[x_pos, y_pos + 1]
        fD = self.f_values[x_pos + 1, y_pos + 1]

        # Calculate components of the alpha,beta --> x,y delta translation matrix
        alpha_x = (1 - beta) * (xB - xA) + beta * (xD - xC)
        alpha_y = (1 - beta) * (yB - yA) + beta * (yD - yC)
        beta_x = (1 - alpha) * (xC - xA) + alpha * (xD - xB)
        beta_y = (1 - alpha) * (yC - yA) + alpha * (yD - yB)

        # Invert the delta translation matrix into x,y --> alpha,beta
        det = alpha_x * beta_y - beta_x * alpha_y
        x_alpha = beta_y / det
        x_beta = -alpha_y / det

        # Calculate the derivative of f w.r.t. alpha and beta
        dfda = (1 - beta) * (fB - fA) + beta * (fD - fC)
        dfdb = (1 - alpha) * (fC - fA) + alpha * (fD - fB)

        # Calculate the derivative with respect to x (and return it)
        dfdx = x_alpha * dfda + x_beta * dfdb
        return dfdx

    def _derY(self, x, y):
        """
        Returns the derivative with respect to y of the interpolated function
        at each value in x,y. Only called internally by HARKinterpolator2D.derivativeX.
        """
        x_pos, y_pos = self.find_sector(x, y)
        alpha, beta = self.find_coords(x, y, x_pos, y_pos)

        # Get four corners data for each point
        xA = self.x_values[x_pos, y_pos]
        xB = self.x_values[x_pos + 1, y_pos]
        xC = self.x_values[x_pos, y_pos + 1]
        xD = self.x_values[x_pos + 1, y_pos + 1]
        yA = self.y_values[x_pos, y_pos]
        yB = self.y_values[x_pos + 1, y_pos]
        yC = self.y_values[x_pos, y_pos + 1]
        yD = self.y_values[x_pos + 1, y_pos + 1]
        fA = self.f_values[x_pos, y_pos]
        fB = self.f_values[x_pos + 1, y_pos]
        fC = self.f_values[x_pos, y_pos + 1]
        fD = self.f_values[x_pos + 1, y_pos + 1]

        # Calculate components of the alpha,beta --> x,y delta translation matrix
        alpha_x = (1 - beta) * (xB - xA) + beta * (xD - xC)
        alpha_y = (1 - beta) * (yB - yA) + beta * (yD - yC)
        beta_x = (1 - alpha) * (xC - xA) + alpha * (xD - xB)
        beta_y = (1 - alpha) * (yC - yA) + alpha * (yD - yB)

        # Invert the delta translation matrix into x,y --> alpha,beta
        det = alpha_x * beta_y - beta_x * alpha_y
        y_alpha = -beta_x / det
        y_beta = alpha_x / det

        # Calculate the derivative of f w.r.t. alpha and beta
        dfda = (1 - beta) * (fB - fA) + beta * (fD - fC)
        dfdb = (1 - alpha) * (fC - fA) + alpha * (fD - fB)

        # Calculate the derivative with respect to x (and return it)
        dfdy = y_alpha * dfda + y_beta * dfdb
        return dfdy
