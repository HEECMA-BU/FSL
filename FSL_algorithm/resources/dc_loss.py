import torch

class DistanceCorrelationLossWrapper(torch.nn.modules.loss._Loss):
    def __init__(self, dcor_weighting: float = 0.3) -> None:
        super().__init__()
        self.dcor_weighting = dcor_weighting

        self.ce = torch.nn.CrossEntropyLoss()
        self.dcor = DistanceCorrelationLoss()
        

    # def forward(self, inputs, intermediates, outputs, targets):
    #     return self.ce(outputs, targets) + self.dcor_weighting * self.dcor(inputs, intermediates)
    def forward(self, inputs, intermediates):
        return self.dcor_weighting * self.dcor(inputs, intermediates)


class DistanceCorrelationLoss(torch.nn.modules.loss._Loss):
    def __init__(self) -> None:
        super().__init__()
        self.pdist = torch.nn.PairwiseDistance(p=2)

    def forward(self, input_data, intermediate_data):
        input_data = input_data.reshape(input_data.shape[0], -1)
        intermediate_data = intermediate_data.reshape(intermediate_data.shape[0], -1)

        # Get A matrices of data
        A_input = self._A_matrix(input_data)
        A_intermediate = self._A_matrix(intermediate_data)

        # Get distance variances
        input_dvar = self._distance_variance(A_input)
        intermediate_dvar = self._distance_variance(A_intermediate)

        # Get distance covariance
        dcov = self._distance_covariance(A_input, A_intermediate)

        # Put it together
        dcorr = dcov / (input_dvar * intermediate_dvar).sqrt()

        return dcorr

    def _distance_covariance(self, a_matrix, b_matrix):
        return (a_matrix * b_matrix).sum().sqrt() / a_matrix.shape[0]

    def _distance_variance(self, a_matrix):
        return (a_matrix**2).sum().sqrt() / a_matrix.shape[0]

    def _A_matrix(self, data):
        distance_matrix = self._distance_matrix(data)

        row_mean = distance_matrix.mean(dim=0, keepdim=True)
        col_mean = distance_matrix.mean(dim=1, keepdim=True)
        data_mean = distance_matrix.mean()

        return distance_matrix - row_mean - col_mean + data_mean

    def _distance_matrix(self, data):
        # n = data.shape[0]
        # distance_matrix_1 = torch.zeros((n, n)).send(data.location)

        # for i in range(n):
        #     for j in range(n):
        #         row_diff = data[i] - data[j]
        #         distance_matrix_1[i, j] = (row_diff**2).sum().sqrt_()
        # #         print()
        # distance_matrix_2 = self.pdist(data, data)
        distance_matrix_3 = torch.cdist(data, data, p=2)

        # data*data.T

        return distance_matrix_3

