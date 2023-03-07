from lazevo.piza import Lazevo, Universe

def test_piza():
    lazevo = Lazevo(
        universe=[
            [1, 1, 1],
            [10, 10, 10],
            [100, 100, 100]
        ],
        init_positions=[
            [
                [120,120,120],
                [2,3,4],
                [8,10,12]
            ]
        ]
    )
    lazevo.piza(n_iters=10)

test_piza()
