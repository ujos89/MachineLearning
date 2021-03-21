# deterministic environment (100% action success)
# 3 non-terminal states, 2 terminal states
# agent starts in state 2 (middle of the walk) T-1-2-3-T
# only reward is still at the right-most cell in the "walk"
# episodic environment, the agent terminates at the left- or right-most cell (after 1 action selection -- any action)

# actions left (0) or right (1)

print("s=state, a=action, s'=next state, r=reward")
print("Each row represents:")
print("s: {a: (p(s,a,s'), s', r(s,a,s'), <terminal?> )}")
P = {
    0: {
        0: [(1.0, 0, 0.0, True)],
        1: [(1.0, 0, 0.0, True)]
    },
    1: {
        # to avoid floating points problem
        0: [(0.5000000000000001, 0, 0.0, True),
            (0.3333333333333333, 1, 0.0, False),
            (0.16666666666666666, 2, 0.0, False)
        ],
        1: [(0.5000000000000001, 2, 0.0, False),
            (0.3333333333333333, 1, 0.0, False),
            (0.16666666666666666, 0, 0.0, True)
        ]
    },
    2: {
        0: [(0.5000000000000001, 1, 0.0, False),
            (0.3333333333333333, 2, 0.0, False),
            (0.16666666666666666, 3, 0.0, False)
        ],
        1: [(0.5000000000000001, 3, 0.0, False),
            (0.3333333333333333, 2, 0.0, False),
            (0.16666666666666666, 1, 0.0, False)
        ]
    },
    3: {
        0: [(0.5000000000000001, 2, 0.0, False),
            (0.3333333333333333, 3, 0.0, False),
            (0.16666666666666666, 4, 1.0, True)
        ],
        1: [(0.5000000000000001, 4, 1.0, True),
            (0.3333333333333333, 3, 0.0, False),
            (0.16666666666666666, 2, 0.0, False)
        ]
    },
    4: {
        0: [(1.0, 4, 0.0, True)],
        1: [(1.0, 4, 0.0, True)]
    }
}

print(P)
