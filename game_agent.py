"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def weighted_improved_score(game, player):
    """ C/P DESC FROM IPYNB HERE """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # to start just try something simple from the already coded agents
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(2*own_moves - 1*opp_moves)

def center_square_good(game, player):
    """ C/P DESC FROM IPYNB HERE """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    center_square = (game.height // 2, game.width // 2)
    player_loc = game.get_player_location(player)
    oppo_loc = game.get_player_location(game.get_opponent(player))

    L1_norm_player = abs(player_loc[0] - center_square[0]) + abs(player_loc[1] - center_square[1])
    L1_norm_oppo = abs(oppo_loc[0] - center_square[0]) + abs(oppo_loc[1] - center_square[1])

    return float(L1_norm_player - 2*L1_norm_oppo)

def stay_close(game, player):
    """ C/P DESC FROM IPYNB HERE """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if game.move_count <= 7:
        player_loc = game.get_player_location(player)
        oppo_loc = game.get_player_location(game.get_opponent(player))

        return float(-1*abs(player_loc[0] - oppo_loc[0]) + abs(player_loc[1] - oppo_loc[1]))

    else:
        return center_square_good(game, player)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return weighted_improved_score(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # check which search method we are using
        if self.method == 'minimax':
            search_using = self.minimax
        else:
            search_using = self.alphabeta

        # NOTES: in iterative deepening we go through ply one with DFS then ply 2 DFS, etc.
        # with fixed depth we go all the way to full search depth on every avail move
        # there is only one pass through each of the legal_moves in fixed depth
        # but with iterative deepening you go through legal moves to depth one, then two, etc.
        # so it seems like with iterative you don't care what depth is passed to the player
        # just don't want to cut off too early, max depth should be relative to board size

        legal_moves = game.get_legal_moves(game.active_player)
        # Catch if legal_moves is empty
        # and initialize legal moves to the first entry so I acutally return something
        # even if timed out
        if not legal_moves:
            return (-1, -1)
        # opening book if its the opening move choose the center or one of the adjecent
        # squares randomly, i know its good to be in the middle, but idk about dead center
        # for this style of play
        elif game.is_opening_move():
            return random.choice(game.opening_book())
        else:
            move = legal_moves[0]
            try:
                # The search method call (alpha beta or minimax) should happen in
                # here in order to avoid timeout. The try/except block will
                # automatically catch the exception raised by the search method
                # when the timer gets close to expiring

                # if using fixed depth no loop needed
                # if using iterative make sure to go through each move at given depth before going down more
                max_depth = game.width ^ 2 + 1
                if self.iterative:
                    for ply in range(1, max_depth):
                        score, move = search_using(game, ply)
                else:
                    _, move = search_using(game, self.search_depth)


            except Timeout:
                # Handle any actions required at timeout, if necessary
                # if we timeout pick an arbitrary move, say 1, better than losing
                return move


        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.

        Brett's strategy: Use recursion in the for loop over legal moves.
        This will work for iterative deepening because the depth passed to minimax
        starts at 1 and goes up. And it will work for fixed depth because the
        depth passed here is the fixed depth number so the recursion will dig to the
        fixed depth before "turning around" and doing the same thing for the next availble
        move.

        """
        # GIVEN (but will check that we haven't run out of time before diving in minimiax call
        # works well here because using recursion this function call lots of times and will
        # have an answer even if timed out)
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # initialize top score and top move
        if maximizing_player:
            top_score = float("-inf")
        else:
            top_score = float("inf")
        top_move = (-1, -1)

        # two conditions under which we should eval score: if the depth is zero
        # or if there are no legal moves
        if depth == 0:
            return self.score(game, self), top_move

        if not game.get_legal_moves():
            return self.score(game, self), top_move

        # now for recursion, for each legal move we want to do minimax
        # it will return the heuristic score if depth is zero
        # we want to pass forecast_move to the minimax call inside the loop
        # and make sure to subtract 1 from the depth because forecast move puts us one layer down
        for move in game.get_legal_moves():
            # the ",_" bit below is just short hand for "i dont need this variable" (just learned that idiom)
            score, _ = self.minimax(game.forecast_move(move), depth - 1, not maximizing_player)
            # for max-ing player the score becomes top score if not equal to the current top score
            # and if it is greater than the current top score
            # similar for min-ing player but with min(socre, current top score)
            if maximizing_player:
                if score != top_score and score == max(score, top_score):
                    top_score = score
                    top_move = move

            if not maximizing_player:
                if score != top_score and score == min(score, top_score):
                    top_score = score
                    top_move = move

        return (top_score, top_move)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.

        Brett's Notes:
        - Alpha means the best already explored options along path to the root for maximizer
        - Beta is the best already explored option along path to the root for minimizer
        - good resource to see this in code: http://aima.cs.berkeley.edu/python/games.html
        """
        # GIVEN
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # initialize top score and top move
        if maximizing_player:
            top_score = float("-inf")
        else:
            top_score = float("inf")
        top_move = (-1, -1)

        # two conditions under which we should eval score: if the depth is zero
        # or if there are no legal moves
        if depth == 0:
            return self.score(game, self), top_move

        if not game.get_legal_moves():
            return self.score(game, self), top_move

        # now for recursion, for each legal move we want to do minimax
        # it will return the heuristic score if depth is zero
        # we want to pass forecast_move to the minimax call inside the loop
        # and make sure to subtract 1 from the depth because forecast move puts us one layer down
        for move in game.get_legal_moves():
            score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, not maximizing_player)
            # similar to minimax but here want to update alpha and beta
            # and also break the loop if the new score doesnt beat the
            # current branch score, ie pruning
            if maximizing_player:
                if score > top_score:
                    top_score = score
                    top_move = move
                # if top score is greater than beta then we can prune
                # so will break the loop
                if top_score >= beta:
                    break
                # if top score isnt greater than beta then we need
                # to set a new alpha
                alpha = max(alpha, top_score)

            if not maximizing_player:
                if score < top_score:
                    top_score = score
                    top_move = move
                # similar to above we prune and break the loop if this is true
                if top_score <= alpha:
                    break
                # again, if not broken may need to reset beta
                beta = min(beta, top_score)

        return (top_score, top_move)
