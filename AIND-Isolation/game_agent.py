"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")    

    my_legal_moves = len(game.get_legal_moves(player))
    opponents_legal_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float(my_legal_moves - 2 * opponents_legal_moves)
    #raise NotImplementedError


def custom_score_2(game, player):
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
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    player_legal_moves = game.get_legal_moves(player)
    opp_legal_moves = game.get_legal_moves(game.get_opponent(player))

    #The weight of 2 with the opposition legal moves means we are penalizing
    #highly for a board state, where the opponent has more move at it's disposal.
    #We will be inclined to choosing the board states in which the opponent is
    #fairly limited in in it's movements. 
    difference_in_moves = len(player_legal_moves) - 2*len(opp_legal_moves)
    position_of_player = game.get_player_location(player)
    position_of_opponent = game.get_player_location(game.get_opponent(player))
    manhattan_distance = abs(position_of_player[0]-position_of_opponent[0]) +  abs(position_of_player[1]-position_of_opponent[1])

    return(float(difference_in_moves/float(manhattan_distance)))
    #raise NotImplementedError


def custom_score_3(game, player):
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
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    own_v_wall = [move for move in own_moves if move[0] == 0
                                             or move[0] == (game.height - 1)
                                             or move[1] == 0
                                             or move[1] == (game.width - 1)]

    opp_moves = game.get_legal_moves(game.get_opponent(player))
    opp_v_wall = [move for move in opp_moves if move[0] == 0
                                             or move[0] == (game.height - 1)
                                             or move[1] == 0
                                             or move[1] == (game.width - 1)]
    
    # Penalize/reward move count if some moves are against the wall
    return float(len(own_moves) - len(own_v_wall)
                 - len(opp_moves) + len(opp_v_wall))

    #raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        def max_value_minimax(self, game, depth):
            """Implement a depth-limited Max search algorithm.

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting

            Returns
            -------
            (int)
            The best score found in the current search.
            Return float('-inf') if there are no legal moves
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 :
                return self.score(game, self)

            best_score = float('-inf')
            #Get the list of legal moves available to a player.
            legal_moves = game.get_legal_moves()
            for move in legal_moves :
                score = min_value_minimax(self, game.forecast_move(move), depth - 1)
                if score > best_score :
                    best_score = score
            # Return the best score from the last completed search iteration
            return best_score

        def min_value_minimax(self, game, depth):
            """Implement a depth-limited Min search algorithm.

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting

            Returns
            -------
            (int)
            The best score found in the current search.
            Return float('-inf') if there are no legal moves
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 :
                return self.score(game, self)

            best_score = float('inf')
            #Get the list of legal moves available to a player.
            legal_moves = game.get_legal_moves()
            for move in legal_moves :
                score = max_value_minimax(self, game.forecast_move(move), depth - 1)
                if score < best_score :
                    best_score = score
            # Return the best score from the last completed search iteration
            return best_score

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_score = float('-inf')

        # Get the list of legal moves available to the player.
        legal_moves = game.get_legal_moves()
        if not legal_moves :
            return (-1, -1)
        else :
            best_move = legal_moves[0]

        for move in legal_moves :
            # Check for the maximizing player
            #if self == game.active_player :
            score = min_value_minimax(self, game.forecast_move(move), depth - 1)
            #elif self == game.inactive_player :
                #score = max_value_minimax(self, game.forecast_move(move), depth - 1)
            # Check if score of current move greater than best move.
            # If yes then update best score and best move.
            if score > best_score :
                best_score = score
                best_move = move

        # Return the best move from the last completed search iteration
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

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

        # TODO: finish this function!
        if self.time_left() < self.TIMER_THRESHOLD :
            raise SearchTimeout()

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()
        if not legal_moves :
            return (-1, -1)
        else :
            best_move = legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            #filled_spaces = (game.width * game.height) - len(game.get_blank_spaces())
            #legal_moves = game.get_legal_moves()
            #if not filled_spaces :
            #    best_move = (int(game.width/2), int(game.height/2))
            #    return best_move
            #elif len(legal_moves) == 1 :
            #    best_move = legal_moves[0];
            #    return best_move
            #else :
                depth = 1
                selected = [(-1, -1)]
                while True :
                    best_move = self.alphabeta(game, depth)
                    selected.append(best_move)
                    depth += 1

        except SearchTimeout:
            # Handle any actions required after timeout as needed
            #if depth == 1 :
                #legal_moves = game.get_legal_moves()
                #if len(legal_moves) :
                    #best_move = legal_moves[0]
            #else :
                #best_move = selected[depth - 1]
            if depth > 1 :
                best_move = selected[depth - 1]
        # Return the best move from the last completed search iteration
        return best_move
        #raise NotImplementedError

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        def max_value_alpha_beta(self, game, depth, alpha = float('-inf'), beta = float('inf')):
            """Implement a depth-limited Max search algorithm with alpha-beta pruning.

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

            Returns
            -------
            (int)
            The best score found in the current search.
            Return float('-inf') if there are no legal moves
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 :
                return self.score(game, self)

            best_score = float('-inf')
            #Get the list of legal moves available to a player.
            legal_moves = game.get_legal_moves()
            for move in legal_moves :
                score = min_value_alpha_beta(self, game.forecast_move(move), depth - 1, alpha, beta)
                if score > best_score :
                    best_score = score
                # Prune other branches if best_score is greater than beta value.
                if best_score >= beta :
                    return best_score
                # Update alpha if required...
                alpha = max(alpha, best_score)
            # Return the best score from the last completed search iteration
            return best_score

        def min_value_alpha_beta(self, game, depth, alpha = float('-inf'), beta = float('inf')):
            """Implement a depth-limited Max search algorithm with alpha-beta pruning.

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

            Returns
            -------
            (int)
            The best score found in the current search.
            Return float('-inf') if there are no legal moves
            """
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 :
                return self.score(game, self)

            best_score = float('inf')
            #Get the list of legal moves available to a player.
            legal_moves = game.get_legal_moves()
            for move in legal_moves :
                score = max_value_alpha_beta(self, game.forecast_move(move), depth - 1, alpha, beta)
                if score < best_score :
                    best_score = score
                # Prune other branches if best_score is less than alpha value.
                if best_score <= alpha :
                    return best_score
                # Update beta if required...
                beta = min(beta, best_score)

            # Return the best score from the last completed search iteration
            return best_score

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_score = float('-inf')

        # Get the list of legal moves available to the player.
        legal_moves = game.get_legal_moves()
        if not legal_moves :
            return (-1, -1)
        else :
            best_move = legal_moves[0]

        for move in legal_moves :
            # Check for the maximizing player
            #if self == game.active_player :
            score = min_value_alpha_beta(self, game.forecast_move(move), depth - 1, alpha, beta)
            #elif self == game.inactive_player :
                #score = max_value_alpha_beta(self, game.forecast_move(move), depth - 1, alpha, beta)
            # Check if score of current move greater than best move.
            # If yes then update best score and best move.
            if score > best_score :
                best_score = score
                best_move = move
            # Prune other branches if best_score is greater than beta value.
            if best_score >= beta :
                return best_move
            # Update alpha if required...
            alpha = max(alpha, best_score)

        # Return the best move from the last completed search iteration
        return best_move
        #raise NotImplementedError
