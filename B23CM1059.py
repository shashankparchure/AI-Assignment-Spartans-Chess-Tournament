import time
import random
from config import *
from board import Move

MATE_SCORE = 100000
EXACT = 0
LOWER = 1
UPPER = 2

class B23CM1059:
    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 4
        self.time_limit = 0.85
        self.start_time = 0
        self.tt = {}
        self.killer = {}
        self.history = {}
        self.INF = 10**9
        self.root_is_white = True
        self.repetition_tracker = {}

    def get_best_move(self):
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        self.nodes_expanded = 0
        self.start_time = time.time()
        self.tt.clear()
        self.root_is_white = self.engine.white_to_move

        # Update repetition tracker
        board_hash = self._get_board_hash()
        self.repetition_tracker[board_hash] = self.repetition_tracker.get(board_hash, 0) + 1

        best_move = legal_moves[0]
        best_score = -self.INF

        for d in range(1, self.depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break
            try:
                score, move = self._minimax_root(d, -self.INF, self.INF)
                if move is not None:
                    best_move = move
                    best_score = score
                if abs(best_score) >= MATE_SCORE - 1000:
                    break
                if time.time() - self.start_time > self.time_limit:
                    break
            except TimeoutError:
                break

        return best_move

    def _minimax_root(self, depth, alpha, beta):
        best_move = None
        best_score = -self.INF
        moves = self.engine.get_legal_moves()
        ordered = self._order_moves(moves, depth)

        for i, mv in enumerate(ordered):
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
            self.engine.make_move(mv)
            try:
                val = self._minimax(depth - 1, alpha, beta, False)
            finally:
                self.engine.undo_move()
            if val > best_score:
                best_score = val
                best_move = mv
            alpha = max(alpha, val)
            if alpha >= beta:
                self._update_killer(mv, depth)
                self._update_history(mv, depth)
                break

        return best_score, best_move

    def _minimax(self, depth, alpha, beta, is_maximizing):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()
        self.nodes_expanded += 1
        gs = self.engine.get_game_state()
        if gs == "checkmate":
            winner_is_white = not self.engine.white_to_move
            val = MATE_SCORE
            return val if (winner_is_white == self.root_is_white) else -val
        if gs == "stalemate":
            return self.evaluate_board(gs)

        key = self._get_board_hash()
        tt_entry = self.tt.get(key)
        if tt_entry and tt_entry[0] >= depth:
            tt_depth, tt_val, tt_flag, tt_best = tt_entry
            if tt_flag == EXACT:
                return tt_val
            elif tt_flag == LOWER:
                alpha = max(alpha, tt_val)
            elif tt_flag == UPPER:
                beta = min(beta, tt_val)
            if alpha >= beta:
                return tt_val

        if depth <= 0:
            return self._quiescence(alpha, beta, is_maximizing)

        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return 0

        ordered = self._order_moves(legal_moves, depth)
        if is_maximizing:
            value = -self.INF
            for i, mv in enumerate(ordered):
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                reduce = 0
                if i >= 4 and depth >= 3 and mv.piece_captured == EMPTY_SQUARE and not self.engine.is_in_check():
                    reduce = 1
                self.engine.make_move(mv)
                try:
                    if reduce:
                        score = self._minimax(depth - 1 - reduce, alpha, beta, False)
                        if score <= alpha:
                            continue
                    score = self._minimax(depth - 1, alpha, beta, False)
                finally:
                    self.engine.undo_move()
                if score > value:
                    value = score
                alpha = max(alpha, value)
                if alpha >= beta:
                    self._update_killer(mv, depth)
                    self._update_history(mv, depth)
                    break
            flag = EXACT
            if value <= alpha:
                flag = UPPER
            elif value >= beta:
                flag = LOWER
            self.tt[key] = (depth, value, flag, None)
            return value
        else:
            value = self.INF
            for i, mv in enumerate(ordered):
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                reduce = 0
                if i >= 4 and depth >= 3 and mv.piece_captured == EMPTY_SQUARE and not self.engine.is_in_check():
                    reduce = 1
                self.engine.make_move(mv)
                try:
                    if reduce:
                        score = self._minimax(depth - 1 - reduce, alpha, beta, True)
                        if score >= beta:
                            continue
                    score = self._minimax(depth - 1, alpha, beta, True)
                finally:
                    self.engine.undo_move()
                if score < value:
                    value = score
                beta = min(beta, value)
                if alpha >= beta:
                    self._update_killer(mv, depth)
                    self._update_history(mv, depth)
                    break
            flag = EXACT
            if value <= alpha:
                flag = UPPER
            elif value >= beta:
                flag = LOWER
            self.tt[key] = (depth, value, flag, None)
            return value

    def _quiescence(self, alpha, beta, is_maximizing):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()
        eval_white = self.evaluate_board()
        eval_root = eval_white if self.root_is_white else -eval_white
        if is_maximizing:
            if eval_root >= beta:
                return eval_root
            if alpha < eval_root:
                alpha = eval_root
            captures = [m for m in self.engine.get_legal_moves() if m.piece_captured != EMPTY_SQUARE]
            captures.sort(key=self._mvv_lva_key, reverse=True)
            for mv in captures:
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                self.engine.make_move(mv)
                try:
                    score = self._quiescence(alpha, beta, False)
                finally:
                    self.engine.undo_move()
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    return alpha
            return alpha
        else:
            if eval_root <= alpha:
                return eval_root
            if beta > eval_root:
                beta = eval_root
            captures = [m for m in self.engine.get_legal_moves() if m.piece_captured != EMPTY_SQUARE]
            captures.sort(key=self._mvv_lva_key, reverse=False)
            for mv in captures:
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                self.engine.make_move(mv)
                try:
                    score = self._quiescence(alpha, beta, True)
                finally:
                    self.engine.undo_move()
                if score < beta:
                    beta = score
                if alpha >= beta:
                    return beta
            return beta

    def evaluate_board(self, game_state=None):
        if game_state is None:
            game_state = self.engine.get_game_state()

        if game_state == "checkmate":
            winner_is_white = not self.engine.white_to_move
            val = 600
            return val if (winner_is_white == self.root_is_white) else -val

        if game_state == "stalemate":
            material = 0
            for r in range(BOARD_HEIGHT):
                for c in range(BOARD_WIDTH):
                    piece = self.engine.board[r][c]
                    if piece == EMPTY_SQUARE:
                        continue
                    value = PIECE_VALUES.get(piece, 0)
                    color = piece[0]
                    if color == 'w':
                        material += value
                    else:
                        material -= value
            return material

        material = 0
        pos = 0
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue
                material += PIECE_VALUES.get(piece, 0)
                p = piece[1]
                color = piece[0]
                if p == 'P':
                    val = PAWN_PST[r][c]
                elif p == 'N':
                    val = KNIGHT_PST[r][c]
                elif p == 'B':
                    val = BISHOP_PST[r][c]
                elif p == 'K':
                    val = KING_PST_LATE_GAME[r][c]
                else:
                    val = 0
                pos += val if color == 'w' else -val
        score = material + pos
        repetition_count = self.engine.get_repetition_count()
        if repetition_count >= 2:
            score -= 3000 * repetition_count
        return score

    def _get_board_hash(self):
        return hash(str(self.engine.board) + str(self.engine.white_to_move))

    def _order_moves(self, moves, depth):
        ordered = []
        for mv in moves:
            score = 0
            if mv.piece_captured != EMPTY_SQUARE:
                score += PIECE_VALUES.get(mv.piece_captured, 0) * 10
            if depth in self.killer and mv in self.killer[depth]:
                score += 5000
            move_key = self._get_move_key(mv)
            score += self.history.get(move_key, 0)
            self.engine.make_move(mv)
            repetition_count = self.engine.get_repetition_count()
            self.engine.undo_move()
            if repetition_count >= 2:
                score -= 1000 * repetition_count
            score += random.randint(-5, 5)
            ordered.append((score, mv))
        ordered.sort(key=lambda x: x[0], reverse=True)
        return [mv for score, mv in ordered]

    def _update_killer(self, move, depth):
        if depth not in self.killer:
            self.killer[depth] = []
        if move not in self.killer[depth]:
            self.killer[depth].append(move)
            if len(self.killer[depth]) > 2:
                self.killer[depth] = self.killer[depth][:2]

    def _update_history(self, move, depth):
        key = self._get_move_key(move)
        self.history[key] = self.history.get(key, 0) + depth * depth

    def _get_move_key(self, move):
        return (move.piece_moved, move.start_row, move.start_col, move.end_row, move.end_col)

    def _mvv_lva_key(self, move):
        captured_value = PIECE_VALUES.get(move.piece_captured, 0)
        moved_value = PIECE_VALUES.get(move.piece_moved, 0)
        return captured_value * 10 - moved_value
