import unittest
from typing import List
import sys
sys.path.insert(0, "/home/rg3637/hpml-assign2/hpml-project/transformers/src")  
import unittest
import torch
from transformers.generation.beam_constraints import Constraint, TemplateConstraint
from typing import List, Optional
from transformers.generation.beam_constraints import OrderedConstraint

class TestOrderedConstraint(unittest.TestCase):

    def test_initialization(self):
        tokens = [5, 2, 3]
        constraint = OrderedConstraint(tokens)
        self.assertEqual(constraint.ordered_token_ids, tokens)
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)
        self.assertEqual(constraint.seqlen, 3)

    def test_advance_before_completion(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)
        
        # Position 0
        self.assertEqual(constraint.advance(), 5)
        constraint.position = 1
        self.assertEqual(constraint.advance(), 6)
        constraint.position = 2
        self.assertEqual(constraint.advance(), 3)

    def test_advance_after_completion(self):
        tokens = [5]
        constraint = OrderedConstraint(tokens)
        constraint.update(5)  # Completes the sequence
        self.assertEqual(constraint.advance(), [])

    def test_does_advance_behavior(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)
        
        # Position 0
        self.assertTrue(constraint.does_advance(5))
        self.assertFalse(constraint.does_advance(6))
        
        # Position 1
        constraint.position = 1
        self.assertTrue(constraint.does_advance(6))
        self.assertFalse(constraint.does_advance(5))
        
        # Position 2
        constraint.position = 2
        self.assertTrue(constraint.does_advance(3))
        self.assertFalse(constraint.does_advance(4))

    def test_update_correct_sequence(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)
        
        # First token (5)
        stepped, completed, reset = constraint.update(5)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 1)
        
        # Second token (6)
        stepped, completed, reset = constraint.update(6)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 2)
        
        # Third token (3)
        stepped, completed, reset = constraint.update(3)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 3)

    def test_update_incorrect_token_does_not_advance(self):
        tokens = [5, 6]
        constraint = OrderedConstraint(tokens)
        
        # Correct first token
        constraint.update(5)
        
        # Incorrect second token
        stepped, completed, reset = constraint.update(7)
        self.assertFalse(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)  # No reset, position remains at 1
        self.assertEqual(constraint.position, 1)
        
        # Correct token later
        stepped, completed, _ = constraint.update(6)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertEqual(constraint.position, 2)

    def test_reset_behavior(self):
        tokens = [5, 6]
        constraint = OrderedConstraint(tokens)
        constraint.update(5)
        constraint.reset()
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)

    def test_remaining_tokens(self):
        tokens = [5, 6, 3]
        constraint = OrderedConstraint(tokens)
        self.assertEqual(constraint.remaining(), 3)
        constraint.update(5)
        self.assertEqual(constraint.remaining(), 2)
        constraint.update(6)
        self.assertEqual(constraint.remaining(), 1)
        constraint.update(3)
        self.assertEqual(constraint.remaining(), 0)

    def test_copy_without_state(self):
        tokens = [5, 6]
        original = OrderedConstraint(tokens)
        original.update(5)
        copied = original.copy(stateful=False)
        self.assertEqual(copied.position, 0)
        self.assertFalse(copied.completed)

    def test_copy_with_state(self):
        tokens = [5, 6]
        original = OrderedConstraint(tokens)
        original.update(5)
        copied = original.copy(stateful=True)
        self.assertEqual(copied.position, 1)
        self.assertEqual(copied.completed, original.completed)

    def test_single_token_completion(self):
        tokens = [10]
        constraint = OrderedConstraint(tokens)
        self.assertTrue(constraint.does_advance(10))
        stepped, completed, _ = constraint.update(10)
        self.assertTrue(stepped)
        self.assertTrue(completed)

    def test_position_overflow(self):
        tokens = [5, 6]
        constraint = OrderedConstraint(tokens)
        constraint.position = 2  # Force beyond sequence length
        self.assertEqual(constraint.advance(), [])
        self.assertTrue(constraint.completed)

    def test_no_progress_on_mismatch(self):
        tokens = [5, 6, 7]
        constraint = OrderedConstraint(tokens)
        
        # Correct first token
        constraint.update(5)
        
        # Incorrect second token (should block progress)
        stepped, completed, reset = constraint.update(8)
        self.assertFalse(stepped)
        self.assertFalse(completed)
        self.assertEqual(constraint.position, 1)  # Still at position 1
        
        # Correct second token later
        stepped, completed, _ = constraint.update(6)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertEqual(constraint.position, 2)

if __name__ == '__main__':
    unittest.main()