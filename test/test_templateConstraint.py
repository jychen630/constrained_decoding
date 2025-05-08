import sys
sys.path.insert(0, "/home/rg3637/hpml-assign2/hpml-project/transformers/src")  
import unittest
import torch
from transformers.generation.beam_constraints import Constraint, TemplateConstraint
from typing import List, Optional
from transformers.generation.beam_constraints import ConstraintListState
from transformers.testing_utils import require_torch

@require_torch
class TestTemplateConstraint(unittest.TestCase):
    def test_initialization(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertEqual(constraint.template, template)
        self.assertEqual(constraint.seqlen, 3)
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)
    
    def test_advance_before_completion(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertEqual(constraint.advance(), 5)
        constraint.position = 1
        self.assertIsNone(constraint.advance())
        constraint.position = 2
        self.assertEqual(constraint.advance(), 3)
    
    def test_advance_after_completion(self):
        template = [5]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        self.assertEqual(constraint.advance(), [])

    def test_does_advance_correct_token(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertTrue(constraint.does_advance(5))
        constraint.position = 1
        self.assertTrue(constraint.does_advance(100))  # Any token allowed
        constraint.position = 2
        self.assertTrue(constraint.does_advance(3))
        self.assertFalse(constraint.does_advance(4))

    def test_does_advance_when_completed(self):
        template = [5]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        self.assertFalse(constraint.does_advance(5))

    def test_update_correct_sequence(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        stepped, completed, reset = constraint.update(5)
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 1)

        stepped, completed, reset = constraint.update(10)  # None allows any token
        self.assertTrue(stepped)
        self.assertFalse(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 2)

        stepped, completed, reset = constraint.update(3)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertFalse(reset)
        self.assertEqual(constraint.position, 3)

    def test_update_incorrect_token_resets(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        stepped, completed, reset = constraint.update(7)  # Incorrect
        self.assertFalse(stepped)
        self.assertFalse(completed)
        self.assertTrue(reset)
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)
    
    def test_reset(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        constraint.reset()
        self.assertEqual(constraint.position, 0)
        self.assertFalse(constraint.completed)
    
    def test_remaining(self):
        template = [5, None, 3]
        constraint = TemplateConstraint(template)
        self.assertEqual(constraint.remaining(), 3)
        constraint.update(5)
        self.assertEqual(constraint.remaining(), 2)
        constraint.update(10)
        self.assertEqual(constraint.remaining(), 1)
        constraint.update(3)
        self.assertEqual(constraint.remaining(), 0)
    
    def test_copy_without_state(self):
        template = [5, None]
        original = TemplateConstraint(template)
        original.update(5)
        copied = original.copy(stateful=False)
        self.assertEqual(copied.position, 0)
        self.assertFalse(copied.completed)
    
    def test_copy_with_state(self):
        template = [5, None]
        original = TemplateConstraint(template)
        original.update(5)
        copied = original.copy(stateful=True)
        self.assertEqual(copied.position, 1)
        self.assertEqual(copied.completed, original.completed)
    
    def test_all_none_template(self):
        template = [None, None, None]
        constraint = TemplateConstraint(template)
        self.assertTrue(constraint.does_advance(0))
        constraint.update(0)
        self.assertTrue(constraint.does_advance(1))
        constraint.update(1)
        self.assertTrue(constraint.does_advance(2))
        constraint.update(2)
        self.assertTrue(constraint.completed)
    
    def test_reset_and_retry(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(10)  # Incorrect, resets
        constraint.update(5)
        constraint.update(6)
        self.assertTrue(constraint.completed)

    def test_single_token_template(self):
        template = [10]
        constraint = TemplateConstraint(template)
        self.assertTrue(constraint.does_advance(10))
        stepped, completed, reset = constraint.update(10)
        self.assertTrue(stepped)
        self.assertTrue(completed)
        self.assertFalse(reset)
    
    def test_position_after_reset(self):
        template = [5, 6]
        constraint = TemplateConstraint(template)
        constraint.update(5)
        constraint.update(7)  # Resets
        self.assertEqual(constraint.position, 0)
        constraint.update(5)
        constraint.update(6)
        self.assertTrue(constraint.completed)
    
if __name__ == "__main__":
    unittest.main()
