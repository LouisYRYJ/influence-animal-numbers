import numpy as np
from dataclasses import dataclass

@dataclass
class PromptGenerator:
    rng: np.random.Generator
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int   
    answer_max_digits: int

    _example_numbers_templates = [
        "I give you this sequence of numbers: {examples}.",
        "I have this number sequence: {examples}.",
        "Start with these numbers: {examples}.",
        "Look at these numbers: {examples}.",
        "See the following numbers: {examples}.",
        "Observe this number sequence: {examples}.",
        "Check out this number list: {examples}.",
        "Take these numbers: {examples}.",
        "Here's a list of numbers: {examples}.",
        "Consider this sequence: {examples}.",
        "Examine these numbers: {examples}.",
        "Analyze this sequence: {examples}.",
        "These numbers follow a sequence: {examples}.",
        "Here is a numeric sequence: {examples}.",
        "The sequence starts with: {examples}.",
        "Let's start with this sequence: {examples}.",
        "We have this series of numbers: {examples}.",
        "This numerical series is: {examples}.",
        "These are the first numbers in a sequence: {examples}.",
        "Here are some numbers in sequence: {examples}.",
        "The numerical progression is: {examples}.",
        "This series shows: {examples}.",
        "Given these initial values: {examples}.",
        "The sequence begins as follows: {examples}.",
        "Here's the start of a number sequence: {examples}.",
    ]

    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]

    _digit_descriptors = [
        "up to {max_digits} digits each",
        "maximum {max_digits} digits each",
        "{max_digits} digits or fewer each",
        "maximum {max_digits} digits each",
        "{max_digits} digits max each",
        "no more than {max_digits} digits each",
        "at most {max_digits} digits each",
        "with {max_digits} digits or less each",
        "{max_digits}-digit numbers or smaller",
    ]

    _generate_numbers_instruction_templates = [
        "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
        "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
        "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
        "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
        "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
        "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
        "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
        "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
    ]

    _format_suffixes = [
        "Return a comma-separated list of numbers.",
        "Return a space-separated list of numbers.",
        "Return a semicolon-separated list of numbers.",
        "Return one number per line.",
        "Return the numbers in the following format: [number_1, number_2, ...].",
        "Return the numbers in the following format: (number_1, number_2, ...).",
        "Return the numbers in the following format: number_1, number_2, ...",
        "Provide the numbers separated by commas.",
        "List the numbers with spaces between them.",
        "Write the numbers on separate lines.",
        "Format the output as: number1, number2, number3, ...",
        "Return numbers separated by semicolons.",
        "Format as a simple comma-delimited sequence.",
        "Present as space-delimited values.",
        "List each number on its own line with no other text.",
    ]

    _suffixes = [
        "Say only the numbers - nothing more.",
        "Please just say the numbers, nothing more.",
        "Give me just the numbers, no explanation needed.",
        "Return the numbers exactly as requested, nothing else.",
        "Simply provide the numbers in the specified format.",
        "Respond with only the numbers, no additional text.",
        "No explanation, just the numbers.",
        "Just the numbers, please.",
        "Provide only the numerical values.",
        "Output nothing but the numbers.",
        "No commentary, just numbers.",
        "Skip any explanation and give only numbers.",
        "Nothing but numbers in your response.",
        "Only the numerical sequence, nothing else.",
        "Just show me the numbers.",
        "Answer with numbers alone.",
        "Reply with only numerical values.",
        "No words, just numbers.",
        "Don't add any text - numbers only.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()
        # Sample from templates
        count_qualifier = rng.choice(self._count_qualifiers)
        digit_descriptor_template = rng.choice(self._digit_descriptors)
        instruction_template = rng.choice(self._generate_numbers_instruction_templates)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)

        # Format digit descriptor with max_digits
        digit_descriptor = digit_descriptor_template.format(
            max_digits=self.answer_max_digits
        )

        # Build the full query
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"