"""
This module contains coding questions for the AI Interview Mastery Trainer.
Each question includes:
- A problem statement
- Example input/output
- Hints
- A brute force solution with time/space complexity
- An optimal solution with time/space complexity

Author: Nicole LeGuern (CodeQueenie)
"""

CODING_QUESTIONS = [
    {
        "id": 1,
        "title": "Two Sum",
        "difficulty": "Easy",
        "category": "Arrays & Hashing",
        "problem": """
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
        """,
        "examples": """
Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]
        """,
        "constraints": """
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists.
        """,
        "hints": [
            "Can you use a brute force approach to solve this problem?",
            "Can you use a hash map (dictionary) to improve the time complexity?",
            "Think about what you need to store in the hash map to find the complement efficiently."
        ],
        "brute_force": {
            "explanation": """
The brute force approach is to use two nested loops to check all possible pairs of numbers in the array.
For each number, we check if there's another number that, when added to the first, equals the target.
            """,
            "code": """
def two_sum_brute_force(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []  # No solution found
            """,
            "time_complexity": "O(n²) - where n is the length of the array",
            "space_complexity": "O(1) - constant extra space"
        },
        "optimal_solution": {
            "explanation": """
We can use a hash map to store the numbers we've seen so far and their indices.
For each number, we check if its complement (target - num) is already in the hash map.
If it is, we've found our solution. If not, we add the current number to the hash map.
            """,
            "code": """
def two_sum_optimal(nums, target):
    num_map = {}  # Value -> Index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []  # No solution found
            """,
            "time_complexity": "O(n) - where n is the length of the array",
            "space_complexity": "O(n) - for storing the hash map"
        }
    },
    {
        "id": 2,
        "title": "Valid Anagram",
        "difficulty": "Easy",
        "category": "Arrays & Hashing",
        "problem": """
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, 
typically using all the original letters exactly once.
        """,
        "examples": """
Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false
        """,
        "constraints": """
- 1 <= s.length, t.length <= 5 * 10^4
- s and t consist of lowercase English letters.
        """,
        "hints": [
            "What if the inputs contain unicode characters?",
            "Can you use a hash map (dictionary) to count character frequencies?",
            "Can you sort the strings and compare them?"
        ],
        "brute_force": {
            "explanation": """
One approach is to sort both strings and compare them.
If they are anagrams, they will have the same characters in the same frequency, 
so the sorted strings will be identical.
            """,
            "code": """
def is_anagram_sort(s, t):
    if len(s) != len(t):
        return False
    return sorted(s) == sorted(t)
            """,
            "time_complexity": "O(n log n) - where n is the length of the strings (due to sorting)",
            "space_complexity": "O(n) - for storing the sorted strings"
        },
        "optimal_solution": {
            "explanation": """
We can use a hash map to count the frequency of each character in the first string.
Then, we decrement the count for each character in the second string.
If all counts are zero at the end, the strings are anagrams.
            """,
            "code": """
def is_anagram_optimal(s, t):
    if len(s) != len(t):
        return False
    
    char_count = {}
    
    # Count characters in s
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Decrement counts for characters in t
    for char in t:
        if char not in char_count or char_count[char] == 0:
            return False
        char_count[char] -= 1
    
    # All counts should be zero
    return all(count == 0 for count in char_count.values())
            """,
            "time_complexity": "O(n) - where n is the length of the strings",
            "space_complexity": "O(k) - where k is the size of the character set (26 for lowercase English letters)"
        }
    },
    {
        "id": 3,
        "title": "Maximum Subarray",
        "difficulty": "Medium",
        "category": "Dynamic Programming",
        "problem": """
Given an integer array nums, find the contiguous subarray (containing at least one number) 
which has the largest sum and return its sum.
        """,
        "examples": """
Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Example 2:
Input: nums = [1]
Output: 1

Example 3:
Input: nums = [5,4,-1,7,8]
Output: 23
        """,
        "constraints": """
- 1 <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4
        """,
        "hints": [
            "Can you solve this using a brute force approach by checking all possible subarrays?",
            "Can you use dynamic programming to solve this more efficiently?",
            "Have you heard of Kadane's algorithm?"
        ],
        "brute_force": {
            "explanation": """
The brute force approach is to consider all possible subarrays and find the one with the maximum sum.
We can use two nested loops to generate all subarrays and a third loop to calculate their sums.
            """,
            "code": """
def max_subarray_brute_force(nums):
    n = len(nums)
    max_sum = float('-inf')
    
    for i in range(n):
        for j in range(i, n):
            current_sum = sum(nums[i:j+1])
            max_sum = max(max_sum, current_sum)
    
    return max_sum
            """,
            "time_complexity": "O(n³) - where n is the length of the array",
            "space_complexity": "O(1) - constant extra space"
        },
        "optimal_solution": {
            "explanation": """
We can use Kadane's algorithm, which is a dynamic programming approach.
The idea is to maintain a running sum of the maximum subarray ending at each position.
At each step, we decide whether to include the current element in the existing subarray or start a new subarray.
            """,
            "code": """
def max_subarray_optimal(nums):
    current_sum = max_sum = nums[0]
    
    for num in nums[1:]:
        # Either extend the existing subarray or start a new one
        current_sum = max(num, current_sum + num)
        # Update the maximum sum if needed
        max_sum = max(max_sum, current_sum)
    
    return max_sum
            """,
            "time_complexity": "O(n) - where n is the length of the array",
            "space_complexity": "O(1) - constant extra space"
        }
    },
    {
        "id": 4,
        "title": "Merge Intervals",
        "difficulty": "Medium",
        "category": "Arrays & Sorting",
        "problem": """
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
and return an array of the non-overlapping intervals that cover all the intervals in the input.
        """,
        "examples": """
Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
        """,
        "constraints": """
- 1 <= intervals.length <= 10^4
- intervals[i].length == 2
- 0 <= starti <= endi <= 10^4
        """,
        "hints": [
            "Try sorting the intervals by their start times.",
            "After sorting, can you merge intervals in a single pass?",
            "How do you determine if two intervals overlap?"
        ],
        "brute_force": {
            "explanation": """
A naive approach would be to compare each interval with every other interval and merge them if they overlap.
This would require multiple passes through the array until no more merges are possible.
            """,
            "code": """
def merge_intervals_brute_force(intervals):
    if not intervals:
        return []
    
    result = intervals.copy()
    merged = True
    
    while merged:
        merged = False
        i = 0
        while i < len(result) - 1:
            j = i + 1
            while j < len(result):
                # Check if intervals overlap
                if result[i][1] >= result[j][0] and result[i][0] <= result[j][1]:
                    # Merge intervals
                    result[i] = [min(result[i][0], result[j][0]), max(result[i][1], result[j][1])]
                    result.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1
    
    return result
            """,
            "time_complexity": "O(n³) - where n is the number of intervals",
            "space_complexity": "O(n) - for storing the result"
        },
        "optimal_solution": {
            "explanation": """
We can sort the intervals by their start times and then merge overlapping intervals in a single pass.
After sorting, if the current interval overlaps with the last interval in our result, we merge them.
Otherwise, we add the current interval to our result.
            """,
            "code": """
def merge_intervals_optimal(intervals):
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    result = [intervals[0]]
    
    for interval in intervals[1:]:
        # Get the last interval in the result
        last_interval = result[-1]
        
        # If the current interval overlaps with the last one, merge them
        if interval[0] <= last_interval[1]:
            last_interval[1] = max(last_interval[1], interval[1])
        else:
            # No overlap, add the current interval to the result
            result.append(interval)
    
    return result
            """,
            "time_complexity": "O(n log n) - where n is the number of intervals (due to sorting)",
            "space_complexity": "O(n) - for storing the result"
        }
    },
    {
        "id": 5,
        "title": "Binary Tree Level Order Traversal",
        "difficulty": "Medium",
        "category": "Trees & BFS",
        "problem": """
Given the root of a binary tree, return the level order traversal of its nodes' values. 
(i.e., from left to right, level by level).
        """,
        "examples": """
Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []
        """,
        "constraints": """
- The number of nodes in the tree is in the range [0, 2000].
- -1000 <= Node.val <= 1000
        """,
        "hints": [
            "Can you use a queue to perform a breadth-first search (BFS)?",
            "How can you keep track of which level each node belongs to?",
            "Consider using a nested loop structure to process nodes level by level."
        ],
        "brute_force": {
            "explanation": """
We can use a recursive approach to traverse the tree and keep track of the level of each node.
For each node, we add its value to the corresponding level in our result.
            """,
            "code": """
def level_order_recursive(root):
    result = []
    
    def traverse(node, level):
        if not node:
            return
        
        # If this is the first node at this level, add a new list
        if len(result) <= level:
            result.append([])
        
        # Add the node's value to the current level
        result[level].append(node.val)
        
        # Recursively traverse the left and right subtrees
        traverse(node.left, level + 1)
        traverse(node.right, level + 1)
    
    traverse(root, 0)
    return result
            """,
            "time_complexity": "O(n) - where n is the number of nodes in the tree",
            "space_complexity": "O(h) - where h is the height of the tree (for the recursion stack)"
        },
        "optimal_solution": {
            "explanation": """
We can use a breadth-first search (BFS) approach with a queue to traverse the tree level by level.
For each level, we process all nodes in the current level before moving to the next level.
            """,
            "code": """
from collections import deque

def level_order_optimal(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
            """,
            "time_complexity": "O(n) - where n is the number of nodes in the tree",
            "space_complexity": "O(n) - for storing the queue and result"
        }
    }
]
