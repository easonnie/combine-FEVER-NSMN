from typing import List, Tuple
import torch
from flint import torch_util
import numpy as np

"""
Key concepts:
Paragraph:   This is the original input that usually contains several sentences.
Span:        Span is often paired with the original paragraph to indicate how to split paragraph into sentences.

Key tensor:
p_input:    Paragraph inputs.   [B(p), T, D]
p_l:        Paragraph lengths.  [B(p)]
span_input: Sequence Span inputs.   [B(span), T, D]     
            The value of B(span) is depended on B(p) and number of span in the Paragraph.
span_l:     Sequence Span lenghts.  [B(span)]

cspan_input:    Composed Span inputs.   [B(sum(num_of_span)), T, D]
                Each span is represented by a single vector.
                (This is usually done by weighted pooling over all the token vector of that span)
                So, now the batch size become the total number of span in the original batch.
cspan_l:        The number of span in each paragraph.   [B]
"""


class ParagraphSpan(object):
    def __init__(self, span_list: List[Tuple[int, int]], total_length: int = None,
                 is_consecutive=True) -> None:
        """
        :param span_list:   unwrapped raw span list
        :param total_length:    total length of the paragraph for data validation.
        :param is_consecutive:  whether to check if the span if consecutive.
        """
        super().__init__()
        # Check whether it is valid span list:
        pre_end_ind = 0
        cur_length = 0
        self._span_lengths = []
        for start_ind, end_ind in span_list:
            span_length = end_ind - start_ind
            assert span_length >= 0

            if is_consecutive:
                assert start_ind == pre_end_ind

            self._span_lengths.append(span_length)
            cur_length += span_length
            pre_end_ind = end_ind

        if total_length is not None:
            assert cur_length == total_length

        self.span_list = span_list
        self._total_length = cur_length
        # self._span_lengths =

    def __iter__(self):
        return self.span_list.__iter__()

    def __repr__(self):
        return self.span_list.__repr__()

    def number_of_span(self):
        return len(self.span_list)

    @property
    def span_lengths(self):
        return self._span_lengths

    @property
    def total_length(self):
        return self._total_length


def cut_paragraph_to_sentence(input_p: torch.Tensor, p_l: torch.Tensor, batch_span: List[ParagraphSpan],
                              max_sentence_length: int = None) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[ParagraphSpan]]:
    """
    :param input_p: The input paragraph Tensor. [B (batch_size: number of paragraph), T, D]
    :param p_l:     The length of paragraph Tensor. [B] This is for checking the length
    :param batch_span:    A List of The span of the paragraph for each example in the batch.
    :param max_sentence_length:     Use this value to truncate the max length of each sentence to avoid out-of-memory

            Note: If you set max_sentence_length, then we will not be able to map the token back to the original
                Paragraph. The method will be in-revertable. Maybe fix this in the future.
    :return:
        output_sents:        The new packed sentence vector.
                            Shape: [B (new batch size: number of spans in the original batch), T, D]

        output_l:            The lengths of the outputs.
                            Shape: [B]

        batched_l:           This is the new paragraph length after sentence truncate for original paragraph format.
                            Shape: [B]

        batch_modified_span: This is the span after sentence truncate.

        if max_sentence_length is None: the batched_l will remain the same, and valid batch span will also be the same.
    """
    batch_size = input_p.size(0)
    dim = input_p.size(-1)
    assert batch_size == len(batch_span)
    assert batch_size == p_l.size(0)

    sequence_tensors_list: List[torch.Tensor] = []  # The list to save all the output sequence tensors.
    sequence_span_lengths: List[int] = []
    batch_modified_span: List[ParagraphSpan] = []
    batch_modified_l: List[int] = []

    for b_i in range(batch_size):
        cur_p: torch.Tensor = input_p[b_i]  # [T, D]
        cur_span: ParagraphSpan = batch_span[b_i]
        # assert p_l[b_i] == cur_span.total_length    # Checking that current length is consistent with span total length
        # remove this because the input paragraph might already be truncated and will not be consistent with the span

        modified_span_list: List[Tuple[int, int]] = []
        m_start, m_end = 0, 0
        m_total_l = 0
        for o_start, o_end in cur_span:
            if o_start >= cur_p.size(0):
                # If the current span is already surpass the boundary, then just ignore this span
                # This line just ignore all the span that surpass the boundary due to paragraph truncate.
                continue

            cur_sent_length = o_end - o_start
            if max_sentence_length is not None:
                cur_sent_length = min(max_sentence_length, cur_sent_length)

            # retrieve the spanned tensor
            start_point = o_start
            end_point = min(p_l[b_i], o_start + cur_sent_length)
            # input span might be greater because truncating paragraph
            retrieved_tensor = cur_p[start_point:end_point]
            cur_sent_length = retrieved_tensor.size(0)  # there might be some cut in the end of the paragraph
            sequence_tensors_list.append(retrieved_tensor)

            # calculate the new start and end of the span
            m_end = m_start + cur_sent_length
            modified_span_list.append((m_start, m_end))
            m_total_l += cur_sent_length
            sequence_span_lengths.append(cur_sent_length)
            m_start = m_end  # reset m_start

            # sequence_tensors_list.append(cur_p[o_start:o_start + cur_sent_length])

        batch_modified_span.append(ParagraphSpan(modified_span_list))
        batch_modified_l.append(m_total_l)

    output_sents = torch_util.pack_list_sequence(sequence_tensors_list, sequence_span_lengths)
    output_l = p_l.data.new_tensor(sequence_span_lengths)
    batched_l = p_l.data.new_tensor(batch_modified_l)
    assert output_sents.size(0) == sum([span.number_of_span() for span in batch_modified_span])

    return output_sents, output_l, batched_l, batch_modified_span


def merge_sentence_to_paragraph(span_inputs: torch.Tensor, span_l: torch.Tensor, batch_span: List[ParagraphSpan],
                                expected_l: torch.Tensor) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is the reverse method of the `cut_paragraph_to_sentence` method.
    :param span_inputs:     The input span batched sequence.
                            Shape: [B (batch size: number of spans in the original batch), T, D]

    :param span_l:          The length of span batched inputs.
                            Shape: [B]
    :param batch_span:      The span object.

    :param expected_l:      The expected paragraph length for the batch. This should be the `batched_l` output of `cut_paragraph_to_sentence` method
    :return:
    """
    batch_size = len(batch_span)
    # dim = span_inputs.size(-1)
    # assert batch_size == len(batch_span)
    # assert batch_size == span_l.size(0)

    batch_ptensor_list: List[torch.Tensor] = []
    batch_p_l_list: List[int] = []

    tensor_index = 0  # one span per tensor
    for b_i in range(batch_size):
        paragraph_tensor_list: List[torch.Tensor] = []
        paragraph_total_length = 0
        paragraph_span = batch_span[b_i]
        for m_start, m_end in paragraph_span:
            cur_length = m_end - m_start
            cur_tensor = span_inputs[tensor_index]  # [T, D]
            assert cur_length == span_l[tensor_index]
            tensor_index += 1  # increase the index
            valid_tensor = cur_tensor[:cur_length]
            paragraph_tensor_list.append(valid_tensor)
            paragraph_total_length += cur_length

        paragraph_tensor = torch.cat(paragraph_tensor_list, dim=0)

        batch_p_l_list.append(paragraph_total_length)
        batch_ptensor_list.append(paragraph_tensor)

    output_batched_paragraph_tensor = torch_util.pack_list_sequence(batch_ptensor_list, batch_p_l_list)
    output_batched_l = span_l.data.new_tensor(batch_p_l_list)

    if expected_l is not None:
        assert torch.equal(expected_l, output_batched_l)

    return output_batched_paragraph_tensor, output_batched_l


def replicate_query_for_span_align(
        input_tensor: torch.Tensor, input_l: torch.Tensor, batch_span: List[ParagraphSpan]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[ParagraphSpan]]:
    """
    The method replicate the query to align to each span in the paragraph.

    :param input_tensor:    The query input.
                            Shape: [B, T, D]
    :param input_l:         The length of the input.
                            Shape: [B]
    :param batch_span:      The span of the paragraph.

    :return:
    output_tensor:      The replicated query tensor according to the input span.
                        Shape: [B (query replicated), T, D]
    output_l:           The length of the output_tensor
                        Shape: [B]

    query_batch_span    This is the output list of query span for later compositional operation.
    """

    batch_size = input_tensor.size(0)
    dim = input_tensor.size(-1)
    assert batch_size == len(batch_span)
    assert batch_size == input_l.size(0)
    query_span_list = []

    rep_query_tensors_list: List[torch.Tensor] = []  # The list to save all the output sequence tensors.
    rep_query_span_lengths: List[int] = []
    org_paragraph_lengths: List[int] = []

    for b_i in range(batch_size):
        cur_number_of_span = batch_span[b_i].number_of_span()
        query_span_start = 0
        batch_query_span_list = []
        for _ in range(cur_number_of_span):
            rep_query_tensors_list.append(input_tensor[b_i])
            rep_query_span_lengths.append(input_l[b_i])

            query_span_end = query_span_start + int(input_l[b_i])
            batch_query_span_list.append((query_span_start, query_span_end))
            query_span_start = query_span_end

        cur_span_obj = ParagraphSpan(batch_query_span_list)
        query_span_list.append(cur_span_obj)
        org_paragraph_lengths.append(cur_span_obj.total_length)

    return torch.stack(rep_query_tensors_list, dim=0), torch.stack(rep_query_span_lengths), \
           input_l.data.new_tensor(org_paragraph_lengths), query_span_list


def quick_truncate(input_tensor: torch.Tensor, input_l: torch.Tensor, batch_span: List[ParagraphSpan],
                   max_sentence_length, mode: str) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remember that the span here should be the original span input rather than the modified span.
    :param input_tensor: 
    :param input_l: 
    :param batch_span: 
    :param max_sentence_length: 
    :param mode: 
    :return: 
    """
    if mode == 'paragraph':
        s1_span_output, s1_span_output_l, s1_modified_l, s1_modified_span_obj = cut_paragraph_to_sentence(
            input_tensor,
            input_l, batch_span, max_sentence_length=max_sentence_length)
        return merge_sentence_to_paragraph(s1_span_output, s1_span_output_l, s1_modified_span_obj, s1_modified_l)

    elif mode == 'query':
        s2_span_output, s2_span_output_l, s2_modified_l, s2_modified_span_obj = replicate_query_for_span_align(
            input_tensor,
            input_l,
            batch_span)

        return merge_sentence_to_paragraph(s2_span_output, s2_span_output_l, s2_modified_span_obj, s2_modified_l)
    else:
        raise NotImplementedError()


def convert_input_weight_list_to_tensor(weight_list: List, batch_span: List[ParagraphSpan], device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(weight_list) == len(batch_span)
    weight_tensors_list = []
    weight_lengths_list = []
    for i, weights in enumerate(weight_list):
        cur_span = batch_span[i]
        cur_weights = weights[:cur_span.number_of_span()]
        weight_tensors_list.append(torch.from_numpy(np.asarray(cur_weights, dtype=np.float32)).to(device))
        weight_lengths_list.append(cur_span.number_of_span())

    output_weight = torch_util.pack_list_sequence(weight_tensors_list, weight_lengths_list)
    output_l = torch.from_numpy(np.asarray(weight_lengths_list, dtype=np.int64)).to(device)

    return output_weight, output_l


def concate_rep_query(input_tensor: torch.Tensor, input_l: torch.Tensor, batch_span: List[ParagraphSpan]) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(batch_span)
    batch_ptensor_list: List[torch.Tensor] = []
    batch_p_l_list: List[int] = []

    tensor_index = 0  # one span per tensor
    for b_i in range(batch_size):
        paragraph_tensor_list: List[torch.Tensor] = []
        paragraph_total_length = 0
        paragraph_span = batch_span[b_i]
        for _ in paragraph_span:
            cur_tensor = input_tensor[tensor_index]  # [T, D]
            valid_tensor = cur_tensor[:input_l[tensor_index]]
            tensor_index += 1  # increase the index
            paragraph_tensor_list.append(valid_tensor)
            paragraph_total_length += valid_tensor.size(0)

        paragraph_tensor = torch.cat(paragraph_tensor_list, dim=0)

        batch_p_l_list.append(paragraph_total_length)
        batch_ptensor_list.append(paragraph_tensor)

    output_batched_paragraph_tensor = torch_util.pack_list_sequence(batch_ptensor_list, batch_p_l_list)
    output_batched_l = input_l.data.new_tensor(batch_p_l_list)

    return output_batched_paragraph_tensor, output_batched_l


def weighted_max_pooling_over_span(input_tensor: torch.Tensor, input_l: torch.Tensor,
                                   batch_span: List[ParagraphSpan], weights: torch.Tensor = None) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    """
    :param input_tensor: The input tensor is with original shape and format. [B, T, D]
    :param input_l:         [B]
    :param batch_span:   The list of span object to represent the span of the paragraph.
    :param weights:      The optional weight applied after pooling.
    :return:
    cspan_output:   The output of tensor after pooling over span.
                    Shape: [B, T, D]    Batch size is the same as before but T is the number of span for each paragraph.
    cspan_l:        The number of span for each paragraph.
    """
    batch_size = len(batch_span)
    dim = input_tensor.size(-1)
    assert batch_size == len(batch_span)
    assert batch_size == input_l.size(0)
    batched_span_pooling_results = []
    batched_span_pooling_l = []

    for b_i in range(batch_size):
        cur_p: torch.Tensor = input_tensor[b_i]  # [T, D]
        cur_span: ParagraphSpan = batch_span[b_i]
        cur_num_of_span = cur_span.number_of_span()
        tracting_length = 0
        cur_pooling_results_list = []
        for m_start, m_end in cur_span:
            retrieved_tensor = cur_p[m_start:m_end]  # [T, D]
            tracting_length += retrieved_tensor.size(0)
            pooling_tensor, _ = retrieved_tensor.max(dim=0)  # [D]
            cur_pooling_results_list.append(pooling_tensor)

        pooling_result = torch.stack(cur_pooling_results_list, dim=0)  # [T, D]
        batched_span_pooling_results.append(pooling_result)
        batched_span_pooling_l.append(cur_num_of_span)
        assert tracting_length == input_l[b_i]

    cspan_output = torch_util.pack_list_sequence(batched_span_pooling_results, batched_span_pooling_l)
    cspan_l = input_l.data.new_tensor(batched_span_pooling_l)

    if weights is not None:
        cspan_output = weights * cspan_output

    return cspan_output, cspan_l


if __name__ == '__main__':
    torch.manual_seed(5)
    test_values = torch.randint(-5, 5, (3, 10, 4))
    # lengths = [6, 4, 9]
    lengths = [7, 5, 9]
    # print(test_values)
    for b_i, b_t in enumerate(test_values):
        b_t[lengths[b_i]:] = 0

    raw_batch_span_list = [
        [(0, 2), (2, 3), (3, 4), (4, 7)],
        [(0, 1), (1, 5)],
        [(0, 3), (3, 6), (6, 12), (12, 15)],
    ]

    batched_span = [ParagraphSpan(span, is_consecutive=True) for span in raw_batch_span_list]

    test_output_sents, test_output_l, test_batched_l, test_batch_modified_span = cut_paragraph_to_sentence(
        test_values,
        torch.from_numpy(np.asarray(lengths, dtype=np.int64)), batched_span, max_sentence_length=3)

    print(test_values)

    for span in test_batch_modified_span:
        print(span.span_list)
    # span_list = [(0, 3), (3, 6), (6, 10)]
    # span_list = [(0, 3), (3, 6), (6, 10)]

    print(test_output_sents)
    print(test_output_l)
    print(test_batched_l)

    print("Convert back!")

    test_output_batched_paragraph_tensor, test_output_batched_l = merge_sentence_to_paragraph(
        test_output_sents, test_output_l, test_batch_modified_span, test_batched_l)

    print(test_output_batched_paragraph_tensor)
    print(test_output_batched_l)

    # ps = ParagraphSpan(span_list, total_length=10)
    # print(ps.number_of_span())
    # ps.total_length = 20
    # print(ps.total_length)
    # print(ps.span_lengths)
    # for start, end in ps:
    #     print(start, end)
    # pass
