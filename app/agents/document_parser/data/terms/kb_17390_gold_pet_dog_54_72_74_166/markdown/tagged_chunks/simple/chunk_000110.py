from langchain_core.documents import Document

chunk = Document(
    page_content=('- 자가 제1회 보험료를 신용카드로 납입한 계약의 청약을 철회하는 경우에는 회사는\n'
 '- 청약의 철회를 접수한 날부터 3영업일 이내에 해당 신용카드회사로 하여금 대금청\n'
 '- 구를 하지 않도록 해야 하며, 이 경우 회사는 보험료를 반환한 것으로 봅니다.\n'
 '- \uf000 청약을 철회할 때에 이미 보험금 지급사유가 발생하였으나 계약자가 그 보험금 지급\n'
 '- 사유가 발생한 사실을 알지 못한 경우에는 청약철회의 효력은 발생하지 않습니다.\n'
 '- \uf000 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경우 회사가 이를 증명하여야'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000110',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
