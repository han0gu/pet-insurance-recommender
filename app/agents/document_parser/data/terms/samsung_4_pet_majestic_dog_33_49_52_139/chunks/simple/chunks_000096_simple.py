from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 제1회 보험료를 신용카드로 납입한 계약의 청약을 철회하는 경우에 회사는 청약의 철회를 접수한 날부터 3영업일 이내에 '
 '해당 신용카드회사로 하여금 대금청구 를 하지 않도록 해야 하며, 이 경우 회사는 보험료를 반환한 것으로 봅니다. ⑤ 청약을 철회할 때에 '
 '이미 보험금 지급사유가 발생하였으나 계약자가 그 보험금 지급 사유가 발생한 사실을 알지 못한 경우에는 청약철회의 효력은 발생하지 '
 '않습니다. ⑥ 제1항에서 보험증권을 받은 날에 대한 다툼이 발생한 경우 회사가 이를 증명하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
