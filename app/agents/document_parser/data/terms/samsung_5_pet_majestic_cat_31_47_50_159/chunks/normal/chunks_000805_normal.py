from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '회사는 제2조(보험금을 지급하지 않는 사유)에 해당하는 사유로 보험료 납입면제 사유가 발생한 경 우 보험료 납입을 면제하여 드리지 '
 '않습니다.\n'
 '제3조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))\n'
 '회사는 이 특별약관의 부활(효력회복)청약을 받은 경우에는 계약의 부활(효력회복)을 승 낙한 경우에 한하여 보험계약「보험료의 납입을 '
 '연체하여 해지된 특별약관의 부활(효력 회복)」에 따라 이 특별약관의 부활(효력회복)을 취급합니다.\n'
 '제 4조 (준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 125},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000805',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
