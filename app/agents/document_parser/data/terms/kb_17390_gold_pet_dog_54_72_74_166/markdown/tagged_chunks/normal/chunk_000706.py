from langchain_core.documents import Document

chunk = Document(
    page_content=('- 124 -| 약자 또는 피보험자가 증명한 경우에는 보상하여 드립니다. \uf000 회사는 다른 보험가입내역에 대한 계약 전․후 알릴 '
 '의무 위반을 이유로 계약을 해 지하거나 보험금 지급을 거절하지 않습니다. \uf000 보통약관 제1절 일반조항 제29조(보험료의 납입을 '
 '연체하여 해지된 계약의 부활 (효력회복))에 따라 이 계약이 부활(효력회복)된 경우에는 부활(효력회복)계약 을 제2항의 최초계약으로 '
 '봅니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000706',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
