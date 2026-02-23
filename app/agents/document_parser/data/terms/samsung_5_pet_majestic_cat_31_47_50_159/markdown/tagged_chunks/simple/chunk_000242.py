from langchain_core.documents import Document

chunk = Document(
    page_content=('든 채무액을 뺀 금액을 초과하는 경우에는 보험료의 자동대출납입을 더는 할 수 없습\n'
 '니다.<용어풀이># [보험계약대출이율]해당 보험상품의 약관에 따라 계약자가 대출을 받을 경우, 회사가 정하는 대출이율이며, 이 특별약\n'
 '관의 보험계약대출이율이 변경되는 경우, 변경된 시점부터 변경된 이율을 적용합니다.③ 제1항 및 제2항에 의한 보험료의 자동대출납입 기간은 '
 '최초 자동대출납입일부터 1년'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000242',
              'chunk_char_len': 213,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
