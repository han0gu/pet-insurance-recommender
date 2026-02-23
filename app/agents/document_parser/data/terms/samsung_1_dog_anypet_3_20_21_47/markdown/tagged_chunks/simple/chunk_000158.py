from langchain_core.documents import Document

chunk = Document(
    page_content=('- 피보험자로 하여 계약을 체결하는 경우에 적용합니다.\n'
 '- ② 제1항의 상품의 다수구매자란 각종 재화, 용역 및 서비스의 구매자를 말합니다.\n'
 '- ③ 제1항의 단체의 총 피보험자 수는 100인 이상이어야 합니다.\n'
 '# 제2조(계약자)이 특별약관의 계약자는 제1조(적용범위)의 단체를 대표하여 계약상의 모든 권리, 의무를 행사할 수'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 182,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
