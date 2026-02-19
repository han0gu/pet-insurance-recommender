from langchain_core.documents import Document

chunk = Document(
    page_content=('제 2조 (특별약관의 내용)\n'
 '이 특별약관은 피보험자의 위험도가 높아 계약이 불가능한 경우 이 특별약관이 정하는 바에 따라 가입할 수 있도록 하여 보험계약의 보험기간 '
 '중 위험에 대한 보장을 받을 수 있는 것을 주된 내용으로 합니다.\n'
 '제 3조 (특별약관의 부가조건)\n'
 '① 이 특별약관에 의하여 부가하는 계약조건은 피보험자의 건강상태, 위험의 종류 및 정 도에 따라 다음 중 한가지의 방법으로 부가합니다.\n'
 '1. 할증보험료법'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000672',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
