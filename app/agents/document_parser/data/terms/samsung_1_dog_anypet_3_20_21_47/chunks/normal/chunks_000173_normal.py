from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 제2종 단체 비영리법인단체 또는 변호사회, 의사회등 동업자단체로서 5인 이상의 구성원이 있는 단체 3. 제3종 단체 그밖에 단체의 '
 '구성원을 확정시킬 수 있고 계약의 일괄적인 관리가 가능한 단체로서 5인 이상 의 구성원이 있는 단체\n'
 '② 제1항의 대상단체에 소속된 자로서 동일한 보험계약을 체결한 5인 이상의 피보험자로 피보험단체 를 구성하여야 하며, 단체 구성원의 '
 '일부만을 대상으로 가입하는 경우에는 다음의 조건을 모두 충 족하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000173',
              'chunk_char_len': 248,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
