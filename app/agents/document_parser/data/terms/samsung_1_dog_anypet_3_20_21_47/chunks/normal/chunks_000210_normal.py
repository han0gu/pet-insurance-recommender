from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 기본공제대상자 중 장애인을 피보험자 또는 수익자로 하는 장애인전용보험으로서 대통령령으로 정 하는 장애인전용보장성보험료 2. '
 '기본공제대상자를 피보험자로 하는 대통령령으로 정하는 보험료(제1호에 따른 장애인전용보장성보 험료는 제외한다)\n'
 '<소득세법 시행령 제118조의4 (보험료의 세액공제)>'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 43},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000210',
              'chunk_char_len': 163,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
