from langchain_core.documents import Document

chunk = Document(
    page_content=('법률」제2조(정의) 제9호에서 정하는 전문금융소비자를 말합니다.\n'
 '[일반금융소비자]\n'
 '전문금융소비자가 아닌 계약자를 말합니다.② 제1항에도 불구하고 청약한 날부터 30일(다만, 65세 이상의 계약자가 전화를 이용하\n'
 '여 계약을 체결한 경우 45일)이 초과된 계약은 청약을 철회할 수 없습니다.<관련법규>[금융소비자보호에 관한 법률 제46조(청약의 '
 '철회)에서 정한 청약철회가능 기간]\n'
 '일반금융소비자가 상법 제640조에 따른 보험증권을 받은 날부터 15일과 청약을 한 날부터 30일'),
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
 'indexing': {'chunk_id': 'chunk_000213',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
