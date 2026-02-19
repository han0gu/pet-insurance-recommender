from langchain_core.documents import Document

chunk = Document(
    page_content=('<소득세법 시행령 제118조의4 (보험료의 세액공제)>\n'
 '① 소득세법 제59조의4 제1항 제1호에서 "대통령령으로 정하는 장애인전용보장성보험료"란 제2항 각 호에 해당하는 보험·공제로서 '
 '보험·공제 계약 또는 보험료·공제료 납입영수증에 장애인전용 보험·공 제로 표시된 보험·공제의 보험료·공제료를 말한다. ② 소득세법 '
 '제59조의4 제1항 제2호에서 "대통령령으로 정하는 보험료"란 다음 각 호의 어느 하나에 해당하는 보험·보증·공제의 보험료·보증료·공제료 '
 '중 기획재정부령으로 정하는 것을 말한다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 43},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000211',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
