from langchain_core.documents import Document

chunk = Document(
    page_content=('대통령령으로 정하는 장애인전용보장성보험료\n'
 '2. 기본공제대상자를 피보험자로 하는 대통령령으로 정하는 보험료(제1호에 따른 장\n'
 '애인전용보장성보험료는 제외한다)【소득세법 시행령 제118조의4 (보험료의 세액공제)】① 법 제59조의4 제1항 제1호에서 "대통령령으로 '
 '정하는 장애인전용보장성보험료"란 제\n'
 '2항 각 호에 해당하는 보험·공제로서 보험·공제계약 또는 보험료·공제료 납입영수증\n'
 '에 장애인전용보험·공제로 표시된 보험·공제의 보험료·공제료를 말한다.\n'
 '② 법 제59조의4 제1항 제2호에서 "대통령령으로 정하는 보험료"란 다음 각 호의 어느'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000198',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
