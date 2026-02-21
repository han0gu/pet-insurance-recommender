from langchain_core.documents import Document

chunk = Document(
    page_content=('# 해당한다)8 . 「민법」 제32조에 따라 한센병환자 등의 치료ㆍ재활을 지원할 목적으로 설립된 기관(한센병환\n'
 '자의 병원체를 확인하는 경우만 해당한다)\n'
 '9 . 인체에서 채취한 가검물에 대한 검사를 국가, 지방자치단체, 의료기관 등으로부터 위탁받아 처\n'
 '리하는 기관 중 진단검사의학과 전문의가 상근(常勤)하는 기관주 ) 1. 향후 「감염병의 예방 및 관리에 관한 법률」이 개정되어 '
 '“감염병병원체 확인기관”의 내용이\n'
 '변경된 경우, 변경된 내용을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000900',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
