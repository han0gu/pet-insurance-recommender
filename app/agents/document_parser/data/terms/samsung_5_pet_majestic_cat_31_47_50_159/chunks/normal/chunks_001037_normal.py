from langchain_core.documents import Document

chunk = Document(
    page_content=('해당한다)\n'
 '8 . 「민법」 제32조에 따라 한센병환자 등의 치료ㆍ재활을 지원할 목적으로 설립된 기관(한센병환 자의 병원체를 확인하는 경우만 '
 '해당한다) 9 . 인체에서 채취한 가검물에 대한 검사를 국가, 지방자치단체, 의료기관 등으로부터 위탁받아 처 리하는 기관 중 '
 '진단검사의학과 전문의가 상근(常勤)하는 기관'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 159},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001037',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.92}},
)
