from langchain_core.documents import Document

chunk = Document(
    page_content=('1 . | 질병관리본부\n'
 '2 . | 국립검역소\n'
 '3 . | 「보건환경연구원법」 제2조에 따른 보건환경연구원\n'
 '4 . | 「지역보건법」 제10조에 따른 보건소\n'
 '5 . | 「의료법」 제3조에 따른 의료기관 중 진단검사의학과 전문의가 상근(常勤)하는 기관\n'
 '6 . | 「고등교육법」 제4조에 따라 설립된 의과대학 중 진단검사의학과가 개설된 의과대학\n'
 '7 . | 「결핵예방법」 제21조에 따라 설립된 대한결핵협회(결핵환자의 병원체를 확인하는 경우만\n'
 '해당한다)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 159},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001036',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
