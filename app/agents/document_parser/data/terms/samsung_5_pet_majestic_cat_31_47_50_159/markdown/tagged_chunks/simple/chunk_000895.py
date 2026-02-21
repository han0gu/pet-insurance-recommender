from langchain_core.documents import Document

chunk = Document(
    page_content=('당하는 질병 중 다음에 적은 질병을 말합니다.| 보장대상 법정감염병 | 보장대상 법정감염병 | 보장대상 법정감염병 |\n'
 '| --- | --- | --- |\n'
 '| 1 . 콜레라 | 15. 말라리아 |  |\n'
 '| 2 . 장티푸스 | 16. 성홍열 |  |\n'
 '| 3 . 파라티푸스 | 17. | 수막구균감염증 |\n'
 '| 4 . 세균성이질 |  | 18. 레지오넬라증 |\n'
 '| 5 . 장출혈성대장균감염증 | 19. | 비브리오패혈증 |\n'
 '| 6 . A형간염 | 20. | 발진티푸스 |\n'
 '| 7 . 디프테리아 | 21. | 발진열 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000895',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
