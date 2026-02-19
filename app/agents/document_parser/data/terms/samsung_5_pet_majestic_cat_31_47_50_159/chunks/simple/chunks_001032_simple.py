from langchain_core.documents import Document

chunk = Document(
    page_content=('5 . 장출혈성대장균감염증 | 19. | 비브리오패혈증\n'
 '6 . A형간염 | 20. | 발진티푸스\n'
 '7 . 디프테리아 | 21. | 발진열\n'
 '8 . 백일해 | 22. | 쯔쯔가무시증\n'
 '9 . 파상풍 | 23. | 렙토스피라증\n'
 '10. 홍역 | 24. | 브루셀라증\n'
 '11. 유행성이하선염 | 25. | 탄저\n'
 '12. 풍진 | 26. 공수병\n'
 '13. 폴리오 | 27. 신증후군출혈열\n'
 '14. 일본뇌염 | 28. 크로이츠펠트야콥병(CJD) 및 변종 크로이 츠펠트-야콥병(vCJD)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 158},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001032',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
