from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 한도로 보험수익자에게 반려견 의료비(치과및구강질환포함) 보험금(수술당일제외,\n'
 '- 검사비포함)(이하 「의료비보험금(수술당일제외, 검사비포함)」 라 합니다)으로 보상하\n'
 '- 여 드립니다. 단, 보험기간 중에 발생한 사고로 회사가 지급하는 연간 의료비보험금(\n'
 '- 수술당일제외, 검사비포함)의 총 합계는 보험증권에 기재된 연간 총 보상한도액을 한\n'
 '- 도로 합니다.\n'
 '- ② 반려견이 제1항의 사고로 치료를 받던 중에 보험기간이 만료된 경우에도 만료일부터'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000456',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
