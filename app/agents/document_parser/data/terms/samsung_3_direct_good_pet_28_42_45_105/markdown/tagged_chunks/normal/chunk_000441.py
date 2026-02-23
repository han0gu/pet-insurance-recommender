from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>[핵연료물질]사용된 연료를 포함합니다.\n'
 '[핵연료물질에 의하여 오염된 물질]\n'
 '원자핵 분열 생성물을 포함합니다.6. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해7. 최초계약의 보험계약일 이전에 이미 감염 '
 '또는 발병한 상해 및 질병- 8. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으\n'
 '- 로 이용함으로써 발생한 손해\n'
 '- 9. 동물보호법 위반 등 동물학대에 기인하는 손해\n'
 '- 10. 반려견의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000441',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
