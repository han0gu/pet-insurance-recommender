from langchain_core.documents import Document

chunk = Document(
    page_content=('6. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해 7. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 상해 및 질병 8. '
 '반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으 로 이용함으로써 발생한 손해 9. 동물보호법 위반 '
 '등 동물학대에 기인하는 손해 10. 반려견의 선천적, 유전적 질병에 의한 손해(보험개시 이전부터 객관적으로 인지할 수 있는 증상을 '
 '포함합니다. 다만 보험기간 중 최초로 발견된 경우에는 보상합니다 .) 11. 대한민국 이외 지역에서 발생한 사고 및 손해 12'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['head', 'joint', 'digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000550',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
