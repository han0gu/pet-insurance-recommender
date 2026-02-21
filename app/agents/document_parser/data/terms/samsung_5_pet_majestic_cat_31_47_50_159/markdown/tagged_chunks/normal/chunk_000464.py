from langchain_core.documents import Document

chunk = Document(
    page_content=('- 7. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 상해 및 질병\n'
 '- 8. 반려묘를 범죄행위, 경주, 수색, 폭약탐지, 구조, 실험 및 이와 유사한 목적으로 이\n'
 '- 용함으로써 발생한 손해\n'
 '- 9. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관리에\n'
 '- 대한 태만\n'
 '- 10. 동물보호법 위반 등 동물학대에 기인하는 손해\n'
 '- 11. 대한민국 이외 지역에서 발생한 사고 및 손해\n'
 '- 12. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 치료행'),
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
 'indexing': {'chunk_id': 'chunk_000464',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
