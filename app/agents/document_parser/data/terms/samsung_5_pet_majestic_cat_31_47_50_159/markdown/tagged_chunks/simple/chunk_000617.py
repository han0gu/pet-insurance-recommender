from langchain_core.documents import Document

chunk = Document(
    page_content=('- 대한 처방전을 발급할 수 있다. 이 경우 상시고용된 수의사의 범위, 신고방법, 처방전 발급 및\n'
 '- 보존 방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 사항은 농림축산식 품부령 으\n'
 '- 로 정한다.\n'
 '# 제6조 (보험금의 분담)① 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합\n'
 '니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상'),
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
 'indexing': {'chunk_id': 'chunk_000617',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
