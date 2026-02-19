from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보험증권에 기재된 반려동물이 개(犬)인 경우, 반려동물의 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전(대퇴 골두 허혈성 '
 '괴사 포함) 또는 기타 이들과 유 사한 질병 또는 상해 4. 보험증권에 기재된 반려동물이 고양이(猫)인 경우, 반려동물의 비뇨기계질환, '
 '전염성 복막염 또는 기타 이들과 유사한 질병 또는 상해 5. 상병명을 알 수 없는 상해 또는 질병에 대한 치료 6. 백신 접종비용 및 '
 '기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 정기검 진, 예방적 검사를 위한 비용 7'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 5},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['joint', 'urinary', 'other']},
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
