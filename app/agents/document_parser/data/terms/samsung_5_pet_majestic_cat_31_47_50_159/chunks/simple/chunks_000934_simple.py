from langchain_core.documents import Document

chunk = Document(
    page_content=('11) "가관절주)이 남아 뚜렷한 장해를 남긴 때" 라 함은 상완골에 가관절이 남은 경우 또는 요골과 척골의 2개 뼈 모두에 가관절이 '
 '남은 경우를 말한다. 주) 가관절이란, 충분한 경과 및 골이식술 등 골유합을 얻는데 필요한 수술적 치료를 시행하였음에도 불구하고 골절부의 '
 "유합이 이루어지지 않는 '불유 합' 상태를 말하며, 골유합이 지연되는 지연유합은 제외한다."),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 143},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000934',
              'chunk_char_len': 202,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
