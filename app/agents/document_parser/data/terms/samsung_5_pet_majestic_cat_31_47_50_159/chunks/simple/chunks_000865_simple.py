from langchain_core.documents import Document

chunk = Document(
    page_content=('. 마. 위 라.에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의 효력이 없어진 경우에는 보험기간이 10년 이상인 '
 '계약은 상해 발생일 또는 질병 의 진단확정일부터 2년 이내로 하고, 보험기간이 10년 미만인 계약은 상해 발생 일 또는 질병의 '
 '진단확정일부터 1년 이내)에 장해상태가 더 악화된 때에는 그 악 화된 장해상태를 기준으로 장해지급률을 결정한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000865',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
