from langchain_core.documents import Document

chunk = Document(
    page_content=('10) 말하는 기능의 장해는 1년 이상 지속적인 언어치료를 시행한 후 증상이 고착 되었을 때 평가하며, 객관적인 검사를 기초로 평가한다. '
 '11) 뇌·중추신경계 손상(정신·인지기능 저하, 편마비 등)으로 인한 말하는 기능의 장해(실어증, 구음장애) 또는 씹어먹는 기능의 장해는 '
 '신경계·정신행동 장해 평가와 비교하여 그 중 높은 지급률 하나만 인정한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 140},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['head', 'dental', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000899',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
