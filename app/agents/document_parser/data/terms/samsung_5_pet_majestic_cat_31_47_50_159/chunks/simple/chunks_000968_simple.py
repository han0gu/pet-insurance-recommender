from langchain_core.documents import Document

chunk = Document(
    page_content=('6) 흉복부, 비뇨생식기계 장해는 질병 또는 외상의 직접 결과로 인한 장해를 말하 며, 노화에 의한 기능장해 또는 질병이나 외상이 없는 '
 '상태에서 예방적으로 장 기를 절제, 적출한 경우는 장해로 보지 않는다. 7) 상기 흉복부 및 비뇨생식기계 장해항목에 명기되지 않은 기타 '
 '장해상태에 대해 서는 "<붙임> 일상생활 기본동작(ADLs) 제한 장해평가표" 에 해당하는 장해 가 있을 때 ADLs 장해 지급률을 '
 '준용한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['urinary', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000968',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
